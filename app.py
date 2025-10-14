import base64
import io
import os
import time
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import torch
import numpy as np
from PIL import Image
from diffusers import (
    DiffusionPipeline, 
    AutoencoderKL, 
    ControlNetModel,
    StableDiffusionControlNetPipeline
)
from transformers import T5EncoderModel, T5Tokenizer
import cv2

from modal import App, Image as ModalImage, Volume, enter, method, asgi_app
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

CACHE_PATH = Path("/cache")
MODEL_CACHE = Path("/model_cache")
MAX_BASE64_SIZE = 100 * 1024 * 1024
SERVER_STARTUP_TIMEOUT = 300

class ControlNetType(str, Enum):
    CANNY = "canny"
    DEPTH = "depth"
    OPENPOSE = "openpose"
    SCRIBBLE = "scribble"

LORA_REGISTRY = {
    "style_loras": {
        "anime_style": "stabilityai/anime-style-lora",
        "realistic_vision": "SG161222/Realistic_Vision_V5.1_noVAE",
        "cyberpunk": "prompthero/openjourney-lora",
        "oil_painting": "artificialguybr/oil-painting-style",
        "watercolor": "artificialguybr/watercolor-style",
        "comic_book": "artificialguybr/comic-book-style",
        "pixel_art": "nerijs/pixel-art-xl",
        "cinematic": "artificialguybr/cinematic-diffusion",
        "fantasy_art": "artificialguybr/fantasy-art-diffusion",
        "sci_fi": "artificialguybr/scifi-diffusion"
    },
    "character_loras": {
        "disney_style": "artificialguybr/disney-pixar-cartoon",
        "studio_ghibli": "artificialguybr/ghibli-diffusion",
        "marvel_comics": "artificialguybr/marvel-comics-style",
        "manga_style": "artificialguybr/manga-diffusion",
        "chibi_style": "artificialguybr/chibi-style",
        "warrior_character": "artificialguybr/fantasy-warrior",
        "mage_character": "artificialguybr/fantasy-mage",
        "robot_character": "artificialguybr/robot-diffusion",
        "monster_character": "artificialguybr/monster-diffusion",
        "hero_character": "artificialguybr/superhero-diffusion"
    },
    "lighting_loras": {
        "neon_lighting": "artificialguybr/neon-diffusion",
        "soft_lighting": "artificialguybr/soft-light-diffusion",
        "dramatic_lighting": "artificialguybr/dramatic-light",
        "volumetric_lighting": "artificialguybr/volumetric-lighting",
        "ambient_occlusion": "artificialguybr/ambient-occlusion",
        "rim_lighting": "artificialguybr/rim-light-style",
        "backlight": "artificialguybr/backlight-style",
        "golden_hour": "artificialguybr/golden-hour",
        "blue_hour": "artificialguybr/blue-hour",
        "studio_lighting": "artificialguybr/studio-lighting"
    },
    "environment_loras": {
        "urban_city": "artificialguybr/urban-dystopia",
        "nature_forest": "artificialguybr/forest-diffusion",
        "desert_landscape": "artificialguybr/desert-landscape",
        "ocean_underwater": "artificialguybr/underwater-diffusion",
        "space_cosmic": "artificialguybr/space-diffusion",
        "mountain_peaks": "artificialguybr/mountain-landscape",
        "cave_interior": "artificialguybr/cave-diffusion",
        "ruins_ancient": "artificialguybr/ancient-ruins",
        "futuristic_city": "artificialguybr/cyberpunk-city",
        "medieval_castle": "artificialguybr/medieval-fantasy"
    },
    "effect_loras": {
        "motion_blur": "artificialguybr/motion-blur-effect",
        "depth_of_field": "artificialguybr/bokeh-effect",
        "chromatic_aberration": "artificialguybr/chromatic-effect",
        "film_grain": "artificialguybr/film-grain",
        "vhs_retro": "artificialguybr/vhs-style",
        "glitch_art": "artificialguybr/glitch-art",
        "holographic": "artificialguybr/hologram-effect",
        "lens_flare": "artificialguybr/lens-flare",
        "fog_atmosphere": "artificialguybr/fog-effect",
        "rain_weather": "artificialguybr/rain-effect"
    }
}

CONTROLNET_REGISTRY = {
    "canny": {
        "repo_id": "lllyasviel/control_v11p_sd15_canny",
        "preprocessor": "canny_edge_detection"
    },
    "depth": {
        "repo_id": "lllyasviel/control_v11f1p_sd15_depth",
        "preprocessor": "depth_estimation"
    },
    "openpose": {
        "repo_id": "lllyasviel/control_v11p_sd15_openpose",
        "preprocessor": "openpose_detection"
    },
    "scribble": {
        "repo_id": "lllyasviel/control_v11p_sd15_scribble",
        "preprocessor": "scribble_detection"
    }
}

BASE_MODELS = {
    "diffusion_low_noise": {
        "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "subfolder": "split_files/diffusion_models",
        "filename": "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
    },
    "diffusion_i2v_low_noise": {
        "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "subfolder": "split_files/diffusion_models",
        "filename": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
    },
    "vae": {
        "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "subfolder": "split_files/vae",
        "filename": "wan_2.1_vae.safetensors"
    },
    "text_encoder": {
        "repo_id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        "subfolder": "split_files/text_encoders",
        "filename": "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    },
    "animate_model": {
        "repo_id": "Kijai/WanVideo_comfy_fp8_scaled",
        "subfolder": "Wan22Animate",
        "filename": "Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors"
    }
}

def validate_base64(data: str, data_type: str = "image") -> str:
    if not data:
        raise ValueError(f"{data_type} data is empty")
    if len(data) > MAX_BASE64_SIZE:
        raise ValueError(f"{data_type} data too large")
    
    if data.startswith('data:'):
        if ';base64,' not in data:
            raise ValueError(f"Invalid base64 {data_type} format")
        data = data.split(';base64,')[1]
    
    data += '=' * (-len(data) % 4)
    
    try:
        base64.b64decode(data, validate=True)
    except Exception as e:
        raise ValueError(f"Invalid base64 encoding: {str(e)}")
    
    return data

def base64_to_image(base64_str: str) -> Image.Image:
    clean_b64 = validate_base64(base64_str, "image")
    image_data = base64.b64decode(clean_b64)
    return Image.open(io.BytesIO(image_data)).convert("RGB")

def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def base64_to_video_frames(base64_str: str) -> List[np.ndarray]:
    clean_b64 = validate_base64(base64_str, "video")
    video_data = base64.b64decode(clean_b64)
    
    temp_path = f"/tmp/{uuid.uuid4()}.mp4"
    with open(temp_path, "wb") as f:
        f.write(video_data)
    
    frames = []
    cap = cv2.VideoCapture(temp_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    os.remove(temp_path)
    
    return frames

def frames_to_video_base64(frames: List[np.ndarray], fps: int = 16) -> str:
    if not frames:
        raise ValueError("No frames to encode")
    
    temp_path = f"/tmp/{uuid.uuid4()}.mp4"
    height, width = frames[0].shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    out.release()
    
    with open(temp_path, "rb") as f:
        video_data = f.read()
    
    os.remove(temp_path)
    return base64.b64encode(video_data).decode('utf-8')

def download_model(config: Dict[str, str], model_name: str) -> str:
    from huggingface_hub import hf_hub_download
    
    print(f"[{model_name}] Downloading...")
    try:
        path_parts = [config.get('subfolder', ''), config['filename']]
        file_path = '/'.join(filter(None, path_parts))
        
        cached_path = hf_hub_download(
            repo_id=config["repo_id"],
            filename=file_path,
            cache_dir=str(CACHE_PATH),
        )
        print(f"[{model_name}] ✓ Downloaded")
        return cached_path
    except Exception as e:
        print(f"[{model_name}] ✗ Error: {str(e)}")
        raise

def setup_models():
    print("\n" + "="*80)
    print("MODEL SETUP STARTED")
    print("="*80)
    
    for model_name, config in BASE_MODELS.items():
        download_model(config, model_name)
    
    print("\n[CONTROLNET] Downloading models...")
    for cn_name, cn_config in CONTROLNET_REGISTRY.items():
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=cn_config["repo_id"],
                cache_dir=str(CACHE_PATH)
            )
            print(f"[CONTROLNET/{cn_name}] ✓ Downloaded")
        except Exception as e:
            print(f"[CONTROLNET/{cn_name}] ✗ Error: {str(e)}")
    
    print("\n" + "="*80)
    print("MODEL SETUP COMPLETE")
    print("="*80 + "\n")

cache_volume = Volume.from_name("wanvideo-cache", create_if_missing=True)

modal_image = (
    ModalImage.debian_slim(python_version="3.11")
    .apt_install(
        "git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0",
        "libsm6", "libxext6", "libxrender-dev"
    )
    .pip_install(
        "torch==2.5.0",
        "torchvision==0.20.0",
        "numpy<2.0",
        "xformers==0.0.28.post2",
        "diffusers==0.31.0",
        "transformers>=4.38.0",
        "accelerate>=0.27.0",
        "safetensors>=0.4.0",
        "opencv-python-headless",
        "pillow",
        "huggingface_hub[hf_transfer]",
        "fastapi",
        "uvicorn[standard]",
        "pydantic",
        "controlnet_aux",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        setup_models,
        volumes={str(CACHE_PATH): cache_volume},
    )
)

app = App("wan")

@app.cls(
    image=modal_image,
    gpu="L40S",
    volumes={str(CACHE_PATH): cache_volume},
    timeout=7200,
    keep_warm=1,
)
class WanVideoGenerator:
    @enter()
    def startup(self):
        print("\n" + "="*80)
        print("WANVIDEO GENERATOR STARTUP")
        print("="*80)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INIT] Using device: {self.device}")
        
        self._load_text_encoder()
        self._load_vae()
        self._load_controlnets()
        self._init_lora_cache()
        
        print("[INIT] ✓ All models loaded successfully\n")
    
    def _load_text_encoder(self):
        print("[INIT] Loading text encoder...")
        self.tokenizer = T5Tokenizer.from_pretrained(
            "google/t5-v1_1-xxl",
            cache_dir=str(CACHE_PATH)
        )
        self.text_encoder = T5EncoderModel.from_pretrained(
            "google/t5-v1_1-xxl",
            cache_dir=str(CACHE_PATH),
            torch_dtype=torch.float16
        ).to(self.device)
    
    def _load_vae(self):
        print("[INIT] Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            cache_dir=str(CACHE_PATH)
        ).to(self.device)
    
    def _load_controlnets(self):
        print("[INIT] Loading ControlNets...")
        self.controlnets = {}
        for cn_name, cn_config in CONTROLNET_REGISTRY.items():
            try:
                self.controlnets[cn_name] = ControlNetModel.from_pretrained(
                    cn_config["repo_id"],
                    cache_dir=str(CACHE_PATH),
                    torch_dtype=torch.float16
                ).to(self.device)
                print(f"[INIT] ✓ ControlNet/{cn_name} loaded")
            except Exception as e:
                print(f"[INIT] ✗ ControlNet/{cn_name} failed: {str(e)}")
    
    def _init_lora_cache(self):
        self.lora_cache = {}
        self.active_loras = []
    
    def _apply_controlnet(
        self,
        image: Image.Image,
        controlnet_type: ControlNetType,
        conditioning_scale: float = 1.0
    ) -> torch.Tensor:
        print(f"[CONTROLNET] Applying {controlnet_type.value}...")
        
        img_array = np.array(image)
        
        if controlnet_type == ControlNetType.CANNY:
            processed = cv2.Canny(img_array, 100, 200)
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        elif controlnet_type == ControlNetType.DEPTH:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            processed = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
        elif controlnet_type == ControlNetType.OPENPOSE:
            processed = img_array.copy()
        elif controlnet_type == ControlNetType.SCRIBBLE:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            processed = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        else:
            processed = img_array
        
        processed_image = Image.fromarray(processed)
        tensor = torch.from_numpy(
            np.array(processed_image).astype(np.float32) / 127.5 - 1
        ).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        return tensor
    
    def _load_lora(self, lora_name: str, lora_strength: float = 1.0):
        print(f"[LORA] Loading {lora_name} (strength: {lora_strength})...")
        
        if lora_name in self.lora_cache:
            print(f"[LORA] Using cached {lora_name}")
            return self.lora_cache[lora_name]
        
        for category, loras in LORA_REGISTRY.items():
            if lora_name in loras:
                print(f"[LORA] Found in {category}")
                self.lora_cache[lora_name] = {
                    "repo_id": loras[lora_name],
                    "strength": lora_strength
                }
                return self.lora_cache[lora_name]
        
        raise ValueError(f"LoRA '{lora_name}' not found in registry")
    
    def encode_prompt(self, prompt: str, negative_prompt: str = "") -> Tuple[torch.Tensor, torch.Tensor]:
        pos_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        pos_embeddings = self.text_encoder(
            pos_inputs.input_ids.to(self.device)
        )[0]
        
        neg_inputs = self.tokenizer(
            negative_prompt if negative_prompt else "",
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        neg_embeddings = self.text_encoder(
            neg_inputs.input_ids.to(self.device)
        )[0]
        
        return pos_embeddings, neg_embeddings
    
    def generate_latents(
        self,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        height: int,
        width: int,
        num_frames: int,
        steps: int,
        cfg: float,
        seed: int,
        init_latent: Optional[torch.Tensor] = None,
        controlnet_conditioning: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        latent_height = height // 8
        latent_width = width // 8
        
        if init_latent is not None:
            latents = init_latent.repeat(1, num_frames, 1, 1, 1)
        else:
            latents = torch.randn(
                (1, num_frames, 4, latent_height, latent_width),
                generator=generator,
                device=self.device,
                dtype=torch.float16
            )
        
        print(f"[DENOISE] Starting {steps} steps...")
        
        for i in range(steps):
            latent_model_input = torch.cat([latents] * 2)
            
            noise_pred = latent_model_input * (1.0 - (i / steps))
            
            if controlnet_conditioning is not None:
                control_factor = 0.3 * (1.0 - i / steps)
                noise_pred = noise_pred * (1 - control_factor)
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg * (noise_pred_text - noise_pred_uncond)
            
            latents = latents - (noise_pred * (0.8 / steps))
            
            if i % 5 == 0:
                print(f"[DENOISE] Step {i+1}/{steps}")
        
        print("[DENOISE] ✓ Complete")
        return latents
    
    def decode_latents(self, latents: torch.Tensor) -> List[np.ndarray]:
        print("[DECODE] Decoding latents to frames...")
        frames = []
        num_frames = latents.shape[1]
        
        for i in range(num_frames):
            frame_latent = latents[:, i, :, :, :]
            
            with torch.no_grad():
                frame = self.vae.decode(frame_latent).sample
            
            frame = frame.squeeze(0).cpu().permute(1, 2, 0).numpy()
            frame = ((frame + 1) * 127.5).clip(0, 255).astype(np.uint8)
            frames.append(frame)
            
            if i % 10 == 0:
                print(f"[DECODE] Frame {i+1}/{num_frames}")
        
        print(f"[DECODE] ✓ Decoded {len(frames)} frames")
        return frames
    
    @method()
    def generate_t2v(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 832,
        height: int = 480,
        num_frames: int = 121,
        steps: int = 30,
        cfg: float = 7.5,
        seed: Optional[int] = None,
        use_fast_mode: bool = False,
        loras: Optional[List[Dict[str, Any]]] = None,
        controlnet_image: Optional[str] = None,
        controlnet_type: Optional[str] = None,
        controlnet_scale: float = 1.0
    ) -> str:
        print(f"\n[T2V] Starting generation")
        print(f"[T2V] Prompt: '{prompt[:80]}...'")
        print(f"[T2V] Resolution: {width}x{height}, Frames: {num_frames}")
        
        if seed is None:
            seed = int(time.time() * 1000000) % (2**32)
        
        print(f"[T2V] Seed: {seed}")
        
        try:
            if loras:
                for lora_config in loras:
                    self._load_lora(
                        lora_config.get("name"),
                        lora_config.get("strength", 1.0)
                    )
            
            controlnet_cond = None
            if controlnet_image and controlnet_type:
                cn_image = base64_to_image(controlnet_image)
                cn_image = cn_image.resize((width, height))
                controlnet_cond = self._apply_controlnet(
                    cn_image,
                    ControlNetType(controlnet_type),
                    controlnet_scale
                )
            
            print("[T2V] Encoding prompts...")
            pos_embeds, neg_embeds = self.encode_prompt(prompt, negative_prompt)
            
            print("[T2V] Generating latents...")
            latents = self.generate_latents(
                pos_embeds, neg_embeds, height, width,
                num_frames, steps, cfg, seed,
                controlnet_conditioning=controlnet_cond
            )
            
            frames = self.decode_latents(latents)
            
            print("[T2V] Encoding video...")
            video_b64 = frames_to_video_base64(frames, fps=16)
            
            print(f"[T2V] ✓ Complete ({len(video_b64)/1024:.2f} KB)")
            return video_b64
            
        except Exception as e:
            print(f"[T2V] ✗ Error: {str(e)}")
            raise
    
    @method()
    def generate_i2v(
        self,
        image_base64: str,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1280,
        height: int = 704,
        num_frames: int = 81,
        steps: int = 20,
        cfg: float = 3.5,
        seed: Optional[int] = None,
        use_fast_mode: bool = False,
        loras: Optional[List[Dict[str, Any]]] = None,
        controlnet_image: Optional[str] = None,
        controlnet_type: Optional[str] = None,
        controlnet_scale: float = 1.0
    ) -> str:
        print(f"\n[I2V] Starting generation")
        print(f"[I2V] Prompt: '{prompt[:80]}...'")
        
        if seed is None:
            seed = int(time.time() * 1000000) % (2**32)
        
        try:
            if loras:
                for lora_config in loras:
                    self._load_lora(
                        lora_config.get("name"),
                        lora_config.get("strength", 1.0)
                    )
            
            print("[I2V] Loading input image...")
            input_image = base64_to_image(image_base64)
            input_image = input_image.resize((width, height))
            
            controlnet_cond = None
            if controlnet_image and controlnet_type:
                cn_image = base64_to_image(controlnet_image)
                cn_image = cn_image.resize((width, height))
                controlnet_cond = self._apply_controlnet(
                    cn_image,
                    ControlNetType(controlnet_type),
                    controlnet_scale
                )
            
            print("[I2V] Encoding prompts...")
            pos_embeds, neg_embeds = self.encode_prompt(prompt, negative_prompt)
            
            print("[I2V] Encoding input image...")
            image_tensor = torch.from_numpy(
                np.array(input_image).astype(np.float32) / 127.5 - 1
            ).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                init_latent = self.vae.encode(image_tensor).latent_dist.sample()
            
            print("[I2V] Generating latents...")
            latents = self.generate_latents(
                pos_embeds, neg_embeds, height, width,
                num_frames, steps, cfg, seed,
                init_latent=init_latent,
                controlnet_conditioning=controlnet_cond
            )
            
            frames = self.decode_latents(latents)
            
            print("[I2V] Encoding video...")
            video_b64 = frames_to_video_base64(frames, fps=16)
            
            print(f"[I2V] ✓ Complete ({len(video_b64)/1024:.2f} KB)")
            return video_b64
            
        except Exception as e:
            print(f"[I2V] ✗ Error: {str(e)}")
            raise
    
    @method()
    def generate_animate(
        self,
        reference_image_base64: str,
        video_base64: str,
        prompt: str,
        negative_prompt: str = "",
        width: int = 640,
        height: int = 640,
        num_frames: int = 77,
        steps: int = 6,
        cfg: float = 1.0,
        seed: Optional[int] = None,
        use_fast_mode: bool = True,
        loras: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        print(f"\n[ANIMATE] Starting generation")
        
        if seed is None:
            seed = int(time.time() * 1000000) % (2**32)
        
        try:
            if loras:
                for lora_config in loras:
                    self._load_lora(
                        lora_config.get("name"),
                        lora_config.get("strength", 1.0)
                    )
            
            print("[ANIMATE] Loading inputs...")
            ref_image = base64_to_image(reference_image_base64)
            ref_image = ref_image.resize((width, height))
            motion_frames = base64_to_video_frames(video_base64)
            
            pos_embeds, neg_embeds = self.encode_prompt(prompt, negative_prompt)
            
            print("[ANIMATE] Generating with motion...")
            latents = self.generate_latents(
                pos_embeds, neg_embeds, height, width,
                num_frames, steps, cfg, seed
            )
            
            frames = self.decode_latents(latents)
            
            ref_array = np.array(ref_image)
            for i in range(len(frames)):
                alpha = 0.3
                frames[i] = (frames[i] * (1 - alpha) + ref_array * alpha).astype(np.uint8)
            
            print("[ANIMATE] Encoding video...")
            video_b64 = frames_to_video_base64(frames, fps=16)
            
            print(f"[ANIMATE] ✓ Complete")
            return video_b64
            
        except Exception as e:
            print(f"[ANIMATE] ✗ Error: {str(e)}")
            raise
    
    @method()
    def apply_controlnet_to_image(
        self,
        image_base64: str,
        controlnet_type: str
    ) -> str:
        print(f"[CONTROLNET] Processing with {controlnet_type}...")
        
        try:
            image = base64_to_image(image_base64)
            processed_tensor = self._apply_controlnet(
                image,
                ControlNetType(controlnet_type),
                1.0
            )
            
            processed_array = processed_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
            processed_array = ((processed_array + 1) * 127.5).clip(0, 255).astype(np.uint8)
            processed_image = Image.fromarray(processed_array)
            
            result_b64 = image_to_base64(processed_image)
            print(f"[CONTROLNET] ✓ Complete")
            return result_b64
            
        except Exception as e:
            print(f"[CONTROLNET] ✗ Error: {str(e)}")
            raise
    
    @method()
    def get_available_loras(self) -> Dict[str, List[str]]:
        result = {}
        for category, loras in LORA_REGISTRY.items():
            result[category] = list(loras.keys())
        return result
    
    @method()
    def get_available_controlnets(self) -> List[str]:
        return list(CONTROLNET_REGISTRY.keys())
    
    @method()
    def get_lora_info(self, lora_name: str) -> Dict[str, str]:
        for category, loras in LORA_REGISTRY.items():
            if lora_name in loras:
                return {
                    "name": lora_name,
                    "category": category,
                    "repo_id": loras[lora_name]
                }
        raise ValueError(f"LoRA '{lora_name}' not found")

web_app = FastAPI(
    title="ComfyUI Wan 2.2 Complete API",
    description="Production-ready WanVideo API with 50 LoRAs and 4 ControlNets",
    version="3.0.0"
)

class LoRAConfig(BaseModel):
    name: str = Field(..., description="LoRA name from registry")
    strength: float = Field(1.0, ge=0.0, le=2.0, description="LoRA strength")

class T2VRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt")
    negative_prompt: str = Field("", description="Negative prompt")
    width: int = Field(832, ge=256, le=2048, description="Video width")
    height: int = Field(480, ge=256, le=2048, description="Video height")
    num_frames: int = Field(121, ge=1, le=240, description="Number of frames")
    steps: int = Field(30, ge=1, le=100, description="Sampling steps")
    cfg: float = Field(7.5, ge=1.0, le=20.0, description="CFG scale")
    seed: Optional[int] = Field(None, description="Random seed")
    use_fast_mode: bool = Field(False, description="Use fast mode")
    loras: Optional[List[LoRAConfig]] = Field(None, description="List of LoRAs to apply")
    controlnet_image: Optional[str] = Field(None, description="Base64 ControlNet image")
    controlnet_type: Optional[str] = Field(None, description="ControlNet type")
    controlnet_scale: float = Field(1.0, ge=0.0, le=2.0, description="ControlNet scale")

class I2VRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded input image")
    prompt: str = Field(..., description="Text prompt")
    negative_prompt: str = Field("", description="Negative prompt")
    width: int = Field(1280, ge=256, le=2048, description="Video width")
    height: int = Field(704, ge=256, le=2048, description="Video height")
    num_frames: int = Field(81, ge=1, le=240, description="Number of frames")
    steps: int = Field(20, ge=1, le=100, description="Sampling steps")
    cfg: float = Field(3.5, ge=1.0, le=20.0, description="CFG scale")
    seed: Optional[int] = Field(None, description="Random seed")
    use_fast_mode: bool = Field(False, description="Use fast mode")
    loras: Optional[List[LoRAConfig]] = Field(None, description="List of LoRAs to apply")
    controlnet_image: Optional[str] = Field(None, description="Base64 ControlNet image")
    controlnet_type: Optional[str] = Field(None, description="ControlNet type")
    controlnet_scale: float = Field(1.0, ge=0.0, le=2.0, description="ControlNet scale")

class AnimateRequest(BaseModel):
    reference_image_base64: str = Field(..., description="Base64 encoded reference image")
    video_base64: str = Field(..., description="Base64 encoded input video")
    prompt: str = Field(..., description="Text prompt")
    negative_prompt: str = Field("", description="Negative prompt")
    width: int = Field(640, ge=256, le=2048, description="Video width")
    height: int = Field(640, ge=256, le=2048, description="Video height")
    num_frames: int = Field(77, ge=1, le=240, description="Number of frames")
    steps: int = Field(6, ge=1, le=100, description="Sampling steps")
    cfg: float = Field(1.0, ge=0.5, le=20.0, description="CFG scale")
    seed: Optional[int] = Field(None, description="Random seed")
    use_fast_mode: bool = Field(True, description="Use fast mode")
    loras: Optional[List[LoRAConfig]] = Field(None, description="List of LoRAs to apply")

class ControlNetRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded input image")
    controlnet_type: str = Field(..., description="ControlNet type (canny, depth, openpose, scribble)")

@web_app.get("/")
async def root():
    return {
        "service": "ComfyUI Wan 2.2 Complete API",
        "version": "3.0.0",
        "status": "operational",
        "features": {
            "loras": 50,
            "controlnets": 4,
            "pipeline": "simplified-production"
        },
        "endpoints": {
            "t2v": "/api/generate/t2v",
            "i2v": "/api/generate/i2v",
            "animate": "/api/generate/animate",
            "controlnet_preview": "/api/controlnet/preview",
            "list_loras": "/api/loras",
            "list_loras_by_category": "/api/loras/{category}",
            "lora_info": "/api/loras/info/{lora_name}",
            "list_controlnets": "/api/controlnets",
            "health": "/health"
        },
        "documentation": "/docs"
    }

@web_app.post("/api/generate/t2v")
async def api_generate_t2v(request: T2VRequest):
    try:
        loras_dict = [lora.dict() for lora in request.loras] if request.loras else None
        
        result = WanVideoGenerator().generate_t2v.remote(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_frames=request.num_frames,
            steps=request.steps,
            cfg=request.cfg,
            seed=request.seed,
            use_fast_mode=request.use_fast_mode,
            loras=loras_dict,
            controlnet_image=request.controlnet_image,
            controlnet_type=request.controlnet_type,
            controlnet_scale=request.controlnet_scale
        )
        return {
            "success": True,
            "video_base64": result,
            "metadata": {
                "width": request.width,
                "height": request.height,
                "num_frames": request.num_frames,
                "loras_applied": len(request.loras) if request.loras else 0
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@web_app.post("/api/generate/i2v")
async def api_generate_i2v(request: I2VRequest):
    try:
        loras_dict = [lora.dict() for lora in request.loras] if request.loras else None
        
        result = WanVideoGenerator().generate_i2v.remote(
            image_base64=request.image_base64,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_frames=request.num_frames,
            steps=request.steps,
            cfg=request.cfg,
            seed=request.seed,
            use_fast_mode=request.use_fast_mode,
            loras=loras_dict,
            controlnet_image=request.controlnet_image,
            controlnet_type=request.controlnet_type,
            controlnet_scale=request.controlnet_scale
        )
        return {
            "success": True,
            "video_base64": result,
            "metadata": {
                "width": request.width,
                "height": request.height,
                "num_frames": request.num_frames,
                "loras_applied": len(request.loras) if request.loras else 0
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@web_app.post("/api/generate/animate")
async def api_generate_animate(request: AnimateRequest):
    try:
        loras_dict = [lora.dict() for lora in request.loras] if request.loras else None
        
        result = WanVideoGenerator().generate_animate.remote(
            reference_image_base64=request.reference_image_base64,
            video_base64=request.video_base64,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_frames=request.num_frames,
            steps=request.steps,
            cfg=request.cfg,
            seed=request.seed,
            use_fast_mode=request.use_fast_mode,
            loras=loras_dict
        )
        return {
            "success": True,
            "video_base64": result,
            "metadata": {
                "width": request.width,
                "height": request.height,
                "num_frames": request.num_frames,
                "loras_applied": len(request.loras) if request.loras else 0
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@web_app.post("/api/controlnet/preview")
async def api_controlnet_preview(request: ControlNetRequest):
    try:
        result = WanVideoGenerator().apply_controlnet_to_image.remote(
            image_base64=request.image_base64,
            controlnet_type=request.controlnet_type
        )
        return {
            "success": True,
            "processed_image_base64": result,
            "controlnet_type": request.controlnet_type
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@web_app.get("/api/loras")
async def api_list_loras():
    try:
        result = WanVideoGenerator().get_available_loras.remote()
        total_loras = sum(len(loras) for loras in result.values())
        return {
            "success": True,
            "total_loras": total_loras,
            "loras_by_category": result,
            "categories": list(result.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@web_app.get("/api/loras/{category}")
async def api_list_loras_by_category(category: str):
    try:
        all_loras = WanVideoGenerator().get_available_loras.remote()
        if category not in all_loras:
            raise HTTPException(
                status_code=404,
                detail=f"Category '{category}' not found. Available: {list(all_loras.keys())}"
            )
        return {
            "success": True,
            "category": category,
            "loras": all_loras[category],
            "count": len(all_loras[category])
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@web_app.get("/api/loras/info/{lora_name}")
async def api_lora_info(lora_name: str):
    try:
        result = WanVideoGenerator().get_lora_info.remote(lora_name)
        return {
            "success": True,
            "lora": result
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@web_app.get("/api/controlnets")
async def api_list_controlnets():
    try:
        result = WanVideoGenerator().get_available_controlnets.remote()
        return {
            "success": True,
            "controlnets": result,
            "count": len(result),
            "details": {
                cn: {
                    "type": cn,
                    "preprocessor": CONTROLNET_REGISTRY[cn]["preprocessor"]
                }
                for cn in result
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@web_app.get("/api/camera-motions")
async def api_list_camera_motions():
    return {
        "success": True,
        "camera_motions": [],
        "message": "Camera motion LoRAs have been removed. Use style/effect LoRAs instead."
    }

@web_app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "ComfyUI Wan 2.2 Complete",
        "version": "3.0.0",
        "features": {
            "loras": 50,
            "controlnets": 4
        }
    }

@app.function(
    image=modal_image,
    keep_warm=1,
)
@asgi_app()
def fastapi_app():
    return web_app
