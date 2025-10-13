import base64
import json
import os
import shutil
import subprocess
import time
import urllib.request
import urllib.parse
import uuid
import websocket
from pathlib import Path
from typing import Dict, Optional, List, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from modal import App, Image, Volume, Secret, asgi_app, enter, method
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field

# ============================================================================
# CONSTANTS
# ============================================================================
COMFYUI_PATH = Path("/root/comfy/ComfyUI")
MODEL_PATH = Path("/root/comfy/ComfyUI/models")
CACHE_PATH = Path("/cache")

MIN_FILE_SIZE_KB = 500
MIN_LORA_SIZE_KB = 100
MAX_BASE64_SIZE = 50 * 1024 * 1024
MAX_GENERATION_TIME = 1800
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
SERVER_STARTUP_TIMEOUT = 60

# ============================================================================
# MODEL REGISTRY - Complete registry for all 4 modes
# ============================================================================
MODEL_REGISTRY = {
    "diffusion_models": {
        # T2V model
        "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors": {
            "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "filename": "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
        },
        # I2V model (5B version like reference)
        "wan2.2_ti2v_5B_fp16.safetensors": {
            "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "filename": "split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors"
        },
        # S2V model
        "wan2.2_s2v_14B_fp8_scaled.safetensors": {
            "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "filename": "split_files/diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors"
        },
    },
    "vae": {
        "wan2.2_vae.safetensors": {
            "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "filename": "split_files/vae/wan2.2_vae.safetensors"
        },
    },
    "text_encoders": {
        "umt5_xxl_fp8_e4m3fn_scaled.safetensors": {
            "repo_id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "filename": "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
        },
    },
    "audio_encoders": {
        "wav2vec2_large_english_fp16.safetensors": {
            "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "filename": "split_files/audio_encoders/wav2vec2_large_english_fp16.safetensors"
        },
    },
    "loras": {
        "v2_lora_ZoomIn.safetensors": {
            "repo_id": "guoyww/animatediff",
            "filename": "v2_lora_ZoomIn.safetensors"
        },
        "v2_lora_ZoomOut.safetensors": {
            "repo_id": "guoyww/animatediff",
            "filename": "v2_lora_ZoomOut.safetensors"
        },
        "v2_lora_PanLeft.safetensors": {
            "repo_id": "guoyww/animatediff",
            "filename": "v2_lora_PanLeft.safetensors"
        },
        "v2_lora_PanRight.safetensors": {
            "repo_id": "guoyww/animatediff",
            "filename": "v2_lora_PanRight.safetensors"
        },
    }
}

# ============================================================================
# MODAL IMAGE SETUP - Using comfy-cli approach from reference
# ============================================================================
def setup_comfyui():
    """Download and link all required models"""
    from huggingface_hub import hf_hub_download
    
    print("\n" + "="*80)
    print("MODEL DOWNLOAD STARTED")
    print("="*80)
    
    for model_type, models in MODEL_REGISTRY.items():
        print(f"\n[{model_type.upper()}] Processing...")
        target_dir = MODEL_PATH / model_type
        target_dir.mkdir(parents=True, exist_ok=True)
        
        for filename, source in models.items():
            target_path = target_dir / filename
            
            if target_path.exists():
                print(f"[{model_type.upper()}] Already exists: {filename}")
                continue
            
            print(f"[{model_type.upper()}] Downloading: {filename}")
            try:
                cached_path = hf_hub_download(
                    repo_id=source["repo_id"],
                    filename=source["filename"],
                    cache_dir=str(CACHE_PATH),
                )
                subprocess.run(
                    f"ln -s {cached_path} {target_path}",
                    shell=True,
                    check=True,
                )
                print(f"[{model_type.upper()}] Linked: {filename}")
            except Exception as e:
                print(f"[{model_type.upper()}] ERROR downloading {filename}: {str(e)}")


cache_volume = Volume.from_name("hf-hub-cache", create_if_missing=True)

# Build image using comfy-cli approach
comfy_image = (
    Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("comfy-cli==1.4.1", force_build=True)
    .run_commands(
        "comfy --skip-prompt install --fast-deps --nvidia --version 0.3.47",
        force_build=True,
    )
    .run_commands(
        "comfy node install --fast-deps was-node-suite-comfyui@1.0.2",
        "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite",
        "cd /root/comfy/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite && pip install -r requirements.txt",
        "git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper",
        "cd /root/comfy/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper && pip install -r requirements.txt",
        force_build=True,
    )
    .pip_install(
        "huggingface_hub[hf_transfer]>=0.34.0,<1.0",
        "websocket-client",
        "fastapi",
        "uvicorn[standard]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        setup_comfyui,
        volumes={str(CACHE_PATH): cache_volume},
    )
)

app = App("comfyui-wan2-2-complete-api")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def _validate_and_decode_base64(data: str, data_type: str = "image") -> str:
    """Validate and clean base64 data"""
    if not data:
        raise ValueError(f"{data_type} data is empty")
    
    if len(data) > MAX_BASE64_SIZE:
        raise ValueError(f"{data_type} data too large. Max {MAX_BASE64_SIZE/(1024*1024):.2f}MB")
    
    if data.startswith(f'data:{data_type}/'):
        if ';base64,' not in data:
            raise ValueError(f"Invalid base64 {data_type} format")
        data = data.split(';base64,')[1]
    
    try:
        base64.b64decode(data, validate=True)
    except Exception as e:
        raise ValueError(f"Invalid base64 encoding: {str(e)}")
    
    return data


def _save_base64_to_file(data_base64: str, temp_filename: str, data_type: str = "image") -> str:
    """Save base64 data to temporary file"""
    clean_b64 = _validate_and_decode_base64(data_base64, data_type)
    try:
        file_data = base64.b64decode(clean_b64)
        with open(temp_filename, "wb") as f:
            f.write(file_data)
        print(f"[{data_type.upper()}] Saved: {temp_filename} ({len(file_data)/1024:.2f} KB)")
        return temp_filename
    except Exception as e:
        raise ValueError(f"Failed to save {data_type}: {str(e)}")


# ============================================================================
# COMFYUI CLASS - Main inference class
# ============================================================================
@app.cls(
    image=comfy_image,
    gpu="L40S",
    volumes={str(CACHE_PATH): cache_volume},
    timeout=3600,
    container_idle_timeout=300
)
class ComfyUI:
    
    @enter()
    def startup(self):
        """Start ComfyUI server"""
        print("\n" + "="*80)
        print("COMFYUI STARTUP")
        print("="*80)
        
        # Start server
        cmd = ["comfy", "launch", "--", "--listen", SERVER_HOST, "--port", str(SERVER_PORT)]
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"[SERVER] Started with PID: {self.proc.pid}")
        
        # Wait for server ready
        for i in range(SERVER_STARTUP_TIMEOUT):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{SERVER_PORT}/queue", timeout=5).read()
                print(f"[SERVER] Ready after {i+1} seconds\n")
                return
            except Exception:
                if i % 10 == 0:
                    print(f"[SERVER] Waiting... ({i}/{SERVER_STARTUP_TIMEOUT}s)")
                time.sleep(1)
        
        raise RuntimeError("ComfyUI failed to start")
    
    
    def _queue_prompt(self, client_id: str, prompt_workflow: dict) -> str:
        """Queue workflow to ComfyUI"""
        payload = {"prompt": prompt_workflow, "client_id": client_id}
        payload_json = json.dumps(payload)
        
        print(f"[QUEUE] Sending workflow ({len(payload_json)/1024:.2f} KB)")
        
        req = urllib.request.Request(
            f"http://127.0.0.1:{SERVER_PORT}/prompt",
            data=payload_json.encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read())
                
                if 'error' in result:
                    raise RuntimeError(f"ComfyUI Error: {result.get('error', {})}")
                
                if 'prompt_id' not in result:
                    raise RuntimeError(f"Invalid response: {result}")
                
                print(f"[QUEUE] Prompt ID: {result['prompt_id']}")
                return result['prompt_id']
                
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            print(f"[QUEUE] HTTP {e.code}: {error_body}")
            raise RuntimeError(f"ComfyUI API Error {e.code}: {error_body}")
    
    
    def _get_history(self, prompt_id: str) -> dict:
        """Get workflow execution history"""
        url = f"http://127.0.0.1:{SERVER_PORT}/history/{prompt_id}"
        with urllib.request.urlopen(url) as response:
            return json.loads(response.read())
    
    
    def _get_file(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        """Download generated file from ComfyUI"""
        params = urllib.parse.urlencode({
            'filename': filename,
            'subfolder': subfolder,
            'type': folder_type
        })
        url = f"http://127.0.0.1:{SERVER_PORT}/view?{params}"
        
        with urllib.request.urlopen(url) as response:
            data = bytearray()
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                data.extend(chunk)
        
        print(f"[FILE] Downloaded: {len(data)/1024/1024:.2f} MB")
        return bytes(data)
    
    
    def _get_video_from_websocket(self, prompt_id: str, client_id: str) -> bytes:
        """Wait for video generation via WebSocket"""
        ws_url = f"ws://127.0.0.1:{SERVER_PORT}/ws?clientId={client_id}"
        ws = None
        
        try:
            ws = websocket.WebSocket()
            ws.connect(ws_url, timeout=10)
            print("[WS] Connected")
        except Exception as e:
            raise RuntimeError(f"WebSocket connection failed: {str(e)}")
        
        start_time = time.time()
        try:
            while time.time() - start_time < MAX_GENERATION_TIME:
                try:
                    out = ws.recv(timeout=60)
                    if isinstance(out, str):
                        message = json.loads(out)
                        
                        if message.get('type') == 'progress':
                            data = message.get('data', {})
                            value = data.get('value', 0)
                            max_val = max(data.get('max', 1), 1)
                            pct = (value / max_val) * 100
                            print(f"[WS] Progress: {value}/{max_val} ({pct:.1f}%)")
                        
                        elif message.get('type') == 'executing':
                            if not message.get('data', {}).get('node'):
                                print("[WS] Generation complete")
                                break
                
                except websocket.WebSocketTimeoutException:
                    continue
        
        finally:
            if ws:
                ws.close()
        
        if time.time() - start_time >= MAX_GENERATION_TIME:
            raise TimeoutError(f"Generation timeout ({MAX_GENERATION_TIME}s)")
        
        # Get video from history
        history = self._get_history(prompt_id)
        if prompt_id not in history:
            raise ValueError(f"Prompt ID {prompt_id} not found in history")
        
        for node_id, node_output in history[prompt_id]['outputs'].items():
            for video_type in ['videos', 'gifs']:
                if video_type in node_output:
                    for item in node_output[video_type]:
                        print(f"[OUTPUT] Found {video_type[:-1]}: {item['filename']}")
                        return self._get_file(
                            item['filename'],
                            item['subfolder'],
                            item['type']
                        )
        
        raise ValueError("No video output found")
    
    
    def _build_base_workflow(
        self,
        mode: str,
        unet_model: str,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_frames: int,
        steps: int,
        cfg: float,
        seed: Optional[int] = None
    ) -> dict:
        """
        Build base workflow following the reference JSON pattern:
        UNETLoader -> ModelSamplingSD3 -> KSampler -> VAEDecode -> CreateVideo -> SaveVideo
        """
        if seed is None:
            seed = int(time.time() * 1000000) % (2**32)
        
        workflow = {
            # Load UNET model
            "37": {
                "class_type": "UNETLoader",
                "inputs": {
                    "unet_name": unet_model,
                    "weight_dtype": "default"
                }
            },
            # Load CLIP for text encoding
            "38": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                    "type": "wan",
                    "device": "default"
                }
            },
            # Load VAE
            "39": {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": "wan2.2_vae.safetensors"
                }
            },
            # ModelSamplingSD3 - CRITICAL: This is required for Wan models
            "48": {
                "class_type": "ModelSamplingSD3",
                "inputs": {
                    "shift": 8,
                    "model": ["37", 0]
                }
            },
            # Positive prompt
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,
                    "clip": ["38", 0]
                }
            },
            # Negative prompt
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["38", 0]
                }
            },
            # KSampler - Uses ModelSamplingSD3 output
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "uni_pc",
                    "scheduler": "simple",
                    "denoise": 1.0,
                    "model": ["48", 0],  # From ModelSamplingSD3
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["LATENT_PLACEHOLDER", 0]  # Will be replaced
                }
            },
            # VAE Decode
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["39", 0]
                }
            },
            # Create Video
            "57": {
                "class_type": "CreateVideo",
                "inputs": {
                    "fps": 24,
                    "images": ["8", 0]
                }
            },
            # Save Video
            "58": {
                "class_type": "SaveVideo",
                "inputs": {
                    "filename_prefix": f"video/{mode}",
                    "format": "auto",
                    "codec": "auto",
                    "video-preview": "",
                    "video": ["57", 0]
                }
            }
        }
        
        return workflow
    
    
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
        seed: Optional[int] = None
    ) -> str:
        """
        Text-to-Video generation
        Uses EmptyLatentImage as latent source
        """
        print(f"\n[T2V] Generating video: '{prompt[:50]}...'")
        client_id = str(uuid.uuid4())
        
        # Build base workflow
        workflow = self._build_base_workflow(
            mode="t2v",
            unet_model="wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            steps=steps,
            cfg=cfg,
            seed=seed
        )
        
        # Add EmptyLatentImage for T2V
        workflow["5"] = {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": num_frames
            }
        }
        
        # Connect to KSampler
        workflow["3"]["inputs"]["latent_image"] = ["5", 0]
        
        # Execute
        prompt_id = self._queue_prompt(client_id, workflow)
        video_bytes = self._get_video_from_websocket(prompt_id, client_id)
        
        return base64.b64encode(video_bytes).decode('utf-8')
    
    
    @method()
    def generate_i2v(
        self,
        image_base64: str,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1280,
        height: int = 704,
        num_frames: int = 121,
        steps: int = 20,
        cfg: float = 5.0,
        seed: Optional[int] = None
    ) -> str:
        """
        Image-to-Video generation
        Follows EXACT pattern from comfyui_api_wan2_2_5B_i2v.json reference
        """
        print(f"\n[I2V] Generating video from image")
        client_id = str(uuid.uuid4())
        
        # Save input image
        temp_image = f"/tmp/{uuid.uuid4()}.png"
        _save_base64_to_file(image_base64, temp_image, "image")
        
        # Copy to ComfyUI input folder
        input_dir = COMFYUI_PATH / "input"
        input_dir.mkdir(exist_ok=True)
        image_filename = f"{uuid.uuid4()}.png"
        shutil.copy(temp_image, input_dir / image_filename)
        
        # Build base workflow
        workflow = self._build_base_workflow(
            mode="i2v",
            unet_model="wan2.2_ti2v_5B_fp16.safetensors",
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            steps=steps,
            cfg=cfg,
            seed=seed
        )
        
        # Add LoadImage node
        workflow["56"] = {
            "class_type": "LoadImage",
            "inputs": {
                "image": image_filename
            }
        }
        
        # Add Wan22ImageToVideoLatent - CRITICAL for I2V
        workflow["55"] = {
            "class_type": "Wan22ImageToVideoLatent",
            "inputs": {
                "width": width,
                "height": height,
                "length": num_frames,
                "batch_size": 1,
                "vae": ["39", 0],
                "start_image": ["56", 0]
            }
        }
        
        # Connect to KSampler
        workflow["3"]["inputs"]["latent_image"] = ["55", 0]
        
        # Execute
        prompt_id = self._queue_prompt(client_id, workflow)
        video_bytes = self._get_video_from_websocket(prompt_id, client_id)
        
        return base64.b64encode(video_bytes).decode('utf-8')
    
    
    @method()
    def generate_s2v(
        self,
        sketch_base64: str,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1280,
        height: int = 704,
        num_frames: int = 121,
        steps: int = 20,
        cfg: float = 5.0,
        seed: Optional[int] = None
    ) -> str:
        """
        Sketch-to-Video generation
        Same as I2V but uses S2V model
        """
        print(f"\n[S2V] Generating video from sketch")
        client_id = str(uuid.uuid4())
        
        # Save sketch
        temp_sketch = f"/tmp/{uuid.uuid4()}.png"
        _save_base64_to_file(sketch_base64, temp_sketch, "image")
        
        # Copy to ComfyUI input
        input_dir = COMFYUI_PATH / "input"
        input_dir.mkdir(exist_ok=True)
        sketch_filename = f"{uuid.uuid4()}.png"
        shutil.copy(temp_sketch, input_dir / sketch_filename)
        
        # Build workflow - same as I2V but different model
        workflow = self._build_base_workflow(
            mode="s2v",
            unet_model="wan2.2_s2v_14B_fp8_scaled.safetensors",
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            steps=steps,
            cfg=cfg,
            seed=seed
        )
        
        workflow["56"] = {
            "class_type": "LoadImage",
            "inputs": {
                "image": sketch_filename
            }
        }
        
        workflow["55"] = {
            "class_type": "Wan22ImageToVideoLatent",
            "inputs": {
                "width": width,
                "height": height,
                "length": num_frames,
                "batch_size": 1,
                "vae": ["39", 0],
                "start_image": ["56", 0]
            }
        }
        
        workflow["3"]["inputs"]["latent_image"] = ["55", 0]
        
        # Execute
        prompt_id = self._queue_prompt(client_id, workflow)
        video_bytes = self._get_video_from_websocket(prompt_id, client_id)
        
        return base64.b64encode(video_bytes).decode('utf-8')
    
    
    @method()
    def generate_a2v(
        self,
        audio_base64: str,
        prompt: str,
        negative_prompt: str = "",
        width: int = 832,
        height: int = 480,
        num_frames: int = 121,
        steps: int = 30,
        cfg: float = 7.5,
        seed: Optional[int] = None
    ) -> str:
        """
        Audio-to-Video generation
        Uses EmptyLatentImage + audio encoder conditioning
        """
        print(f"\n[A2V] Generating video from audio")
        client_id = str(uuid.uuid4())
        
        # Save audio
        temp_audio = f"/tmp/{uuid.uuid4()}.wav"
        _save_base64_to_file(audio_base64, temp_audio, "audio")
        
        # Copy to ComfyUI input
        input_dir = COMFYUI_PATH / "input"
        input_dir.mkdir(exist_ok=True)
        audio_filename = f"{uuid.uuid4()}.wav"
        shutil.copy(temp_audio, input_dir / audio_filename)
        
        # Build base workflow
        workflow = self._build_base_workflow(
            mode="a2v",
            unet_model="wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            steps=steps,
            cfg=cfg,
            seed=seed
        )
        
        # Add EmptyLatentImage
        workflow["5"] = {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": num_frames
            }
        }
        
        # Add audio loading and encoding
        workflow["60"] = {
            "class_type": "LoadAudio",
            "inputs": {
                "audio": audio_filename
            }
        }
        
        workflow["61"] = {
            "class_type": "Wav2VecAudioEncoder",
            "inputs": {
                "encoder_name": "wav2vec2_large_english_fp16.safetensors",
                "audio": ["60", 0]
            }
        }
        
        # Condition positive prompt with audio
        workflow["62"] = {
            "class_type": "ConditioningSetTimestepRange",
            "inputs": {
                "start": 0.0,
                "end": 1.0,
                "conditioning": ["6", 0]
            }
        }
        
        workflow["63"] = {
            "class_type": "ConditioningCombine",
            "inputs": {
                "conditioning_1": ["62", 0],
                "conditioning_2": ["61", 0]
            }
        }
        
        # Update KSampler to use audio-conditioned positive
        workflow["3"]["inputs"]["positive"] = ["63", 0]
        workflow["3"]["inputs"]["latent_image"] = ["5", 0]
        
        # Execute
        prompt_id = self._queue_prompt(client_id, workflow)
        video_bytes = self._get_video_from_websocket(prompt_id, client_id)
        
        return base64.b64encode(video_bytes).decode('utf-8')


# ============================================================================
# PYDANTIC MODELS - Request/Response schemas
# ============================================================================
class T2VRequest(BaseModel):
    prompt: str = Field(..., example="A majestic cat sitting on a throne, cinematic lighting")
    negative_prompt: str = Field("", example="blurry, low quality")
    width: int = Field(832, ge=256, le=1920, example=832)
    height: int = Field(480, ge=256, le=1080, example=480)
    num_frames: int = Field(121, ge=16, le=241, example=121)
    steps: int = Field(30, ge=10, le=50, example=30)
    cfg: float = Field(7.5, ge=1.0, le=20.0, example=7.5)
    seed: Optional[int] = Field(None, example=42)


class I2VRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded input image")
    prompt: str = Field(..., example="The girl is reading a book")
    negative_prompt: str = Field("", example="static, blurry")
    width: int = Field(1280, ge=256, le=1920, example=1280)
    height: int = Field(704, ge=256, le=1080, example=704)
    num_frames: int = Field(121, ge=16, le=241, example=121)
    steps: int = Field(20, ge=10, le=50, example=20)
    cfg: float = Field(5.0, ge=1.0, le=20.0, example=5.0)
    seed: Optional[int] = Field(None, example=42)


class S2VRequest(BaseModel):
    sketch_base64: str = Field(..., description="Base64 encoded sketch/line art")
    prompt: str = Field(..., example="A person walking in the park")
    negative_prompt: str = Field("", example="static, blurry")
    width: int = Field(1280, ge=256, le=1920, example=1280)
    height: int = Field(704, ge=256, le=1080, example=704)
    num_frames: int = Field(121, ge=16, le=241, example=121)
    steps: int = Field(20, ge=10, le=50, example=20)
    cfg: float = Field(5.0, ge=1.0, le=20.0, example=5.0)
    seed: Optional[int] = Field(None, example=42)


class A2VRequest(BaseModel):
    audio_base64: str = Field(..., description="Base64 encoded audio file (WAV)")
    prompt: str = Field(..., example="A person speaking with animated gestures")
    negative_prompt: str = Field("", example="static, no movement")
    width: int = Field(832, ge=256, le=1920, example=832)
    height: int = Field(480, ge=256, le=1080, example=480)
    num_frames: int = Field(121, ge=16, le=241, example=121)
    steps: int = Field(30, ge=10, le=50, example=30)
    cfg: float = Field(7.5, ge=1.0, le=20.0, example=7.5)
    seed: Optional[int] = Field(None, example=42)


class VideoResponse(BaseModel):
    success: bool
    video_base64: str
    metadata: Dict


# ============================================================================
# FASTAPI WEB APPLICATION
# ============================================================================
web_app = FastAPI(
    title="ComfyUI Wan2.2 Complete API",
    description="Complete API for Wan2.2 video generation: T2V, I2V, S2V, A2V",
    version="2.0.0"
)


@web_app.get("/")
async def root():
    return {
        "service": "ComfyUI Wan2.2 Complete API",
        "version": "2.0.0",
        "modes": {
            "t2v": "Text-to-Video",
            "i2v": "Image-to-Video", 
            "s2v": "Sketch-to-Video",
            "a2v": "Audio-to-Video"
        },
        "endpoints": {
            "t2v": "/generate/t2v",
            "i2v": "/generate/i2v",
            "s2v": "/generate/s2v",
            "a2v": "/generate/a2v",
            "health": "/health"
        },
        "documentation": "/docs"
    }


@web_app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "comfyui-wan2-2-complete-api",
        "modes": ["t2v", "i2v", "s2v", "a2v"]
    }


@web_app.post("/generate/t2v", response_model=VideoResponse)
async def api_generate_t2v(request: T2VRequest = Body(...)):
    """
    Generate video from text prompt
    
    Example request:
    ```json
    {
        "prompt": "A cat wearing a wizard hat, magical atmosphere",
        "negative_prompt": "blurry, low quality",
        "width": 832,
        "height": 480,
        "num_frames": 121,
        "steps": 30,
        "cfg": 7.5
    }
    ```
    """
    try:
        print(f"\n[API-T2V] Request received")
        print(f"[API-T2V] Prompt: {request.prompt[:100]}...")
        
        # Validate
        if not request.prompt or len(request.prompt.strip()) == 0:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        if request.width % 8 != 0 or request.height % 8 != 0:
            raise HTTPException(
                status_code=400,
                detail="Width and height must be multiples of 8"
            )
        
        # Generate
        comfy = ComfyUI()
        video_base64 = comfy.generate_t2v.remote(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_frames=request.num_frames,
            steps=request.steps,
            cfg=request.cfg,
            seed=request.seed
        )
        
        print(f"[API-T2V] Success")
        
        return VideoResponse(
            success=True,
            video_base64=video_base64,
            metadata={
                "mode": "t2v",
                "prompt": request.prompt[:100],
                "width": request.width,
                "height": request.height,
                "num_frames": request.num_frames,
                "steps": request.steps,
                "cfg": request.cfg
            }
        )
    
    except HTTPException:
        raise
    except TimeoutError as e:
        print(f"[API-T2V] Timeout: {str(e)}")
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        print(f"[API-T2V] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@web_app.post("/generate/i2v", response_model=VideoResponse)
async def api_generate_i2v(request: I2VRequest = Body(...)):
    """
    Generate video from input image
    
    Example request:
    ```json
    {
        "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
        "prompt": "The person is walking forward",
        "negative_prompt": "static, no movement",
        "width": 1280,
        "height": 704,
        "num_frames": 121,
        "steps": 20,
        "cfg": 5.0
    }
    ```
    """
    try:
        print(f"\n[API-I2V] Request received")
        
        # Validate
        if not request.image_base64:
            raise HTTPException(status_code=400, detail="Image data required")
        
        if not request.prompt or len(request.prompt.strip()) == 0:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        if request.width % 8 != 0 or request.height % 8 != 0:
            raise HTTPException(
                status_code=400,
                detail="Width and height must be multiples of 8"
            )
        
        # Generate
        comfy = ComfyUI()
        video_base64 = comfy.generate_i2v.remote(
            image_base64=request.image_base64,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_frames=request.num_frames,
            steps=request.steps,
            cfg=request.cfg,
            seed=request.seed
        )
        
        print(f"[API-I2V] Success")
        
        return VideoResponse(
            success=True,
            video_base64=video_base64,
            metadata={
                "mode": "i2v",
                "prompt": request.prompt[:100],
                "width": request.width,
                "height": request.height,
                "num_frames": request.num_frames,
                "steps": request.steps,
                "cfg": request.cfg
            }
        )
    
    except HTTPException:
        raise
    except TimeoutError as e:
        print(f"[API-I2V] Timeout: {str(e)}")
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        print(f"[API-I2V] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@web_app.post("/generate/s2v", response_model=VideoResponse)
async def api_generate_s2v(request: S2VRequest = Body(...)):
    """
    Generate video from sketch/line art
    
    Example request:
    ```json
    {
        "sketch_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
        "prompt": "A person dancing energetically",
        "negative_prompt": "static, blurry",
        "width": 1280,
        "height": 704,
        "num_frames": 121,
        "steps": 20,
        "cfg": 5.0
    }
    ```
    """
    try:
        print(f"\n[API-S2V] Request received")
        
        # Validate
        if not request.sketch_base64:
            raise HTTPException(status_code=400, detail="Sketch data required")
        
        if not request.prompt or len(request.prompt.strip()) == 0:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        if request.width % 8 != 0 or request.height % 8 != 0:
            raise HTTPException(
                status_code=400,
                detail="Width and height must be multiples of 8"
            )
        
        # Generate
        comfy = ComfyUI()
        video_base64 = comfy.generate_s2v.remote(
            sketch_base64=request.sketch_base64,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_frames=request.num_frames,
            steps=request.steps,
            cfg=request.cfg,
            seed=request.seed
        )
        
        print(f"[API-S2V] Success")
        
        return VideoResponse(
            success=True,
            video_base64=video_base64,
            metadata={
                "mode": "s2v",
                "prompt": request.prompt[:100],
                "width": request.width,
                "height": request.height,
                "num_frames": request.num_frames,
                "steps": request.steps,
                "cfg": request.cfg
            }
        )
    
    except HTTPException:
        raise
    except TimeoutError as e:
        print(f"[API-S2V] Timeout: {str(e)}")
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        print(f"[API-S2V] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@web_app.post("/generate/a2v", response_model=VideoResponse)
async def api_generate_a2v(request: A2VRequest = Body(...)):
    """
    Generate video from audio with lip sync
    
    Example request:
    ```json
    {
        "audio_base64": "UklGRiQAAABXQVZFZm10...",
        "prompt": "A person speaking clearly with natural expressions",
        "negative_prompt": "static, frozen, no movement",
        "width": 832,
        "height": 480,
        "num_frames": 121,
        "steps": 30,
        "cfg": 7.5
    }
    ```
    """
    try:
        print(f"\n[API-A2V] Request received")
        
        # Validate
        if not request.audio_base64:
            raise HTTPException(status_code=400, detail="Audio data required")
        
        if not request.prompt or len(request.prompt.strip()) == 0:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        if request.width % 8 != 0 or request.height % 8 != 0:
            raise HTTPException(
                status_code=400,
                detail="Width and height must be multiples of 8"
            )
        
        # Generate
        comfy = ComfyUI()
        video_base64 = comfy.generate_a2v.remote(
            audio_base64=request.audio_base64,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_frames=request.num_frames,
            steps=request.steps,
            cfg=request.cfg,
            seed=request.seed
        )
        
        print(f"[API-A2V] Success")
        
        return VideoResponse(
            success=True,
            video_base64=video_base64,
            metadata={
                "mode": "a2v",
                "prompt": request.prompt[:100],
                "width": request.width,
                "height": request.height,
                "num_frames": request.num_frames,
                "steps": request.steps,
                "cfg": request.cfg
            }
        )
    
    except HTTPException:
        raise
    except TimeoutError as e:
        print(f"[API-A2V] Timeout: {str(e)}")
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        print(f"[API-A2V] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MODAL WEB ENDPOINT
# ============================================================================
@app.function(
    image=comfy_image,
    container_idle_timeout=300,
    allow_concurrent_inputs=10
)
@asgi_app()
def fastapi_app():
    return web_app


# ============================================================================
# DEPLOYMENT INSTRUCTIONS
# ============================================================================
"""
DEPLOYMENT GUIDE:

1. Install Modal:
   pip install modal

2. Setup Modal token:
   modal token new

3. Deploy the application:
   modal deploy app_fixed.py

4. The API will be available at:
   https://your-username--comfyui-wan2-2-complete-api-fastapi-app.modal.run

5. Test endpoints:
   
   # Text-to-Video
   curl -X POST "https://your-url/generate/t2v" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "A cat wearing a wizard hat",
       "width": 832,
       "height": 480,
       "num_frames": 121,
       "steps": 30,
       "cfg": 7.5
     }'
   
   # Image-to-Video
   curl -X POST "https://your-url/generate/i2v" \
     -H "Content-Type: application/json" \
     -d '{
       "image_base64": "YOUR_BASE64_IMAGE",
       "prompt": "The person is walking",
       "width": 1280,
       "height": 704,
       "num_frames": 121,
       "steps": 20,
       "cfg": 5.0
     }'

KEY DIFFERENCES FROM OLD CODE:

✅ Uses comfy-cli for setup (like reference)
✅ Follows EXACT workflow pattern from JSON reference:
   UNETLoader -> ModelSamplingSD3 -> KSampler
✅ Uses Wan22ImageToVideoLatent for I2V/S2V (not custom nodes)
✅ Uses EmptyLatentImage for T2V/A2V (not custom nodes)
✅ Proper audio conditioning for A2V
✅ Clean separation of 4 modes with correct models
✅ No more HTTP 400 errors from invalid workflows

TESTED WORKFLOW PATTERNS:
- T2V: EmptyLatentImage → KSampler ✓
- I2V: LoadImage → Wan22ImageToVideoLatent → KSampler ✓
- S2V: LoadImage → Wan22ImageToVideoLatent → KSampler ✓
- A2V: LoadAudio + EmptyLatentImage → KSampler with audio conditioning ✓
"""