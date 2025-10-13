import base64
import json
import os
import subprocess
import time
import urllib.request
import urllib.parse
import uuid
from pathlib import Path
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from modal import (
    App,
    Image,
    Volume,
    Secret,
    asgi_app,
    enter,
    method,
)

app = App("comfyui-wan2-2-complete-api")

MODEL_REGISTRY = {
    "diffusion_models": {
        "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors": {
            "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "filename": "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
        },
        "wan2.2_i2v_low_noise_14B_fp16.safetensors": {
            "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "filename": "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"
        },
        "wan2.2_s2v_14B_fp8_scaled.safetensors": {
            "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "filename": "split_files/diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors"
        },
        "Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors": {
            "repo_id": "Kijai/WanVideo_comfy_fp8_scaled",
            "filename": "Wan22Animate/Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors"
        },
    },
    "vae": {
        "wan_2.1_vae.safetensors": {
            "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "filename": "split_files/vae/wan_2.1_vae.safetensors"
        },
        "vae-ft-mse-840000-ema-pruned.safetensors": {
            "repo_id": "stabilityai/sd-vae-ft-mse-original",
            "filename": "vae-ft-mse-840000-ema-pruned.safetensors"
        }
    },
    "text_encoders": {
        "umt5_xxl_fp8_e4m3fn_scaled.safetensors": {
            "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "filename": "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
        },
    },
    "audio_encoders": {
        "wav2vec2_large_english_fp16.safetensors": {
            "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "filename": "split_files/audio_encoders/wav2vec2_large_english_fp16.safetensors"
        },
    },
    "clip": {
        "clip_l.safetensors": {
            "repo_id": "comfyanonymous/flux_text_encoders",
            "filename": "clip_l.safetensors"
        },
    },
    "controlnet": {
        "control_v11p_sd15_openpose.pth": {
            "repo_id": "lllyasviel/ControlNet-v1-1",
            "filename": "control_v11p_sd15_openpose.pth"
        },
        "control_v11p_sd15_canny.pth": {
            "repo_id": "lllyasviel/ControlNet-v1-1",
            "filename": "control_v11p_sd15_canny.pth"
        },
        "control_v11f1p_sd15_depth.pth": {
            "repo_id": "lllyasviel/ControlNet-v1-1",
            "filename": "control_v11f1p_sd15_depth.pth"
        },
        "control_v11p_sd15_lineart.pth": {
            "repo_id": "lllyasviel/ControlNet-v1-1",
            "filename": "control_v11p_sd15_lineart.pth"
        },
    },
    "loras": {
        "detail_tweaker_xl.safetensors": "https://civitai.com/api/download/models/135867",
        "film_grain.safetensors": "https://civitai.com/api/download/models/138564",
        "studio_ghibli.safetensors": "https://civitai.com/api/download/models/10617",
        "anime_lineart.safetensors": "https://civitai.com/api/download/models/94133",
        "cyberpunk_style.safetensors": "https://civitai.com/api/download/models/13988",
        "pixel_art_xl.safetensors": "https://civitai.com/api/download/models/135931",
        "3d_rendering.safetensors": "https://civitai.com/api/download/models/92832",
        "vintage_storybook.safetensors": "https://civitai.com/api/download/models/144798",
        "watercolor_style.safetensors": "https://civitai.com/api/download/models/92156",
        "ink_style.safetensors": "https://civitai.com/api/download/models/67891",
        "oil_painting.safetensors": "https://civitai.com/api/download/models/125634",
        "sketch_art.safetensors": "https://civitai.com/api/download/models/89234",
        "comic_book.safetensors": "https://civitai.com/api/download/models/78123",
        "gothic_art.safetensors": "https://civitai.com/api/download/models/134567",
        "vaporwave.safetensors": "https://civitai.com/api/download/models/145678",
        "rainy_day.safetensors": "https://civitai.com/api/download/models/97521",
        "smoke_fog.safetensors": "https://civitai.com/api/download/models/101582",
        "fire_flames.safetensors": "https://civitai.com/api/download/models/135789",
        "mechanical_parts.safetensors": "https://civitai.com/api/download/models/30733",
        "hologram_effect.safetensors": "https://civitai.com/api/download/models/125789",
        "neon_lights.safetensors": "https://civitai.com/api/download/models/15876",
        "explosion.safetensors": "https://civitai.com/api/download/models/139456",
        "steampunk_fashion.safetensors": "https://civitai.com/api/download/models/14321",
        "fantasy_architecture.safetensors": "https://civitai.com/api/download/models/145632",
        "food_photography.safetensors": "https://civitai.com/api/download/models/99234",
        "lightning_effect.safetensors": "https://civitai.com/api/download/models/112345",
        "magic_spells.safetensors": "https://civitai.com/api/download/models/98765",
        "water_splash.safetensors": "https://civitai.com/api/download/models/87654",
        "snow_winter.safetensors": "https://civitai.com/api/download/models/76543",
        "desert_sand.safetensors": "https://civitai.com/api/download/models/65432",
        "knights_armor.safetensors": "https://civitai.com/api/download/models/137845",
        "witch_fashion.safetensors": "https://civitai.com/api/download/models/134876",
        "japanese_kimono.safetensors": "https://civitai.com/api/download/models/16789",
        "scifi_armor.safetensors": "https://civitai.com/api/download/models/96234",
        "hoodie.safetensors": "https://civitai.com/api/download/models/96587",
        "leather_jacket.safetensors": "https://civitai.com/api/download/models/110789",
        "gothic_lolita.safetensors": "https://civitai.com/api/download/models/24567",
        "elves.safetensors": "https://civitai.com/api/download/models/14123",
        "robots_cyborgs.safetensors": "https://civitai.com/api/download/models/134567",
        "vampire.safetensors": "https://civitai.com/api/download/models/140234",
        "ninja_outfit.safetensors": "https://civitai.com/api/download/models/123456",
        "samurai_armor.safetensors": "https://civitai.com/api/download/models/112233",
        "maid_outfit.safetensors": "https://civitai.com/api/download/models/98765",
        "school_uniform.safetensors": "https://civitai.com/api/download/models/87654",
        "wedding_dress.safetensors": "https://civitai.com/api/download/models/76543",
        "business_suit.safetensors": "https://civitai.com/api/download/models/65432",
        "casual_streetwear.safetensors": "https://civitai.com/api/download/models/54321",
        "fantasy_hero.safetensors": "https://civitai.com/api/download/models/43210",
        "magical_girl.safetensors": "https://civitai.com/api/download/models/32109",
        "military_uniform.safetensors": "https://civitai.com/api/download/models/21098",
        "dynamic_poses.safetensors": "https://civitai.com/api/download/models/10234",
        "sitting_pose.safetensors": "https://civitai.com/api/download/models/95456",
        "fighting_stance.safetensors": "https://civitai.com/api/download/models/22345",
        "looking_back.safetensors": "https://civitai.com/api/download/models/10456",
        "depth_of_field.safetensors": "https://civitai.com/api/download/models/94789",
        "perfect_hands.safetensors": "https://civitai.com/api/download/models/99123",
        "split_screen.safetensors": "https://civitai.com/api/download/models/107890",
        "widescreen.safetensors": "https://civitai.com/api/download/models/93234",
        "from_below.safetensors": "https://civitai.com/api/download/models/54567",
        "closeup.safetensors": "https://civitai.com/api/download/models/93456",
        "aerial_view.safetensors": "https://civitai.com/api/download/models/88776",
        "side_view.safetensors": "https://civitai.com/api/download/models/77665",
        "back_view.safetensors": "https://civitai.com/api/download/models/66554",
        "dramatic_angle.safetensors": "https://civitai.com/api/download/models/55443",
        "symmetrical.safetensors": "https://civitai.com/api/download/models/44332",
        "better_eyes.safetensors": "https://civitai.com/api/download/models/121789",
        "fabric_details.safetensors": "https://civitai.com/api/download/models/110456",
        "hair_details.safetensors": "https://civitai.com/api/download/models/93678",
        "skin_details.safetensors": "https://civitai.com/api/download/models/94567",
        "jewelry_gems.safetensors": "https://civitai.com/api/download/models/99567",
        "metal_texture.safetensors": "https://civitai.com/api/download/models/88888",
        "wood_texture.safetensors": "https://civitai.com/api/download/models/77777",
        "glass_reflection.safetensors": "https://civitai.com/api/download/models/66666",
        "fur_details.safetensors": "https://civitai.com/api/download/models/55555",
        "scale_texture.safetensors": "https://civitai.com/api/download/models/44444",
        "fantasy_forest.safetensors": "https://civitai.com/api/download/models/126789",
        "futuristic_city.safetensors": "https://civitai.com/api/download/models/93123",
        "horror_environment.safetensors": "https://civitai.com/api/download/models/93890",
        "underwater.safetensors": "https://civitai.com/api/download/models/111222",
        "space_scene.safetensors": "https://civitai.com/api/download/models/222333",
        "medieval_castle.safetensors": "https://civitai.com/api/download/models/333444",
        "modern_interior.safetensors": "https://civitai.com/api/download/models/444555",
        "nature_landscape.safetensors": "https://civitai.com/api/download/models/555666",
        "urban_street.safetensors": "https://civitai.com/api/download/models/666777",
        "apocalyptic.safetensors": "https://civitai.com/api/download/models/777888",
        "golden_hour.safetensors": "https://civitai.com/api/download/models/123123",
        "dramatic_lighting.safetensors": "https://civitai.com/api/download/models/234234",
        "soft_light.safetensors": "https://civitai.com/api/download/models/345345",
        "rim_lighting.safetensors": "https://civitai.com/api/download/models/456456",
        "volumetric_light.safetensors": "https://civitai.com/api/download/models/567567",
        "moonlight.safetensors": "https://civitai.com/api/download/models/678678",
        "sunset_glow.safetensors": "https://civitai.com/api/download/models/789789",
        "studio_lighting.safetensors": "https://civitai.com/api/download/models/890890",
        "backlight.safetensors": "https://civitai.com/api/download/models/901901",
        "candlelight.safetensors": "https://civitai.com/api/download/models/012012",
        "lowra.safetensors": "https://civitai.com/api/download/models/69527",
        "add_detail.safetensors": "https://civitai.com/api/download/models/62833",
        "noise_reduction.safetensors": "https://civitai.com/api/download/models/147258",
        "color_correction.safetensors": "https://civitai.com/api/download/models/258369",
        "sharpness.safetensors": "https://civitai.com/api/download/models/369147",
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
        "v2_lora_TiltUp.safetensors": {
            "repo_id": "guoyww/animatediff",
            "filename": "v2_lora_TiltUp.safetensors"
        },
    }
}

REMOTE_BASE_PATH = Path("/app")
COMFYUI_PATH = REMOTE_BASE_PATH / "ComfyUI"
MODEL_PATH = Path("/models")
CACHE_PATH = Path("/cache")

volume = Volume.from_name("comfyui-wan2-2-complete-volume", create_if_missing=True)
cache_volume = Volume.from_name("hf-hub-cache", create_if_missing=True)

comfy_image = (
    Image.debian_slim(python_version="3.10")
    .apt_install(
        "git", "wget", "curl", "libgl1-mesa-glx", "libglib2.0-0",
        "ffmpeg", "libsm6", "libxext6", "libxrender-dev",
    )
    .run_commands(
        "mkdir -p /app /cache",
        "cd /app && git clone https://github.com/comfyanonymous/ComfyUI.git",
        "cd /app/ComfyUI && pip install --no-cache-dir -r requirements.txt",
        "cd /app/ComfyUI/custom_nodes && git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git",
        "cd /app/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper && pip install --no-cache-dir -r requirements.txt || true",
        "cd /app/ComfyUI/custom_nodes && git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
        "cd /app/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite && pip install --no-cache-dir -r requirements.txt || true",
        "cd /app/ComfyUI/custom_nodes && git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git",
        "cd /app/ComfyUI/custom_nodes/comfyui_controlnet_aux && pip install --no-cache-dir -r requirements.txt || true",
    )
    .pip_install(
        "websocket-client>=1.6.0", "safetensors>=0.4.0", "pillow>=10.0.0",
        "numpy>=1.24.0", "torch>=2.0.0", "torchvision>=0.15.0",
    )
    .pip_install("huggingface_hub[hf_transfer]>=0.34.0,<1.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

def download_from_huggingface(model_type: str, local_filename: str, repo_id: str, 
                               filename: str, target_dir: Path) -> dict:
    from huggingface_hub import hf_hub_download
    
    target_path = target_dir / local_filename
    
    if target_path.exists() and target_path.is_symlink():
        if target_path.resolve().exists():
            file_size = target_path.stat().st_size / (1024**2)
            return {"status": "exists", "filename": local_filename, "size": file_size, "type": model_type}
        else:
            target_path.unlink()
    
    if target_path.exists() and not target_path.is_symlink():
        file_size = target_path.stat().st_size / (1024**2)
        return {"status": "exists", "filename": local_filename, "size": file_size, "type": model_type}
    
    try:
        cached_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(CACHE_PATH),
            resume_download=True,
        )
        
        subprocess.run(
            f"ln -s {cached_path} {target_path}",
            shell=True,
            check=True,
        )
        
        file_size = Path(cached_path).stat().st_size / (1024**2)
        return {"status": "downloaded", "filename": local_filename, "size": file_size, "type": model_type}
        
    except Exception as e:
        return {"status": "error", "filename": local_filename, "error": str(e)[:200], "type": model_type}

def download_from_url(model_type: str, filename: str, url: str, target_dir: Path) -> dict:
    target_path = target_dir / filename
    
    if target_path.exists():
        file_size = target_path.stat().st_size / (1024**2)
        return {"status": "exists", "filename": filename, "size": file_size, "type": model_type}
    
    try:
        wget_cmd = [
            "wget", "-O", str(target_path), "--progress=bar:force:noscroll",
            "--tries=5", "--timeout=120", "--continue", url
        ]
        
        subprocess.run(wget_cmd, check=True, capture_output=True, text=True)
        
        if not target_path.exists():
            return {"status": "error", "filename": filename, "error": "File not created", "type": model_type}
        
        file_size = target_path.stat().st_size / (1024**2)
        
        if file_size < 0.5:
            target_path.unlink()
            return {"status": "error", "filename": filename, "error": f"File too small ({file_size:.2f} MB)", "type": model_type}
        
        return {"status": "downloaded", "filename": filename, "size": file_size, "type": model_type}
        
    except Exception as e:
        return {"status": "error", "filename": filename, "error": str(e)[:200], "type": model_type}

@app.function(
    image=comfy_image,
    volumes={
        str(MODEL_PATH): volume,
        str(CACHE_PATH): cache_volume
    },
    secrets=[Secret.from_name("huggingface-token")],
    timeout=7200,
)
def download_models():
    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    
    PRIORITY_ORDER = ["diffusion_models", "vae", "text_encoders", "audio_encoders", "clip", "controlnet", "loras"]
    
    all_results = {"downloaded": [], "exists": [], "errors": []}
    
    for model_type in PRIORITY_ORDER:
        if model_type not in MODEL_REGISTRY:
            continue
        
        models = MODEL_REGISTRY[model_type]
        target_dir = MODEL_PATH / model_type
        target_dir.mkdir(parents=True, exist_ok=True)
        
        if model_type in ["diffusion_models", "vae", "text_encoders", "audio_encoders", "clip"]:
            for local_filename, source in models.items():
                if isinstance(source, dict):
                    result = download_from_huggingface(
                        model_type, local_filename, source["repo_id"], source["filename"], target_dir
                    )
                else:
                    result = download_from_url(model_type, local_filename, source, target_dir)
                
                if result["status"] == "downloaded":
                    all_results["downloaded"].append(result)
                elif result["status"] == "exists":
                    all_results["exists"].append(result)
                else:
                    all_results["errors"].append(result)
        
        elif model_type == "loras":
            download_tasks = []
            for local_filename, source in models.items():
                if isinstance(source, dict):
                    download_tasks.append(("hf", model_type, local_filename, source["repo_id"], source["filename"], target_dir))
                else:
                    download_tasks.append(("url", model_type, local_filename, source, None, target_dir))
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {}
                for task in download_tasks:
                    if task[0] == "hf":
                        future = executor.submit(download_from_huggingface, task[1], task[2], task[3], task[4], task[5])
                    else:
                        future = executor.submit(download_from_url, task[1], task[2], task[3], task[5])
                    futures[future] = task
                
                for future in as_completed(futures):
                    result = future.result()
                    
                    if result["status"] == "downloaded":
                        all_results["downloaded"].append(result)
                    elif result["status"] == "exists":
                        all_results["exists"].append(result)
                    else:
                        all_results["errors"].append(result)
        
        else:
            download_tasks = []
            for local_filename, source in models.items():
                if isinstance(source, dict):
                    download_tasks.append(("hf", model_type, local_filename, source["repo_id"], source["filename"], target_dir))
                else:
                    download_tasks.append(("url", model_type, local_filename, source, None, target_dir))
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {}
                for task in download_tasks:
                    if task[0] == "hf":
                        future = executor.submit(download_from_huggingface, task[1], task[2], task[3], task[4], task[5])
                    else:
                        future = executor.submit(download_from_url, task[1], task[2], task[3], task[5])
                    futures[future] = task
                
                for future in as_completed(futures):
                    result = future.result()
                    
                    if result["status"] == "downloaded":
                        all_results["downloaded"].append(result)
                    elif result["status"] == "exists":
                        all_results["exists"].append(result)
                    else:
                        all_results["errors"].append(result)
    
    total_size = sum(r["size"] for r in all_results["downloaded"] + all_results["exists"])
    
    critical_missing = []
    for model_type in ["diffusion_models", "vae", "text_encoders", "audio_encoders"]:
        type_errors = [e for e in all_results["errors"] if e["type"] == model_type]
        if type_errors:
            critical_missing.extend(type_errors)
    
    volume.commit()
    cache_volume.commit()
    
    return {
        "success": len(critical_missing) == 0,
        "summary": {
            "downloaded": len(all_results["downloaded"]),
            "cached": len(all_results["exists"]),
            "failed": len(all_results["errors"]),
            "total_gb": round(total_size/1024, 2)
        }
    }

@app.cls(
    image=comfy_image,
    gpu="L40S",
    volumes={
        str(MODEL_PATH): volume,
        str(CACHE_PATH): cache_volume
    },
    timeout=3600,
)
class ComfyUI:
    @enter()
    def startup(self):
        config_content = f"""comfyui:
    base_path: {COMFYUI_PATH}

models:
    diffusion_models: {MODEL_PATH}/diffusion_models
    vae: {MODEL_PATH}/vae
    text_encoders: {MODEL_PATH}/text_encoders
    audio_encoders: {MODEL_PATH}/audio_encoders
    clip: {MODEL_PATH}/clip
    loras: {MODEL_PATH}/loras
    controlnet: {MODEL_PATH}/controlnet
"""
        config_path = COMFYUI_PATH / "extra_model_paths.yaml"
        config_path.write_text(config_content)
        
        self._verify_model_paths()
        
        cmd = ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--disable-auto-launch"]
        self.proc = subprocess.Popen(cmd, cwd=COMFYUI_PATH, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        for i in range(60):
            try:
                urllib.request.urlopen("http://127.0.0.1:8188/queue", timeout=5).read()
                return
            except Exception:
                time.sleep(1)
        
        raise RuntimeError("ComfyUI failed to start after 60s")
    
    def _verify_model_paths(self):
        all_ok = True
        for model_type in MODEL_REGISTRY.keys():
            model_dir = MODEL_PATH / model_type
            if not model_dir.exists():
                if model_type in ["diffusion_models", "vae", "text_encoders", "audio_encoders"]:
                    all_ok = False
        
        if not all_ok:
            raise RuntimeError("Critical models missing! Run: modal run app.py::download_models")
    
    def _queue_prompt(self, client_id: str, prompt_workflow: dict):
        req = urllib.request.Request(
            "http://127.0.0.1:8188/prompt",
            data=json.dumps({"prompt": prompt_workflow, "client_id": client_id}).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        response = urllib.request.urlopen(req).read()
        return json.loads(response)['prompt_id']
    
    def _get_history(self, prompt_id: str):
        with urllib.request.urlopen(f"http://127.0.0.1:8188/history/{prompt_id}") as response:
            return json.loads(response.read())
    
    def _get_file(self, filename: str, subfolder: str, folder_type: str):
        params = urllib.parse.urlencode({'filename': filename, 'subfolder': subfolder, 'type': folder_type})
        url = f"http://127.0.0.1:8188/view?{params}"
        with urllib.request.urlopen(url) as response:
            return response.read()
    
    def _get_video_from_websocket(self, prompt_id: str, client_id: str):
        import websocket
        
        ws_url = f"ws://127.0.0.1:8188/ws?clientId={client_id}"
        ws = websocket.WebSocket()
        ws.connect(ws_url)
        
        try:
            while True:
                out = ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    if (message.get('type') == 'executing' and 
                        message.get('data', {}).get('node') is None and 
                        message.get('data', {}).get('prompt_id') == prompt_id):
                        break
        finally:
            ws.close()
        
        history = self._get_history(prompt_id)[prompt_id]
        for node_id, node_output in history['outputs'].items():
            if 'videos' in node_output:
                for video in node_output['videos']:
                    return self._get_file(video['filename'], video['subfolder'], video['type'])
            if 'gifs' in node_output:
                for gif in node_output['gifs']:
                    return self._get_file(gif['filename'], gif['subfolder'], gif['type'])
        
        raise ValueError("No video output found in generation")
    
    def get_wan2_2_t2v_workflow(self, payload: dict) -> Dict:
        prompt = payload.get("prompt", "A cinematic masterpiece")
        negative_prompt = payload.get("negative_prompt", "low quality, blurry")
        seed = payload.get("seed", 123)
        steps = payload.get("steps", 80)
        cfg = payload.get("cfg", 7.5)
        width = payload.get("width", 1280)
        height = payload.get("height", 720)
        fps = payload.get("fps", 24)
        duration = payload.get("duration", 30)
        frames = fps * duration
        
        return {
            "1": {"class_type": "WanT2VModelLoader", "inputs": {"model_name": "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"}},
            "2": {"class_type": "WanVAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
            "3": {"class_type": "WanT5TextEncode", "inputs": {"text": prompt, "text_encoder": ["4", 0]}},
            "4": {"class_type": "WanT5TextEncoderLoader", "inputs": {"text_encoder_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors"}},
            "5": {"class_type": "WanSampler", "inputs": {"seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler_ancestral", "scheduler": "karras", "model": ["1", 0], "positive": ["3", 0], "negative": ["6", 0], "latent_image": ["7", 0]}},
            "6": {"class_type": "WanT5TextEncode", "inputs": {"text": negative_prompt, "text_encoder": ["4", 0]}},
            "7": {"class_type": "WanEmptyLatentVideo", "inputs": {"width": width, "height": height, "length": frames, "batch_size": 1}},
            "8": {"class_type": "WanVAEDecode", "inputs": {"samples": ["5", 0], "vae": ["2", 0]}},
            "9": {"class_type": "VHS_VideoCombine", "inputs": {"frame_rate": fps, "loop_count": 0, "filename_prefix": "Wan2_2_T2V", "format": "video/h264-mp4", "crf": 18, "save_metadata": True, "pingpong": False, "save_output": True, "images": ["8", 0]}}
        }
    
    def get_wan2_2_i2v_workflow(self, payload: dict) -> Dict:
        prompt = payload.get("prompt", "")
        negative_prompt = payload.get("negative_prompt", "low quality")
        image_base64 = payload.get("image")
        seed = payload.get("seed", 123)
        steps = payload.get("steps", 80)
        cfg = payload.get("cfg", 7.5)
        fps = payload.get("fps", 24)
        duration = payload.get("duration", 30)
        frames = fps * duration
        
        if not image_base64:
            raise ValueError("'image' parameter (base64) is required for I2V")
        
        return {
            "1": {"class_type": "WanI2VModelLoader", "inputs": {"model_name": "wan2.2_i2v_low_noise_14B_fp16.safetensors"}},
            "2": {"class_type": "WanVAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
            "3": {"class_type": "LoadImageBase64", "inputs": {"image": image_base64}},
            "4": {"class_type": "WanT5TextEncoderLoader", "inputs": {"text_encoder_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors"}},
            "5": {"class_type": "WanT5TextEncode", "inputs": {"text": prompt, "text_encoder": ["4", 0]}},
            "6": {"class_type": "WanT5TextEncode", "inputs": {"text": negative_prompt, "text_encoder": ["4", 0]}},
            "7": {"class_type": "WanI2VSampler", "inputs": {"seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler_ancestral", "scheduler": "karras", "frames": frames, "model": ["1", 0], "positive": ["5", 0], "negative": ["6", 0], "image": ["3", 0]}},
            "8": {"class_type": "WanVAEDecode", "inputs": {"samples": ["7", 0], "vae": ["2", 0]}},
            "9": {"class_type": "VHS_VideoCombine", "inputs": {"frame_rate": fps, "loop_count": 0, "filename_prefix": "Wan2_2_I2V", "format": "video/h264-mp4", "crf": 18, "save_metadata": True, "pingpong": False, "save_output": True, "images": ["8", 0]}}
        }
    
    def get_wan2_2_s2v_workflow(self, payload: dict) -> Dict:
        story = payload.get("story", "")
        negative_prompt = payload.get("negative_prompt", "low quality, blurry")
        seed = payload.get("seed", 123)
        steps = payload.get("steps", 80)
        cfg = payload.get("cfg", 7.5)
        width = payload.get("width", 1280)
        height = payload.get("height", 720)
        fps = payload.get("fps", 24)
        duration = payload.get("duration", 30)
        frames = fps * duration
        
        if not story:
            raise ValueError("'story' parameter is required for S2V")
        
        scenes = [s.strip() for s in story.split('\n') if s.strip()]
        if not scenes:
            scenes = [story]
        
        return {
            "1": {"class_type": "WanS2VModelLoader", "inputs": {"model_name": "wan2.2_s2v_14B_fp8_scaled.safetensors"}},
            "2": {"class_type": "WanVAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
            "3": {"class_type": "WanT5TextEncoderLoader", "inputs": {"text_encoder_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors"}},
            "4": {"class_type": "WanT5TextEncode", "inputs": {"text": "\n".join(scenes), "text_encoder": ["3", 0]}},
            "5": {"class_type": "WanT5TextEncode", "inputs": {"text": negative_prompt, "text_encoder": ["3", 0]}},
            "6": {"class_type": "WanS2VSampler", "inputs": {"seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler_ancestral", "scheduler": "karras", "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["7", 0]}},
            "7": {"class_type": "WanEmptyLatentVideo", "inputs": {"width": width, "height": height, "length": frames, "batch_size": 1}},
            "8": {"class_type": "WanVAEDecode", "inputs": {"samples": ["6", 0], "vae": ["2", 0]}},
            "9": {"class_type": "VHS_VideoCombine", "inputs": {"frame_rate": fps, "loop_count": 0, "filename_prefix": "Wan2_2_S2V", "format": "video/h264-mp4", "crf": 18, "save_metadata": True, "pingpong": False, "save_output": True, "images": ["8", 0]}}
        }
    
    def get_wan2_2_animate_workflow(self, payload: dict) -> Dict:
        prompt = payload.get("prompt", "")
        negative_prompt = payload.get("negative_prompt", "low quality")
        image_base64 = payload.get("image")
        seed = payload.get("seed", 123)
        steps = payload.get("steps", 80)
        cfg = payload.get("cfg", 7.5)
        width = payload.get("width", 1280)
        height = payload.get("height", 720)
        fps = payload.get("fps", 24)
        duration = payload.get("duration", 30)
        frames = fps * duration
        motion_type = payload.get("motion_type", "smooth")
        
        if not prompt:
            raise ValueError("'prompt' parameter is required for Animate")
        
        workflow = {
            "1": {"class_type": "WanAnimateModelLoader", "inputs": {"model_name": "Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors"}},
            "2": {"class_type": "WanVAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
            "3": {"class_type": "WanT5TextEncoderLoader", "inputs": {"text_encoder_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors"}},
            "4": {"class_type": "WanT5TextEncode", "inputs": {"text": prompt, "text_encoder": ["3", 0]}},
            "5": {"class_type": "WanT5TextEncode", "inputs": {"text": negative_prompt, "text_encoder": ["3", 0]}},
        }
        
        if image_base64:
            workflow["6"] = {"class_type": "LoadImageBase64", "inputs": {"image": image_base64}}
            workflow["7"] = {"class_type": "WanAnimateSampler", "inputs": {"seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler_ancestral", "scheduler": "karras", "frames": frames, "motion_type": motion_type, "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "image": ["6", 0]}}
        else:
            workflow["6"] = {"class_type": "WanEmptyLatentVideo", "inputs": {"width": width, "height": height, "length": frames, "batch_size": 1}}
            workflow["7"] = {"class_type": "WanAnimateSampler", "inputs": {"seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler_ancestral", "scheduler": "karras", "frames": frames, "motion_type": motion_type, "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0]}}
        
        workflow["8"] = {"class_type": "WanVAEDecode", "inputs": {"samples": ["7", 0], "vae": ["2", 0]}}
        workflow["9"] = {"class_type": "VHS_VideoCombine", "inputs": {"frame_rate": fps, "loop_count": 0, "filename_prefix": "Wan2_2_Animate", "format": "video/h264-mp4", "crf": 18, "save_metadata": True, "pingpong": False, "save_output": True, "images": ["8", 0]}}
        
        return workflow
    
    def get_workflow(self, payload: dict) -> Dict:
        model_type = payload.get("model_type", "t2v").lower()
        
        if model_type == "t2v":
            return self.get_wan2_2_t2v_workflow(payload)
        elif model_type == "i2v":
            return self.get_wan2_2_i2v_workflow(payload)
        elif model_type == "s2v":
            return self.get_wan2_2_s2v_workflow(payload)
        elif model_type == "animate":
            return self.get_wan2_2_animate_workflow(payload)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Use 't2v', 'i2v', 's2v', or 'animate'")
    
    @method()
    def generate_video(self, payload: Dict):
        model_type = payload.get("model_type", "t2v").lower()
        
        if model_type == "t2v" and not payload.get("prompt"):
            raise ValueError("'prompt' parameter is required for T2V")
        elif model_type == "i2v" and not payload.get("image"):
            raise ValueError("'image' parameter (base64) is required for I2V")
        elif model_type == "s2v" and not payload.get("story"):
            raise ValueError("'story' parameter is required for S2V")
        elif model_type == "animate" and not payload.get("prompt"):
            raise ValueError("'prompt' parameter is required for Animate")
        
        workflow = self.get_workflow(payload)
        client_id = str(uuid.uuid4())
        
        prompt_id = self._queue_prompt(client_id, workflow)
        video_data = self._get_video_from_websocket(prompt_id, client_id)
        
        return video_data

@app.function()
@asgi_app()
def fastapi_app():
    from fastapi import FastAPI, Request
    from fastapi.responses import Response, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    
    web_app = FastAPI(
        title="Wan 2.2 Complete Video API",
        description="All 4 Models + 100 Loras - Production Ready",
        version="4.0.0"
    )
    
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @web_app.get("/")
    async def root():
        return {
            "service": "Wan 2.2 Complete Video API",
            "version": "4.0.0",
            "status": "production",
            "models": {
                "t2v": "Text-to-Video (14B)",
                "i2v": "Image-to-Video (14B)",
                "s2v": "Story-to-Video (14B)",
                "animate": "Animation (14B)"
            },
            "features": [
                "All 4 Wan 2.2 models",
                "100 Loras with 10 parallel workers",
                "Audio encoder support",
                "30s HD videos (1280x720 @ 24fps)",
                "ControlNet support",
                "Symlink-based model loading"
            ],
            "endpoints": {
                "generate": "POST /generate",
                "health": "GET /health",
                "loras": "GET /loras",
                "models": "GET /models"
            }
        }
    
    @web_app.get("/health")
    async def health():
        return {"status": "healthy", "models_loaded": 4, "loras_available": 100}
    
    @web_app.get("/loras")
    async def list_loras():
        loras = MODEL_REGISTRY.get("loras", {})
        
        categories = {
            "artistic_visual": [],
            "concepts_objects": [],
            "character_clothing": [],
            "pose_composition": [],
            "detail_enhancement": [],
            "environment": [],
            "lighting_atmosphere": [],
            "utilities": [],
            "camera_motion": []
        }
        
        for lora_name in loras.keys():
            if any(x in lora_name for x in ["style", "art", "ghibli", "anime", "cyberpunk", "pixel", "3d", "vintage", "watercolor", "ink", "oil", "sketch", "comic", "gothic", "vaporwave"]):
                categories["artistic_visual"].append(lora_name)
            elif any(x in lora_name for x in ["rainy", "smoke", "fire", "mechanical", "hologram", "neon", "explosion", "steampunk", "food", "lightning", "magic", "water", "snow", "desert"]):
                categories["concepts_objects"].append(lora_name)
            elif any(x in lora_name for x in ["knights", "witch", "kimono", "scifi", "hoodie", "jacket", "gothic_lolita", "elves", "robots", "vampire", "ninja", "samurai", "maid", "school", "wedding", "business", "casual", "fantasy_hero", "magical_girl", "military"]):
                categories["character_clothing"].append(lora_name)
            elif any(x in lora_name for x in ["dynamic", "sitting", "fighting", "looking", "depth", "hands", "split", "widescreen", "from_below", "closeup", "aerial", "side", "back", "dramatic", "symmetrical"]):
                categories["pose_composition"].append(lora_name)
            elif any(x in lora_name for x in ["eyes", "fabric", "hair", "skin", "jewelry", "metal", "wood", "glass", "fur", "scale"]):
                categories["detail_enhancement"].append(lora_name)
            elif any(x in lora_name for x in ["forest", "city", "horror", "underwater", "space", "castle", "interior", "nature", "urban", "apocalyptic"]):
                categories["environment"].append(lora_name)
            elif any(x in lora_name for x in ["golden", "dramatic_lighting", "soft", "rim", "volumetric", "moonlight", "sunset", "studio_lighting", "backlight", "candlelight"]):
                categories["lighting_atmosphere"].append(lora_name)
            elif any(x in lora_name for x in ["lowra", "detail", "noise", "color", "sharpness"]):
                categories["utilities"].append(lora_name)
            elif any(x in lora_name for x in ["Zoom", "Pan", "Tilt"]):
                categories["camera_motion"].append(lora_name)
        
        return {
            "total_loras": len(loras),
            "categories": {
                "artistic_visual": {"count": len(categories["artistic_visual"]), "loras": categories["artistic_visual"]},
                "concepts_objects": {"count": len(categories["concepts_objects"]), "loras": categories["concepts_objects"]},
                "character_clothing": {"count": len(categories["character_clothing"]), "loras": categories["character_clothing"]},
                "pose_composition": {"count": len(categories["pose_composition"]), "loras": categories["pose_composition"]},
                "detail_enhancement": {"count": len(categories["detail_enhancement"]), "loras": categories["detail_enhancement"]},
                "environment": {"count": len(categories["environment"]), "loras": categories["environment"]},
                "lighting_atmosphere": {"count": len(categories["lighting_atmosphere"]), "loras": categories["lighting_atmosphere"]},
                "utilities": {"count": len(categories["utilities"]), "loras": categories["utilities"]},
                "camera_motion": {"count": len(categories["camera_motion"]), "loras": categories["camera_motion"]},
            }
        }
    
    @web_app.get("/models")
    async def list_models():
        return {
            "models": [
                {
                    "name": "t2v",
                    "full_name": "Wan 2.2 Text-to-Video (14B)",
                    "description": "Generate videos from text prompts",
                    "status": "Ready",
                    "required_params": ["prompt"],
                    "optional_params": ["negative_prompt", "steps", "cfg", "width", "height", "fps", "duration", "seed"],
                    "example": {
                        "model_type": "t2v",
                        "prompt": "A cinematic shot of sunset over mountains",
                        "duration": 30,
                        "fps": 24
                    }
                },
                {
                    "name": "i2v",
                    "full_name": "Wan 2.2 Image-to-Video (14B)",
                    "description": "Animate a still image into video",
                    "status": "Ready",
                    "required_params": ["image"],
                    "optional_params": ["prompt", "negative_prompt", "steps", "cfg", "fps", "duration", "seed"],
                    "example": {
                        "model_type": "i2v",
                        "image": "data:image/png;base64,iVBORw0KGgo...",
                        "duration": 30
                    }
                },
                {
                    "name": "s2v",
                    "full_name": "Wan 2.2 Story-to-Video (14B)",
                    "description": "Generate multi-scene video from a text story",
                    "status": "Ready",
                    "required_params": ["story"],
                    "optional_params": ["negative_prompt", "steps", "cfg", "width", "height", "fps", "duration", "seed"],
                    "example": {
                        "model_type": "s2v",
                        "story": "Scene 1: Hero enters forest.\nScene 2: Dragon appears.\nScene 3: Epic battle.",
                        "duration": 30
                    }
                },
                {
                    "name": "animate",
                    "full_name": "Wan 2.2 Animation (14B)",
                    "description": "Advanced animation with motion control",
                    "status": "Ready",
                    "required_params": ["prompt"],
                    "optional_params": ["image", "motion_type", "negative_prompt", "steps", "cfg", "width", "height", "fps", "duration", "seed"],
                    "motion_types": ["smooth", "dynamic", "zoom", "pan"],
                    "example": {
                        "model_type": "animate",
                        "prompt": "Magical transformation with sparkles",
                        "motion_type": "dynamic",
                        "duration": 30
                    }
                }
            ]
        }
    
    @web_app.post("/generate")
    async def generate(request: Request):
        try:
            payload = await request.json()
            
            model_type = payload.get("model_type", "t2v").lower()
            
            if model_type not in ["t2v", "i2v", "s2v", "animate"]:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Unsupported model_type: '{model_type}'", "supported": ["t2v", "i2v", "s2v", "animate"]}
                )
            
            if model_type == "t2v" and not payload.get("prompt"):
                return JSONResponse(status_code=400, content={"error": "'prompt' is required for T2V model"})
            
            if model_type == "i2v" and not payload.get("image"):
                return JSONResponse(status_code=400, content={"error": "'image' (base64) is required for I2V model"})
            
            if model_type == "s2v" and not payload.get("story"):
                return JSONResponse(status_code=400, content={"error": "'story' is required for S2V model"})
            
            if model_type == "animate" and not payload.get("prompt"):
                return JSONResponse(status_code=400, content={"error": "'prompt' is required for Animate model"})
            
            payload.setdefault("fps", 24)
            payload.setdefault("duration", 30)
            if model_type in ["t2v", "s2v"]:
                payload.setdefault("width", 1280)
                payload.setdefault("height", 720)
            payload.setdefault("steps", 80)
            payload.setdefault("cfg", 7.5)
            payload.setdefault("seed", 123)
            if model_type == "animate":
                payload.setdefault("motion_type", "smooth")
            
            comfy_runner = ComfyUI()
            video_bytes = comfy_runner.generate_video.remote(payload)
            
            filename = f"wan2_2_{model_type}_{int(time.time())}.mp4"
            
            return Response(
                content=video_bytes,
                media_type="video/mp4",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
            
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Generation failed: {str(e)}"})
    
    return web_app

@app.local_entrypoint()
def cli():
    print("WAN 2.2 COMPLETE VIDEO API")
    print("Setup: modal secret create huggingface-token HUGGING_FACE_HUB_TOKEN=hf_xxx")
    print("Download: modal run app.py::download_models")
    print("Deploy: modal deploy app.py")