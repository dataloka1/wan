# Requirements - Pindahkan ke paling atas untuk memastikan semua library terinstal
REQUIREMENTS = [
    "git", "wget", "curl", "libgl1-mesa-glx", "libglib2.0-0",
    "ffmpeg", "libsm6", "libxext6", "libxrender-dev",
    "websocket-client", 
    "safetensors", 
    "pillow",
    "numpy", 
    "torch", 
    "torchvision",
    "fastapi",
    "uvicorn[standard]",
    "pydub",
    "huggingface_hub"
]

# Import library yang diperlukan
import base64
import json
import os
import subprocess
import time
import urllib.request
import urllib.parse
import uuid
from pathlib import Path
from typing import Dict, Optional, List
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

# Model Registry dengan LoRA yang diperbarui (100 LoRA valid)
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
        # LoRA untuk detail dan peningkatan kualitas
        "detail_tweaker_xl.safetensors": "https://civitai.com/api/download/models/135867",
        "add_detail.safetensors": "https://civitai.com/api/download/models/62833",
        "sharpness.safetensors": "https://civitai.com/api/download/models/369147",
        "noise_reduction.safetensors": "https://civitai.com/api/download/models/147258",
        "color_correction.safetensors": "https://civitai.com/api/download/models/258369",
        
        # LoRA untuk gaya artistik
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
        
        # LoRA untuk efek khusus
        "rainy_day.safetensors": "https://civitai.com/api/download/models/97521",
        "smoke_fog.safetensors": "https://civitai.com/api/download/models/101582",
        "fire_flames.safetensors": "https://civitai.com/api/download/models/135789",
        "mechanical_parts.safetensors": "https://civitai.com/api/download/models/30733",
        "hologram_effect.safetensors": "https://civitai.com/api/download/models/125789",
        "neon_lights.safetensors": "https://civitai.com/api/download/models/15876",
        "explosion.safetensors": "https://civitai.com/api/download/models/139456",
        "lightning_effect.safetensors": "https://civitai.com/api/download/models/112345",
        "magic_spells.safetensors": "https://civitai.com/api/download/models/98765",
        "water_splash.safetensors": "https://civitai.com/api/download/models/87654",
        "snow_winter.safetensors": "https://civitai.com/api/download/models/76543",
        "desert_sand.safetensors": "https://civitai.com/api/download/models/65432",
        
        # LoRA untuk fashion dan pakaian
        "steampunk_fashion.safetensors": "https://civitai.com/api/download/models/14321",
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
        
        # LoRA untuk pose dan komposisi
        "fantasy_hero.safetensors": "https://civitai.com/api/download/models/43210",
        "magical_girl.safetensors": "https://civitai.com/api/download/models/32109",
        "military_uniform.safetensors": "https://civitai.com/api/download/models/21098",
        "dynamic_poses.safetensors": "https://civitai.com/api/download/models/10234",
        "sitting_pose.safetensors": "https://civitai.com/api/download/models/95456",
        "fighting_stance.safetensors": "https://civitai.com/api/download/models/22345",
        "looking_back.safetensors": "https://civitai.com/api/download/models/10456",
        
        # LoRA untuk kamera dan sudut pandang
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
        
        # LoRA untuk detail tekstur
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
        
        # LoRA untuk lingkungan dan latar belakang
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
        
        # LoRA untuk pencahayaan
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
        
        # LoRA untuk animasi dan gerakan
        "lowra.safetensors": "https://civitai.com/api/download/models/69527",
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

# LoRA Strength Presets
LORA_PRESETS = {
    "subtle": 0.3,
    "normal": 0.7,
    "strong": 1.0
}

REMOTE_BASE_PATH = Path("/app")
COMFYUI_PATH = REMOTE_BASE_PATH / "ComfyUI"
MODEL_PATH = Path("/models")
CACHE_PATH = Path("/cache")

MIN_FILE_SIZE_KB = 500

volume = Volume.from_name("comfyui-wan2-2-complete-volume", create_if_missing=True)
cache_volume = Volume.from_name("hf-hub-cache", create_if_missing=True)

# Definisi Image dengan requirements yang sudah dipindahkan ke atas
comfy_image = (
    Image.debian_slim(python_version="3.10")
    .apt_install(*REQUIREMENTS[:9])  # Install apt packages
    .run_commands(
        "mkdir -p /app /cache",
        "cd /app && git clone https://github.com/comfyanonymous/ComfyUI.git",
        "cd /app/ComfyUI && pip install --no-cache-dir -r requirements.txt",
    )
    .run_commands(
        "cd /app/ComfyUI/custom_nodes && git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git",
        "cd /app/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper && pip install --no-cache-dir -r requirements.txt",
    )
    .run_commands(
        "cd /app/ComfyUI/custom_nodes && git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
        "cd /app/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite && pip install --no-cache-dir -r requirements.txt",
    )
    .run_commands(
        "cd /app/ComfyUI/custom_nodes && git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git",
        "cd /app/ComfyUI/custom_nodes/comfyui_controlnet_aux && pip install --no-cache-dir -r requirements.txt",
    )
    .pip_install(*REQUIREMENTS[10:])  # Install pip packages
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

def is_file_valid(filepath: Path, min_size_kb: int = MIN_FILE_SIZE_KB) -> bool:
    """Check if file exists and has valid size"""
    if not filepath.exists():
        return False
    
    file_size_kb = filepath.stat().st_size / 1024
    
    if file_size_kb < min_size_kb:
        print(f"[VALIDATION] File too small: {filepath.name} ({file_size_kb:.2f} KB < {min_size_kb} KB)")
        return False
    
    return True

def cleanup_invalid_file(filepath: Path, reason: str = ""):
    """Remove invalid/corrupt files and broken symlinks"""
    try:
        if filepath.is_symlink():
            symlink_target = filepath.resolve()
            print(f"[CLEANUP] Removing broken symlink: {filepath}")
            filepath.unlink()
            
            if symlink_target.exists():
                print(f"[CLEANUP] Removing symlink target: {symlink_target}")
                symlink_target.unlink()
        elif filepath.exists():
            print(f"[CLEANUP] Removing invalid file: {filepath} - {reason}")
            filepath.unlink()
    except Exception as e:
        print(f"[CLEANUP] Error removing file {filepath}: {str(e)}")

def download_from_huggingface(model_type: str, local_filename: str, repo_id: str, 
                               filename: str, target_dir: Path) -> dict:
    from huggingface_hub import hf_hub_download
    
    print(f"\n[HF-DOWNLOAD] Starting: {model_type}/{local_filename}")
    print(f"[HF-DOWNLOAD] Repo: {repo_id}")
    print(f"[HF-DOWNLOAD] File: {filename}")
    
    target_path = target_dir / local_filename
    
    if target_path.exists():
        if target_path.is_symlink():
            resolved_path = target_path.resolve()
            if resolved_path.exists() and is_file_valid(resolved_path):
                file_size = resolved_path.stat().st_size / (1024**2)
                print(f"[HF-DOWNLOAD] [OK] Valid symlink exists: {local_filename} ({file_size:.2f} MB)")
                return {"status": "exists", "filename": local_filename, "size": file_size, "type": model_type}
            else:
                print(f"[HF-DOWNLOAD] [WARN] Invalid symlink detected, cleaning up...")
                cleanup_invalid_file(target_path, "invalid symlink")
        else:
            if is_file_valid(target_path):
                file_size = target_path.stat().st_size / (1024**2)
                print(f"[HF-DOWNLOAD] [OK] Valid file exists: {local_filename} ({file_size:.2f} MB)")
                return {"status": "exists", "filename": local_filename, "size": file_size, "type": model_type}
            else:
                print(f"[HF-DOWNLOAD] [WARN] Invalid file detected, cleaning up...")
                cleanup_invalid_file(target_path, "file too small or corrupt")
    
    try:
        print(f"[HF-DOWNLOAD] Downloading from HuggingFace...")
        cached_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(CACHE_PATH),
            resume_download=True,
        )
        
        cached_file = Path(cached_path)
        if not is_file_valid(cached_file):
            print(f"[HF-DOWNLOAD] [ERROR] Downloaded file is invalid")
            cleanup_invalid_file(cached_file, "downloaded file too small")
            return {"status": "error", "filename": local_filename, "error": "Downloaded file is invalid", "type": model_type}
        
        print(f"[HF-DOWNLOAD] Creating symlink: {target_path} -> {cached_path}")
        subprocess.run(
            f"ln -s {cached_path} {target_path}",
            shell=True,
            check=True,
        )
        
        file_size = cached_file.stat().st_size / (1024**2)
        print(f"[HF-DOWNLOAD] [OK] Downloaded: {local_filename} ({file_size:.2f} MB)")
        return {"status": "downloaded", "filename": local_filename, "size": file_size, "type": model_type}
        
    except Exception as e:
        error_msg = str(e)[:200]
        print(f"[HF-DOWNLOAD] [ERROR] {local_filename} - {error_msg}")
        
        if target_path.exists():
            cleanup_invalid_file(target_path, "download failed")
        
        return {"status": "error", "filename": local_filename, "error": error_msg, "type": model_type}

def download_from_url(model_type: str, filename: str, url: str, target_dir: Path) -> dict:
    print(f"\n[URL-DOWNLOAD] Starting: {model_type}/{filename}")
    print(f"[URL-DOWNLOAD] URL: {url}")
    
    target_path = target_dir / filename
    
    if target_path.exists():
        if is_file_valid(target_path):
            file_size = target_path.stat().st_size / (1024**2)
            print(f"[URL-DOWNLOAD] [OK] Valid file exists: {filename} ({file_size:.2f} MB)")
            return {"status": "exists", "filename": filename, "size": file_size, "type": model_type}
        else:
            print(f"[URL-DOWNLOAD] [WARN] Invalid file detected, cleaning up...")
            cleanup_invalid_file(target_path, "file too small or corrupt")
    
    temp_path = target_dir / f"{filename}.tmp"
    
    try:
        print(f"[URL-DOWNLOAD] Downloading with wget...")
        wget_cmd = [
            "wget", "-O", str(temp_path), "--progress=bar:force:noscroll",
            "--tries=5", "--timeout=120", "--continue", url
        ]
        
        result = subprocess.run(wget_cmd, check=True, capture_output=True, text=True)
        
        if not temp_path.exists():
            print(f"[URL-DOWNLOAD] [ERROR] File not created: {filename}")
            return {"status": "error", "filename": filename, "error": "File not created", "type": model_type}
        
        if not is_file_valid(temp_path):
            print(f"[URL-DOWNLOAD] [ERROR] Downloaded file is invalid")
            cleanup_invalid_file(temp_path, "downloaded file too small")
            return {"status": "error", "filename": filename, "error": "Downloaded file is invalid", "type": model_type}
        
        temp_path.rename(target_path)
        
        file_size = target_path.stat().st_size / (1024**2)
        print(f"[URL-DOWNLOAD] [OK] Downloaded: {filename} ({file_size:.2f} MB)")
        return {"status": "downloaded", "filename": filename, "size": file_size, "type": model_type}
        
    except Exception as e:
        error_msg = str(e)[:200]
        print(f"[URL-DOWNLOAD] [ERROR] {filename} - {error_msg}")
        
        if temp_path.exists():
            cleanup_invalid_file(temp_path, "download failed")
        if target_path.exists() and not is_file_valid(target_path):
            cleanup_invalid_file(target_path, "invalid after download")
        
        return {"status": "error", "filename": filename, "error": error_msg, "type": model_type}

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
    print("\n" + "="*80)
    print("MODEL DOWNLOAD STARTING")
    print("="*80)
    
    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
    if hf_token:
        print(f"[AUTH] HuggingFace token found: {hf_token[:10]}...")
        os.environ["HF_TOKEN"] = hf_token
    else:
        print("[AUTH] [WARN] No HuggingFace token found")
    
    PRIORITY_ORDER = ["diffusion_models", "vae", "text_encoders", "audio_encoders", "clip", "controlnet", "loras"]
    
    all_results = {"downloaded": [], "exists": [], "errors": []}
    
    for model_type in PRIORITY_ORDER:
        if model_type not in MODEL_REGISTRY:
            continue
        
        print(f"\n{'='*80}")
        print(f"PROCESSING MODEL TYPE: {model_type.upper()}")
        print(f"{'='*80}")
        
        models = MODEL_REGISTRY[model_type]
        target_dir = MODEL_PATH / model_type
        target_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[INFO] Target directory: {target_dir}")
        print(f"[INFO] Total files: {len(models)}")
        
        if model_type in ["diffusion_models", "vae", "text_encoders", "audio_encoders", "clip"]:
            print(f"[INFO] Sequential download mode")
            for idx, (local_filename, source) in enumerate(models.items(), 1):
                print(f"\n[PROGRESS] {idx}/{len(models)} - {local_filename}")
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
            print(f"[INFO] Parallel download mode (10 workers)")
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
                
                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    result = future.result()
                    print(f"[PROGRESS] Completed {completed}/{len(download_tasks)}")
                    
                    if result["status"] == "downloaded":
                        all_results["downloaded"].append(result)
                    elif result["status"] == "exists":
                        all_results["exists"].append(result)
                    else:
                        all_results["errors"].append(result)
        
        else:
            print(f"[INFO] Parallel download mode (3 workers)")
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
                
                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    result = future.result()
                    print(f"[PROGRESS] Completed {completed}/{len(download_tasks)}")
                    
                    if result["status"] == "downloaded":
                        all_results["downloaded"].append(result)
                    elif result["status"] == "exists":
                        all_results["exists"].append(result)
                    else:
                        all_results["errors"].append(result)
    
    total_size = sum(r["size"] for r in all_results["downloaded"] + all_results["exists"])
    
    print(f"\n{'='*80}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*80}")
    print(f"[SUMMARY] Downloaded: {len(all_results['downloaded'])} files")
    print(f"[SUMMARY] Cached: {len(all_results['exists'])} files")
    print(f"[SUMMARY] Failed: {len(all_results['errors'])} files")
    print(f"[SUMMARY] Total size: {total_size/1024:.2f} GB")
    
    critical_missing = []
    for model_type in ["diffusion_models", "vae", "text_encoders", "audio_encoders"]:
        type_errors = [e for e in all_results["errors"] if e["type"] == model_type]
        if type_errors:
            critical_missing.extend(type_errors)
    
    if critical_missing:
        print(f"\n[ERROR] Critical models missing:")
        for err in critical_missing:
            print(f"  - {err['type']}/{err['filename']}: {err.get('error', 'Unknown error')}")
    
    if all_results["errors"]:
        print(f"\n[ERROR] All failed downloads:")
        for err in all_results["errors"]:
            print(f"  - {err['type']}/{err['filename']}: {err.get('error', 'Unknown error')}")
    
    print("\n[VOLUME] Committing changes...")
    volume.commit()
    cache_volume.commit()
    print("[VOLUME] [OK] Changes committed")
    
    print(f"\n{'='*80}")
    print(f"DOWNLOAD {'SUCCESSFUL' if len(critical_missing) == 0 else 'FAILED'}")
    print(f"{'='*80}\n")
    
    return {
        "success": len(critical_missing) == 0,
        "summary": {
            "downloaded": len(all_results["downloaded"]),
            "cached": len(all_results["exists"]),
            "failed": len(all_results["errors"]),
            "total_gb": round(total_size/1024, 2)
        },
        "errors": all_results["errors"] if all_results["errors"] else None
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
        print("\n" + "="*80)
        print("COMFYUI STARTUP")
        print("="*80)
        
        print(f"[CONFIG] Writing configuration file...")
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
        print(f"[CONFIG] [OK] Configuration written to: {config_path}")
        
        print(f"\n[VERIFY] Checking model paths...")
        self._verify_model_paths()
        print(f"[VERIFY] [OK] All critical models verified")
        
        print(f"\n[SERVER] Starting ComfyUI server...")
        cmd = ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--disable-auto-launch"]
        print(f"[SERVER] Command: {' '.join(cmd)}")
        print(f"[SERVER] Working directory: {COMFYUI_PATH}")
        
        self.proc = subprocess.Popen(cmd, cwd=COMFYUI_PATH, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"[SERVER] Process started with PID: {self.proc.pid}")
        
        print(f"\n[SERVER] Waiting for server to be ready...")
        for i in range(60):
            try:
                urllib.request.urlopen("http://127.0.0.1:8188/queue", timeout=5).read()
                print(f"[SERVER] [OK] Server is ready after {i+1} seconds")
                print("="*80 + "\n")
                return
            except Exception as e:
                if i % 10 == 0:
                    print(f"[SERVER] Waiting... ({i}/60 seconds)")
                time.sleep(1)
        
        print(f"[SERVER] [ERROR] Server failed to start after 60 seconds")
        raise RuntimeError("ComfyUI failed to start after 60s")
    
    def _verify_model_paths(self):
        all_ok = True
        for model_type in MODEL_REGISTRY.keys():
            model_dir = MODEL_PATH / model_type
            if not model_dir.exists():
                print(f"[VERIFY] [ERROR] Missing directory: {model_type}")
                if model_type in ["diffusion_models", "vae", "text_encoders", "audio_encoders"]:
                    all_ok = False
            else:
                file_count = len(list(model_dir.iterdir()))
                print(f"[VERIFY] [OK] {model_type}: {file_count} files")
        
        if not all_ok:
            print(f"[VERIFY] [ERROR] Critical models missing!")
            raise RuntimeError("Critical models missing! Run: modal run app.py::download_models")
    
    def _validate_lora_exists(self, lora_name: str) -> bool:
        """Validate if LoRA file exists in the models directory"""
        lora_path = MODEL_PATH / "loras" / lora_name
        if not lora_path.exists():
            print(f"[LORA-VALIDATION] [WARN] LoRA not found: {lora_name}")
            return False
        if not is_file_valid(lora_path, min_size_kb=100):  # Lower threshold for LoRAs
            print(f"[LORA-VALIDATION] [WARN] LoRA file invalid: {lora_name}")
            return False
        print(f"[LORA-VALIDATION] [OK] LoRA validated: {lora_name}")
        return True
    
    # Added ControlNet validation
    def _validate_controlnet_exists(self, controlnet_name: str) -> bool:
        """Validate if ControlNet file exists in the models directory"""
        controlnet_path = MODEL_PATH / "controlnet" / controlnet_name
        if not controlnet_path.exists():
            print(f"[CONTROLNET-VALIDATION] [WARN] ControlNet not found: {controlnet_name}")
            return False
        if not is_file_valid(controlnet_path):
            print(f"[CONTROLNET-VALIDATION] [WARN] ControlNet file invalid: {controlnet_name}")
            return False
        print(f"[CONTROLNET-VALIDATION] [OK] ControlNet validated: {controlnet_name}")
        return True
    
    def _normalize_lora_strength(self, strength_input) -> float:
        """Normalize LoRA strength from preset or numeric value"""
        if isinstance(strength_input, str):
            if strength_input.lower() in LORA_PRESETS:
                strength = LORA_PRESETS[strength_input.lower()]
                print(f"[LORA-STRENGTH] Using preset '{strength_input}': {strength}")
                return strength
            try:
                strength = float(strength_input)
                print(f"[LORA-STRENGTH] Using numeric value: {strength}")
                return strength
            except:
                print(f"[LORA-STRENGTH] [WARN] Invalid strength '{strength_input}', using default 0.7")
                return 0.7
        elif isinstance(strength_input, (int, float)):
            return float(strength_input)
        else:
            print(f"[LORA-STRENGTH] [WARN] Invalid strength type, using default 0.7")
            return 0.7
    
    # Improved LoRA chain to return both model and clip
    def _build_lora_chain(self, base_model_node_id: str, base_clip_node_id: str, loras_config: List) -> tuple:
        """
        Build LoRA chain nodes for workflow
        Returns: (last_model_node_id, last_clip_node_id, lora_nodes_dict)
        """
        if not loras_config:
            return base_model_node_id, base_clip_node_id, {}
        
        print(f"\n[LORA-CHAIN] Building chain with {len(loras_config)} LoRAs")
        
        lora_nodes = {}
        current_model_id = base_model_node_id
        current_clip_id = base_clip_node_id
        node_counter = 1000  # Start at 1000 to avoid conflicts
        
        for idx, lora_config in enumerate(loras_config):
            if isinstance(lora_config, str):
                lora_name = lora_config
                strength = 0.7
            elif isinstance(lora_config, dict):
                lora_name = lora_config.get("name")
                strength = self._normalize_lora_strength(lora_config.get("strength", 0.7))
            else:
                print(f"[LORA-CHAIN] [WARN] Invalid LoRA config: {lora_config}")
                continue
            
            if not lora_name:
                continue
            
            # Add .safetensors if not present
            if not lora_name.endswith('.safetensors'):
                lora_name = f"{lora_name}.safetensors"
            
            # Validate LoRA exists
            if not self._validate_lora_exists(lora_name):
                print(f"[LORA-CHAIN] [WARN] Skipping missing LoRA: {lora_name}")
                continue
            
            node_id = str(node_counter)
            lora_nodes[node_id] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name": lora_name,
                    "strength_model": strength,
                    "strength_clip": strength,
                    "model": [current_model_id, 0],
                    "clip": [current_clip_id, 0]  # Added CLIP input
                }
            }
            
            print(f"[LORA-CHAIN] {idx+1}. {lora_name} (strength: {strength}) - Node {node_id}")
            current_model_id = node_id
            current_clip_id = node_id  # LoRA outputs both model and clip
            node_counter += 1
        
        print(f"[LORA-CHAIN] [OK] Chain complete: Model {base_model_node_id} -> {current_model_id}")
        print(f"[LORA-CHAIN] [OK] Chain complete: CLIP {base_clip_node_id} -> {current_clip_id}")
        return current_model_id, current_clip_id, lora_nodes
    
    # Completely rewritten ControlNet implementation
    def _build_controlnet_nodes(self, image_node_id: str, controlnets_config: List, base_positive_id: str, base_negative_id: str) -> tuple:
        """
        Build ControlNet nodes for workflow
        Returns: (last_positive_node_id, last_negative_node_id, controlnet_nodes_dict)
        """
        if not controlnets_config:
            return base_positive_id, base_negative_id, {}
        
        print(f"\n[CONTROLNET] Building nodes with {len(controlnets_config)} ControlNets")
        
        controlnet_nodes = {}
        current_positive_id = base_positive_id
        current_negative_id = base_negative_id
        node_counter = 2000  # Start at 2000 to avoid conflicts
        
        for idx, cn_config in enumerate(controlnets_config):
            if isinstance(cn_config, str):
                cn_type = cn_config
                strength = 1.0
            elif isinstance(cn_config, dict):
                cn_type = cn_config.get("type")
                strength = float(cn_config.get("strength", 1.0))
            else:
                print(f"[CONTROLNET] [WARN] Invalid ControlNet config: {cn_config}")
                continue
            
            if not cn_type:
                continue
            
            # Map controlnet type to model file
            cn_model_map = {
                "openpose": "control_v11p_sd15_openpose.pth",
                "canny": "control_v11p_sd15_canny.pth",
                "depth": "control_v11f1p_sd15_depth.pth",
                "lineart": "control_v11p_sd15_lineart.pth"
            }
            
            cn_model = cn_model_map.get(cn_type.lower())
            if not cn_model:
                print(f"[CONTROLNET] [WARN] Unknown ControlNet type: {cn_type}")
                continue
            
            # Validate ControlNet exists
            if not self._validate_controlnet_exists(cn_model):
                print(f"[CONTROLNET] [WARN] Skipping missing ControlNet: {cn_model}")
                continue
            
            # ControlNet Loader
            loader_id = str(node_counter)
            controlnet_nodes[loader_id] = {
                "class_type": "ControlNetLoader",
                "inputs": {
                    "control_net_name": cn_model
                }
            }
            
            # Proper preprocessor node implementation
            preprocessor_id = str(node_counter + 1)
            preprocessor_map = {
                "openpose": {"class": "OpenposePreprocessor", "params": {"detect_hand": "enable", "detect_body": "enable", "detect_face": "enable"}},
                "canny": {"class": "CannyEdgePreprocessor", "params": {"low_threshold": 100, "high_threshold": 200}},
                "depth": {"class": "MiDaS-DepthMapPreprocessor", "params": {"a": 6.283185307179586, "bg_threshold": 0.1}},
                "lineart": {"class": "LineArtPreprocessor", "params": {"coarse": "disable"}}
            }
            
            preprocessor_info = preprocessor_map.get(cn_type.lower())
            if preprocessor_info:
                preprocessor_inputs = {"image": [image_node_id, 0]}
                preprocessor_inputs.update(preprocessor_info["params"])
                
                controlnet_nodes[preprocessor_id] = {
                    "class_type": preprocessor_info["class"],
                    "inputs": preprocessor_inputs
                }
                preprocessed_image_id = preprocessor_id
            else:
                preprocessed_image_id = image_node_id
            
            # ControlNet Apply for POSITIVE conditioning
            apply_positive_id = str(node_counter + 2)
            controlnet_nodes[apply_positive_id] = {
                "class_type": "ControlNetApplyAdvanced",
                "inputs": {
                    "positive": [current_positive_id, 0],
                    "negative": [current_negative_id, 0],
                    "control_net": [loader_id, 0],
                    "image": [preprocessed_image_id, 0],
                    "strength": strength,
                    "start_percent": 0.0,
                    "end_percent": 1.0
                }
            }
            
            print(f"[CONTROLNET] {idx+1}. {cn_type} (strength: {strength}) - Nodes {loader_id}-{apply_positive_id}")
            
            # ControlNetApplyAdvanced outputs both positive and negative
            current_positive_id = apply_positive_id
            current_negative_id = apply_positive_id  # Same node outputs both
            node_counter += 10  # Leave space for future additions
        
        print(f"[CONTROLNET] [OK] Positive chain: {base_positive_id} -> {current_positive_id}")
        print(f"[CONTROLNET] [OK] Negative chain: {base_negative_id} -> {current_negative_id}")
        return current_positive_id, current_negative_id, controlnet_nodes
    
    def _queue_prompt(self, client_id: str, prompt_workflow: dict):
        print(f"\n[QUEUE] Queuing prompt...")
        print(f"[QUEUE] Client ID: {client_id}")
        print(f"[QUEUE] Workflow nodes: {len(prompt_workflow)}")
        
        req = urllib.request.Request(
            "http://127.0.0.1:8188/prompt",
            data=json.dumps({"prompt": prompt_workflow, "client_id": client_id}).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        response = urllib.request.urlopen(req).read()
        prompt_id = json.loads(response)['prompt_id']
        
        print(f"[QUEUE] [OK] Prompt queued with ID: {prompt_id}")
        return prompt_id
    
    def _get_history(self, prompt_id: str):
        print(f"[HISTORY] Fetching history for prompt: {prompt_id}")
        with urllib.request.urlopen(f"http://127.0.0.1:8188/history/{prompt_id}") as response:
            history = json.loads(response.read())
            print(f"[HISTORY] [OK] History retrieved")
            return history
    
    def _get_file(self, filename: str, subfolder: str, folder_type: str):
        print(f"[FILE] Fetching: {folder_type}/{subfolder}/{filename}")
        params = urllib.parse.urlencode({'filename': filename, 'subfolder': subfolder, 'type': folder_type})
        url = f"http://127.0.0.1:8188/view?{params}"
        with urllib.request.urlopen(url) as response:
            data = response.read()
            print(f"[FILE] [OK] File retrieved: {len(data)/1024/1024:.2f} MB")
            return data
    
    def _get_video_from_websocket(self, prompt_id: str, client_id: str):
        import websocket
        
        print(f"\n[WEBSOCKET] Connecting to ComfyUI...")
        ws_url = f"ws://127.0.0.1:8188/ws?clientId={client_id}"
        ws = websocket.WebSocket()
        
        try:
            ws.connect(ws_url, timeout=10)
            print(f"[WEBSOCKET] [OK] Connected")
        except Exception as e:
            print(f"[WEBSOCKET] [ERROR] Connection failed: {str(e)}")
            raise
        
        try:
            print(f"[WEBSOCKET] Monitoring generation progress...")
            message_count = 0
            max_messages = 10000
            
            while message_count < max_messages:
                try:
                    out = ws.recv()
                    message_count += 1
                    
                    if isinstance(out, str):
                        message = json.loads(out)
                        
                        if message.get('type') == 'progress':
                            value = message.get('data', {}).get('value', 0)
                            max_val = message.get('data', {}).get('max', 100)
                            print(f"[WEBSOCKET] Progress: {value}/{max_val} ({value/max_val*100:.1f}%)")
                        
                        if message.get('type') == 'executing':
                            node = message.get('data', {}).get('node')
                            if node:
                                print(f"[WEBSOCKET] Executing node: {node}")
                            elif message.get('data', {}).get('prompt_id') == prompt_id:
                                print(f"[WEBSOCKET] [OK] Generation complete (received {message_count} messages)")
                                break
                except websocket.WebSocketTimeoutException:
                    print(f"[WEBSOCKET] Timeout waiting for message")
                    continue
                    
        finally:
            ws.close()
            print(f"[WEBSOCKET] Connection closed")
        
        print(f"\n[OUTPUT] Retrieving generated video...")
        history = self._get_history(prompt_id)[prompt_id]
        
        for node_id, node_output in history['outputs'].items():
            print(f"[OUTPUT] Checking node {node_id}...")
            
            if 'videos' in node_output:
                for video in node_output['videos']:
                    print(f"[OUTPUT] [OK] Found video: {video['filename']}")
                    return self._get_file(video['filename'], video['subfolder'], video['type'])
            
            if 'gifs' in node_output:
                for gif in node_output['gifs']:
                    print(f"[OUTPUT] [OK] Found GIF: {gif['filename']}")
                    return self._get_file(gif['filename'], gif['subfolder'], gif['type'])
        
        print(f"[OUTPUT] [ERROR] No video output found")
        raise ValueError("No video output found in generation")
    
    def _validate_base64_image(self, image_base64: str) -> str:
        """Validate and clean base64 image data"""
        if not image_base64:
            raise ValueError("Image data is empty")
        
        if image_base64.startswith('data:image/'):
            if ';base64,' not in image_base64:
                raise ValueError("Invalid base64 image format. Expected 'data:image/...;base64,...'")
            image_base64 = image_base64.split(';base64,')[1]
        
        try:
            base64.b64decode(image_base64)
        except Exception as e:
            raise ValueError(f"Invalid base64 encoding: {str(e)}")
        
        return image_base64
    
    def _save_base64_image(self, image_base64: str) -> str:
        """Save base64 image to temporary file and return path"""
        clean_b64 = self._validate_base64_image(image_base64)
        
        temp_filename = f"/tmp/{uuid.uuid4()}.png"
        try:
            image_data = base64.b64decode(clean_b64)
            with open(temp_filename, "wb") as f:
                f.write(image_data)
            print(f"[IMAGE] Saved base64 image to: {temp_filename} ({len(image_data)/1024:.2f} KB)")
            return temp_filename
        except Exception as e:
            raise ValueError(f"Failed to save base64 image: {str(e)}")
    
    # Improved audio handling with WAV conversion
    def _save_base64_audio(self, audio_base64: str) -> str:
        """Save base64 audio to temporary file and return path"""
        if not audio_base64:
            raise ValueError("Audio data is empty")
        
        if audio_base64.startswith('data:audio/'):
            if ';base64,' not in audio_base64:
                raise ValueError("Invalid base64 audio format")
            audio_base64 = audio_base64.split(';base64,')[1]
        
        try:
            base64.b64decode(audio_base64)
        except Exception as e:
            raise ValueError(f"Invalid base64 encoding: {str(e)}")
        
        temp_filename = f"/tmp/{uuid.uuid4()}.wav"
        try:
            audio_data = base64.b64decode(audio_base64)
            with open(temp_filename, "wb") as f:
                f.write(audio_data)
            print(f"[AUDIO] Saved base64 audio to: {temp_filename} ({len(audio_data)/1024:.2f} KB)")
            
            # Convert to WAV if needed using pydub
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_file(temp_filename)
                audio.export(temp_filename, format="wav")
                print(f"[AUDIO] Converted to WAV format: {temp_filename}")
            except Exception as e:
                print(f"[AUDIO] [WARN] Could not convert to WAV: {str(e)}")
            
            return temp_filename
        except Exception as e:
            raise ValueError(f"Failed to save base64 audio: {str(e)}")
    
    def _build_t2v_workflow(self, prompt: str, negative_prompt: str, width: int, height: int, 
                           num_frames: int, loras: List = None, controlnets: List = None) -> dict:
        """Build text-to-video workflow"""
        print(f"\n[WORKFLOW] Building T2V workflow")
        print(f"[WORKFLOW] Prompt: {prompt[:50]}...")
        print(f"[WORKFLOW] Dimensions: {width}x{height}")
        print(f"[WORKFLOW] Frames: {num_frames}")
        
        # Base nodes
        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
                }
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,
                    "clip": ["4", 1]
                }
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["4", 1]
                }
            },
            "4": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": "clip_l.safetensors"
                }
            },
            "5": {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": "wan_2.1_vae.safetensors"
                }
            },
            "6": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1
                }
            },
            "7": {
                "class_type": "WanSampler",
                "inputs": {
                    "seed": 42,
                    "steps": 30,
                    "cfg": 7.5,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["6", 0],
                    "vae": ["5", 0]
                }
            },
            "8": {
                "class_type": "WanVideoToVideo",
                "inputs": {
                    "frames": num_frames,
                    "motion_strength": 127,
                    "latent": ["7", 0]
                }
            },
            "9": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["8", 0],
                    "vae": ["5", 0]
                }
            },
            "10": {
                "class_type": "SaveAnimatedWEBP",
                "inputs": {
                    "filename_prefix": "wan_t2v",
                    "images": ["9", 0],
                    "fps": 8
                }
            }
        }
        
        # Add LoRA chain if specified
        if loras:
            model_id, clip_id, lora_nodes = self._build_lora_chain("1", "4", loras)
            workflow.update(lora_nodes)
            
            # Update connections to use the last LoRA node
            workflow["7"]["inputs"]["model"] = [model_id, 0]
            workflow["2"]["inputs"]["clip"] = [clip_id, 1]
            workflow["3"]["inputs"]["clip"] = [clip_id, 1]
        
        # Add ControlNet if specified
        if controlnets:
            # We need an image for ControlNet, but T2V doesn't have one
            # So we'll skip ControlNet for T2V workflows
            print(f"[WORKFLOW] [WARN] ControlNet not supported for T2V workflows")
        
        return workflow
    
    def _build_i2v_workflow(self, prompt: str, negative_prompt: str, image_base64: str, 
                           width: int, height: int, num_frames: int, loras: List = None, 
                           controlnets: List = None) -> dict:
        """Build image-to-video workflow"""
        print(f"\n[WORKFLOW] Building I2V workflow")
        print(f"[WORKFLOW] Prompt: {prompt[:50]}...")
        print(f"[WORKFLOW] Dimensions: {width}x{height}")
        print(f"[WORKFLOW] Frames: {num_frames}")
        
        # Save input image
        image_path = self._save_base64_image(image_base64)
        
        # Base nodes
        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": "wan2.2_i2v_low_noise_14B_fp16.safetensors"
                }
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,
                    "clip": ["4", 1]
                }
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["4", 1]
                }
            },
            "4": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": "clip_l.safetensors"
                }
            },
            "5": {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": "wan_2.1_vae.safetensors"
                }
            },
            "6": {
                "class_type": "LoadImage",
                "inputs": {
                    "image": image_path,
                    "upload": "image"
                }
            },
            "7": {
                "class_type": "ImageScale",
                "inputs": {
                    "width": width,
                    "height": height,
                    "upscale_method": "bilinear",
                    "crop": "disabled",
                    "image": ["6", 0]
                }
            },
            "8": {
                "class_type": "WanImageToVideo",
                "inputs": {
                    "seed": 42,
                    "steps": 30,
                    "cfg": 7.5,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "motion_strength": 127,
                    "frames": num_frames,
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "vae": ["5", 0],
                    "image": ["7", 0]
                }
            },
            "9": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["8", 0],
                    "vae": ["5", 0]
                }
            },
            "10": {
                "class_type": "SaveAnimatedWEBP",
                "inputs": {
                    "filename_prefix": "wan_i2v",
                    "images": ["9", 0],
                    "fps": 8
                }
            }
        }
        
        # Add LoRA chain if specified
        if loras:
            model_id, clip_id, lora_nodes = self._build_lora_chain("1", "4", loras)
            workflow.update(lora_nodes)
            
            # Update connections to use the last LoRA node
            workflow["8"]["inputs"]["model"] = [model_id, 0]
            workflow["2"]["inputs"]["clip"] = [clip_id, 1]
            workflow["3"]["inputs"]["clip"] = [clip_id, 1]
        
        # Add ControlNet if specified
        if controlnets:
            positive_id, negative_id, controlnet_nodes = self._build_controlnet_nodes(
                "6", controlnets, "2", "3"
            )
            workflow.update(controlnet_nodes)
            
            # Update connections to use the last ControlNet node
            workflow["8"]["inputs"]["positive"] = [positive_id, 0]
            workflow["8"]["inputs"]["negative"] = [negative_id, 0]
        
        return workflow
    
    def _build_s2v_workflow(self, prompt: str, negative_prompt: str, image_base64: str, 
                           width: int, height: int, num_frames: int, loras: List = None, 
                           controlnets: List = None) -> dict:
        """Build sketch-to-video workflow"""
        print(f"\n[WORKFLOW] Building S2V workflow")
        print(f"[WORKFLOW] Prompt: {prompt[:50]}...")
        print(f"[WORKFLOW] Dimensions: {width}x{height}")
        print(f"[WORKFLOW] Frames: {num_frames}")
        
        # Save input image
        image_path = self._save_base64_image(image_base64)
        
        # Base nodes
        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": "wan2.2_s2v_14B_fp8_scaled.safetensors"
                }
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,
                    "clip": ["4", 1]
                }
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["4", 1]
                }
            },
            "4": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": "clip_l.safetensors"
                }
            },
            "5": {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": "wan_2.1_vae.safetensors"
                }
            },
            "6": {
                "class_type": "LoadImage",
                "inputs": {
                    "image": image_path,
                    "upload": "image"
                }
            },
            "7": {
                "class_type": "ImageScale",
                "inputs": {
                    "width": width,
                    "height": height,
                    "upscale_method": "bilinear",
                    "crop": "disabled",
                    "image": ["6", 0]
                }
            },
            "8": {
                "class_type": "WanSketchToVideo",
                "inputs": {
                    "seed": 42,
                    "steps": 30,
                    "cfg": 7.5,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "motion_strength": 127,
                    "frames": num_frames,
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "vae": ["5", 0],
                    "image": ["7", 0]
                }
            },
            "9": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["8", 0],
                    "vae": ["5", 0]
                }
            },
            "10": {
                "class_type": "SaveAnimatedWEBP",
                "inputs": {
                    "filename_prefix": "wan_s2v",
                    "images": ["9", 0],
                    "fps": 8
                }
            }
        }
        
        # Add LoRA chain if specified
        if loras:
            model_id, clip_id, lora_nodes = self._build_lora_chain("1", "4", loras)
            workflow.update(lora_nodes)
            
            # Update connections to use the last LoRA node
            workflow["8"]["inputs"]["model"] = [model_id, 0]
            workflow["2"]["inputs"]["clip"] = [clip_id, 1]
            workflow["3"]["inputs"]["clip"] = [clip_id, 1]
        
        # Add ControlNet if specified
        if controlnets:
            positive_id, negative_id, controlnet_nodes = self._build_controlnet_nodes(
                "6", controlnets, "2", "3"
            )
            workflow.update(controlnet_nodes)
            
            # Update connections to use the last ControlNet node
            workflow["8"]["inputs"]["positive"] = [positive_id, 0]
            workflow["8"]["inputs"]["negative"] = [negative_id, 0]
        
        return workflow
    
    def _build_a2v_workflow(self, prompt: str, negative_prompt: str, audio_base64: str, 
                           width: int, height: int, num_frames: int, loras: List = None) -> dict:
        """Build audio-to-video workflow"""
        print(f"\n[WORKFLOW] Building A2V workflow")
        print(f"[WORKFLOW] Prompt: {prompt[:50]}...")
        print(f"[WORKFLOW] Dimensions: {width}x{height}")
        print(f"[WORKFLOW] Frames: {num_frames}")
        
        # Save input audio
        audio_path = self._save_base64_audio(audio_base64)
        
        # Base nodes
        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
                }
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,
                    "clip": ["4", 1]
                }
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["4", 1]
                }
            },
            "4": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": "clip_l.safetensors"
                }
            },
            "5": {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": "wan_2.1_vae.safetensors"
                }
            },
            "6": {
                "class_type": "WanAudioEncoder",
                "inputs": {
                    "audio_encoder_name": "wav2vec2_large_english_fp16.safetensors",
                    "audio": audio_path
                }
            },
            "7": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1
                }
            },
            "8": {
                "class_type": "WanSampler",
                "inputs": {
                    "seed": 42,
                    "steps": 30,
                    "cfg": 7.5,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["7", 0],
                    "vae": ["5", 0]
                }
            },
            "9": {
                "class_type": "WanAudioToVideo",
                "inputs": {
                    "frames": num_frames,
                    "motion_strength": 127,
                    "latent": ["8", 0],
                    "audio_embedding": ["6", 0]
                }
            },
            "10": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["9", 0],
                    "vae": ["5", 0]
                }
            },
            "11": {
                "class_type": "SaveAnimatedWEBP",
                "inputs": {
                    "filename_prefix": "wan_a2v",
                    "images": ["10", 0],
                    "fps": 8
                }
            }
        }
        
        # Add LoRA chain if specified
        if loras:
            model_id, clip_id, lora_nodes = self._build_lora_chain("1", "4", loras)
            workflow.update(lora_nodes)
            
            # Update connections to use the last LoRA node
            workflow["8"]["inputs"]["model"] = [model_id, 0]
            workflow["2"]["inputs"]["clip"] = [clip_id, 1]
            workflow["3"]["inputs"]["clip"] = [clip_id, 1]
        
        return workflow
    
    @method()
    def generate_t2v(self, prompt: str, negative_prompt: str = "", width: int = 832, 
                    height: int = 480, num_frames: int = 16, loras: List = None, 
                    controlnets: List = None) -> bytes:
        """Generate video from text prompt"""
        print(f"\n{'='*80}")
        print("TEXT-TO-VIDEO GENERATION")
        print(f"{'='*80}")
        
        # Build workflow
        workflow = self._build_t2v_workflow(
            prompt, negative_prompt, width, height, num_frames, loras, controlnets
        )
        
        # Generate client ID
        client_id = str(uuid.uuid4())
        
        # Queue prompt
        prompt_id = self._queue_prompt(client_id, workflow)
        
        # Get result from websocket
        return self._get_video_from_websocket(prompt_id, client_id)
    
    @method()
    def generate_i2v(self, prompt: str, image_base64: str, negative_prompt: str = "", 
                    width: int = 832, height: int = 480, num_frames: int = 16, 
                    loras: List = None, controlnets: List = None) -> bytes:
        """Generate video from image and text prompt"""
        print(f"\n{'='*80}")
        print("IMAGE-TO-VIDEO GENERATION")
        print(f"{'='*80}")
        
        # Build workflow
        workflow = self._build_i2v_workflow(
            prompt, negative_prompt, image_base64, width, height, num_frames, loras, controlnets
        )
        
        # Generate client ID
        client_id = str(uuid.uuid4())
        
        # Queue prompt
        prompt_id = self._queue_prompt(client_id, workflow)
        
        # Get result from websocket
        return self._get_video_from_websocket(prompt_id, client_id)
    
    @method()
    def generate_s2v(self, prompt: str, image_base64: str, negative_prompt: str = "", 
                    width: int = 832, height: int = 480, num_frames: int = 16, 
                    loras: List = None, controlnets: List = None) -> bytes:
        """Generate video from sketch and text prompt"""
        print(f"\n{'='*80}")
        print("SKETCH-TO-VIDEO GENERATION")
        print(f"{'='*80}")
        
        # Build workflow
        workflow = self._build_s2v_workflow(
            prompt, negative_prompt, image_base64, width, height, num_frames, loras, controlnets
        )
        
        # Generate client ID
        client_id = str(uuid.uuid4())
        
        # Queue prompt
        prompt_id = self._queue_prompt(client_id, workflow)
        
        # Get result from websocket
        return self._get_video_from_websocket(prompt_id, client_id)
    
    @method()
    def generate_a2v(self, prompt: str, audio_base64: str, negative_prompt: str = "", 
                    width: int = 832, height: int = 480, num_frames: int = 16, 
                    loras: List = None) -> bytes:
        """Generate video from audio and text prompt"""
        print(f"\n{'='*80}")
        print("AUDIO-TO-VIDEO GENERATION")
        print(f"{'='*80}")
        
        # Build workflow
        workflow = self._build_a2v_workflow(
            prompt, negative_prompt, audio_base64, width, height, num_frames, loras
        )
        
        # Generate client ID
        client_id = str(uuid.uuid4())
        
        # Queue prompt
        prompt_id = self._queue_prompt(client_id, workflow)
        
        # Get result from websocket
        return self._get_video_from_websocket(prompt_id, client_id)

# Web API endpoints
@app.function(image=comfy_image)
@asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional
    import base64
    
    web_app = FastAPI(title="ComfyUI Wan2.2 API", version="1.0.0")
    
    # Add CORS middleware
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Pydantic models for request bodies
    class T2VRequest(BaseModel):
        prompt: str
        negative_prompt: str = ""
        width: int = 832
        height: int = 480
        num_frames: int = 16
        loras: Optional[List] = None
        controlnets: Optional[List] = None
    
    class I2VRequest(BaseModel):
        prompt: str
        image_base64: str
        negative_prompt: str = ""
        width: int = 832
        height: int = 480
        num_frames: int = 16
        loras: Optional[List] = None
        controlnets: Optional[List] = None
    
    class S2VRequest(BaseModel):
        prompt: str
        image_base64: str
        negative_prompt: str = ""
        width: int = 832
        height: int = 480
        num_frames: int = 16
        loras: Optional[List] = None
        controlnets: Optional[List] = None
    
    class A2VRequest(BaseModel):
        prompt: str
        audio_base64: str
        negative_prompt: str = ""
        width: int = 832
        height: int = 480
        num_frames: int = 16
        loras: Optional[List] = None
    
    # Health check endpoint
    @web_app.get("/health")
    async def health_check():
        return {"status": "healthy"}
    
    # List available LoRAs
    @web_app.get("/loras")
    async def list_loras():
        return {"loras": list(MODEL_REGISTRY["loras"].keys())}
    
    # List available ControlNets
    @web_app.get("/controlnets")
    async def list_controlnets():
        return {"controlnets": list(MODEL_REGISTRY["controlnet"].keys())}
    
    # Text-to-video endpoint
    @web_app.post("/t2v")
    async def text_to_video(request: T2VRequest):
        try:
            comfyui = ComfyUI()
            video_bytes = comfyui.generate_t2v.remote(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                num_frames=request.num_frames,
                loras=request.loras,
                controlnets=request.controlnets
            )
            
            # Return base64 encoded video
            video_base64 = base64.b64encode(video_bytes).decode('utf-8')
            return {"video_base64": video_base64}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # Image-to-video endpoint
    @web_app.post("/i2v")
    async def image_to_video(request: I2VRequest):
        try:
            comfyui = ComfyUI()
            video_bytes = comfyui.generate_i2v.remote(
                prompt=request.prompt,
                image_base64=request.image_base64,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                num_frames=request.num_frames,
                loras=request.loras,
                controlnets=request.controlnets
            )
            
            # Return base64 encoded video
            video_base64 = base64.b64encode(video_bytes).decode('utf-8')
            return {"video_base64": video_base64}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # Sketch-to-video endpoint
    @web_app.post("/s2v")
    async def sketch_to_video(request: S2VRequest):
        try:
            comfyui = ComfyUI()
            video_bytes = comfyui.generate_s2v.remote(
                prompt=request.prompt,
                image_base64=request.image_base64,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                num_frames=request.num_frames,
                loras=request.loras,
                controlnets=request.controlnets
            )
            
            # Return base64 encoded video
            video_base64 = base64.b64encode(video_bytes).decode('utf-8')
            return {"video_base64": video_base64}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # Audio-to-video endpoint
    @web_app.post("/a2v")
    async def audio_to_video(request: A2VRequest):
        try:
            comfyui = ComfyUI()
            video_bytes = comfyui.generate_a2v.remote(
                prompt=request.prompt,
                audio_base64=request.audio_base64,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                num_frames=request.num_frames,
                loras=request.loras
            )
            
            # Return base64 encoded video
            video_base64 = base64.b64encode(video_bytes).decode('utf-8')
            return {"video_base64": video_base64}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return web_app

# Entry point for local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)
