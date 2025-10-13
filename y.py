# ==============================================================================
# 1. IMPORTS & KONSTANTAS
# ==============================================================================
import base64
import io
import json
import os
import subprocess
import time
import urllib.request
import urllib.parse
import uuid
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any

from modal import App, Image, Volume, Secret, asgi_app, enter, method
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from PIL import Image as PILImage
from pydub import AudioSegment

# ==============================================================================
# 2. KONFIGURASI UTAMA
# ==============================================================================

app = App("comfyui-wan2-2-complete-api")

# Konstanta untuk menghindari nilai hardcoded
COMFYUI_PORT = 8188
COMFYUI_STARTUP_TIMEOUT = 120  # detik
WEBSOCKET_TIMEOUT = 10  # detik
MAX_WEBSOCKET_MESSAGES = 10000
MAX_CONSECUTIVE_TIMEOUTS = 5
MIN_FILE_SIZE_KB = 500
MIN_LORA_SIZE_KB = 100

# Path konfigurasi
REMOTE_BASE_PATH = Path("/app")
COMFYUI_PATH = REMOTE_BASE_PATH / "ComfyUI"
MODEL_PATH = Path("/models")
CACHE_PATH = Path("/cache")

# Volume
volume = Volume.from_name("comfyui-wan2-2-complete-volume", create_if_missing=True)
cache_volume = Volume.from_name("hf-hub-cache", create_if_missing=True)

# ==============================================================================
# 3. DEFINISI IMAGE (SEMUA DEPENDENSI DI ATAS)
# ==============================================================================

# Daftar semua dependensi Python yang diperlukan
PYTHON_REQUIREMENTS = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "websocket-client>=1.6.0",
    "safetensors>=0.4.0",
    "pillow>=10.0.0",
    "numpy>=1.24.0",
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "pydub>=0.25.1",
    "huggingface_hub[hf_transfer]>=0.34.0,<1.0",
]

comfy_image = (
    Image.debian_slim(python_version="3.10")
    # Instal semua dependensi Python di awal
    .pip_install(*PYTHON_REQUIREMENTS)
    # Instal dependensi sistem
    .apt_install(
        "git",
        "wget",
        "curl",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "ffmpeg",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
    )
    # Set environment variable untuk transfer HuggingFace yang lebih cepat
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # Kloning dan instal ComfyUI
    .run_commands(
        "mkdir -p /app /cache",
        "cd /app && git clone https://github.com/comfyanonymous/ComfyUI.git",
        "cd /app/ComfyUI && pip install --no-cache-dir -r requirements.txt",
    )
    # Kloning dan instal custom nodes
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
)

# ==============================================================================
# 4. MODEL REGISTRY & KONFIGURASI LAINNYA (VERSI REFACTOR LORA)
# ==============================================================================

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
        "control_v11p_sd15_openpose.pth": {"repo_id": "lllyasviel/ControlNet-v1-1", "filename": "control_v11p_sd15_openpose.pth"},
        "control_v11p_sd15_canny.pth": {"repo_id": "lllyasviel/ControlNet-v1-1", "filename": "control_v11p_sd15_canny.pth"},
        "control_v11f1p_sd15_depth.pth": {"repo_id": "lllyasviel/ControlNet-v1-1", "filename": "control_v11f1p_sd15_depth.pth"},
        "control_v11p_sd15_lineart.pth": {"repo_id": "lllyasviel/ControlNet-v1-1", "filename": "control_v11p_sd15_lineart.pth"},
    },
    "loras": {
        # --- KATEGORI: TEKNIS & KUALITAS (10 LoRA) ---
        "detail_tweaker_xl.safetensors": "https://civitai.com/api/download/models/135867",
        "add_detail.safetensors": "https://civitai.com/api/download/models/62833",
        "lowra.safetensors": "https://civitai.com/api/download/models/69527",
        "sharpness.safetensors": "https://civitai.com/api/download/models/369147",
        "noise_reduction.safetensors": "https://civitai.com/api/download/models/147258",
        "color_correction.safetensors": "https://civitai.com/api/download/models/258369",
        "film_grain.safetensors": "https://civitai.com/api/download/models/138564",
        "perfect_hands.safetensors": "https://civitai.com/api/download/models/99123",
        "better_eyes.safetensors": "https://civitai.com/api/download/models/121789",
        "depth_of_field.safetensors": "https://civitai.com/api/download/models/94789",

        # --- KATEGORI: GERAKAN & ANIMASI (5 LoRA) ---
        "v2_lora_ZoomIn.safetensors": {"repo_id": "guoyww/animatediff", "filename": "v2_lora_ZoomIn.safetensors"},
        "v2_lora_ZoomOut.safetensors": {"repo_id": "guoyww/animatediff", "filename": "v2_lora_ZoomOut.safetensors"},
        "v2_lora_PanLeft.safetensors": {"repo_id": "guoyww/animatediff", "filename": "v2_lora_PanLeft.safetensors"},
        "v2_lora_PanRight.safetensors": {"repo_id": "guoyww/animatediff", "filename": "v2_lora_PanRight.safetensors"},
        "v2_lora_TiltUp.safetensors": {"repo_id": "guoyww/animatediff", "filename": "v2_lora_TiltUp.safetensors"},

        # --- KATEGORI: GAYA SENI & MEDIUM (20 LoRA) ---
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
        "steampunk_style.safetensors": "https://civitai.com/api/download/models/14321",
        "impressionism.safetensors": "https://civitai.com/api/download/models/151234",
        "art_nouveau.safetensors": "https://civitai.com/api/download/models/152345",
        "ukiyo_e_style.safetensors": "https://civitai.com/api/download/models/153456",
        "surrealism.safetensors": "https://civitai.com/api/download/models/154567",
        "concept_art.safetensors": "https://civitai.com/api/download/models/155678",
        "matte_painting.safetensors": "https://civitai.com/api/download/models/156789",

        # --- KATEGORI: EFEK & ATMOSFER (25 LoRA) ---
        "rainy_day.safetensors": "https://civitai.com/api/download/models/97521",
        "smoke_fog.safetensors": "https://civitai.com/api/download/models/101582",
        "fire_flames.safetensors": "https://civitai.com/api/download/models/135789",
        "water_splash.safetensors": "https://civitai.com/api/download/models/87654",
        "snow_winter.safetensors": "https://civitai.com/api/download/models/76543",
        "desert_sand.safetensors": "https://civitai.com/api/download/models/65432",
        "lightning_effect.safetensors": "https://civitai.com/api/download/models/112345",
        "magic_spells.safetensors": "https://civitai.com/api/download/models/98765",
        "underwater_scene.safetensors": "https://civitai.com/api/download/models/111222",
        "space_scene.safetensors": "https://civitai.com/api/download/models/222333",
        "apocalyptic_scene.safetensors": "https://civitai.com/api/download/models/777888",
        "fantasy_forest.safetensors": "https://civitai.com/api/download/models/126789",
        "futuristic_city.safetensors": "https://civitai.com/api/download/models/93123",
        "horror_environment.safetensors": "https://civitai.com/api/download/models/93890",
        "medieval_castle.safetensors": "https://civitai.com/api/download/models/333444",
        "modern_interior.safetensors": "https://civitai.com/api/download/models/444555",
        "nature_landscape.safetensors": "https://civitai.com/api/download/models/555666",
        "urban_street.safetensors": "https://civitai.com/api/download/models/666777",
        "golden_hour.safetensors": "https://civitai.com/api/download/models/123123",
        "dramatic_lighting.safetensors": "https://civitai.com/api/download/models/234234",
        "soft_light.safetensors": "https://civitai.com/api/download/models/345345",
        "rim_lighting.safetensors": "https://civitai.com/api/download/models/456456",
        "volumetric_light.safetensors": "https://civitai.com/api/download/models/567567",
        "moonlight_effect.safetensors": "https://civitai.com/api/download/models/678678",

        # --- KATEGORI: KARAKTER & PAKAIAN (25 LoRA) ---
        "steampunk_fashion.safetensors": "https://civitai.com/api/download/models/14321",
        "witch_fashion.safetensors": "https://civitai.com/api/download/models/134876",
        "japanese_kimono.safetensors": "https://civitai.com/api/download/models/16789",
        "scifi_armor.safetensors": "https://civitai.com/api/download/models/96234",
        "hoodie_garment.safetensors": "https://civitai.com/api/download/models/96587",
        "leather_jacket.safetensors": "https://civitai.com/api/download/models/110789",
        "gothic_lolita.safetensors": "https://civitai.com/api/download/models/24567",
        "elves_ear.safetensors": "https://civitai.com/api/download/models/14123",
        "robots_cyborgs.safetensors": "https://civitai.com/api/download/models/134567",
        "vampire_lord.safetensors": "https://civitai.com/api/download/models/140234",
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
        "fabric_details.safetensors": "https://civitai.com/api/download/models/110456",
        "hair_details.safetensors": "https://civitai.com/api/download/models/93678",
        "skin_details.safetensors": "https://civitai.com/api/download/models/94567",
        "jewelry_gems.safetensors": "https://civitai.com/api/download/models/99567",

        # --- KATEGORI: KOMPOSISI & KAMERA (15 LoRA) ---
        "dynamic_poses.safetensors": "https://civitai.com/api/download/models/10234",
        "sitting_pose.safetensors": "https://civitai.com/api/download/models/95456",
        "fighting_stance.safetensors": "https://civitai.com/api/download/models/22345",
        "looking_back_pose.safetensors": "https://civitai.com/api/download/models/10456",
        "from_below_angle.safetensors": "https://civitai.com/api/download/models/54567",
        "closeup_shot.safetensors": "https://civitai.com/api/download/models/93456",
        "aerial_view.safetensors": "https://civitai.com/api/download/models/88776",
        "side_view.safetensors": "https://civitai.com/api/download/models/77665",
        "back_view.safetensors": "https://civitai.com/api/download/models/66554",
        "dramatic_angle.safetensors": "https://civitai.com/api/download/models/55443",
        "symmetrical_composition.safetensors": "https://civitai.com/api/download/models/44332",
        "widescreen_format.safetensors": "https://civitai.com/api/download/models/93234",
        "split_screen.safetensors": "https://civitai.com/api/download/models/107890",
        "dutch_angle.safetensors": "https://civitai.com/api/download/models/108901",
        "macro_photography.safetensors": "https://civitai.com/api/download/models/109012",
    }
}

# LoRA Strength Presets
LORA_PRESETS = {"subtle": 0.3, "normal": 0.7, "strong": 1.0}

# ==============================================================================
# 5. FUNGSI BANTUAN (HELPER FUNCTIONS)
# ==============================================================================

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
    target_path = target_dir / local_filename

    if target_path.exists():
        if target_path.is_symlink() and is_file_valid(target_path.resolve()):
            file_size = target_path.resolve().stat().st_size / (1024**2)
            print(f"[HF-DOWNLOAD] [OK] Valid symlink exists: {local_filename} ({file_size:.2f} MB)")
            return {"status": "exists", "filename": local_filename, "size": file_size, "type": model_type}
        elif is_file_valid(target_path):
            file_size = target_path.stat().st_size / (1024**2)
            print(f"[HF-DOWNLOAD] [OK] Valid file exists: {local_filename} ({file_size:.2f} MB)")
            return {"status": "exists", "filename": local_filename, "size": file_size, "type": model_type}
        else:
            print(f"[HF-DOWNLOAD] [WARN] Invalid file/symlink detected, cleaning up...")
            cleanup_invalid_file(target_path, "invalid file or symlink")

    try:
        print(f"[HF-DOWNLOAD] Downloading from HuggingFace...")
        cached_path = hf_hub_download(
            repo_id=repo_id, filename=filename, cache_dir=str(CACHE_PATH), resume_download=True
        )
        cached_file = Path(cached_path)
        if not is_file_valid(cached_file):
            cleanup_invalid_file(cached_file, "downloaded file too small")
            return {"status": "error", "filename": local_filename, "error": "Downloaded file is invalid", "type": model_type}

        print(f"[HF-DOWNLOAD] Creating symlink: {target_path} -> {cached_path}")
        subprocess.run(["ln", "-s", str(cached_file), str(target_path)], check=True)

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
    target_path = target_dir / filename
    temp_path = target_dir / f"{filename}.tmp"

    if target_path.exists() and is_file_valid(target_path):
        file_size = target_path.stat().st_size / (1024**2)
        print(f"[URL-DOWNLOAD] [OK] Valid file exists: {filename} ({file_size:.2f} MB)")
        return {"status": "exists", "filename": filename, "size": file_size, "type": model_type}
    elif target_path.exists():
        print(f"[URL-DOWNLOAD] [WARN] Invalid file detected, cleaning up...")
        cleanup_invalid_file(target_path, "file too small or corrupt")

    try:
        print(f"[URL-DOWNLOAD] Downloading with wget...")
        wget_cmd = ["wget", "-O", str(temp_path), "--progress=bar:force:noscroll", "--tries=5", "--timeout=120", "--continue", url]
        subprocess.run(wget_cmd, check=True, capture_output=True, text=True)

        if not is_file_valid(temp_path):
            cleanup_invalid_file(temp_path, "downloaded file too small")
            return {"status": "error", "filename": filename, "error": "Downloaded file is invalid", "type": model_type}

        temp_path.rename(target_path)
        file_size = target_path.stat().st_size / (1024**2)
        print(f"[URL-DOWNLOAD] [OK] Downloaded: {filename} ({file_size:.2f} MB)")
        return {"status": "downloaded", "filename": filename, "size": file_size, "type": model_type}
    except Exception as e:
        error_msg = str(e)[:200]
        print(f"[URL-DOWNLOAD] [ERROR] {filename} - {error_msg}")
        if temp_path.exists(): cleanup_invalid_file(temp_path, "download failed")
        if target_path.exists(): cleanup_invalid_file(target_path, "invalid after download")
        return {"status": "error", "filename": filename, "error": error_msg, "type": model_type}

# ==============================================================================
# 6. FUNGSI UNDUH MODEL
# ==============================================================================

@app.function(
    image=comfy_image,
    volumes={str(MODEL_PATH): volume, str(CACHE_PATH): cache_volume},
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
        if model_type not in MODEL_REGISTRY: continue
        
        print(f"\n{'='*80}\nPROCESSING MODEL TYPE: {model_type.upper()}\n{'='*80}")
        models = MODEL_REGISTRY[model_type]
        target_dir = MODEL_PATH / model_type
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Target directory: {target_dir}, Total files: {len(models)}")

        # Logika unduhan yang disederhanakan
        for idx, (local_filename, source) in enumerate(models.items(), 1):
            print(f"\n[PROGRESS] {idx}/{len(models)} - {local_filename}")
            if isinstance(source, dict):
                result = download_from_huggingface(model_type, local_filename, source["repo_id"], source["filename"], target_dir)
            else:
                result = download_from_url(model_type, local_filename, source, target_dir)
            
            if result["status"] == "downloaded": all_results["downloaded"].append(result)
            elif result["status"] == "exists": all_results["exists"].append(result)
            else: all_results["errors"].append(result)

    total_size = sum(r["size"] for r in all_results["downloaded"] + all_results["exists"])
    critical_missing = [e for e in all_results["errors"] if e["type"] in ["diffusion_models", "vae", "text_encoders", "audio_encoders"]]

    print(f"\n{'='*80}\nDOWNLOAD SUMMARY\n{'='*80}")
    print(f"[SUMMARY] Downloaded: {len(all_results['downloaded'])}, Cached: {len(all_results['exists'])}, Failed: {len(all_results['errors'])}")
    print(f"[SUMMARY] Total size: {total_size/1024:.2f} GB")
    if critical_missing:
        print(f"\n[ERROR] Critical models missing: {critical_missing}")
    if all_results["errors"]:
        print(f"\n[ERROR] All failed downloads: {all_results['errors']}")

    print("\n[VOLUME] Committing changes...")
    volume.commit()
    cache_volume.commit()
    print("[VOLUME] [OK] Changes committed")

    print(f"\n{'='*80}\nDOWNLOAD {'SUCCESSFUL' if not critical_missing else 'FAILED'}\n{'='*80}\n")
    return {"success": not critical_missing, "summary": {"downloaded": len(all_results["downloaded"]), "cached": len(all_results["exists"]), "failed": len(all_results["errors"]), "total_gb": round(total_size/1024, 2)}, "errors": all_results["errors"] if all_results["errors"] else None}

# ==============================================================================
# 7. KELAS UTAMA COMFYUI
# ==============================================================================

@app.cls(
    image=comfy_image,
    gpu="L40S",
    volumes={str(MODEL_PATH): volume, str(CACHE_PATH): cache_volume},
    timeout=3600,
    allow_concurrent_inputs=10,
)
class ComfyUI:
    @enter()
    def startup(self):
        print("\n" + "="*80)
        print("COMFYUI STARTUP")
        print("="*80)

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
        (COMFYUI_PATH / "extra_model_paths.yaml").write_text(config_content)
        print("[CONFIG] [OK] Configuration written")
        self._verify_model_paths()
        print("[VERIFY] [OK] All critical models verified")

        print(f"\n[SERVER] Starting ComfyUI server on port {COMFYUI_PORT}...")
        cmd = ["python", "main.py", "--listen", "0.0.0.0", "--port", str(COMFYUI_PORT), "--disable-auto-launch"]
        self.proc = subprocess.Popen(cmd, cwd=COMFYUI_PATH, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(f"[SERVER] Process started with PID: {self.proc.pid}")

        print(f"\n[SERVER] Waiting for server to be ready...")
        for i in range(COMFYUI_STARTUP_TIMEOUT):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{COMFYUI_PORT}/queue", timeout=5).read()
                print(f"[SERVER] [OK] Server is ready after {i+1} seconds")
                print("[SERVER] Waiting for models to fully load...")
                time.sleep(30) 
                print("="*80 + "\n")
                return
            except Exception as e:
                if i % 10 == 0: print(f"[SERVER] Waiting... ({i}/{COMFYUI_STARTUP_TIMEOUT} seconds)")
                time.sleep(1)
        
        print(f"[SERVER] [ERROR] Server failed to start after {COMFYUI_STARTUP_TIMEOUT} seconds")
        raise RuntimeError("ComfyUI failed to start")

    def _verify_model_paths(self):
        for model_type in ["diffusion_models", "vae", "text_encoders", "audio_encoders"]:
            model_dir = MODEL_PATH / model_type
            if not model_dir.exists() or not any(model_dir.iterdir()):
                print(f"[VERIFY] [ERROR] Critical model directory missing or empty: {model_type}")
                raise RuntimeError(f"Critical models missing! Run: modal run app.py::download_models")
            print(f"[VERIFY] [OK] {model_type}: {len(list(model_dir.iterdir()))} files")

    def _validate_lora_exists(self, lora_name: str) -> bool:
        lora_path = MODEL_PATH / "loras" / lora_name
        if not lora_path.exists(): return False
        if not is_file_valid(lora_path, MIN_LORA_SIZE_KB): return False
        return True

    def _validate_controlnet_exists(self, controlnet_name: str) -> bool:
        controlnet_path = MODEL_PATH / "controlnet" / controlnet_name
        if not controlnet_path.exists(): return False
        if not is_file_valid(controlnet_path): return False
        return True

    def _normalize_lora_strength(self, strength_input) -> float:
        if isinstance(strength_input, str) and strength_input.lower() in LORA_PRESETS:
            return LORA_PRESETS[strength_input.lower()]
        try: return float(strength_input)
        except: return 0.7

    def _build_lora_chain(self, base_model_node_id: str, base_clip_node_id: str, loras_config: List) -> Tuple[str, str, dict]:
        if not loras_config: return base_model_node_id, base_clip_node_id, {}
        print(f"\n[LORA-CHAIN] Building chain with {len(loras_config)} LoRAs")
        lora_nodes = {}
        current_model_id, current_clip_id = base_model_node_id, base_clip_node_id
        node_counter = 1000

        for idx, lora_config in enumerate(loras_config):
            lora_name = lora_config if isinstance(lora_config, str) else lora_config.get("name")
            strength = self._normalize_lora_strength(lora_config.get("strength", 0.7) if isinstance(lora_config, dict) else 0.7)
            
            if not lora_name: continue
            if not lora_name.endswith('.safetensors'): lora_name = f"{lora_name}.safetensors"
            if not self._validate_lora_exists(lora_name):
                print(f"[LORA-CHAIN] [WARN] Skipping missing LoRA: {lora_name}")
                continue

            node_id = str(node_counter)
            lora_nodes[node_id] = {
                "class_type": "LoraLoader",
                "inputs": {"lora_name": lora_name, "strength_model": strength, "strength_clip": strength, "model": [current_model_id, 0], "clip": [current_clip_id, 0]}
            }
            print(f"[LORA-CHAIN] {idx+1}. {lora_name} (strength: {strength}) - Node {node_id}")
            current_model_id, current_clip_id = node_id, node_id
            node_counter += 1
        
        return current_model_id, current_clip_id, lora_nodes

    def _build_controlnet_nodes(self, image_node_id: str, controlnets_config: List, base_positive_id: str, base_negative_id: str) -> Tuple[str, str, dict]:
        if not controlnets_config: return base_positive_id, base_negative_id, {}
        
        print(f"\n[CONTROLNET] Building nodes with {len(controlnets_config)} ControlNets")
        controlnet_nodes = {}
        conditioning_outputs = []
        node_counter = 2000

        cn_model_map = {"openpose": "control_v11p_sd15_openpose.pth", "canny": "control_v11p_sd15_canny.pth", "depth": "control_v11f1p_sd15_depth.pth", "lineart": "control_v11p_sd15_lineart.pth"}
        preprocessor_map = {
            "openpose": {"class": "OpenposePreprocessor", "params": {"detect_hand": "enable", "detect_body": "enable", "detect_face": "enable"}},
            "canny": {"class": "CannyEdgePreprocessor", "params": {"low_threshold": 100, "high_threshold": 200}},
            "depth": {"class": "MiDaS-DepthMapPreprocessor", "params": {"a": 6.283185307179586, "bg_threshold": 0.1}},
            "lineart": {"class": "LineArtPreprocessor", "params": {"coarse": "disable"}}
        }

        for idx, cn_config in enumerate(controlnets_config):
            cn_type = cn_config if isinstance(cn_config, str) else cn_config.get("type")
            strength = float(cn_config.get("strength", 1.0) if isinstance(cn_config, dict) else 1.0)
            if not cn_type: continue

            cn_model = cn_model_map.get(cn_type.lower())
            if not cn_model or not self._validate_controlnet_exists(cn_model):
                print(f"[CONTROLNET] [WARN] Skipping missing/unknown ControlNet: {cn_type}")
                continue

            loader_id = str(node_counter)
            controlnet_nodes[loader_id] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": cn_model}}
            
            preprocessor_id = str(node_counter + 1)
            preprocessor_info = preprocessor_map.get(cn_type.lower())
            if preprocessor_info:
                preprocessor_inputs = {"image": [image_node_id, 0]}
                preprocessor_inputs.update(preprocessor_info["params"])
                controlnet_nodes[preprocessor_id] = {"class_type": preprocessor_info["class"], "inputs": preprocessor_inputs}
                preprocessed_image_id = preprocessor_id
            else:
                preprocessed_image_id = image_node_id
            
            apply_id = str(node_counter + 2)
            controlnet_nodes[apply_id] = {
                "class_type": "ControlNetApplyAdvanced",
                "inputs": {
                    "positive": [base_positive_id, 0], 
                    "negative": [base_negative_id, 0], 
                    "control_net": [loader_id, 0],
                    "image": [preprocessed_image_id, 0],
                    "strength": strength, "start_percent": 0.0, "end_percent": 1.0
                }
            }
            conditioning_outputs.append(apply_id)
            print(f"[CONTROLNET] {idx+1}. {cn_type} (strength: {strength}) - Nodes {loader_id}-{apply_id}")
            node_counter += 10
        
        if len(conditioning_outputs) > 1:
            avg_id = str(node_counter)
            controlnet_nodes[avg_id] = {
                "class_type": "ConditioningAverage",
                "inputs": {"conditioning_1": [conditioning_outputs[0], 0], "conditioning_2": [conditioning_outputs[1], 0], "conditioning_to": [conditioning_outputs[0], 0]}
            }
            for i in range(2, len(conditioning_outputs)):
                next_avg_id = str(node_counter + i)
                controlnet_nodes[next_avg_id] = {
                    "class_type": "ConditioningAverage",
                    "inputs": {"conditioning_1": [avg_id, 0], "conditioning_2": [conditioning_outputs[i], 0], "conditioning_to": [avg_id, 0]}
                }
                avg_id = next_avg_id
            return avg_id, avg_id, controlnet_nodes
        elif len(conditioning_outputs) == 1:
            return conditioning_outputs[0], conditioning_outputs[0], controlnet_nodes
        else:
            return base_positive_id, base_negative_id, controlnet_nodes

    def _queue_prompt(self, client_id: str, prompt_workflow: dict) -> str:
        req = urllib.request.Request(f"http://127.0.0.1:{COMFYUI_PORT}/prompt", data=json.dumps({"prompt": prompt_workflow, "client_id": client_id}).encode('utf-8'), headers={'Content-Type': 'application/json'})
        response = urllib.request.urlopen(req).read()
        return json.loads(response)['prompt_id']

    def _get_history(self, prompt_id: str) -> dict:
        with urllib.request.urlopen(f"http://127.0.0.1:{COMFYUI_PORT}/history/{prompt_id}") as response:
            return json.loads(response.read())

    def _get_file(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        params = urllib.parse.urlencode({'filename': filename, 'subfolder': subfolder, 'type': folder_type})
        url = f"http://127.0.0.1:{COMFYUI_PORT}/view?{params}"
        with urllib.request.urlopen(url) as response:
            return response.read()

    def _get_video_from_websocket(self, prompt_id: str, client_id: str) -> bytes:
        import websocket
        ws_url = f"ws://127.0.0.1:{COMFYUI_PORT}/ws?clientId={client_id}"
        ws = websocket.WebSocket()
        consecutive_timeouts = 0

        try:
            ws.connect(ws_url, timeout=WEBSOCKET_TIMEOUT)
            print("[WEBSOCKET] [OK] Connected")
        except Exception as e:
            raise RuntimeError(f"WebSocket connection failed: {str(e)}")

        try:
            for _ in range(MAX_WEBSOCKET_MESSAGES):
                try:
                    out = ws.recv()
                    consecutive_timeouts = 0
                    if isinstance(out, str):
                        message = json.loads(out)
                        if message.get('type') == 'progress':
                            data = message.get('data', {})
                            print(f"[WEBSOCKET] Progress: {data.get('value', 0)}/{data.get('max', 100)}")
                        elif message.get('type') == 'executing' and message.get('data', {}).get('node') is None and message.get('data', {}).get('prompt_id') == prompt_id:
                            print("[WEBSOCKET] [OK] Generation complete")
                            break
                except websocket.WebSocketTimeoutException:
                    consecutive_timeouts += 1
                    print(f"[WEBSOCKET] Timeout ({consecutive_timeouts}/{MAX_CONSECUTIVE_TIMEOUTS})")
                    if consecutive_timeouts >= MAX_CONSECUTIVE_TIMEOUTS:
                        raise RuntimeError("WebSocket connection timed out repeatedly")
        finally:
            ws.close()

        history = self._get_history(prompt_id)[prompt_id]
        for node_id, node_output in history['outputs'].items():
            if 'videos' in node_output:
                return self._get_file(node_output['videos'][0]['filename'], node_output['videos'][0]['subfolder'], node_output['videos'][0]['type'])
            if 'gifs' in node_output:
                return self._get_file(node_output['gifs'][0]['filename'], node_output['gifs'][0]['subfolder'], node_output['gifs'][0]['type'])
        
        raise ValueError("No video output found in generation")

    def _save_base64_image(self, image_b64: str) -> str:
        clean_b64 = image_b64.split(';base64,')[-1] if ';base64,' in image_b64 else image_b64
        temp_filename = f"/tmp/{uuid.uuid4()}.png"
        try:
            image_data = base64.b64decode(clean_b64)
            img = PILImage.open(io.BytesIO(image_data))
            img.verify()
            with open(temp_filename, "wb") as f: f.write(image_data)
            print(f"[IMAGE] Saved base64 image to: {temp_filename}")
            return temp_filename
        except Exception as e:
            raise ValueError(f"Invalid base64 image data: {str(e)}")

    def _save_base64_audio(self, audio_b64: str) -> str:
        clean_b64 = audio_b64.split(';base64,')[-1] if ';base64,' in audio_b64 else audio_b64
        temp_filename = f"/tmp/{uuid.uuid4()}.wav"
        try:
            audio_data = base64.b64decode(clean_b64)
            audio = AudioSegment.from_file(io.BytesIO(audio_data))
            audio.export(temp_filename, format="wav")
            print(f"[AUDIO] Saved base64 audio to: {temp_filename}")
            return temp_filename
        except Exception as e:
            raise ValueError(f"Invalid base64 audio data: {str(e)}")

    @method()
    def generate(self, prompt: str, negative_prompt: str = "", width: int = 832, height: int = 480,
                 num_frames: int = 81, steps: int = 30, cfg: float = 2.5, seed: int = -1,
                 image_b64: Optional[str] = None, audio_b64: Optional[str] = None,
                 loras: Optional[List[Any]] = None, controlnets: Optional[List[Any]] = None) -> bytes:
        
        client_id = str(uuid.uuid4())
        workflow = {"4": {"inputs": {"ckpt_name": "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"}, "class_type": "CheckpointLoaderSimple"},
                    "5": {"inputs": {"text": prompt, "clip": ["4", 1]}, "class_type": "CLIPTextEncode"},
                    "6": {"inputs": {"text": negative_prompt, "clip": ["4", 1]}, "class_type": "CLIPTextEncode"},
                    "7": {"inputs": {"filename_prefix": "WanVideo", "frames": num_frames, "fps": 16}, "class_type": "WanVidSave"},
                    "8": {"inputs": {"seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0, "model": ["4", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["10", 0]}, "class_type": "KSampler"},
                    "10": {"inputs": {"width": width, "height": height, "batch_size": 1, "length": num_frames}, "class_type": "EmptyLatentImage"},
                    "11": {"inputs": {"vae_name": "wan_2.1_vae.safetensors"}, "class_type": "VAELoader"},
                    "12": {"inputs": {"samples": ["8", 0], "vae": ["11", 0]}, "class_type": "VAEDecode"},
                    "13": {"inputs": {"filename_prefix": "WanVideo", "images": ["12", 0]}, "class_type": "SaveImage"}}

        if loras:
            final_model_id, final_clip_id, lora_nodes = self._build_lora_chain("4", "4", loras)
            workflow.update(lora_nodes)
            workflow["8"]["inputs"]["model"] = [final_model_id, 0]
            workflow["5"]["inputs"]["clip"] = [final_clip_id, 0]
            workflow["6"]["inputs"]["clip"] = [final_clip_id, 0]

        if image_b64 and controlnets:
            image_path = self._save_base64_image(image_b64)
            workflow["3"] = {"inputs": {"image": image_path, "upload": "image"}, "class_type": "LoadImage"}
            final_positive_id, final_negative_id, cn_nodes = self._build_controlnet_nodes("3", controlnets, "5", "6")
            workflow.update(cn_nodes)
            workflow["8"]["inputs"]["positive"] = [final_positive_id, 0]
            workflow["8"]["inputs"]["negative"] = [final_negative_id, 0]

        # TODO: Tambahkan logika untuk image_to_video dan audio_to_video

        prompt_id = self._queue_prompt(client_id, workflow)
        return self._get_video_from_websocket(prompt_id, client_id)

# ==============================================================================
# 8. FASTAPI ENDPOINTS
# ==============================================================================

web_app = FastAPI()

class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 832
    height: int = 480
    num_frames: int = 81
    steps: int = 30
    cfg: float = 2.5
    seed: int = -1
    image_b64: Optional[str] = None
    audio_b64: Optional[str] = None
    loras: Optional[List[Any]] = None
    controlnets: Optional[List[Any]] = None

@web_app.post("/generate_video")
async def generate_video_api(request: GenerationRequest):
    try:
        comfyui = ComfyUI()
        video_bytes = comfyui.generate.remote(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_frames=request.num_frames,
            steps=request.steps,
            cfg=request.cfg,
            seed=request.seed,
            image_b64=request.image_b64,
            audio_b64=request.audio_b64,
            loras=request.loras,
            controlnets=request.controlnets
        )
        return Response(content=video_bytes, media_type="video/mp4")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@asgi_app()
def fastapi_app():
    return web_app