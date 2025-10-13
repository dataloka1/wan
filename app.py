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
from typing import Dict, Optional, List, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from modal import App, Image, Volume, Secret, asgi_app, enter, method
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field

REQUIREMENTS = [
    "git", "wget", "curl", "libgl1-mesa-glx", "libglib2.0-0",
    "ffmpeg", "libsm6", "libxext6", "libxrender-dev", "pkg-config", "libcairo2-dev",
    "websocket-client",
    "safetensors",
    "pillow",
    "numpy",
    "torch",
    "torchvision",
    "fastapi",
    "uvicorn[standard]",
    "pydub",
    "huggingface-hub[hf_transfer]"
]

REMOTE_BASE_PATH = Path("/app")
COMFYUI_PATH = REMOTE_BASE_PATH / "ComfyUI"
MODEL_PATH = Path("/models")
CACHE_PATH = Path("/cache")

MIN_FILE_SIZE_KB = 500
MIN_LORA_SIZE_KB = 100
MAX_BASE64_SIZE = 50 * 1024 * 1024
MAX_GENERATION_TIME = 1800
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8188
SERVER_STARTUP_TIMEOUT = 60

LORA_PRESETS = {
    "subtle": 0.3,
    "normal": 0.7,
    "strong": 1.0
}

PRIORITY_ORDER = ["diffusion_models", "vae", "text_encoders", "audio_encoders", "clip", "controlnet", "loras"]

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
        "add_detail.safetensors": "https://civitai.com/api/download/models/62833",
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

volume = Volume.from_name("comfyui-wan2-2-complete-volume", create_if_missing=True)
cache_volume = Volume.from_name("hf-hub-cache", create_if_missing=True)

comfy_image = (
    Image.debian_slim(python_version="3.10")
    .apt_install(*REQUIREMENTS[:11])
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
    .pip_install(*REQUIREMENTS[11:])
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

def is_file_valid(filepath: Path, min_size_kb: int = MIN_FILE_SIZE_KB) -> bool:
    if not filepath.exists():
        return False
    file_size_kb = filepath.stat().st_size / 1024
    return file_size_kb >= min_size_kb

def cleanup_invalid_file(filepath: Path, reason: str = ""):
    try:
        if filepath.is_symlink():
            symlink_target = filepath.resolve()
            print(f"[CLEANUP] Menghapus symlink rusak: {filepath}")
            filepath.unlink()
            if symlink_target.exists():
                print(f"[CLEANUP] Menghapus target symlink: {symlink_target}")
                symlink_target.unlink()
        elif filepath.exists():
            print(f"[CLEANUP] Menghapus file tidak valid: {filepath} - {reason}")
            filepath.unlink()
    except Exception as e:
        print(f"[CLEANUP] Error menghapus file {filepath}: {str(e)}")

def _validate_and_decode_base64(data: str, data_type: str = "image") -> str:
    if not data:
        raise ValueError(f"Data {data_type} kosong.")
    if len(data) > MAX_BASE64_SIZE:
        raise ValueError(f"Data {data_type} terlalu besar. Maksimal {MAX_BASE64_SIZE / (1024*1024):.2f}MB.")
    if data.startswith(f'data:{data_type}/'):
        if ';base64,' not in data:
            raise ValueError(f"Format base64 {data_type} tidak valid. Seharusnya 'data:{data_type}/...;base64,...'")
        data = data.split(';base64,')[1]
    try:
        base64.b64decode(data, validate=True)
    except Exception as e:
        raise ValueError(f"Encoding base64 tidak valid: {str(e)}")
    return data

def _save_base64_to_file(data_base64: str, temp_filename: str, data_type: str = "image") -> str:
    clean_b64 = _validate_and_decode_base64(data_base64, data_type)
    try:
        file_data = base64.b64decode(clean_b64)
        with open(temp_filename, "wb") as f:
            f.write(file_data)
        print(f"[{data_type.upper()}] Disimpan ke: {temp_filename} ({len(file_data)/1024:.2f} KB)")
        return temp_filename
    except Exception as e:
        raise ValueError(f"Gagal menyimpan {data_type} base64: {str(e)}")

def _download_and_link_hf(model_type: str, local_filename: str, repo_id: str, filename: str, target_dir: Path) -> dict:
    from huggingface_hub import hf_hub_download
    print(f"\n[HF-DOWNLOAD] Memulai: {model_type}/{local_filename}")
    target_path = target_dir / local_filename
    if target_path.is_symlink() and is_file_valid(target_path.resolve()):
        file_size = target_path.resolve().stat().st_size / (1024**2)
        print(f"[HF-DOWNLOAD] [OK] Symlink valid ada: {local_filename} ({file_size:.2f} MB)")
        return {"status": "exists", "filename": local_filename, "size": file_size, "type": model_type}
    if target_path.exists() and is_file_valid(target_path):
        file_size = target_path.stat().st_size / (1024**2)
        print(f"[HF-DOWNLOAD] [OK] File valid ada: {local_filename} ({file_size:.2f} MB)")
        return {"status": "exists", "filename": local_filename, "size": file_size, "type": model_type}
    cleanup_invalid_file(target_path, "file tidak valid atau rusak")
    try:
        cached_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=str(CACHE_PATH), resume_download=True)
        if not is_file_valid(Path(cached_path)):
            cleanup_invalid_file(Path(cached_path), "file yang diunduh terlalu kecil")
            return {"status": "error", "filename": local_filename, "error": "File yang diunduh tidak valid", "type": model_type}
        subprocess.run(f"ln -s {cached_path} {target_path}", shell=True, check=True)
        file_size = Path(cached_path).stat().st_size / (1024**2)
        print(f"[HF-DOWNLOAD] [OK] Diunduh: {local_filename} ({file_size:.2f} MB)")
        return {"status": "downloaded", "filename": local_filename, "size": file_size, "type": model_type}
    except Exception as e:
        error_msg = str(e)[:200]
        print(f"[HF-DOWNLOAD] [ERROR] {local_filename} - {error_msg}")
        cleanup_invalid_file(target_path, "download gagal")
        return {"status": "error", "filename": local_filename, "error": error_msg, "type": model_type}

def _download_and_link_url(model_type: str, filename: str, url: str, target_dir: Path) -> dict:
    print(f"\n[URL-DOWNLOAD] Memulai: {model_type}/{filename}")
    target_path = target_dir / filename
    temp_path = target_dir / f"{filename}.tmp"
    if target_path.exists() and is_file_valid(target_path):
        file_size = target_path.stat().st_size / (1024**2)
        print(f"[URL-DOWNLOAD] [OK] File valid ada: {filename} ({file_size:.2f} MB)")
        return {"status": "exists", "filename": filename, "size": file_size, "type": model_type}
    cleanup_invalid_file(target_path, "file tidak valid atau rusak")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'ComfyUI-Modal-App/1.0'})
        with urllib.request.urlopen(req, timeout=300) as response, open(temp_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        if not is_file_valid(temp_path):
            cleanup_invalid_file(temp_path, "file yang diunduh terlalu kecil")
            return {"status": "error", "filename": filename, "error": "File yang diunduh tidak valid", "type": model_type}
        temp_path.rename(target_path)
        file_size = target_path.stat().st_size / (1024**2)
        print(f"[URL-DOWNLOAD] [OK] Diunduh: {filename} ({file_size:.2f} MB)")
        return {"status": "downloaded", "filename": filename, "size": file_size, "type": model_type}
    except Exception as e:
        error_msg = str(e)[:200]
        print(f"[URL-DOWNLOAD] [ERROR] {filename} - {error_msg}")
        cleanup_invalid_file(temp_path, "download gagal")
        return {"status": "error", "filename": filename, "error": error_msg, "type": model_type}

@app.function(
    image=comfy_image,
    volumes={str(MODEL_PATH): volume, str(CACHE_PATH): cache_volume},
    secrets=[Secret.from_name("huggingface-token")],
    timeout=7200,
)
def download_models():
    print("\n" + "="*80)
    print("MODEL DOWNLOAD DIMULAI")
    print("="*80)
    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    all_results = {"downloaded": [], "exists": [], "errors": []}
    for model_type in PRIORITY_ORDER:
        if model_type not in MODEL_REGISTRY: continue
        print(f"\n{'='*80}\nPROSES TIPE MODEL: {model_type.upper()}\n{'='*80}")
        models = MODEL_REGISTRY[model_type]
        target_dir = MODEL_PATH / model_type
        target_dir.mkdir(parents=True, exist_ok=True)
        download_tasks = []
        for local_filename, source in models.items():
            if isinstance(source, dict):
                download_tasks.append(("hf", model_type, local_filename, source["repo_id"], source["filename"], target_dir))
            else:
                download_tasks.append(("url", model_type, local_filename, source, None, target_dir))
        max_workers = 10 if model_type == "loras" else 3
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_download_and_link_hf, *t[1:]) if t[0] == "hf" else executor.submit(_download_and_link_url, *t[1:]): t for t in download_tasks}
            for future in as_completed(futures):
                result = future.result()
                if result["status"] == "downloaded": all_results["downloaded"].append(result)
                elif result["status"] == "exists": all_results["exists"].append(result)
                else: all_results["errors"].append(result)
        print(f"\n[VOLUME] Melakukan commit untuk {model_type}...")
        volume.commit()
        cache_volume.commit()
        print(f"[VOLUME] [OK] Commit untuk {model_type} selesai.")
    total_size = sum(r["size"] for r in all_results["downloaded"] + all_results["exists"])
    critical_missing = [e for e in all_results["errors"] if e["type"] in ["diffusion_models", "vae", "text_encoders", "audio_encoders"]]
    print(f"\n{'='*80}\nRINGKASAN DOWNLOAD\n{'='*80}")
    print(f"Diunduh: {len(all_results['downloaded'])} file")
    print(f"Dalam cache: {len(all_results['exists'])} file")
    print(f"Gagal: {len(all_results['errors'])} file")
    print(f"Total ukuran: {total_size/1024:.2f} GB")
    if critical_missing:
        print("\n[ERROR] Model kritis hilang:")
        for err in critical_missing: print(f"  - {err['type']}/{err['filename']}: {err.get('error', 'Error tidak diketahui')}")
    print(f"\n{'='*80}\nDOWNLOAD {'BERHASIL' if not critical_missing else 'GAGAL'}\n{'='*80}")
    return {"success": not critical_missing, "summary": {"downloaded": len(all_results["downloaded"]), "cached": len(all_results["exists"]), "failed": len(all_results["errors"]), "total_gb": round(total_size/1024, 2)}, "errors": all_results["errors"] if all_results["errors"] else None}

@app.cls(image=comfy_image, gpu="L40S", volumes={str(MODEL_PATH): volume, str(CACHE_PATH): cache_volume}, timeout=3600)
class ComfyUI:
    @enter()
    def startup(self):
        print("\n" + "="*80)
        print("COMFYUI STARTUP")
        print("="*80)
        config_content = f"""comfyui:\n    base_path: {COMFYUI_PATH}\nmodels:\n    diffusion_models: {MODEL_PATH}/diffusion_models\n    vae: {MODEL_PATH}/vae\n    text_encoders: {MODEL_PATH}/text_encoders\n    audio_encoders: {MODEL_PATH}/audio_encoders\n    clip: {MODEL_PATH}/clip\n    loras: {MODEL_PATH}/loras\n    controlnet: {MODEL_PATH}/controlnet\n"""
        (COMFYUI_PATH / "extra_model_paths.yaml").write_text(config_content)
        self._verify_model_paths()
        cmd = ["python", "main.py", "--listen", SERVER_HOST, "--port", str(SERVER_PORT), "--disable-auto-launch"]
        self.proc = subprocess.Popen(cmd, cwd=COMFYUI_PATH, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"[SERVER] Memulai server di {SERVER_HOST}:{SERVER_PORT} dengan PID: {self.proc.pid}")
        for i in range(SERVER_STARTUP_TIMEOUT):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{SERVER_PORT}/queue", timeout=5).read()
                print(f"[SERVER] [OK] Server siap setelah {i+1} detik\n")
                return
            except Exception:
                if i % 10 == 0:
                    print(f"[SERVER] Menunggu... ({i}/{SERVER_STARTUP_TIMEOUT} detik)")
                time.sleep(1)
        raise RuntimeError("ComfyUI gagal start")

    def _verify_model_paths(self):
        for model_type in MODEL_REGISTRY.keys():
            model_dir = MODEL_PATH / model_type
            if not model_dir.exists():
                print(f"[VERIFY] [ERROR] Direktori hilang: {model_type}")
                if model_type in ["diffusion_models", "vae", "text_encoders", "audio_encoders"]:
                    raise RuntimeError("Model kritis hilang! Jalankan: modal run app.py::download_models")
            else:
                print(f"[VERIFY] [OK] {model_type}: {len(list(model_dir.iterdir()))} file")

    def _validate_model_file(self, model_type: str, filename: str, min_size_kb: int) -> bool:
        model_path = MODEL_PATH / model_type / filename
        if not model_path.exists():
            print(f"[{model_type.upper()}-VALIDATION] [WARN] Model tidak ditemukan: {filename}")
            return False
        if not is_file_valid(model_path, min_size_kb):
            print(f"[{model_type.upper()}-VALIDATION] [WARN] Model tidak valid: {filename}")
            return False
        print(f"[{model_type.upper()}-VALIDATION] [OK] Model divalidasi: {filename}")
        return True

    def _validate_lora_exists(self, lora_name: str) -> bool:
        if not lora_name.endswith('.safetensors'):
            lora_name = f"{lora_name}.safetensors"
        return self._validate_model_file("loras", lora_name, MIN_LORA_SIZE_KB)

    def _validate_controlnet_exists(self, controlnet_name: str) -> bool:
        return self._validate_model_file("controlnet", controlnet_name, MIN_FILE_SIZE_KB)

    def _normalize_lora_strength(self, strength_input) -> float:
        if isinstance(strength_input, str) and strength_input.lower() in LORA_PRESETS:
            return LORA_PRESETS[strength_input.lower()]
        try:
            return float(strength_input)
        except:
            return 0.7

    def _build_lora_chain(self, base_model_node_id: str, base_clip_node_id: str, loras_config: List) -> tuple:
        if not loras_config:
            return base_model_node_id, base_clip_node_id, {}
        print(f"\n[LORA-CHAIN] Membangun rantai dengan {len(loras_config)} LoRA")
        lora_nodes = {}
        current_model_id, current_clip_id = base_model_node_id, base_clip_node_id
        for idx, lora_config in enumerate(loras_config):
            lora_name = lora_config if isinstance(lora_config, str) else lora_config.get("name")
            strength = self._normalize_lora_strength(lora_config.get("strength", 0.7) if isinstance(lora_config, dict) else 0.7)
            if not lora_name:
                continue
            if not lora_name.endswith('.safetensors'):
                lora_name = f"{lora_name}.safetensors"
            if not self._validate_lora_exists(lora_name):
                continue
            node_id = str(1000 + idx)
            lora_nodes[node_id] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name": lora_name,
                    "strength_model": strength,
                    "strength_clip": strength,
                    "model": [current_model_id, 0],
                    "clip": [current_clip_id, 0]
                }
            }
            print(f"[LORA-CHAIN] {idx+1}. {lora_name} (strength: {strength}) - Node {node_id}")
            current_model_id, current_clip_id = node_id, node_id
        return current_model_id, current_clip_id, lora_nodes

    def _build_controlnet_nodes(self, image_node_id: str, controlnets_config: List, base_positive_id: str, base_negative_id: str) -> tuple:
        if not controlnets_config:
            return base_positive_id, base_negative_id, {}
        print(f"\n[CONTROLNET] Membangun nodes dengan {len(controlnets_config)} ControlNet")
        print("[CONTROLNET] [WARN] Menggunakan ControlNet SD1.5 dengan model Wan2.2. Hasil mungkin tidak dapat diprediksi.")
        controlnet_nodes = {}
        current_positive_id, current_negative_id = base_positive_id, base_negative_id
        cn_model_map = {
            "openpose": "control_v11p_sd15_openpose.pth",
            "canny": "control_v11p_sd15_canny.pth",
            "depth": "control_v11f1p_sd15_depth.pth",
            "lineart": "control_v11p_sd15_lineart.pth"
        }
        preprocessor_map = {
            "openpose": ("OpenposePreprocessor", {"detect_hand": "enable", "detect_body": "enable", "detect_face": "enable"}),
            "canny": ("CannyEdgePreprocessor", {"low_threshold": 100, "high_threshold": 200}),
            "depth": ("MiDaS-DepthMapPreprocessor", {"a": 6.283185307179586, "bg_threshold": 0.1}),
            "lineart": ("LineArtPreprocessor", {"coarse": "disable"})
        }
        for idx, cn_config in enumerate(controlnets_config):
            cn_type = cn_config if isinstance(cn_config, str) else cn_config.get("type")
            strength = float(cn_config.get("strength", 1.0) if isinstance(cn_config, dict) else 1.0)
            if not cn_type:
                continue
            cn_model = cn_model_map.get(cn_type.lower())
            if not cn_model or not self._validate_controlnet_exists(cn_model):
                continue
            loader_id = str(2000 + idx * 10)
            controlnet_nodes[loader_id] = {
                "class_type": "ControlNetLoader",
                "inputs": {"control_net_name": cn_model}
            }
            preprocessor_info = preprocessor_map.get(cn_type.lower())
            preprocessed_image_id = image_node_id
            if preprocessor_info:
                preprocessor_id = str(int(loader_id) + 1)
                preprocessor_inputs = {"image": [image_node_id, 0]}
                preprocessor_inputs.update(preprocessor_info[1])
                controlnet_nodes[preprocessor_id] = {
                    "class_type": preprocessor_info[0],
                    "inputs": preprocessor_inputs
                }
                preprocessed_image_id = preprocessor_id
            apply_id = str(int(loader_id) + 2)
            controlnet_nodes[apply_id] = {
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
            print(f"[CONTROLNET] {idx+1}. {cn_type} (strength: {strength}) - Nodes {loader_id}-{apply_id}")
            current_positive_id, current_negative_id = apply_id, apply_id
        return current_positive_id, current_negative_id, controlnet_nodes

    def _queue_prompt(self, client_id: str, prompt_workflow: dict):
        req = urllib.request.Request(
            f"http://127.0.0.1:{SERVER_PORT}/prompt",
            data=json.dumps({"prompt": prompt_workflow, "client_id": client_id}).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        response = urllib.request.urlopen(req).read()
        return json.loads(response)['prompt_id']

    def _get_history(self, prompt_id: str):
        with urllib.request.urlopen(f"http://127.0.0.1:{SERVER_PORT}/history/{prompt_id}") as response:
            return json.loads(response.read())

    def _get_file(self, filename: str, subfolder: str, folder_type: str):
        params = urllib.parse.urlencode({'filename': filename, 'subfolder': subfolder, 'type': folder_type})
        url = f"http://127.0.0.1:{SERVER_PORT}/view?{params}"
        with urllib.request.urlopen(url) as response:
            data = bytearray()
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                data.extend(chunk)
            print(f"[FILE] [OK] File diambil: {len(data)/1024/1024:.2f} MB")
            return bytes(data)

    def _get_video_from_websocket(self, prompt_id: str, client_id: str):
        ws_url = f"ws://127.0.0.1:{SERVER_PORT}/ws?clientId={client_id}"
        ws = websocket.WebSocket()
        try:
            ws.connect(ws_url, timeout=10)
            print("[WEBSOCKET] [OK] Terhubung")
        except Exception as e:
            raise RuntimeError(f"Koneksi WebSocket gagal: {str(e)}")
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
                            percentage = (value / max_val) * 100
                            print(f"[WEBSOCKET] Progress: {value}/{max_val} ({percentage:.1f}%)")
                        elif message.get('type') == 'executing' and not message.get('data', {}).get('node'):
                            print("[WEBSOCKET] [OK] Generasi selesai")
                            break
                except websocket.WebSocketTimeoutException:
                    print("[WEBSOCKET] Timeout, mencoba lagi...")
                    continue
        finally:
            ws.close()
        if time.time() - start_time >= MAX_GENERATION_TIME:
            raise TimeoutError(f"Generasi video melebihi batas waktu {MAX_GENERATION_TIME} detik")
        history = self._get_history(prompt_id)[prompt_id]
        for node_id, node_output in history['outputs'].items():
            for video_type in ['videos', 'gifs']:
                if video_type in node_output:
                    for item in node_output[video_type]:
                        print(f"[OUTPUT] [OK] Ditemukan {video_type[:-1]}: {item['filename']}")
                        return self._get_file(item['filename'], item['subfolder'], item['type'])
        raise ValueError("Tidak ada output video ditemukan")

    def _save_base64_image(self, image_base64: str) -> str:
        temp_filename = f"/tmp/{uuid.uuid4()}.png"
        return _save_base64_to_file(image_base64, temp_filename, "image")

    def _save_base64_audio(self, audio_base64: str) -> str:
        temp_filename = f"/tmp/{uuid.uuid4()}.wav"
        return _save_base64_to_file(audio_base64, temp_filename, "audio")

    @method()
    def generate_video(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 832,
        height: int = 480,
        num_frames: int = 16,
        loras: Optional[List[Union[str, Dict]]] = None,
        controlnets: Optional[List[Union[str, Dict]]] = None
    ):
        """Membangun workflow dan menghasilkan video dari teks."""
        print(f"\n[WORKFLOW] Membangun T2V workflow untuk: '{prompt[:50]}...'")
        client_id = str(uuid.uuid4())

        # Base workflow menggunakan WanTextToVideo
        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"}
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": prompt, "clip": ["4", 1]}
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": negative_prompt, "clip": ["4", 1]}
            },
            "4": {
                "class_type": "CLIPLoader",
                "inputs": {"clip_name": "clip_l.safetensors"}
            },
            "5": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": "wan_2.1_vae.safetensors"}
            },
            "6": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": width, "height": height, "batch_size": 1}
            },
            "7": {
                "class_type": "WanTextToVideo",
                "inputs": {
                    "seed": int(time.time()),
                    "steps": 30,
                    "cfg": 7.5,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "frames": num_frames,
                    "motion_strength": 127,
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "vae": ["5", 0],
                    "latent_image": ["6", 0]
                }
            },
            "9": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["7", 0], "vae": ["5", 0]}
            },
            "10": {
                "class_type": "SaveAnimatedWEBP",
                "inputs": {
                    "filename_prefix": "wan_t2v_output",
                    "images": ["9", 0],
                    "fps": 8,
                    "lossless": False
                }
            }
        }

        # Terapkan LoRA jika ada
        if loras:
            final_model_id, final_clip_id, lora_nodes = self._build_lora_chain("1", "4", loras)
            workflow.update(lora_nodes)
            workflow["7"]["inputs"]["model"] = [final_model_id, 0]
            workflow["2"]["inputs"]["clip"] = [final_clip_id, 1]
            workflow["3"]["inputs"]["clip"] = [final_clip_id, 1]

        # Peringatan untuk ControlNet (T2V tidak punya input gambar)
        if controlnets:
            print("[WORKFLOW] [WARN] ControlNet tidak didukung untuk Text-to-Video murni.")

        # Jalankan workflow
        prompt_id = self._queue_prompt(client_id, workflow)
        video_data = self._get_video_from_websocket(prompt_id, client_id)
        return base64.b64encode(video_data).decode('utf-8')


# Pydantic models untuk request body
class GenerateVideoRequest(BaseModel):
    prompt: str = Field(..., example="A majestic cat sitting on a throne, cinematic lighting")
    negative_prompt: str = Field("", example="blurry, low quality")
    width: int = Field(832, ge=256, le=1920, example=832)
    height: int = Field(480, ge=256, le=1080, example=480)
    num_frames: int = Field(16, ge=8, le=64, example=16)
    loras: Optional[List[Union[str, Dict]]] = Field(
        None,
        example=[
            "detail_tweaker_xl.safetensors",
            {"name": "v2_lora_ZoomIn.safetensors", "strength": "strong"}
        ]
    )
    controlnets: Optional[List[Union[str, Dict]]] = Field(
        None,
        example=[
            "canny",
            {"type": "openpose", "strength": 0.8}
        ]
    )


# FastAPI Web Application
web_app = FastAPI(
    title="ComfyUI Wan2.2 Video API",
    description="API untuk generate video menggunakan Wan2.2 model",
    version="1.0.0"
)


@web_app.get("/")
async def root():
    return {
        "message": "ComfyUI Wan2.2 API is running",
        "endpoints": {
            "generate_video": "/generate_video",
            "health": "/health"
        },
        "documentation": "/docs"
    }


@web_app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "comfyui-wan2-2-api"}


@web_app.post("/generate_video")
async def api_generate_video(request: GenerateVideoRequest = Body(...)):
    """
    Generate video dari text prompt menggunakan Wan2.2 model.
    
    Request body harus dalam format JSON dengan structure:
    {
        "prompt": "A cat wearing a wizard hat",
        "negative_prompt": "blurry, low quality",
        "width": 832,
        "height": 480,
        "num_frames": 16,
        "loras": ["detail_tweaker_xl.safetensors"],
        "controlnets": null
    }
    """
    try:
        print(f"\n[API] Menerima request generate_video")
        print(f"[API] Prompt: {request.prompt[:100]}...")
        
        # Validasi input
        if not request.prompt or len(request.prompt.strip()) == 0:
            raise HTTPException(status_code=400, detail="Prompt tidak boleh kosong")
        
        if request.width % 8 != 0 or request.height % 8 != 0:
            raise HTTPException(
                status_code=400,
                detail="Width dan height harus kelipatan 8"
            )
        
        # Call Modal function
        comfyui_instance = ComfyUI()
        result = comfyui_instance.generate_video.remote(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_frames=request.num_frames,
            loras=request.loras,
            controlnets=request.controlnets
        )
        
        print(f"[API] [OK] Video berhasil di-generate")
        return {
            "success": True,
            "video_base64": result,
            "metadata": {
                "prompt": request.prompt,
                "width": request.width,
                "height": request.height,
                "num_frames": request.num_frames,
                "loras_count": len(request.loras) if request.loras else 0
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        print(f"[API] [ERROR] {error_msg}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating video: {error_msg}"
        )


@app.function(image=comfy_image, keep_warm=1)
@asgi_app()
def fastapi_app():
    return web_app