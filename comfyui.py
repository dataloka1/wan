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

COMFYUI_PATH = Path("/root/comfy/ComfyUI")
MODEL_PATH = Path("/root/comfy/ComfyUI/models")
CACHE_PATH = Path("/cache")

MIN_FILE_SIZE_KB = 500
MIN_LORA_SIZE_KB = 100
MAX_BASE64_SIZE = 100 * 1024 * 1024
MAX_GENERATION_TIME = 3600
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
SERVER_STARTUP_TIMEOUT = 120

MODEL_REGISTRY = {
    "diffusion_models": {
        "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors": {
            "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "filename": "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
        },
        "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors": {
            "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "filename": "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
        },
        "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors": {
            "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "filename": "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
        },
        "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors": {
            "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "filename": "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
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
    },
    "text_encoders": {
        "umt5_xxl_fp8_e4m3fn_scaled.safetensors": {
            "repo_id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "filename": "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
        },
    },
    "clip_vision": {
        "clip_vision_h.safetensors": {
            "repo_id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "filename": "split_files/clip_vision/clip_vision_h.safetensors"
        },
    },
    "loras": {
        "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors": {
            "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "filename": "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
        },
        "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors": {
            "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "filename": "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"
        },
        "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors": {
            "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "filename": "split_files/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors"
        },
        "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors": {
            "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "filename": "split_files/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors"
        },
        "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors": {
            "repo_id": "Kijai/WanVideo_comfy",
            "filename": "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
        },
        "WanAnimate_relight_lora_fp16.safetensors": {
            "repo_id": "Kijai/WanVideo_comfy",
            "filename": "LoRAs/Wan22_relight/WanAnimate_relight_lora_fp16.safetensors"
        },
        "v2_lora_ZoomIn.safetensors": {
            "repo_id": "guoyww/animatediff",
            "filename": "motion_lora_v2/v2_lora_ZoomIn.safetensors"
        },
        "v2_lora_ZoomOut.safetensors": {
            "repo_id": "guoyww/animatediff",
            "filename": "motion_lora_v2/v2_lora_ZoomOut.safetensors"
        },
        "v2_lora_PanLeft.safetensors": {
            "repo_id": "guoyww/animatediff",
            "filename": "motion_lora_v2/v2_lora_PanLeft.safetensors"
        },
        "v2_lora_PanRight.safetensors": {
            "repo_id": "guoyww/animatediff",
            "filename": "motion_lora_v2/v2_lora_PanRight.safetensors"
        },
        "v2_lora_TiltUp.safetensors": {
            "repo_id": "guoyww/animatediff",
            "filename": "motion_lora_v2/v2_lora_TiltUp.safetensors"
        },
        "v2_lora_TiltDown.safetensors": {
            "repo_id": "guoyww/animatediff",
            "filename": "motion_lora_v2/v2_lora_TiltDown.safetensors"
        },
        "v2_lora_RollingClockwise.safetensors": {
            "repo_id": "guoyww/animatediff",
            "filename": "motion_lora_v2/v2_lora_RollingClockwise.safetensors"
        },
        "v2_lora_RollingAnticlockwise.safetensors": {
            "repo_id": "guoyww/animatediff",
            "filename": "motion_lora_v2/v2_lora_RollingAnticlockwise.safetensors"
        },
    }
}

def _validate_and_decode_base64(data: str, data_type: str = "image") -> str:
    if not data:
        raise ValueError(f"{data_type} data is empty")
    if len(data) > MAX_BASE64_SIZE:
        raise ValueError(f"{data_type} data too large. Max {MAX_BASE64_SIZE/(1024*1024):.2f}MB")
    if data.startswith(f'data:{data_type}/') or data.startswith('data:audio/'):
        if ';base64,' not in data:
            raise ValueError(f"Invalid base64 {data_type} format")
        data = data.split(';base64,')[1]
    try:
        data += '=' * (-len(data) % 4)
        base64.b64decode(data, validate=True)
    except Exception as e:
        raise ValueError(f"Invalid base64 encoding: {str(e)}")
    return data

def _save_base64_to_file(data_base64: str, temp_filename: str, data_type: str = "image") -> str:
    clean_b64 = _validate_and_decode_base64(data_base64, data_type)
    try:
        file_data = base64.b64decode(clean_b64)
        with open(temp_filename, "wb") as f:
            f.write(file_data)
        print(f"[{data_type.upper()}] Saved: {temp_filename} ({len(file_data)/1024:.2f} KB)")
        return temp_filename
    except Exception as e:
        raise ValueError(f"Failed to save {data_type}: {str(e)}")

def _download_model_safe(repo_id: str, filename: str, target_path: Path, model_type: str) -> bool:
    from huggingface_hub import hf_hub_download
    try:
        print(f"[{model_type.upper()}] Downloading: {filename}")
        cached_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(CACHE_PATH),
        )
        cached_file = Path(cached_path)
        if not cached_file.exists():
            print(f"[{model_type.upper()}] ERROR: Downloaded file doesn't exist")
            return False
        file_size_kb = cached_file.stat().st_size / 1024
        min_size = MIN_LORA_SIZE_KB if model_type == "loras" else MIN_FILE_SIZE_KB
        if file_size_kb < min_size:
            print(f"[{model_type.upper()}] ERROR: File too small ({file_size_kb:.2f} KB < {min_size} KB)")
            try:
                cached_file.unlink()
            except:
                pass
            return False
        subprocess.run(
            f"ln -sf {cached_path} {target_path}",
            shell=True,
            check=True,
        )
        print(f"[{model_type.upper()}] ✓ Linked: {filename} ({file_size_kb:.2f} KB)")
        return True
    except Exception as e:
        print(f"[{model_type.upper()}] ✗ Failed: {filename} - {str(e)}")
        return False

def setup_comfyui():
    print("\n" + "="*80)
    print("MODEL DOWNLOAD STARTED")
    print("="*80)
    stats = {"total": 0, "success": 0, "failed": 0}
    for model_type, models in MODEL_REGISTRY.items():
        print(f"\n[{model_type.upper()}] Processing {len(models)} models...")
        target_dir = MODEL_PATH / model_type
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename, source in models.items():
            stats["total"] += 1
            target_path = target_dir / filename
            if target_path.exists() and target_path.is_symlink():
                try:
                    file_size = target_path.stat().st_size / 1024
                    min_size = MIN_LORA_SIZE_KB if model_type == "loras" else MIN_FILE_SIZE_KB
                    if file_size > min_size:
                        print(f"[{model_type.upper()}] Already exists: {filename}")
                        stats["success"] += 1
                        continue
                    else:
                        target_path.unlink()
                except:
                    target_path.unlink()
            if _download_model_safe(
                source["repo_id"],
                source["filename"],
                target_path,
                model_type
            ):
                stats["success"] += 1
            else:
                stats["failed"] += 1
    print("\n" + "="*80)
    print(f"MODEL DOWNLOAD COMPLETE: {stats['success']}/{stats['total']} successful, {stats['failed']} failed")
    print("="*80 + "\n")

cache_volume = Volume.from_name("hf-hub-cache", create_if_missing=True)

comfy_image = (
    Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("comfy-cli==1.4.1", force_build=True)
    .run_commands(
        "comfy --skip-prompt install --fast-deps --nvidia --version 0.3.59",
        force_build=True,
    )
    .run_commands(
        "comfy node install comfyui_controlnet_aux",
        "comfy node install ComfyUI-KJNodes",
        "comfy node install ComfyUI-segment-anything-2",
        "comfy node install ComfyUI-WanVideoWrapper",
        "comfy node install ComfyUI-VideoHelperSuite",
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

app = App("comfyui-wan2-2-complete-production")

@app.cls(
    image=comfy_image,
    gpu="L40S",
    volumes={str(CACHE_PATH): cache_volume},
    timeout=7200,
    keep_warm=1,
)
class ComfyUI:
    @enter()
    def startup(self):
        print("\n" + "="*80)
        print("COMFYUI STARTUP")
        print("="*80)
        cmd = ["comfy", "launch", "--", "--listen", SERVER_HOST, "--port", str(SERVER_PORT)]
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"[SERVER] Started with PID: {self.proc.pid}")
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
                    error_detail = result.get('error', {})
                    print(f"[QUEUE] ERROR: {json.dumps(error_detail, indent=2)}")
                    raise RuntimeError(f"ComfyUI Error: {error_detail}")
                if 'prompt_id' not in result:
                    raise RuntimeError(f"Invalid response: {result}")
                print(f"[QUEUE] Prompt ID: {result['prompt_id']}")
                return result['prompt_id']
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            print(f"[QUEUE] HTTP {e.code}: {error_body}")
            raise RuntimeError(f"ComfyUI API Error {e.code}: {error_body}")

    def _get_history(self, prompt_id: str) -> dict:
        url = f"http://127.0.0.1:{SERVER_PORT}/history/{prompt_id}"
        with urllib.request.urlopen(url) as response:
            return json.loads(response.read())

    def _get_file(self, filename: str, subfolder: str, folder_type: str) -> bytes:
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
        ws_url = f"ws://127.0.0.1:{SERVER_PORT}/ws?clientId={client_id}"
        ws = None
        try:
            ws = websocket.WebSocket()
            ws.connect(ws_url, timeout=10)
            ws.settimeout(60)
            print("[WS] Connected")
        except Exception as e:
            raise RuntimeError(f"WebSocket connection failed: {str(e)}")
        
        start_time = time.time()
        generation_done = False
        
        try:
            while time.time() - start_time < MAX_GENERATION_TIME:
                try:
                    out = ws.recv()
                    if isinstance(out, str):
                        message = json.loads(out)
                        if message.get('type') == 'progress':
                            data = message.get('data', {})
                            value = data.get('value', 0)
                            max_val = max(data.get('max', 1), 1)
                            pct = (value / max_val) * 100
                            print(f"[WS] Progress: {value}/{max_val} ({pct:.1f}%)")
                        elif message.get('type') == 'executing':
                            data = message.get('data', {})
                            node = data.get('node')
                            if node is None:
                                print("[WS] Execution complete signal received")
                                generation_done = True
                                break
                            else:
                                print(f"[WS] Executing node: {node}")
                        elif message.get('type') == 'execution_error':
                            error_data = message.get('data', {})
                            print(f"[WS] Execution error: {error_data}")
                            raise RuntimeError(f"ComfyUI execution error: {error_data}")
                except websocket.WebSocketTimeoutException:
                    continue
        finally:
            if ws:
                ws.close()
        
        if not generation_done and (time.time() - start_time >= MAX_GENERATION_TIME):
            raise TimeoutError(f"Generation timeout ({MAX_GENERATION_TIME}s)")
        
        # Extended polling with more detailed logging
        POLL_TIMEOUT = 120  # Increased from 60 to 120 seconds
        poll_start_time = time.time()
        poll_count = 0
        
        while time.time() - poll_start_time < POLL_TIMEOUT:
            poll_count += 1
            print(f"[POLL] Attempt {poll_count}, elapsed: {time.time() - poll_start_time:.1f}s")
            
            try:
                history = self._get_history(prompt_id)
                
                if prompt_id not in history:
                    print(f"[POLL] Prompt ID {prompt_id} not in history yet")
                    time.sleep(2)
                    continue
                
                prompt_history = history[prompt_id]
                print(f"[POLL] History found for prompt {prompt_id}")
                print(f"[POLL] Status: {prompt_history.get('status', {})}")
                
                # Check if there was an error
                if 'status' in prompt_history:
                    status_data = prompt_history['status']
                    if status_data.get('status_str') == 'error':
                        error_info = status_data.get('messages', [])
                        raise RuntimeError(f"Generation failed: {error_info}")
                
                # Look for outputs
                outputs = prompt_history.get('outputs', {})
                print(f"[POLL] Found {len(outputs)} output nodes")
                
                for node_id, node_output in outputs.items():
                    print(f"[POLL] Node {node_id} output keys: {list(node_output.keys())}")
                    
                    # Check for videos or gifs
                    for media_type in ['videos', 'gifs']:
                        if media_type in node_output and node_output[media_type]:
                            items = node_output[media_type]
                            print(f"[POLL] Found {len(items)} {media_type} in node {node_id}")
                            
                            for item in items:
                                print(f"[OUTPUT] Downloading {media_type[:-1]}: {item.get('filename')}")
                                return self._get_file(
                                    item['filename'],
                                    item.get('subfolder', ''),
                                    item.get('type', 'output')
                                )
                    
                    # Also check for images as fallback
                    if 'images' in node_output and node_output['images']:
                        print(f"[POLL] Found images in node {node_id} (expected video)")
                
                print(f"[POLL] No video output found yet, waiting...")
                time.sleep(2)
                
            except Exception as e:
                print(f"[POLL] Error during polling: {str(e)}")
                time.sleep(2)
        
        # If we get here, we've timed out
        print(f"[ERROR] Polling timeout after {POLL_TIMEOUT}s")
        print(f"[ERROR] Last history state: {json.dumps(history.get(prompt_id, {}), indent=2)}")
        raise ValueError("No video output found after polling. Check ComfyUI logs for workflow errors.")

    def _copy_file_to_comfyui_input(self, base64_data: str, extension: str, data_type: str = "image") -> str:
        temp_file = f"/tmp/{uuid.uuid4()}.{extension}"
        _save_base64_to_file(base64_data, temp_file, data_type)
        input_dir = COMFYUI_PATH / "input"
        input_dir.mkdir(exist_ok=True)
        filename = f"{uuid.uuid4()}.{extension}"
        shutil.copy(temp_file, input_dir / filename)
        return filename

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
        use_fast_mode: bool = False
    ) -> str:
        print(f"\n[T2V] Mode: {'FAST (4-steps LoRA)' if use_fast_mode else 'STANDARD (fp8_scaled)'}")
        print(f"[T2V] Prompt: '{prompt[:50]}...'")
        client_id = str(uuid.uuid4())
        
        if seed is None:
            seed = int(time.time() * 1000000) % (2**32)
        
        if use_fast_mode:
            # Fast mode: dengan LoRA, shift 5.0, max 4 steps
            high_noise_model = "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
            low_noise_model = "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
            high_lora = "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors"
            low_lora = "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors"
            shift = 5.0
            steps = min(steps, 4)
        else:
            # Standard mode: tanpa LoRA, shift 8.0
            high_noise_model = "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
            low_noise_model = "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
            high_lora = None
            low_lora = None
            shift = 8.0
        
        # Base workflow nodes
        workflow = {
            "71": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
            "73": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
            "72": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["71", 0]}},
            "89": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["71", 0]}},
            "74": {"class_type": "EmptyHunyuanLatentVideo", "inputs": {"width": width, "height": height, "length": num_frames, "batch_size": 1}},
        }
        
        if use_fast_mode:
            # Fast mode dengan LoRA
            workflow.update({
                "75": {"class_type": "UNETLoader", "inputs": {"unet_name": high_noise_model, "weight_dtype": "default"}},
                "76": {"class_type": "UNETLoader", "inputs": {"unet_name": low_noise_model, "weight_dtype": "default"}},
                "83": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": high_lora, "strength_model": 1.0, "model": ["75", 0]}},
                "85": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": low_lora, "strength_model": 1.0, "model": ["76", 0]}},
                "82": {"class_type": "ModelSamplingSD3", "inputs": {"shift": shift, "model": ["83", 0]}},
                "86": {"class_type": "ModelSamplingSD3", "inputs": {"shift": shift, "model": ["85", 0]}},
                "81": {"class_type": "KSamplerAdvanced", "inputs": {
                    "add_noise": "enable", "noise_seed": seed, "steps": steps, "cfg": cfg,
                    "sampler_name": "euler", "scheduler": "simple",
                    "start_at_step": 0, "end_at_step": steps // 2, "return_with_leftover_noise": "enable",
                    "model": ["82", 0], "positive": ["89", 0], "negative": ["72", 0], "latent_image": ["74", 0]
                }},
                "78": {"class_type": "KSamplerAdvanced", "inputs": {
                    "add_noise": "disable", "noise_seed": 0, "steps": steps, "cfg": cfg,
                    "sampler_name": "euler", "scheduler": "simple",
                    "start_at_step": steps // 2, "end_at_step": steps, "return_with_leftover_noise": "disable",
                    "model": ["86", 0], "positive": ["89", 0], "negative": ["72", 0], "latent_image": ["81", 0]
                }},
                "87": {"class_type": "VAEDecode", "inputs": {"samples": ["78", 0], "vae": ["73", 0]}},
                "88": {"class_type": "CreateVideo", "inputs": {"fps": 16, "images": ["87", 0]}},
                "80": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "video/t2v_fast", "format": "auto", "codec": "auto", "video": ["88", 0]}}
            })
        else:
            # Standard mode tanpa LoRA
            workflow.update({
                "101": {"class_type": "UNETLoader", "inputs": {"unet_name": high_noise_model, "weight_dtype": "default"}},
                "102": {"class_type": "UNETLoader", "inputs": {"unet_name": low_noise_model, "weight_dtype": "default"}},
                "93": {"class_type": "ModelSamplingSD3", "inputs": {"shift": shift, "model": ["101", 0]}},
                "94": {"class_type": "ModelSamplingSD3", "inputs": {"shift": shift, "model": ["102", 0]}},
                "96": {"class_type": "KSamplerAdvanced", "inputs": {
                    "add_noise": "enable", "noise_seed": seed, "steps": steps, "cfg": cfg,
                    "sampler_name": "euler", "scheduler": "simple",
                    "start_at_step": 0, "end_at_step": 10, "return_with_leftover_noise": "enable",
                    "model": ["93", 0], "positive": ["89", 0], "negative": ["72", 0], "latent_image": ["74", 0]
                }},
                "95": {"class_type": "KSamplerAdvanced", "inputs": {
                    "add_noise": "disable", "noise_seed": 0, "steps": steps, "cfg": cfg,
                    "sampler_name": "euler", "scheduler": "simple",
                    "start_at_step": 10, "end_at_step": 10000, "return_with_leftover_noise": "disable",
                    "model": ["94", 0], "positive": ["89", 0], "negative": ["72", 0], "latent_image": ["96", 0]
                }},
                "97": {"class_type": "VAEDecode", "inputs": {"samples": ["95", 0], "vae": ["73", 0]}},
                "100": {"class_type": "CreateVideo", "inputs": {"fps": 16, "images": ["97", 0]}},
                "98": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "video/t2v_std", "format": "auto", "codec": "auto", "video": ["100", 0]}}
            })
        
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
        num_frames: int = 81,
        steps: int = 20,
        cfg: float = 3.5,
        seed: Optional[int] = None,
        use_fast_mode: bool = False
    ) -> str:
        print(f"\n[I2V] Mode: {'FAST (4-steps LoRA)' if use_fast_mode else 'STANDARD (fp8_scaled)'}")
        client_id = str(uuid.uuid4())
        
        if seed is None:
            seed = int(time.time() * 1000000) % (2**32)
        
        image_filename = self._copy_file_to_comfyui_input(image_base64, "png", "image")
        
        if use_fast_mode:
            # Fast mode dengan LoRA
            high_noise_model = "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
            low_noise_model = "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
            high_lora = "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
            low_lora = "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"
            shift = 5.0
            steps = min(steps, 4)
        else:
            # Standard mode tanpa LoRA
            high_noise_model = "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
            low_noise_model = "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
            high_lora = None
            low_lora = None
            shift = 8.0
        
        # Base nodes
        workflow = {
            "38": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
            "39": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["38", 0]}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["38", 0]}},
            "62": {"class_type": "LoadImage", "inputs": {"image": image_filename}},
        }
        
        if use_fast_mode:
            # Fast mode dengan LoRA (menggunakan node 95, 96, 101, 102 sesuai JSON)
            workflow.update({
                "95": {"class_type": "UNETLoader", "inputs": {"unet_name": high_noise_model, "weight_dtype": "default"}},
                "96": {"class_type": "UNETLoader", "inputs": {"unet_name": low_noise_model, "weight_dtype": "default"}},
                "101": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": high_lora, "strength_model": 1.0, "model": ["95", 0]}},
                "102": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": low_lora, "strength_model": 1.0, "model": ["96", 0]}},
                "104": {"class_type": "ModelSamplingSD3", "inputs": {"shift": shift, "model": ["101", 0]}},
                "103": {"class_type": "ModelSamplingSD3", "inputs": {"shift": shift, "model": ["102", 0]}},
                "98": {"class_type": "WanImageToVideo", "inputs": {
                    "width": width, "height": height, "length": num_frames, "batch_size": 1,
                    "positive": ["6", 0], "negative": ["7", 0], "vae": ["39", 0], "start_image": ["62", 0]
                }},
                "86": {"class_type": "KSamplerAdvanced", "inputs": {
                    "add_noise": "enable", "noise_seed": seed, "steps": steps, "cfg": cfg,
                    "sampler_name": "euler", "scheduler": "simple",
                    "start_at_step": 0, "end_at_step": 2, "return_with_leftover_noise": "enable",
                    "model": ["104", 0], "positive": ["98", 0], "negative": ["98", 1], "latent_image": ["98", 2]
                }},
                "85": {"class_type": "KSamplerAdvanced", "inputs": {
                    "add_noise": "disable", "noise_seed": 0, "steps": steps, "cfg": cfg,
                    "sampler_name": "euler", "scheduler": "simple",
                    "start_at_step": 2, "end_at_step": 4, "return_with_leftover_noise": "disable",
                    "model": ["103", 0], "positive": ["98", 0], "negative": ["98", 1], "latent_image": ["86", 0]
                }},
                "87": {"class_type": "VAEDecode", "inputs": {"samples": ["85", 0], "vae": ["39", 0]}},
                "94": {"class_type": "CreateVideo", "inputs": {"fps": 16, "images": ["87", 0]}},
                "108": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "video/i2v_fast", "format": "auto", "codec": "auto", "video": ["94", 0]}}
            })
        else:
            # Standard mode tanpa LoRA (menggunakan node 37, 56, 57, 58 sesuai JSON)
            workflow.update({
                "37": {"class_type": "UNETLoader", "inputs": {"unet_name": high_noise_model, "weight_dtype": "default"}},
                "56": {"class_type": "UNETLoader", "inputs": {"unet_name": low_noise_model, "weight_dtype": "default"}},
                "54": {"class_type": "ModelSamplingSD3", "inputs": {"shift": shift, "model": ["37", 0]}},
                "55": {"class_type": "ModelSamplingSD3", "inputs": {"shift": shift, "model": ["56", 0]}},
                "63": {"class_type": "WanImageToVideo", "inputs": {
                    "width": width, "height": height, "length": num_frames, "batch_size": 1,
                    "positive": ["6", 0], "negative": ["7", 0], "vae": ["39", 0], "start_image": ["62", 0]
                }},
                "57": {"class_type": "KSamplerAdvanced", "inputs": {
                    "add_noise": "enable", "noise_seed": seed, "steps": steps, "cfg": cfg,
                    "sampler_name": "euler", "scheduler": "simple",
                    "start_at_step": 0, "end_at_step": 10, "return_with_leftover_noise": "enable",
                    "model": ["54", 0], "positive": ["63", 0], "negative": ["63", 1], "latent_image": ["63", 2]
                }},
                "58": {"class_type": "KSamplerAdvanced", "inputs": {
                    "add_noise": "disable", "noise_seed": 0, "steps": steps, "cfg": cfg,
                    "sampler_name": "euler", "scheduler": "simple",
                    "start_at_step": 10, "end_at_step": 10000, "return_with_leftover_noise": "disable",
                    "model": ["55", 0], "positive": ["63", 0], "negative": ["63", 1], "latent_image": ["57", 0]
                }},
                "8": {"class_type": "VAEDecode", "inputs": {"samples": ["58", 0], "vae": ["39", 0]}},
                "109": {"class_type": "CreateVideo", "inputs": {"fps": 16, "images": ["8", 0]}},
                "61": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "video/i2v_std", "format": "auto", "codec": "auto", "video": ["109", 0]}}
            })
        
        prompt_id = self._queue_prompt(client_id, workflow)
        video_bytes = self._get_video_from_websocket(prompt_id, client_id)
        return base64.b64encode(video_bytes).decode('utf-8')

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
        use_fast_mode: bool = True
    ) -> str:
        print(f"\n[ANIMATE] Mode: {'FAST' if use_fast_mode else 'STANDARD'}")
        client_id = str(uuid.uuid4())
        
        if seed is None:
            seed = int(time.time() * 1000000) % (2**32)
        
        ref_image_filename = self._copy_file_to_comfyui_input(reference_image_base64, "png", "image")
        video_filename = self._copy_file_to_comfyui_input(video_base64, "mp4", "video")
        
        workflow = {
            "20": {"class_type": "UNETLoader", "inputs": {"unet_name": "Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors", "weight_dtype": "default"}},
            "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
            "4": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "clip_vision_h.safetensors"}},
            "21": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
            "1": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["2", 0]}},
            "10": {"class_type": "LoadImage", "inputs": {"image": ref_image_filename}},
            "9": {"class_type": "CLIPVisionEncode", "inputs": {"crop": "none", "clip_vision": ["4", 0], "image": ["10", 0]}},
            "145": {"class_type": "LoadVideo", "inputs": {"video": video_filename}},
            "23": {"class_type": "GetVideoComponents", "inputs": {"video": ["145", 0]}},
            "212": {"class_type": "ImageScale", "inputs": {"upscale_method": "lanczos", "width": width, "height": height, "crop": "center", "image": ["23", 0]}},
            "100": {"class_type": "DWPreprocessor", "inputs": {"detect_hand": "disable", "detect_body": "disable", "detect_face": "enable", "resolution": 512, "bbox_detector": "yolox_l.onnx", "pose_estimator": "dw-ll_ucoco_384_bs5.torchscript.pt", "scale_stick_for_xinsr_cn": "disable", "image": ["212", 0]}},
            "101": {"class_type": "DWPreprocessor", "inputs": {"detect_hand": "enable", "detect_body": "enable", "detect_face": "disable", "resolution": 512, "bbox_detector": "yolox_l.onnx", "pose_estimator": "dw-ll_ucoco_384_bs5.torchscript.pt", "scale_stick_for_xinsr_cn": "disable", "image": ["212", 0]}}
        }
        
        if use_fast_mode:
            workflow["18"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", "strength_model": 1.0, "model": ["20", 0]}}
            workflow["99"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "WanAnimate_relight_lora_fp16.safetensors", "strength_model": 1.0, "model": ["18", 0]}}
            model_source = ["99", 0]
        else:
            model_source = ["20", 0]
        
        workflow.update({
            "60": {"class_type": "ModelSamplingSD3", "inputs": {"shift": 8.0, "model": model_source}},
            "232": {"class_type": "WanAnimateToVideo", "inputs": {"width": width, "height": height, "length": num_frames, "batch_size": 1, "mode": 5, "video_frame_offset": 0, "positive": ["21", 0], "negative": ["1", 0], "vae": ["3", 0], "clip_vision_output": ["9", 0], "reference_image": ["10", 0], "face_video": ["100", 0], "pose_video": ["101", 0]}},
            "63": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0, "model": ["60", 0], "positive": ["232", 0], "negative": ["232", 1], "latent_image": ["232", 2]}},
            "58": {"class_type": "VAEDecode", "inputs": {"samples": ["63", 0], "vae": ["3", 0]}},
            "15": {"class_type": "CreateVideo", "inputs": {"fps": 16, "images": ["58", 0], "audio": ["23", 1]}},
            "19": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "video/animate", "format": "auto", "codec": "auto", "video": ["15", 0]}}
        })
        
        prompt_id = self._queue_prompt(client_id, workflow)
        video_bytes = self._get_video_from_websocket(prompt_id, client_id)
        return base64.b64encode(video_bytes).decode('utf-8')

    @method()
    def apply_camera_lora(
        self,
        image_base64: str,
        prompt: str,
        camera_motion: str = "ZoomIn",
        lora_strength: float = 1.0,
        negative_prompt: str = "",
        width: int = 1280,
        height: int = 704,
        num_frames: int = 81,
        steps: int = 20,
        cfg: float = 3.5,
        seed: Optional[int] = None
    ) -> str:
        print(f"\n[CAMERA] Applying: {camera_motion} (strength: {lora_strength})")
        client_id = str(uuid.uuid4())
        
        if seed is None:
            seed = int(time.time() * 1000000) % (2**32)
        
        valid_motions = ["ZoomIn", "ZoomOut", "PanLeft", "PanRight", "TiltUp", "TiltDown", "RollingClockwise", "RollingAnticlockwise"]
        if camera_motion not in valid_motions:
            raise ValueError(f"Invalid camera motion. Must be one of: {valid_motions}")
        
        lora_name = f"v2_lora_{camera_motion}.safetensors"
        image_filename = self._copy_file_to_comfyui_input(image_base64, "png", "image")
        
        workflow = {
            "37": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}},
            "56": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}},
            "200": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": lora_name, "strength_model": lora_strength, "model": ["37", 0]}},
            "201": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": lora_name, "strength_model": lora_strength, "model": ["56", 0]}},
            "38": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
            "39": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["38", 0]}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["38", 0]}},
            "62": {"class_type": "LoadImage", "inputs": {"image": image_filename}},
            "63": {"class_type": "WanImageToVideo", "inputs": {"width": width, "height": height, "length": num_frames, "batch_size": 1, "positive": ["6", 0], "negative": ["7", 0], "vae": ["39", 0], "start_image": ["62", 0]}},
            "54": {"class_type": "ModelSamplingSD3", "inputs": {"shift": 8.0, "model": ["200", 0]}},
            "55": {"class_type": "ModelSamplingSD3", "inputs": {"shift": 8.0, "model": ["201", 0]}},
            "57": {"class_type": "KSamplerAdvanced", "inputs": {"add_noise": "enable", "noise_seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "simple", "start_at_step": 0, "end_at_step": 10, "return_with_leftover_noise": "enable", "model": ["54", 0], "positive": ["63", 0], "negative": ["63", 1], "latent_image": ["63", 2]}},
            "58": {"class_type": "KSamplerAdvanced", "inputs": {"add_noise": "disable", "noise_seed": 0, "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "simple", "start_at_step": 10, "end_at_step": 10000, "return_with_leftover_noise": "disable", "model": ["55", 0], "positive": ["63", 0], "negative": ["63", 1], "latent_image": ["57", 0]}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["58", 0], "vae": ["39", 0]}},
            "109": {"class_type": "CreateVideo", "inputs": {"fps": 16, "images": ["8", 0]}},
            "61": {"class_type": "SaveVideo", "inputs": {"filename_prefix": f"video/camera_{camera_motion}", "format": "auto", "codec": "auto", "video": ["109", 0]}}
        }
        
        prompt_id = self._queue_prompt(client_id, workflow)
        video_bytes = self._get_video_from_websocket(prompt_id, client_id)
        return base64.b64encode(video_bytes).decode('utf-8')

    @method()
    def get_available_loras(self) -> List[str]:
        return list(MODEL_REGISTRY["loras"].keys())

    @method()
    def get_available_camera_motions(self) -> List[str]:
        return ["ZoomIn", "ZoomOut", "PanLeft", "PanRight", "TiltUp", "TiltDown", "RollingClockwise", "RollingAnticlockwise"]

# FastAPI Web Application
web_app = FastAPI(title="ComfyUI Wan 2.2 Complete API")

class T2VRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for video generation")
    negative_prompt: str = Field("", description="Negative prompt")
    width: int = Field(832, ge=256, le=2048, description="Video width (must be multiple of 16)")
    height: int = Field(480, ge=256, le=2048, description="Video height (must be multiple of 16)")
    num_frames: int = Field(121, ge=1, le=240, description="Number of frames")
    steps: int = Field(30, ge=1, le=100, description="Sampling steps")
    cfg: float = Field(7.5, ge=1.0, le=20.0, description="CFG scale")
    seed: Optional[int] = Field(None, description="Random seed")
    use_fast_mode: bool = Field(False, description="Use fast mode (4 steps with LoRA)")

class I2VRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded input image")
    prompt: str = Field(..., description="Text prompt for video generation")
    negative_prompt: str = Field("", description="Negative prompt")
    width: int = Field(1280, ge=256, le=2048, description="Video width")
    height: int = Field(704, ge=256, le=2048, description="Video height")
    num_frames: int = Field(81, ge=1, le=240, description="Number of frames")
    steps: int = Field(20, ge=1, le=100, description="Sampling steps")
    cfg: float = Field(3.5, ge=1.0, le=20.0, description="CFG scale")
    seed: Optional[int] = Field(None, description="Random seed")
    use_fast_mode: bool = Field(False, description="Use fast mode")

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

class CameraLoraRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded input image")
    prompt: str = Field(..., description="Text prompt")
    camera_motion: str = Field(..., description="Camera motion type")
    lora_strength: float = Field(1.0, ge=0.0, le=2.0, description="LoRA strength")
    negative_prompt: str = Field("", description="Negative prompt")
    width: int = Field(1280, ge=256, le=2048, description="Video width")
    height: int = Field(704, ge=256, le=2048, description="Video height")
    num_frames: int = Field(81, ge=1, le=240, description="Number of frames")
    steps: int = Field(20, ge=1, le=100, description="Sampling steps")
    cfg: float = Field(3.5, ge=1.0, le=20.0, description="CFG scale")
    seed: Optional[int] = Field(None, description="Random seed")

@web_app.get("/")
async def root():
    return {
        "service": "ComfyUI Wan 2.2 Complete API",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "t2v": "/api/generate/t2v",
            "i2v": "/api/generate/i2v",
            "animate": "/api/generate/animate",
            "camera_lora": "/api/generate/camera-lora",
            "list_loras": "/api/loras",
            "list_camera_motions": "/api/camera-motions",
            "health": "/health"
        },
        "documentation": "/docs"
    }

@web_app.post("/api/generate/t2v")
async def api_generate_t2v(request: T2VRequest):
    try:
        result = ComfyUI().generate_t2v.remote(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_frames=request.num_frames,
            steps=request.steps,
            cfg=request.cfg,
            seed=request.seed,
            use_fast_mode=request.use_fast_mode
        )
        return {"success": True, "video_base64": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@web_app.post("/api/generate/i2v")
async def api_generate_i2v(request: I2VRequest):
    try:
        result = ComfyUI().generate_i2v.remote(
            image_base64=request.image_base64,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_frames=request.num_frames,
            steps=request.steps,
            cfg=request.cfg,
            seed=request.seed,
            use_fast_mode=request.use_fast_mode
        )
        return {"success": True, "video_base64": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@web_app.post("/api/generate/animate")
async def api_generate_animate(request: AnimateRequest):
    try:
        result = ComfyUI().generate_animate.remote(
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
            use_fast_mode=request.use_fast_mode
        )
        return {"success": True, "video_base64": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@web_app.post("/api/generate/camera-lora")
async def api_apply_camera_lora(request: CameraLoraRequest):
    try:
        result = ComfyUI().apply_camera_lora.remote(
            image_base64=request.image_base64,
            prompt=request.prompt,
            camera_motion=request.camera_motion,
            lora_strength=request.lora_strength,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_frames=request.num_frames,
            steps=request.steps,
            cfg=request.cfg,
            seed=request.seed
        )
        return {"success": True, "video_base64": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@web_app.get("/api/loras")
async def api_list_loras():
    try:
        result = ComfyUI().get_available_loras.remote()
        return {"success": True, "loras": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@web_app.get("/api/camera-motions")
async def api_list_camera_motions():
    try:
        result = ComfyUI().get_available_camera_motions.remote()
        return {"success": True, "camera_motions": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@web_app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ComfyUI Wan 2.2 Complete"}

@app.function(
    image=comfy_image,
    keep_warm=1,
)
@asgi_app()
def fastapi_app():
    return web_app

if __name__ == "__main__":
    print("ComfyUI Wan 2.2 Complete Production API")
    print("=" * 80)
    print("\nTo deploy this application:")
    print("1. Install Modal: pip install modal")
    print("2. Set up Modal: modal setup")
    print("3. Deploy: modal deploy app.py")
    print("\nThe API will be available at the URL provided by Modal")
    print("\nAvailable endpoints:")
    print("  POST /api/generate/t2v - Text-to-Video")
    print("  POST /api/generate/i2v - Image-to-Video")
    print("  POST /api/generate/animate - Animate with pose transfer")
    print("  POST /api/generate/camera-lora - Apply camera motion")
    print("  GET  /api/loras - List available LoRAs")
    print("  GET  /api/camera-motions - List camera motions")
    print("  GET  /health - Health check")
    print("\nFor detailed usage examples, see the deployment notes in the code")
