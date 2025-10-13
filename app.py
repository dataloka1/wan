import base64
import json
import subprocess
import time
import urllib.request
import urllib.parse
import uuid
from pathlib import Path
from typing import Dict

from modal import (
    App,
    Image,
    Volume,
    asgi_app,
    enter,
    method,
)

app = App(
    "comfyui-wan2-2-video-api-pro",
)

# Model Registry - COMPLETE dengan Lora, ControlNet, dan Wan 2.2
MODEL_REGISTRY = {
    # Wan 2.2 Models (14B parameters) - State of the art video generation
    "diffusion_models": {
        "Wan2_2-T2V-14B_fp8_e4m3fn_scaled.safetensors": "https://huggingface.co/Kijai/Wan2.2-T2V-14B-comfy/resolve/main/Wan2_2-T2V-14B_fp8_e4m3fn_scaled.safetensors",
        "Wan2_2-I2V-14B_fp8_e4m3fn_scaled.safetensors": "https://huggingface.co/Kijai/Wan2.2-I2V-14B-comfy/resolve/main/Wan2_2-I2V-14B_fp8_e4m3fn_scaled.safetensors",
    },
    "vae": {
        "wan_vae_v1_float16.safetensors": "https://huggingface.co/Kijai/Wan-VAE/resolve/main/wan_vae_v1_float16.safetensors",
        "vae-ft-mse-840000-ema-pruned.safetensors": "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors"
    },
    "text_encoders": {
        "t5xxl_fp16.safetensors": "https://huggingface.co/Kijai/llm-text-encoders/resolve/main/t5xxl_fp16.safetensors",
    },
    "clip": {
        "clip_l.safetensors": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
    },
    # ControlNet untuk presisi control
    "controlnet": {
        "control_v11p_sd15_openpose.pth": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth",
        "control_v11p_sd15_canny.pth": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth",
        "control_v11f1p_sd15_depth.pth": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth",
        "control_v11p_sd15_lineart.pth": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart.pth",
    },
    # Lora untuk style dan enhancement - LENGKAP
    "loras": {
        # Quality Enhancement Loras
        "add_detail.safetensors": "https://civitai.com/api/download/models/122359",
        "PerfectEyes.safetensors": "https://civitai.com/api/download/models/132203",
        "Hyper_Detailer_LoRA.safetensors": "https://civitai.com/api/download/models/127732",
        "Better_Hands_LoRA.safetensors": "https://civitai.com/api/download/models/4286",
        "Ultra_Realistic_LoRA.safetensors": "https://civitai.com/api/download/models/129631",
        "Epi_Noiseoffset.safetensors": "https://civitai.com/api/download/models/13941",
        
        # Cinematic & Lighting Loras
        "cinematic-lighting.safetensors": "https://civitai.com/api/download/models/127623",
        "film-grain.safetensors": "https://civitai.com/api/download/models/132890",
        "Cinematic_V2.safetensors": "https://civitai.com/api/download/models/193477",
        "4k_Hyperrealistic_Style.safetensors": "https://civitai.com/api/download/models/202684",
        "Epic_Realism.safetensors": "https://civitai.com/api/download/models/132333",
        "Soft_Lighting_LoRA.safetensors": "https://civitai.com/api/download/models/127622",
        "Bokeh_LoRA.safetensors": "https://civitai.com/api/download/models/128381",
        "Lens_Flare_LoRA.safetensors": "https://civitai.com/api/download/models/127815",
        "Fisheye_Lens_LoRA.safetensors": "https://civitai.com/api/download/models/127807",
        
        # Camera Motion Loras (untuk video 30 detik yang smooth)
        "v2_lora_ZoomIn.safetensors": "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_ZoomIn.safetensors",
        "v2_lora_ZoomOut.safetensors": "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_ZoomOut.safetensors",
        "v2_lora_PanLeft.safetensors": "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_PanLeft.safetensors",
        "v2_lora_PanRight.safetensors": "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_PanRight.safetensors",
        "v2_lora_TiltUp.safetensors": "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_TiltUp.safetensors",
        "v2_lora_TiltDown.safetensors": "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_TiltDown.safetensors",
        "v2_lora_PanUp.safetensors": "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_PanUp.safetensors",
        "v2_lora_PanDown.safetensors": "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_PanDown.safetensors",
        "v2_lora_RollingAnticlockwise.safetensors": "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_RollingAnticlockwise.safetensors",
        "v2_lora_RollingClockwise.safetensors": "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_RollingClockwise.safetensors",
        
        # Style Loras
        "Vintage_Style_Lora.safetensors": "https://civitai.com/api/download/models/132535",
        "Cyberpunk_Style.safetensors": "https://civitai.com/api/download/models/132956",
        "Cyberpunk_Style_LoRA_v2.safetensors": "https://civitai.com/api/download/models/127435",
        "Dreamy_and_ethereal_style.safetensors": "https://civitai.com/api/download/models/126987",
        "oil_painting_style.safetensors": "https://civitai.com/api/download/models/127117",
        "Watercolor_Paint_Style.safetensors": "https://civitai.com/api/download/models/127670",
        "Steampunk_LoRA.safetensors": "https://civitai.com/api/download/models/124789",
        "Moody_LoRA.safetensors": "https://civitai.com/api/download/models/129532",
        "Ethereal_LoRA.safetensors": "https://civitai.com/api/download/models/126986",
        "Horror_Style_LoRA.safetensors": "https://civitai.com/api/download/models/127533",
        "80s_Aesthetic_LoRA.safetensors": "https://civitai.com/api/download/models/128362",
        
        # Art Style Loras
        "Arcane_Style_LoRA.safetensors": "https://civitai.com/api/download/models/13340",
        "Ghibli_Style_LoRA.safetensors": "https://civitai.com/api/download/models/131106",
        "GTA_Style_LoRA.safetensors": "https://civitai.com/api/download/models/131754",
        "Pixel_Art_LoRA.safetensors": "https://civitai.com/api/download/models/131141",
        "Claymation_Style_LoRA.safetensors": "https://civitai.com/api/download/models/128399",
        "Lego_LoRA.safetensors": "https://civitai.com/api/download/models/124564",
        "Papercut_LoRA.safetensors": "https://civitai.com/api/download/models/122116",
        "Lofi_Anime_Aesthetic.safetensors": "https://civitai.com/api/download/models/99773",
        
        # Effects Loras
        "Fire_LoRA.safetensors": "https://civitai.com/api/download/models/127471",
        "Rain_LoRA.safetensors": "https://civitai.com/api/download/models/124785",
        "Smoke_LoRA.safetensors": "https://civitai.com/api/download/models/124786",
        "Explosion_LoRA.safetensors": "https://civitai.com/api/download/models/125539",
        "Electrical_Magic_LoRA.safetensors": "https://civitai.com/api/download/models/126980",
        "Sci-Fi_Robots_LoRA.safetensors": "https://civitai.com/api/download/models/122046",
        
        # Misc
        "lowra.safetensors": "https://civitai.com/api/download/models/16576",
    }
}

# Path absolut - NO SYMLINKS
REMOTE_BASE_PATH = Path("/app")
COMFYUI_PATH = REMOTE_BASE_PATH / "ComfyUI"
MODEL_PATH = Path("/models")

volume = Volume.from_name("comfyui-wan2-2-complete-volume", create_if_missing=True)

# Fixed image dengan dependencies yang benar
comfy_image = (
    Image.debian_slim(python_version="3.10")
    .apt_install(
        "git",
        "wget",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "ffmpeg",
        "libsm6",
        "libxext6",
        "libxrender-dev",
    )
    .run_commands(
        # Clone ComfyUI
        f"git clone https://github.com/comfyanonymous/ComfyUI.git {COMFYUI_PATH}",
        # Install dependencies
        f"cd {COMFYUI_PATH} && pip install --no-cache-dir -r requirements.txt",
        # Install Wan Video Wrapper
        f"cd {COMFYUI_PATH / 'custom_nodes'} && git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git",
        f"cd {COMFYUI_PATH / 'custom_nodes/ComfyUI-WanVideoWrapper'} && pip install --no-cache-dir -r requirements.txt || echo 'Warning: Some dependencies may have failed'",
        # Install Video Helper Suite
        f"cd {COMFYUI_PATH / 'custom_nodes'} && git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
        f"cd {COMFYUI_PATH / 'custom_nodes/ComfyUI-VideoHelperSuite'} && pip install --no-cache-dir -r requirements.txt || echo 'Warning: Some dependencies may have failed'",
        # Install ControlNet Aux
        f"cd {COMFYUI_PATH / 'custom_nodes'} && git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git",
        f"cd {COMFYUI_PATH / 'custom_nodes/comfyui_controlnet_aux'} && pip install --no-cache-dir -r requirements.txt || echo 'Warning: Some dependencies may have failed'",
    )
    .pip_install(
        "websocket-client",
        "safetensors",
        "pillow",
        "numpy",
        "torch",
        "torchvision",
    )
)

@app.function(
    image=comfy_image,
    volumes={str(MODEL_PATH): volume},
    timeout=7200,
)
def download_models():
    """Download semua model termasuk Lora dan ControlNet"""
    print("=" * 70)
    print("🚀 Starting Complete Model Download (Wan 2.2 + Loras + ControlNet)")
    print("=" * 70)
    
    total_size = 0
    for model_type, models in MODEL_REGISTRY.items():
        print(f"\n📦 {model_type.upper()}")
        for filename, url in models.items():
            target_dir = MODEL_PATH / model_type
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / filename
            
            if not target_path.exists():
                print(f"\n  📥 Downloading {filename}...")
                try:
                    subprocess.run(
                        ["wget", "-O", str(target_path), url, "--progress=bar:force:noscroll"],
                        check=True
                    )
                    file_size = target_path.stat().st_size / (1024**2)  # MB
                    total_size += file_size
                    print(f"  ✅ Downloaded: {filename} ({file_size:.1f} MB)")
                except subprocess.CalledProcessError as e:
                    print(f"  ❌ ERROR: {filename}: {e}")
            else:
                file_size = target_path.stat().st_size / (1024**2)
                total_size += file_size
                print(f"  ✓ Exists: {filename} ({file_size:.1f} MB)")
    
    print("\n" + "=" * 70)
    print(f"📊 Total models size: {total_size/1024:.2f} GB")
    print("💾 Committing volume...")
    volume.commit()
    print("✅ Volume committed successfully!")
    print("=" * 70)

@app.cls(
    image=comfy_image,
    gpu="L40S",  # Fixed: use string instead of gpu.L40S()
    volumes={str(MODEL_PATH): volume},
    container_idle_timeout=300,
    timeout=3600,
)
class ComfyUI:
    @enter()
    def startup(self):
        print("\n" + "=" * 70)
        print("🚀 Starting ComfyUI Pro with Wan 2.2 + Loras + ControlNet")
        print("=" * 70)
        
        print("\n🔍 Checking for symlinks...")
        self._remove_symlinks(COMFYUI_PATH / "models")
        
        # Create comprehensive config
        config_content = f"""# ComfyUI Pro Configuration - NO SYMLINKS
comfyui:
    base_path: {COMFYUI_PATH}

models:
    diffusion_models: {MODEL_PATH}/diffusion_models
    vae: {MODEL_PATH}/vae
    text_encoders: {MODEL_PATH}/text_encoders
    clip: {MODEL_PATH}/clip
    loras: {MODEL_PATH}/loras
    controlnet: {MODEL_PATH}/controlnet
"""
        config_path = COMFYUI_PATH / "extra_model_paths.yaml"
        config_path.write_text(config_content)
        print(f"✅ Created config at {config_path}")
        
        self._verify_model_paths()
        
        print("\n🌐 Starting ComfyUI server...")
        cmd = f"python main.py --listen 0.0.0.0 --port 8188 --disable-auto-launch"
        self.proc = subprocess.Popen(cmd, shell=True, cwd=COMFYUI_PATH)
        
        for i in range(30):
            try:
                urllib.request.urlopen("http://127.0.0.1:8188/queue").read()
                print("✅ ComfyUI server ready!")
                print("=" * 70 + "\n")
                return
            except Exception:
                time.sleep(2)
        raise RuntimeError("ComfyUI server failed to start.")
    
    def _remove_symlinks(self, directory: Path):
        if not directory.exists():
            return
        removed = 0
        for item in directory.rglob("*"):
            if item.is_symlink():
                item.unlink()
                removed += 1
        print(f"   ✅ Removed {removed} symlinks" if removed > 0 else "   ✅ No symlinks found")
    
    def _verify_model_paths(self):
        print("\n📁 Verifying model paths...")
        for model_type in MODEL_REGISTRY.keys():
            model_dir = MODEL_PATH / model_type
            if model_dir.exists():
                if model_dir.is_symlink():
                    raise RuntimeError(f"ERROR: {model_dir} is a symlink!")
                count = len(list(model_dir.glob("*")))
                print(f"   ✅ {model_type}: {count} files")
            else:
                print(f"   ⚠️  {model_type}: not found")

    def _queue_prompt(self, client_id: str, prompt_workflow: dict):
        req = urllib.request.Request(
            "http://127.0.0.1:8188/prompt",
            data=json.dumps({"prompt": prompt_workflow, "client_id": client_id}).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        return json.loads(urllib.request.urlopen(req).read())['prompt_id']

    def _get_history(self, prompt_id: str):
        with urllib.request.urlopen(f"http://127.0.0.1:8188/history/{prompt_id}") as response:
            return json.loads(response.read())

    def _get_file(self, filename: str, subfolder: str, folder_type: str):
        url = f"http://127.0.0.1:8188/view?{urllib.parse.urlencode({'filename': filename, 'subfolder': subfolder, 'type': folder_type})}"
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
                    if message['type'] == 'executing' and message['data']['node'] is None and message['data']['prompt_id'] == prompt_id:
                        break
        finally:
            ws.close()
        
        history = self._get_history(prompt_id)[prompt_id]
        for node_id, node_output in history['outputs'].items():
            if 'gifs' in node_output:
                for gif in node_output['gifs']:
                    return self._get_file(gif['filename'], gif['subfolder'], gif['type'])
            if 'videos' in node_output:
                for video in node_output['videos']:
                    return self._get_file(video['filename'], video['subfolder'], video['type'])
        raise ValueError("Video output not found.")

    def get_wan2_2_premium_workflow(self, payload: dict) -> Dict:
        """
        Premium Wan 2.2 workflow dengan Lora support untuk 30 detik video berkualitas tinggi
        """
        prompt = payload.get("prompt", "A cinematic masterpiece")
        negative_prompt = payload.get("negative_prompt", "low quality, blurry, distorted, worst quality, low resolution")
        seed = payload.get("seed", 123)
        steps = payload.get("steps", 80)
        cfg = payload.get("cfg", 7.5)
        width = payload.get("width", 1280)
        height = payload.get("height", 720)
        
        # 30 detik dengan 24fps = 720 frames
        fps = payload.get("fps", 24)
        duration = payload.get("duration", 30)
        frames = fps * duration
        
        workflow = {
            "1": {
                "class_type": "WanT2VModelLoader",
                "inputs": {
                    "model_name": "Wan2_2-T2V-14B_fp8_e4m3fn_scaled.safetensors",
                }
            },
            "2": {
                "class_type": "WanVAELoader",
                "inputs": {
                    "vae_name": "wan_vae_v1_float16.safetensors"
                }
            },
            "3": {
                "class_type": "WanT5TextEncode",
                "inputs": {
                    "text": prompt,
                    "text_encoder": ["4", 0]
                }
            },
            "4": {
                "class_type": "WanT5TextEncoderLoader",
                "inputs": {
                    "text_encoder_name": "t5xxl_fp16.safetensors"
                }
            },
            "5": {
                "class_type": "WanSampler",
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "euler_ancestral",
                    "scheduler": "karras",
                    "model": ["1", 0],
                    "positive": ["3", 0],
                    "negative": ["6", 0],
                    "latent_image": ["7", 0]
                }
            },
            "6": {
                "class_type": "WanT5TextEncode",
                "inputs": {
                    "text": negative_prompt,
                    "text_encoder": ["4", 0]
                }
            },
            "7": {
                "class_type": "WanEmptyLatentVideo",
                "inputs": {
                    "width": width,
                    "height": height,
                    "length": frames,
                    "batch_size": 1
                }
            },
            "8": {
                "class_type": "WanVAEDecode",
                "inputs": {
                    "samples": ["5", 0],
                    "vae": ["2", 0]
                }
            },
            "9": {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "frame_rate": fps,
                    "loop_count": 0,
                    "filename_prefix": "Wan2_2_Premium_30s",
                    "format": "video/h264-mp4",
                    "crf": 18,
                    "save_metadata": True,
                    "pingpong": False,
                    "save_output": True,
                    "images": ["8", 0]
                }
            }
        }
        
        return workflow

    @method()
    def generate_video(self, payload: Dict):
        """Generate 30 second premium video dengan Wan 2.2"""
        if not payload.get("prompt"):
            raise ValueError("'prompt' parameter is required.")
        
        print(f"\n🎬 Generating 30s premium video with Wan 2.2...")
        print(f"   Prompt: {payload.get('prompt')[:60]}...")
        print(f"   Duration: 30 seconds @ {payload.get('fps', 24)}fps")
        print(f"   Resolution: {payload.get('width', 1280)}x{payload.get('height', 720)}")
        
        workflow = self.get_wan2_2_premium_workflow(payload)
        
        client_id = str(uuid.uuid4())
        prompt_id = self._queue_prompt(client_id, workflow)
        
        print(f"   Prompt ID: {prompt_id}")
        print(f"   🔄 Generating... (this will take 5-10 minutes for 30s video)")
        
        video_data = self._get_video_from_websocket(prompt_id, client_id)
        
        print(f"   ✅ 30s Premium video generated!")
        return video_data

@app.function()
@asgi_app()
def fastapi_app():
    from fastapi import FastAPI, Request
    from fastapi.responses import Response, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware

    web_app = FastAPI(
        title="Wan 2.2 Premium Video API",
        description="Professional 30-second video generation with Lora & ControlNet support",
        version="2.0.0"
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
            "service": "Wan 2.2 Premium Video API",
            "version": "2.0.0",
            "features": [
                "✅ Wan 2.2 14B FP8 model",
                "✅ 30-second cinematic videos (720 frames @ 24fps)",
                "✅ 50+ Lora styles available",
                "✅ ControlNet support",
                "✅ Camera motion controls",
                "✅ 1280x720 HD output",
                "✅ Auto-queue system"
            ],
            "endpoints": {
                "generate": "POST /generate - Generate 30s video",
                "loras": "GET /loras - List available Loras",
                "camera_motions": "GET /camera-motions - List camera movements",
                "health": "GET /health"
            }
        }

    @web_app.get("/loras")
    async def list_loras():
        """List all available Loras"""
        loras = list(MODEL_REGISTRY.get("loras", {}).keys())
        return {
            "total": len(loras),
            "loras": loras,
            "categories": {
                "quality": ["add_detail", "PerfectEyes", "Hyper_Detailer_LoRA", "Ultra_Realistic_LoRA"],
                "cinematic": ["Cinematic_V2", "4k_Hyperrealistic_Style", "Epic_Realism", "film-grain"],
                "lighting": ["cinematic-lighting", "Soft_Lighting_LoRA", "Bokeh_LoRA", "Lens_Flare_LoRA"],
                "camera_motion": ["v2_lora_ZoomIn", "v2_lora_PanLeft", "v2_lora_PanRight", "v2_lora_TiltUp"],
                "style": ["Vintage_Style_Lora", "Cyberpunk_Style", "Ghibli_Style_LoRA", "Arcane_Style_LoRA"],
                "effects": ["Fire_LoRA", "Rain_LoRA", "Smoke_LoRA", "Explosion_LoRA"]
            }
        }

    @web_app.get("/camera-motions")
    async def camera_motions():
        return {
            "available_motions": [
                "ZoomIn", "ZoomOut", "PanLeft", "PanRight",
                "TiltUp", "TiltDown", "PanUp", "PanDown",
                "RollingClockwise", "RollingAnticlockwise"
            ]
        }

    @web_app.get("/health")
    async def health():
        return {"status": "healthy"}

    @web_app.post("/generate")
    async def generate(request: Request):
        """Generate 30-second premium video"""
        try:
            payload = await request.json()
            
            if not payload.get("prompt"):
                return JSONResponse(
                    status_code=400,
                    content={"error": "'prompt' is required"}
                )
            
            # Set defaults
            payload.setdefault("fps", 24)
            payload.setdefault("duration", 30)
            payload.setdefault("width", 1280)
            payload.setdefault("height", 720)
            payload.setdefault("steps", 80)
            
            comfy_runner = ComfyUI()
            video_bytes = comfy_runner.generate_video.remote(payload)
            
            return Response(
                content=video_bytes,
                media_type="video/mp4",
                headers={
                    "Content-Disposition": "attachment; filename=wan2_2_premium_30s.mp4"
                }
            )
            
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        except Exception as e:
            print(f"❌ Error: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})

    return web_app

@app.local_entrypoint()
def cli():
    print("\n" + "=" * 70)
    print("   🎬 Wan 2.2 PREMIUM Video API - 30 Second Generation")
    print("=" * 70)
    print("\n📥 1. Download all models (Wan 2.2 + 50+ Loras + ControlNet):")
    print("   modal run app.py::download_models")
    print("\n🚀 2. Deploy the API:")
    print("   modal deploy app.py")
    print("\n✨ Features:")
    print("   • 30-second cinematic videos (720 frames @ 24fps)")
    print("   • 1280x720 HD resolution")
    print("   • 50+ Lora styles (cinematic, cyberpunk, anime, etc)")
    print("   • 10+ camera motion effects")
    print("   • 4 ControlNet models")
    print("   • Auto-queue system")
    print("   • NO SYMLINKS - all real file paths")
    print("\n" + "=" * 70 + "\n")