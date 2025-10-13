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
    cls,
    enter,
    gpu,
    method,
)

app = App(
    "comfyui-video-prod-api-v2",
)

MODEL_REGISTRY = {
    "checkpoints": {
        "realisticVisionV60_v60B1.safetensors": "https://huggingface.co/segmind/Realistic-Vision-V6.0-B1/resolve/main/Realistic_Vision_V6.0_B1_noVAE.safetensors",
        "svd_xt.safetensors": "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt.safetensors",
    },
    "vae": {
        "vae-ft-mse-840000-ema-pruned.safetensors": "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors"
    },
    "animatediff_models": {
        "mm_sd_v15_v2.ckpt": "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt"
    },
    "controlnet": {
        "control_v11p_sd15_openpose.pth": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth",
        "control_v11p_sd15_canny.pth": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth",
        "control_v11f1p_sd15_depth.pth": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth",
    },
    "loras": {
        "add_detail.safetensors": "https://civitai.com/api/download/models/122359",
        "film-grain.safetensors": "https://civitai.com/api/download/models/132890",
        "PerfectEyes.safetensors": "https://civitai.com/api/download/models/132203",
        "cinematic-lighting.safetensors": "https://civitai.com/api/download/models/127623",
        "lowra.safetensors": "https://civitai.com/api/download/models/16576",
        "v2_lora_ZoomIn.safetensors": "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_ZoomIn.safetensors",
        "v2_lora_PanLeft.safetensors": "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_PanLeft.safetensors",
        "v2_lora_PanRight.safetensors": "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_PanRight.safetensors",
        "4k_Hyperrealistic_Style.safetensors": "https://civitai.com/api/download/models/202684",
        "Cinematic_V2.safetensors": "https://civitai.com/api/download/models/193477",
        "Vintage_Style_Lora.safetensors": "https://civitai.com/api/download/models/132535",
        "Cyberpunk_Style.safetensors": "https://civitai.com/api/download/models/132956",
        "Dreamy_and_ethereal_style.safetensors": "https://civitai.com/api/download/models/126987",
        "oil_painting_style.safetensors": "https://civitai.com/api/download/models/127117",
        "Epic_Realism.safetensors": "https://civitai.com/api/download/models/132333",
        "Arcane_Style_LoRA.safetensors": "https://civitai.com/api/download/models/13340",
        "Ghibli_Style_LoRA.safetensors": "https://civitai.com/api/download/models/131106",
        "GTA_Style_LoRA.safetensors": "https://civitai.com/api/download/models/131754",
        "Cyberpunk_Style_LoRA_v2.safetensors": "https://civitai.com/api/download/models/127435",
        "Pixel_Art_LoRA.safetensors": "https://civitai.com/api/download/models/131141",
        "Claymation_Style_LoRA.safetensors": "https://civitai.com/api/download/models/128399",
        "Lego_LoRA.safetensors": "https://civitai.com/api/download/models/124564",
        "Papercut_LoRA.safetensors": "https://civitai.com/api/download/models/122116",
        "Watercolor_Paint_Style.safetensors": "https://civitai.com/api/download/models/127670",
        "Steampunk_LoRA.safetensors": "https://civitai.com/api/download/models/124789",
        "Epi_Noiseoffset.safetensors": "https://civitai.com/api/download/models/13941",
        "Hyper_Detailer_LoRA.safetensors": "https://civitai.com/api/download/models/127732",
        "Better_Hands_LoRA.safetensors": "https://civitai.com/api/download/models/4286",
        "Soft_Lighting_LoRA.safetensors": "https://civitai.com/api/download/models/127622",
        "Ultra_Realistic_LoRA.safetensors": "https://civitai.com/api/download/models/129631",
        "Bokeh_LoRA.safetensors": "https://civitai.com/api/download/models/128381",
        "Lens_Flare_LoRA.safetensors": "https://civitai.com/api/download/models/127815",
        "v2_lora_TiltUp.safetensors": "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_TiltUp.safetensors",
        "v2_lora_TiltDown.safetensors": "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_TiltDown.safetensors",
        "v2_lora_PanUp.safetensors": "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_PanUp.safetensors",
        "v2_lora_PanDown.safetensors": "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_PanDown.safetensors",
        "v2_lora_RollingAnticlockwise.safetensors": "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_RollingAnticlockwise.safetensors",
        "v2_lora_RollingClockwise.safetensors": "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_RollingClockwise.safetensors",
        "Fisheye_Lens_LoRA.safetensors": "https://civitai.com/api/download/models/127807",
        "Fire_LoRA.safetensors": "https://civitai.com/api/download/models/127471",
        "Rain_LoRA.safetensors": "https://civitai.com/api/download/models/124785",
        "Smoke_LoRA.safetensors": "https://civitai.com/api/download/models/124786",
        "Explosion_LoRA.safetensors": "https://civitai.com/api/download/models/125539",
        "Electrical_Magic_LoRA.safetensors": "https://civitai.com/api/download/models/126980",
        "Sci-Fi_Robots_LoRA.safetensors": "https://civitai.com/api/download/models/122046",
        "Moody_LoRA.safetensors": "https://civitai.com/api/download/models/129532",
        "Ethereal_LoRA.safetensors": "https://civitai.com/api/download/models/126986",
        "Horror_Style_LoRA.safetensors": "https://civitai.com/api/download/models/127533",
        "80s_Aesthetic_LoRA.safetensors": "https://civitai.com/api/download/models/128362",
        "Lofi_Anime_Aesthetic.safetensors": "https://civitai.com/api/download/models/99773",
    }
}

REMOTE_BASE_PATH = Path("/app")
COMFYUI_PATH = REMOTE_BASE_PATH / "ComfyUI"
MODEL_PATH = COMFYUI_PATH / "models"

volume = Volume.from_name("comfyui-models-massive-volume", create_if_missing=True)

comfy_image = (
    Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", setup_environment={"NVIDIA_DRIVER_CAPABILITIES": "all"})
    .apt_install("git", "wget", "libgl1", "libglib2.0-0")
    .run_commands(
        f"git clone https://github.com/comfyanonymous/ComfyUI.git {COMFYUI_PATH}",
        "cd /app/ComfyUI && pip install -r requirements.txt",
        f"cd {COMFYUI_PATH / 'custom_nodes'} && git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git",
        f"cd {COMFYUI_PATH / 'custom_nodes'} && git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
        f"cd {COMFYUI_PATH / 'custom_nodes'} && git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git",
    )
)

@app.function(
    image=comfy_image,
    volumes={str(MODEL_PATH): volume},
    timeout=7200,
)
def download_models():
    for model_type, models in MODEL_REGISTRY.items():
        for filename, url in models.items():
            target_dir = MODEL_PATH / model_type
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / filename
            if not target_path.exists():
                print(f"Downloading {model_type}/{filename}...")
                try:
                    subprocess.run(["wget", "-O", str(target_path), url], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"ERROR: Failed to download {filename}: {e}")
            else:
                print(f"Exists: {filename}")
    print("--- Committing volume... ---")
    volume.commit()
    print("--- ✅ Volume committed. ---")

@app.cls(
    image=comfy_image,
    gpu=gpu.L40S(),
    volumes={str(MODEL_PATH): volume},
    concurrency_limit=100,
    allow_concurrent_inputs=15,
    timeout=3600,
    scaledown_window=120,
)
class ComfyUI:
    @enter()
    def startup(self):
        cmd = "python main.py --listen 0.0.0.0 --port 8188 --disable-auto-launch"
        self.proc = subprocess.Popen(cmd, shell=True, cwd=COMFYUI_PATH)
        for i in range(20):
            try:
                urllib.request.urlopen("http://127.0.0.1:8188/queue").read()
                print("--- ✅ ComfyUI server is active. ---")
                return
            except Exception:
                time.sleep(1)
        raise RuntimeError("ComfyUI server failed to start.")

    def _queue_prompt(self, client_id: str, prompt_workflow: dict):
        req = urllib.request.Request(
            "http://127.0.0.1:8188/prompt",
            data=json.dumps({"prompt": prompt_workflow, "client_id": client_id}).encode('utf-8')
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
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing' and message['data']['node'] is None and message['data']['prompt_id'] == prompt_id:
                    break
        ws.close()
        history = self._get_history(prompt_id)[prompt_id]
        for node_id, node_output in history['outputs'].items():
            if 'gifs' in node_output:
                for gif in node_output['gifs']:
                    return self._get_file(gif['filename'], gif['subfolder'], gif['type'])
        raise ValueError("Video/GIF output not found in execution results.")

    def get_t2v_workflow(self, payload: dict) -> Dict:
        prompt = payload.get("prompt", "a beautiful landscape")
        negative_prompt = payload.get("negative_prompt", "ugly, blurry")
        seed = payload.get("seed", 123)
        steps = payload.get("steps", 25)
        width = payload.get("width", 512)
        height = payload.get("height", 512)
        frames = payload.get("frames", 24)
        lora_name = payload.get("lora_name", "None")
        
        workflow_template = """
        {{
          "3": {{ "class_type": "KSampler", "inputs": {{ "seed": {seed}, "steps": {steps}, "cfg": 8, "sampler_name": "euler", "scheduler": "normal", "denoise": 1, "model": ["17", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0] }} }},
          "5": {{ "class_type": "EmptyLatentImage", "inputs": {{ "width": {width}, "height": {height}, "batch_size": 1 }} }},
          "6": {{ "class_type": "CLIPTextEncode", "inputs": {{ "text": "{prompt}", "clip": ["17", 1] }} }},
          "7": {{ "class_type": "CLIPTextEncode", "inputs": {{ "text": "{negative_prompt}", "clip": ["17", 1] }} }},
          "9": {{ "class_type": "VHS_VideoCombine", "inputs": {{ "frame_rate": 8, "loop_count": 0, "filename_prefix": "ComfyUI_Video", "format": "image/gif", "pingpong": false, "save_image": true, "images": ["16", 0] }} }},
          "10": {{ "class_type": "ADE_AnimateDiffLoader", "inputs": {{ "model_name": "mm_sd_v15_v2.ckpt" }} }},
          "13": {{ "class_type": "CheckpointLoaderSimple", "inputs": {{ "ckpt_name": "realisticVisionV60_v60B1.safetensors" }} }},
          "15": {{ "class_type": "ADE_ApplyAnimateDiff", "inputs": {{ "frame_limit": {frames}, "model": ["13", 0], "clip": ["13", 1], "ad_model": ["10", 0] }} }},
          "16": {{ "class_type": "VAEDecode", "inputs": {{ "samples": ["3", 0], "vae": ["13", 2] }} }},
          "17": {{ "class_type": "LoraLoader", "inputs": {{ "lora_name": "{lora_name}", "strength_model": 0.8, "strength_clip": 0.8, "model": ["15", 0], "clip": ["15", 1] }} }}
        }}
        """
        
        prompt_sanitized = json.dumps(prompt)[1:-1]
        negative_prompt_sanitized = json.dumps(negative_prompt)[1:-1]

        formatted_workflow = workflow_template.format(
            seed=seed, steps=steps, width=width, height=height, 
            prompt=prompt_sanitized, negative_prompt=negative_prompt_sanitized, 
            frames=frames, lora_name=lora_name
        )
        return json.loads(formatted_workflow)

    @method()
    def generate_video(self, payload: Dict):
        if not payload.get("prompt"):
            raise ValueError("'prompt' parameter is required.")
        
        workflow = self.get_t2v_workflow(payload)
        
        client_id = str(uuid.uuid4())
        prompt_id = self._queue_prompt(client_id, workflow)
        video_data = self._get_video_from_websocket(prompt_id, client_id)
        return video_data

@app.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, Request
    from fastapi.responses import Response, JSONResponse

    web_app = FastAPI()
    comfy_runner = ComfyUI()

    @web_app.post("/generate")
    async def generate(request: Request):
        try:
            payload = await request.json()
            video_bytes = comfy_runner.generate_video.remote(payload)
            return Response(content=video_bytes, media_type="image/gif")
        except (ValueError, NotImplementedError) as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        except Exception as e:
            print(f"Internal server error: {e}")
            return JSONResponse(status_code=500, content={"error": "An unexpected internal server error occurred."})

    return web_app

@app.local_entrypoint()
def cli():
    print("--- ComfyUI Video Generation API CLI ---")
    print("\n1. Download all models (run once or to update):")
    print("   modal run app.py::download_models")
    print("\n2. Deploy the API server:")
    print("   modal deploy app.py")
