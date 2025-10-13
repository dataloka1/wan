# app.py
# Single-file, production-grade ComfyUI Video Generation API on Modal.com

import base64
import json
import subprocess
import time
import urllib.request
import urllib.parse
import uuid
from pathlib import Path
from typing import Dict, Optional

from modal import (
    App,
    Image,
    Mount,
    Volume,
    asgi_app,
    cls,
    enter,
    gpu,
    method,
    secret,
)

# --- Konfigurasi Aplikasi Modal ---
# Menggunakan sintaks App terbaru dengan konfigurasi skalabilitas
app = App(
    "comfyui-video-prod-api",
    # Setelah permintaan terakhir, tunggu 120 detik sebelum mematikan kontainer.
    # Ini efisien untuk menangani lalu lintas yang bergelombang.
    scaledown_window=120,
)

# --- Konfigurasi Model ---
# Ganti dengan model, LoRA, dan ControlNet yang Anda inginkan.
MODEL_REGISTRY = {
    # Checkpoint utama. SVD digunakan untuk i2v.
    # Untuk t2v, Anda mungkin ingin model lain seperti RealisticVision.
    "checkpoints": {
        "svd_xt.safetensors": "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt.safetensors",
        "realisticVisionV60_v60B1.safetensors": "https://huggingface.co/segmind/Realistic-Vision-V6.0-B1/resolve/main/Realistic_Vision_V6.0_B1_noVAE.safetensors"
    },
    "loras": {
        "v2_lora_ZoomIn.safetensors": "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_ZoomIn.safetensors"
    },
    "animatediff_models": {
        "mm_sd_v15_v2.ckpt": "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt"
    },
    "vae": {
        "vae-ft-mse-840000-ema-pruned.safetensors": "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors"
    }
}

# --- Path di dalam Kontainer ---
REMOTE_BASE_PATH = Path("/app")
COMFYUI_PATH = REMOTE_BASE_PATH / "ComfyUI"
MODEL_PATH = COMFYUI_PATH / "models"

# --- Definisi Volume & Image Docker ---
# Volume untuk menyimpan model secara persisten
volume = Volume.from_name("comfyui-models-prod-volume", create_if_missing=True)

# Image Docker yang berisi ComfyUI dan semua dependensinya
comfy_image = (
    Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", setup_environment={"NVIDIA_DRIVER_CAPABILITIES": "all"})
    .apt_install("git", "wget", "libgl1", "libglib2.0-0")
    .run_commands(
        "echo 'Cloning ComfyUI repository...'",
        f"git clone https://github.com/comfyanonymous/ComfyUI.git {COMFYUI_PATH}",
        "cd /app/ComfyUI && pip install -r requirements.txt",
        "echo 'Installing custom nodes for video generation...'",
        f"cd {COMFYUI_PATH / 'custom_nodes'} && git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git",
        f"cd {COMFYUI_PATH / 'custom_nodes'} && git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
        "echo 'Image setup complete.'",
    )
)

# --- Fungsi Terpisah untuk Mengunduh Model ---
@app.function(
    image=comfy_image,
    volumes={str(MODEL_PATH): volume},
    timeout=2400,  # Timeout lebih panjang untuk download besar
)
def download_models():
    """
    Jalankan fungsi ini secara manual untuk mengunduh dan menyimpan semua model
    ke dalam volume persisten.
    Perintah: modal run app.py::download_models
    """
    print("--- Memulai proses pengunduhan model ---")
    for model_type, models in MODEL_REGISTRY.items():
        for filename, url in models.items():
            target_dir = MODEL_PATH / model_type
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / filename
            if not target_path.exists():
                print(f"Mengunduh {filename} dari {url}...")
                try:
                    subprocess.run(["wget", "-O", str(target_path), url], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"ERROR: Gagal mengunduh {filename}: {e}")
            else:
                print(f"Model {filename} sudah ada. Melewati.")
    
    print("--- Selesai mengunduh model. Melakukan commit pada volume... ---")
    volume.commit()
    print("--- ✅ Volume telah disimpan. Anda sekarang bisa mendeploy aplikasi. ---")

# --- Kelas Utama untuk Server API ComfyUI ---
@app.cls(
    image=comfy_image,
    gpu=gpu.L40S(), # GPU yang sangat powerful untuk video
    volumes={str(MODEL_PATH): volume},
    # Izinkan hingga 100 kontainer berjalan secara bersamaan
    concurrency_limit=10,
    # Izinkan hingga 15 permintaan diproses oleh satu kontainer
    allow_concurrent_inputs=15,
    # Timeout yang sangat panjang untuk generasi video 30 detik+
    timeout=3600,
    container_idle_timeout=300 # Matikan kontainer setelah 5 menit tidak aktif
)
class ComfyUI:
    @enter()
    def startup(self):
        """
        Metode ini dipanggil saat kontainer dimulai.
        Server ComfyUI dijalankan di latar belakang.
        """
        print("--- Memulai server ComfyUI di latar belakang ---")
        cmd = "python main.py --listen 0.0.0.0 --port 8188 --disable-auto-launch"
        self.proc = subprocess.Popen(cmd, shell=True, cwd=COMFYUI_PATH)
        
        print("Menunggu server ComfyUI siap...")
        # Lakukan health check sederhana
        for i in range(20):
            try:
                urllib.request.urlopen("http://127.0.0.1:8188/queue").read()
                print("--- ✅ Server ComfyUI aktif dan berjalan. ---")
                return
            except Exception:
                time.sleep(1)
        raise RuntimeError("Server ComfyUI gagal dimulai dalam 20 detik.")

    # ... metode helper untuk berkomunikasi dengan ComfyUI via websocket ...
    def _queue_prompt(self, client_id: str, prompt_workflow: dict):
        req = urllib.request.Request(
            "http://127.0.0.1:8188/prompt",
            data=json.dumps({"prompt": prompt_workflow, "client_id": client_id}).encode('utf-8')
        )
        response = json.loads(urllib.request.urlopen(req).read())
        return response['prompt_id']

    def _get_history(self, prompt_id: str):
        with urllib.request.urlopen(f"http://127.0.0.1:8188/history/{prompt_id}") as response:
            return json.loads(response.read())

    def _get_file(self, filename: str, subfolder: str, folder_type: str):
        url = f"http://127.0.0.1:8188/view?{urllib.parse.urlencode({'filename': filename, 'subfolder': subfolder, 'type': folder_type})}"
        with urllib.request.urlopen(url) as response:
            return response.read()

    def _get_video_from_websocket(self, prompt_id: str, client_id: str):
        import websocket # Impor di sini agar tidak perlu diinstal di image global
        
        ws_url = f"ws://127.0.0.1:8188/ws?clientId={client_id}"
        ws = websocket.WebSocket()
        ws.connect(ws_url)
        
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing' and message['data']['node'] is None and message['data']['prompt_id'] == prompt_id:
                    break # Eksekusi selesai
            # Abaikan pesan biner
        
        ws.close()
        history = self._get_history(prompt_id)[prompt_id]
        
        for node_id, node_output in history['outputs'].items():
            if 'gifs' in node_output: # AnimateDiff biasanya output sebagai GIF
                for gif in node_output['gifs']:
                    return self._get_file(gif['filename'], gif['subfolder'], gif['type'])
        
        raise ValueError("Tidak ditemukan output video/gif dalam hasil eksekusi.")

    # --- Endpoint API Publik ---
    @method()
    def generate_video(self, payload: Dict):
        """
        Menjalankan alur kerja ComfyUI berdasarkan payload JSON.
        Secara dinamis memilih workflow T2V atau I2V.
        """
        prompt_text = payload.get("prompt")
        input_image_b64 = payload.get("image_base64")

        if not prompt_text:
            raise ValueError("Parameter 'prompt' wajib diisi.")

        if input_image_b64:
            # TODO: Logika untuk I2V (Image-to-Video)
            # 1. Decode base64 dan simpan gambar ke folder input ComfyUI
            # 2. Gunakan workflow_i2v
            raise NotImplementedError("Workflow I2V belum diimplementasikan. Silakan tambahkan workflow JSON Anda.")
        else:
            # Logika untuk T2V (Text-to-Video)
            print("Memulai job Text-to-Video...")
            workflow = self.get_t2v_workflow(
                prompt=prompt_text,
                negative_prompt=payload.get("negative_prompt", "ugly, blurry"),
                seed=payload.get("seed", 123),
                steps=payload.get("steps", 25),
                width=payload.get("width", 512),
                height=payload.get("height", 512),
                frames=payload.get("frames", 24), # 24 frame ~ 1 detik
            )

        client_id = str(uuid.uuid4())
        prompt_id = self._queue_prompt(client_id, workflow)
        video_data = self._get_video_from_websocket(prompt_id, client_id)

        return video_data

    # --- Definisi Workflow ---
    def get_t2v_workflow(self, prompt: str, negative_prompt: str, seed: int, steps: int, width: int, height: int, frames: int) -> Dict:
        """
        Membangun workflow JSON untuk Text-to-Video menggunakan AnimateDiff.
        Ini adalah contoh yang berfungsi.
        """
        # Anda dapat membuat ini di UI Comfy dan mengekspornya dengan "Save (API Format)"
        # lalu menerjemahkannya ke dalam fungsi ini.
        return json.loads(f"""
{{
  "3": {{
    "inputs": {{
      "seed": {seed},
      "steps": {steps},
      "cfg": 8,
      "sampler_name": "euler",
      "scheduler": "normal",
      "denoise": 1,
      "model": ["15", 0],
      "positive": ["6", 0],
      "negative": ["7", 0],
      "latent_image": ["5", 0]
    }},
    "class_type": "KSampler"
  }},
  "5": {{
    "inputs": {{
      "width": {width},
      "height": {height},
      "batch_size": 1
    }},
    "class_type": "EmptyLatentImage"
  }},
  "6": {{
    "inputs": {{
      "text": "{prompt}",
      "clip": ["15", 1]
    }},
    "class_type": "CLIPTextEncode"
  }},
  "7": {{
    "inputs": {{
      "text": "{negative_prompt}",
      "clip": ["15", 1]
    }},
    "class_type": "CLIPTextEncode"
  }},
  "9": {{
    "inputs": {{
      "frame_rate": 8,
      "loop_count": 0,
      "filename_prefix": "ComfyUI_Video",
      "format": "image/gif",
      "pingpong": false,
      "save_image": true,
      "images": ["16", 0]
    }},
    "class_type": "VHS_VideoCombine"
  }},
  "10": {{
    "inputs": {{
      "model_name": "mm_sd_v15_v2.ckpt"
    }},
    "class_type": "ADE_AnimateDiffLoader"
  }},
  "13": {{
    "inputs": {{
      "ckpt_name": "realisticVisionV60_v60B1.safetensors"
    }},
    "class_type": "CheckpointLoaderSimple"
  }},
  "15": {{
    "inputs": {{
      "frame_limit": {frames},
      "model": ["13", 0],
      "clip": ["13", 1],
      "ad_model": ["10", 0]
    }},
    "class_type": "ADE_ApplyAnimateDiff"
  }},
  "16": {{
    "inputs": {{
      "samples": ["3", 0],
      "vae": ["13", 2]
    }},
    "class_type": "VAEDecode"
  }}
}}
""")

# --- Menjalankan API dengan FastAPI ---
# Ini membuat endpoint web yang memanggil metode di dalam kelas ComfyUI.
@app.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, Request
    from fastapi.responses import Response

    web_app = FastAPI()
    comfy_runner = ComfyUI()

    @web_app.post("/generate")
    async def generate(request: Request):
        payload = await request.json()
        try:
            # Panggil metode kelas secara sinkron di dalam endpoint async
            video_bytes = comfy_runner.generate_video.remote(payload)
            # Hasilnya akan berupa bytes dari file GIF/video
            return Response(content=video_bytes, media_type="image/gif")
        except (ValueError, NotImplementedError) as e:
            return Response(content=json.dumps({"error": str(e)}), status_code=400, media_type="application/json")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return Response(content=json.dumps({"error": "Internal server error"}), status_code=500, media_type="application/json")

    return web_app

@app.local_entrypoint()
def cli():
    print("--- ComfyUI Video Generation API CLI ---")
    print("Gunakan perintah berikut untuk mengelola aplikasi ini:")
    print("\n1. Mengunduh semua model (jalankan sekali atau untuk update):")
    print("   modal run app.py::download_models")
    print("\n2. Mendeploy API server ke cloud Modal:")
    print("   modal deploy app.py")
    print("\n   Setelah deploy, Anda akan mendapatkan URL publik.")
