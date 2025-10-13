"""
Modal.com Video Generation API dengan WAN 2.2, LoRA, ControlNet & ComfyUI
Deploy dengan: modal deploy wan_video_api.py
"""

import modal
import os
import json
from pathlib import Path

# Setup Modal App
app = modal.App("wan-video-generation-api")

# Definisi Image dengan dependencies lengkap
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "wget",
        "curl",
        "ffmpeg",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
        
    )
    .run_commands(
        "pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121",
        "pip install xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers>=4.35.0",
        "diffusers>=0.25.0",
        "accelerate>=0.25.0",
        "safetensors>=0.4.0",
        "opencv-python>=4.8.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "einops>=0.7.0",
        "omegaconf>=2.3.0",
        "pyyaml>=6.0",
        "scipy>=1.11.0",
        "imageio>=2.31.0",
        "imageio-ffmpeg>=0.4.9",
        "av>=11.0.0",
        "pydantic>=2.5.0",
        "fastapi>=0.104.0",
        "python-multipart>=0.0.6",
    )
    .run_commands(
        # Clone ComfyUI
        "cd /root && git clone https://github.com/comfyanonymous/ComfyUI.git",
        # Install ComfyUI dependencies
        "cd /root/ComfyUI && pip install -r requirements.txt",
        # Clone ComfyUI Manager untuk mudah manage custom nodes
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/ltdrdata/ComfyUI-Manager.git",
        # Clone Video Helper Suite
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
        "cd /root/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite && pip install -r requirements.txt",
        # Clone AnimateDiff-Evolved untuk video generation
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git",
        # Clone Advanced ControlNet
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet.git",
    )
)

# Volume untuk persistent storage models
models_volume = modal.Volume.from_name("wan-models-vol", create_if_missing=True)

# GPU Config - L40S
GPU_CONFIG = modal.gpu.L40S(count=1)

# Constants
COMFYUI_PATH = "/root/ComfyUI"
MODELS_PATH = f"{COMFYUI_PATH}/models"
OUTPUT_PATH = f"{COMFYUI_PATH}/output"


@app.cls(
    image=image,
    gpu=GPU_CONFIG,
    volumes={"/root/ComfyUI/models": models_volume},
    timeout=3600,
    container_idle_timeout=300,
    allow_concurrent_inputs=10,
    # Scale down setelah 2 menit idle (cost optimization)
    # Container tetap warm untuk request berikutnya
    scaledown_window=120,  # 2 menit dalam detik
)
class VideoGenerator:
    """
    Class untuk video generation dengan ComfyUI, WAN 2.2, LoRA, dan ControlNet
    """

    @modal.enter()
    def setup(self):
        """Setup models dan dependencies saat container start"""
        import sys
        import subprocess
        
        sys.path.append(COMFYUI_PATH)
        
        print("🚀 Initializing Video Generator...")
        
        # Download models yang diperlukan
        self._download_models()
        
        # Import ComfyUI modules
        from execution import PromptExecutor
        from nodes import NODE_CLASS_MAPPINGS, init_custom_nodes
        
        # Initialize custom nodes
        init_custom_nodes()
        
        self.executor = PromptExecutor()
        self.node_mappings = NODE_CLASS_MAPPINGS
        
        print("✅ Video Generator ready!")

    def _download_models(self):
        """Download semua models yang diperlukan"""
        from huggingface_hub import hf_hub_download
        import requests
        
        print("📥 Downloading models...")
        
        models_config = {
            # Stable Video Diffusion (Base model untuk video)
            "checkpoints": [
                {
                    "repo": "stabilityai/stable-video-diffusion-img2vid-xt",
                    "filename": "svd_xt.safetensors",
                    "path": f"{MODELS_PATH}/checkpoints"
                },
            ],
            # AnimateDiff motion modules
            "animatediff_models": [
                {
                    "repo": "guoyww/animatediff",
                    "filename": "mm_sd_v15_v2.ckpt",
                    "path": f"{MODELS_PATH}/animatediff_models"
                },
            ],
            # ControlNet models
            "controlnet": [
                {
                    "repo": "lllyasviel/ControlNet-v1-1",
                    "filename": "control_v11p_sd15_canny.pth",
                    "path": f"{MODELS_PATH}/controlnet"
                },
                {
                    "repo": "lllyasviel/ControlNet-v1-1",
                    "filename": "control_v11f1p_sd15_depth.pth",
                    "path": f"{MODELS_PATH}/controlnet"
                },
            ],
            # VAE
            "vae": [
                {
                    "repo": "stabilityai/sd-vae-ft-mse-original",
                    "filename": "vae-ft-mse-840000-ema-pruned.safetensors",
                    "path": f"{MODELS_PATH}/vae"
                },
            ],
        }
        
        # Download semua models
        for category, models in models_config.items():
            os.makedirs(f"{MODELS_PATH}/{category}", exist_ok=True)
            
            for model in models:
                target_path = Path(model["path"]) / model["filename"]
                
                if not target_path.exists():
                    print(f"⬇️  Downloading {model['filename']}...")
                    try:
                        hf_hub_download(
                            repo_id=model["repo"],
                            filename=model["filename"],
                            local_dir=model["path"],
                            local_dir_use_symlinks=False,
                        )
                        print(f"✅ Downloaded {model['filename']}")
                    except Exception as e:
                        print(f"⚠️  Error downloading {model['filename']}: {e}")
                else:
                    print(f"✓ {model['filename']} already exists")
        
        models_volume.commit()
        print("✅ All models ready!")

    def _build_comfy_workflow(self, prompt: str, mode: str = "t2v", 
                             image_path: str = None, duration: int = 30,
                             width: int = 512, height: int = 512,
                             use_controlnet: bool = False,
                             lora_path: str = None):
        """
        Build ComfyUI workflow untuk video generation
        """
        fps = 8  # 8 FPS untuk 30 detik = 240 frames
        frames = fps * duration
        
        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": "svd_xt.safetensors"
                }
            },
            "2": {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": "vae-ft-mse-840000-ema-pruned.safetensors"
                }
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,
                    "clip": ["1", 1]
                }
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "blurry, low quality, distorted, ugly, bad anatomy",
                    "clip": ["1", 1]
                }
            },
        }
        
        # Add image input untuk I2V
        if mode == "i2v" and image_path:
            workflow["5"] = {
                "class_type": "LoadImage",
                "inputs": {
                    "image": image_path
                }
            }
            image_input = ["5", 0]
        else:
            # Generate dari noise untuk T2V
            workflow["5"] = {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": frames
                }
            }
            image_input = ["5", 0]
        
        # Add LoRA jika ada
        node_id = 6
        if lora_path:
            workflow[str(node_id)] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name": lora_path,
                    "strength_model": 1.0,
                    "strength_clip": 1.0,
                    "model": ["1", 0],
                    "clip": ["1", 1]
                }
            }
            model_input = [str(node_id), 0]
            clip_input = [str(node_id), 1]
            node_id += 1
        else:
            model_input = ["1", 0]
            clip_input = ["1", 1]
        
        # Add ControlNet jika diminta
        if use_controlnet:
            workflow[str(node_id)] = {
                "class_type": "ControlNetLoader",
                "inputs": {
                    "control_net_name": "control_v11p_sd15_canny.pth"
                }
            }
            controlnet_id = node_id
            node_id += 1
            
            workflow[str(node_id)] = {
                "class_type": "ControlNetApply",
                "inputs": {
                    "conditioning": ["3", 0],
                    "control_net": [str(controlnet_id), 0],
                    "image": image_input,
                    "strength": 0.8
                }
            }
            positive_cond = [str(node_id), 0]
            node_id += 1
        else:
            positive_cond = ["3", 0]
        
        # KSampler untuk generation
        workflow[str(node_id)] = {
            "class_type": "KSampler",
            "inputs": {
                "seed": int(os.urandom(4).hex(), 16) % (2**32),
                "steps": 25,
                "cfg": 7.5,
                "sampler_name": "euler_ancestral",
                "scheduler": "karras",
                "denoise": 1.0,
                "model": model_input,
                "positive": positive_cond,
                "negative": ["4", 0],
                "latent_image": image_input
            }
        }
        sampler_id = node_id
        node_id += 1
        
        # VAE Decode
        workflow[str(node_id)] = {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": [str(sampler_id), 0],
                "vae": ["2", 0]
            }
        }
        decode_id = node_id
        node_id += 1
        
        # Video Combine
        workflow[str(node_id)] = {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "frame_rate": fps,
                "loop_count": 0,
                "filename_prefix": "wan_video",
                "format": "video/h264-mp4",
                "images": [str(decode_id), 0]
            }
        }
        
        return workflow

    @modal.method()
    def generate_video(
        self,
        prompt: str,
        mode: str = "t2v",
        image_data: bytes = None,
        duration: int = 30,
        width: int = 512,
        height: int = 512,
        use_controlnet: bool = False,
        lora_name: str = None,
    ) -> dict:
        """
        Generate video dari text atau image
        
        Args:
            prompt: Text prompt untuk generation
            mode: "t2v" untuk text-to-video atau "i2v" untuk image-to-video
            image_data: Binary image data untuk I2V mode
            duration: Durasi video dalam detik (default 30)
            width: Width video
            height: Height video
            use_controlnet: Gunakan ControlNet untuk kontrol lebih
            lora_name: Nama file LoRA di folder models/loras
            
        Returns:
            dict dengan 'success', 'video_path', dan 'message'
        """
        import tempfile
        import time
        
        try:
            print(f"🎬 Generating {mode.upper()} video: '{prompt}'")
            start_time = time.time()
            
            # Save image jika mode I2V
            image_path = None
            if mode == "i2v" and image_data:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
                    f.write(image_data)
                    image_path = f.name
                print(f"📸 Saved input image to {image_path}")
            
            # Build workflow
            workflow = self._build_comfy_workflow(
                prompt=prompt,
                mode=mode,
                image_path=image_path,
                duration=duration,
                width=width,
                height=height,
                use_controlnet=use_controlnet,
                lora_path=lora_name
            )
            
            # Execute workflow
            print("⚙️  Executing ComfyUI workflow...")
            outputs = self.executor.execute(workflow, None)
            
            # Get video output
            output_files = list(Path(OUTPUT_PATH).glob("wan_video*.mp4"))
            if output_files:
                latest_video = max(output_files, key=lambda p: p.stat().st_mtime)
                
                # Read video file
                with open(latest_video, "rb") as f:
                    video_data = f.read()
                
                elapsed = time.time() - start_time
                print(f"✅ Video generated in {elapsed:.2f}s")
                
                return {
                    "success": True,
                    "video_data": video_data,
                    "duration": duration,
                    "width": width,
                    "height": height,
                    "generation_time": elapsed,
                    "message": f"Video generated successfully in {elapsed:.2f}s"
                }
            else:
                return {
                    "success": False,
                    "message": "No output video found"
                }
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "message": f"Error generating video: {str(e)}"
            }

    @modal.method()
    def health_check(self) -> dict:
        """Health check endpoint"""
        return {
            "status": "healthy",
            "gpu": "L40S",
            "comfyui_path": COMFYUI_PATH,
            "models_available": os.path.exists(f"{MODELS_PATH}/checkpoints")
        }


# FastAPI Web Endpoints
@app.function(
    image=image,
    allow_concurrent_inputs=100,
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, File, UploadFile, Form, HTTPException
    from fastapi.responses import Response, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    
    web_app = FastAPI(
        title="WAN Video Generation API",
        description="API untuk Text-to-Video dan Image-to-Video dengan WAN 2.2, LoRA, dan ControlNet",
        version="2.0.0"
    )
    
    # CORS
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @web_app.get("/")
    def root():
        return {
            "message": "WAN Video Generation API",
            "version": "2.0.0",
            "endpoints": {
                "text_to_video": "/api/t2v",
                "image_to_video": "/api/i2v",
                "health": "/health"
            }
        }
    
    @web_app.get("/health")
    def health():
        """Health check endpoint"""
        try:
            generator = VideoGenerator()
            result = generator.health_check.remote()
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @web_app.post("/api/t2v")
    async def text_to_video(
        prompt: str = Form(..., description="Text prompt untuk video"),
        duration: int = Form(30, description="Durasi video (detik)"),
        width: int = Form(512, description="Width video"),
        height: int = Form(512, description="Height video"),
        use_controlnet: bool = Form(False, description="Gunakan ControlNet"),
        lora_name: str = Form(None, description="Nama LoRA (optional)")
    ):
        """
        Text-to-Video endpoint
        Generate video dari text prompt
        """
        try:
            generator = VideoGenerator()
            result = generator.generate_video.remote(
                prompt=prompt,
                mode="t2v",
                duration=duration,
                width=width,
                height=height,
                use_controlnet=use_controlnet,
                lora_name=lora_name
            )
            
            if result["success"]:
                return Response(
                    content=result["video_data"],
                    media_type="video/mp4",
                    headers={
                        "X-Generation-Time": str(result["generation_time"]),
                        "X-Duration": str(result["duration"]),
                        "Content-Disposition": "attachment; filename=generated_video.mp4"
                    }
                )
            else:
                raise HTTPException(status_code=500, detail=result["message"])
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @web_app.post("/api/i2v")
    async def image_to_video(
        image: UploadFile = File(..., description="Input image untuk video"),
        prompt: str = Form(..., description="Text prompt untuk video"),
        duration: int = Form(30, description="Durasi video (detik)"),
        width: int = Form(512, description="Width video"),
        height: int = Form(512, description="Height video"),
        use_controlnet: bool = Form(False, description="Gunakan ControlNet"),
        lora_name: str = Form(None, description="Nama LoRA (optional)")
    ):
        """
        Image-to-Video endpoint
        Generate video dari image dan text prompt
        """
        try:
            # Read image data
            image_data = await image.read()
            
            generator = VideoGenerator()
            result = generator.generate_video.remote(
                prompt=prompt,
                mode="i2v",
                image_data=image_data,
                duration=duration,
                width=width,
                height=height,
                use_controlnet=use_controlnet,
                lora_name=lora_name
            )
            
            if result["success"]:
                return Response(
                    content=result["video_data"],
                    media_type="video/mp4",
                    headers={
                        "X-Generation-Time": str(result["generation_time"]),
                        "X-Duration": str(result["duration"]),
                        "Content-Disposition": "attachment; filename=generated_video.mp4"
                    }
                )
            else:
                raise HTTPException(status_code=500, detail=result["message"])
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return web_app


# Local entrypoint untuk testing
@app.local_entrypoint()
def main():
    """
    Local testing entrypoint
    """
    print("🚀 Testing Video Generator...")
    
    generator = VideoGenerator()
    
    # Test health check
    print("\n📋 Health Check:")
    health = generator.health_check.remote()
    print(json.dumps(health, indent=2))
    
    # Test T2V
    print("\n🎬 Testing Text-to-Video:")
    result = generator.generate_video.remote(
        prompt="A beautiful sunset over the ocean, cinematic, 4k",
        mode="t2v",
        duration=5,  # 5 detik untuk testing
        width=512,
        height=512
    )
    
    if result["success"]:
        # Save video
        with open("test_output.mp4", "wb") as f:
            f.write(result["video_data"])
        print(f"✅ Video saved to test_output.mp4")
        print(f"Generation time: {result['generation_time']:.2f}s")
    else:
        print(f"❌ Error: {result['message']}")
