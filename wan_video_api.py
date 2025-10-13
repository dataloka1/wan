"""
Modal.com Video Generation API - PRODUCTION READY
Commands:
  - modal run wan_video_api.py::download_all_models  # Download models to storage
  - modal run wan_video_api.py::test_generation      # Test video generation
  - modal deploy wan_video_api.py                    # Deploy API

Models are stored directly in persistent volume (no symlinks)
"""

import modal
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Setup Modal App
app = modal.App("wan-video-generation-api")

# Base image dengan dependencies
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
    # --- PERBAIKAN: Gunakan .run_commands() untuk instalasi dengan index URL ---
    .run_commands(
        "pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121",
        "pip install xformers==0.0.24 --index-url https://download.pytorch.org/whl/cu121"
    )
    # ------------------------------------------------------------------------
    .pip_install(
        "transformers>=4.37.0",
        "diffusers>=0.26.0",
        "accelerate>=0.26.0",
        "safetensors>=0.4.2",
        "opencv-python>=4.9.0",
        "pillow>=10.2.0",
        "numpy>=1.26.0,<2.0.0",
        "einops>=0.7.0",
        "omegaconf>=2.3.0",
        "pyyaml>=6.0",
        "scipy>=1.12.0",
        "imageio>=2.33.0",
        "imageio-ffmpeg>=0.4.9",
        "av>=11.0.0",
        "pydantic>=2.6.0",
        "fastapi>=0.109.0",
        "python-multipart>=0.0.9",
        "huggingface-hub>=0.20.0",
    )
)

# Persistent Volume untuk models
models_volume = modal.Volume.from_name("wan-models-vol", create_if_missing=True)

# GPU Config
GPU_CONFIG = modal.gpu.L40S(count=1)

# Paths - direct storage paths (NO SYMLINKS)
MODELS_PATH = "/models"
OUTPUT_PATH = "/output"


# ============================================================================
# MODEL DOWNLOADER - Separate command to download all models
# ============================================================================

@app.function(
    image=image,
    volumes={MODELS_PATH: models_volume},
    timeout=3600,
)
def download_all_models():
    """
    Download all required models to persistent storage.
    Run with: modal run wan_video_api.py::download_all_models
    
    This only needs to be run ONCE. Models will be stored in persistent volume.
    """
    from huggingface_hub import hf_hub_download
    import shutil
    
    print("=" * 80)
    print("🚀 MODEL DOWNLOADER")
    print("=" * 80)
    print()
    print("This will download all required models to persistent storage.")
    print("You only need to run this once.")
    print()
    
    # Create directories
    checkpoint_dir = Path(MODELS_PATH) / "checkpoints"
    vae_dir = Path(MODELS_PATH) / "vae"
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    vae_dir.mkdir(parents=True, exist_ok=True)
    
    # Model configurations
    models_to_download = [
        {
            "name": "Stable Diffusion v1.5",
            "repo_id": "runwayml/stable-diffusion-v1-5",
            "filename": "v1-5-pruned-emaonly.safetensors",
            "target_dir": checkpoint_dir,
            "size": "~4.0 GB"
        },
        {
            "name": "VAE (Variational AutoEncoder)",
            "repo_id": "stabilityai/sd-vae-ft-mse-original",
            "filename": "vae-ft-mse-840000-ema-pruned.safetensors",
            "target_dir": vae_dir,
            "size": "~330 MB"
        }
    ]
    
    print("📋 Models to download:")
    for i, model in enumerate(models_to_download, 1):
        print(f"  {i}. {model['name']} ({model['size']})")
    print()
    print("-" * 80)
    print()
    
    # Download each model
    for i, model in enumerate(models_to_download, 1):
        target_path = model["target_dir"] / model["filename"]
        
        print(f"[{i}/{len(models_to_download)}] {model['name']}")
        print(f"     Repo: {model['repo_id']}")
        print(f"     File: {model['filename']}")
        print(f"     Size: {model['size']}")
        
        if target_path.exists():
            file_size = target_path.stat().st_size / (1024**3)
            print(f"     ✓ Already exists ({file_size:.2f} GB)")
            print()
            continue
        
        print(f"     ⬇️  Downloading...")
        
        try:
            # Download to temporary location first
            temp_download_dir = Path("/tmp/model_download")
            temp_download_dir.mkdir(exist_ok=True)
            
            # Download file (NO SYMLINKS - direct download)
            downloaded_file = hf_hub_download(
                repo_id=model["repo_id"],
                filename=model["filename"],
                cache_dir=temp_download_dir,
                local_dir=None,  # Don't use local_dir to avoid symlinks
                local_dir_use_symlinks=False,
            )
            
            # Move to final location
            print(f"     📦 Moving to storage...")
            shutil.move(downloaded_file, target_path)
            
            # Verify
            if target_path.exists():
                file_size = target_path.stat().st_size / (1024**3)
                print(f"     ✅ Downloaded successfully ({file_size:.2f} GB)")
            else:
                print(f"     ❌ File not found after download!")
                
        except Exception as e:
            print(f"     ❌ Error: {e}")
            print()
            continue
        
        print()
    
    print("-" * 80)
    print()
    print("📊 Verifying storage...")
    print()
    
    # Verify all files
    all_success = True
    for model in models_to_download:
        target_path = model["target_dir"] / model["filename"]
        if target_path.exists():
            size_gb = target_path.stat().st_size / (1024**3)
            print(f"  ✅ {model['name']}: {size_gb:.2f} GB")
        else:
            print(f"  ❌ {model['name']}: NOT FOUND")
            all_success = False
    
    print()
    
    # Commit to volume
    print("💾 Committing to persistent volume...")
    models_volume.commit()
    print("✅ Committed successfully!")
    print()
    
    print("=" * 80)
    if all_success:
        print("✅ ALL MODELS DOWNLOADED SUCCESSFULLY!")
        print()
        print("Next steps:")
        print("  1. Test generation: modal run wan_video_api.py::test_generation")
        print("  2. Deploy API: modal deploy wan_video_api.py")
    else:
        print("⚠️  SOME MODELS FAILED TO DOWNLOAD")
        print("Please re-run this command to retry failed downloads.")
    print("=" * 80)


# ============================================================================
# VIDEO GENERATOR CLASS
# ============================================================================

@app.cls(
    image=image,
    gpu=GPU_CONFIG,
    volumes={MODELS_PATH: models_volume},
    timeout=3600,
    allow_concurrent_inputs=10,
)
class VideoGenerator:
    """Video Generator with Stable Diffusion"""

    @modal.enter()
    def setup(self):
        """Initialize generator - models should already be in storage"""
        import torch
        
        print("🚀 Initializing Video Generator...")
        
        # Create output directory
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        
        # Verify GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU: {gpu_name}")
            print(f"   CUDA: {torch.version.cuda}")
            print(f"   PyTorch: {torch.__version__}")
        else:
            print("⚠️  No GPU detected")
        
        # Verify models exist
        checkpoint_path = Path(MODELS_PATH) / "checkpoints" / "v1-5-pruned-emaonly.safetensors"
        vae_path = Path(MODELS_PATH) / "vae" / "vae-ft-mse-840000-ema-pruned.safetensors"
        
        print()
        print("📦 Checking models in storage...")
        
        if checkpoint_path.exists():
            size_gb = checkpoint_path.stat().st_size / (1024**3)
            print(f"  ✅ Checkpoint: {size_gb:.2f} GB")
        else:
            print(f"  ❌ Checkpoint NOT FOUND!")
            print(f"     Run: modal run wan_video_api.py::download_all_models")
            raise RuntimeError("Checkpoint not found in storage!")
        
        if vae_path.exists():
            size_mb = vae_path.stat().st_size / (1024**2)
            print(f"  ✅ VAE: {size_mb:.0f} MB")
        else:
            print(f"  ⚠️  VAE NOT FOUND (will use default)")
        
        self.checkpoint_path = checkpoint_path
        self.vae_path = vae_path if vae_path.exists() else None
        self.initialized = True
        
        print()
        print("✅ Video Generator ready!")
        print()

    def _load_pipeline(self):
        """Load Stable Diffusion pipeline from storage"""
        import torch
        from diffusers import StableDiffusionPipeline
        
        print("📦 Loading pipeline from storage...")
        print(f"   Checkpoint: {self.checkpoint_path}")
        
        # Load from safetensors file directly (NO SYMLINKS)
        pipe = StableDiffusionPipeline.from_single_file(
            str(self.checkpoint_path),
            torch_dtype=torch.float16,
            safety_checker=None,
            use_safetensors=True,
            load_safety_checker=False,
        )
        
        # Load custom VAE if available
        if self.vae_path:
            print(f"   VAE: {self.vae_path}")
            from diffusers import AutoencoderKL
            vae = AutoencoderKL.from_single_file(
                str(self.vae_path),
                torch_dtype=torch.float16,
            )
            pipe.vae = vae
        
        pipe = pipe.to("cuda")
        
        # Optimizations
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        
        print("✅ Pipeline loaded!")
        
        return pipe

    def _generate_frames(
        self,
        pipe,
        prompt: str,
        num_frames: int,
        width: int,
        height: int,
    ):
        """Generate video frames"""
        import torch
        import numpy as np
        
        print(f"🎬 Generating {num_frames} frames...")
        
        frames = []
        for i in range(num_frames):
            generator = torch.Generator("cuda").manual_seed(42 + i * 7)
            
            # Add variation
            varied_prompt = f"{prompt}, frame {i+1}"
            
            with torch.inference_mode():
                image = pipe(
                    prompt=varied_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    generator=generator,
                ).images[0]
            
            frame = np.array(image, dtype=np.uint8)
            frames.append(frame)
            
            if (i + 1) % 5 == 0 or i == 0:
                print(f"  ✓ {i + 1}/{num_frames} frames")
        
        return frames

    def _encode_video(self, frames, width, height, fps) -> bytes:
        """Encode frames to MP4"""
        import cv2
        import tempfile
        
        print("🎞️  Encoding video...")
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            video_path = tmp.name
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        
        with open(video_path, 'rb') as f:
            video_data = f.read()
        
        try:
            os.unlink(video_path)
        except:
            pass
        
        return video_data

    @modal.method()
    def generate_video(
        self,
        prompt: str,
        duration: int = 10,
        width: int = 512,
        height: int = 512,
        fps: int = 8,
    ) -> Dict[str, Any]:
        """Generate video from text prompt"""
        import time
        
        try:
            # Validate
            duration = max(1, min(duration, 30))
            width = max(256, min((width // 8) * 8, 768))
            height = max(256, min((height // 8) * 8, 768))
            fps = max(4, min(fps, 12))
            
            num_frames = min(duration * fps, 48)
            actual_duration = num_frames / fps
            
            print("=" * 70)
            print("🎬 VIDEO GENERATION")
            print("=" * 70)
            print(f"Prompt: {prompt}")
            print(f"Duration: {actual_duration:.1f}s ({num_frames} frames @ {fps} fps)")
            print(f"Resolution: {width}x{height}")
            print("=" * 70)
            print()
            
            start_time = time.time()
            
            # Load pipeline
            pipe = self._load_pipeline()
            
            # Generate frames
            frames = self._generate_frames(pipe, prompt, num_frames, width, height)
            
            # Encode video
            video_data = self._encode_video(frames, width, height, fps)
            
            elapsed = time.time() - start_time
            size_mb = len(video_data) / (1024 * 1024)
            
            print()
            print("=" * 70)
            print(f"✅ SUCCESS!")
            print(f"   Time: {elapsed:.1f}s")
            print(f"   Size: {size_mb:.2f} MB")
            print(f"   Speed: {num_frames/elapsed:.1f} fps")
            print("=" * 70)
            
            return {
                "success": True,
                "video_data": video_data,
                "duration": actual_duration,
                "frames": num_frames,
                "fps": fps,
                "width": width,
                "height": height,
                "generation_time": elapsed,
                "size_mb": size_mb,
            }
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "success": False,
                "error": str(e)
            }

    @modal.method()
    def health_check(self) -> Dict[str, Any]:
        """Health check"""
        import torch
        
        checkpoint_exists = self.checkpoint_path.exists()
        
        return {
            "status": "healthy" if self.initialized else "initializing",
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "pytorch_version": torch.__version__,
            "checkpoint_ready": checkpoint_exists,
            "checkpoint_path": str(self.checkpoint_path),
        }


# ============================================================================
# FASTAPI WEB INTERFACE
# ============================================================================

@app.function(image=image, allow_concurrent_inputs=100)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, Form, HTTPException
    from fastapi.responses import Response, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    
    web_app = FastAPI(
        title="WAN Video Generation API",
        version="2.3.0",
        description="Text-to-Video generation with Stable Diffusion"
    )
    
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
            "name": "WAN Video Generation API",
            "version": "2.3.0",
            "status": "operational",
            "endpoints": {
                "health": "GET /health",
                "generate": "POST /api/generate",
                "docs": "GET /docs"
            }
        }
    
    @web_app.get("/health")
    async def health():
        try:
            generator = VideoGenerator()
            return generator.health_check.remote()
        except Exception as e:
            return JSONResponse(
                status_code=503,
                content={"status": "error", "message": str(e)}
            )
    
    @web_app.post("/api/generate")
    async def generate(
        prompt: str = Form(..., min_length=1, max_length=500),
        duration: int = Form(10, ge=1, le=30),
        width: int = Form(512, ge=256, le=768),
        height: int = Form(512, ge=256, le=768),
        fps: int = Form(8, ge=4, le=12),
    ):
        try:
            generator = VideoGenerator()
            result = generator.generate_video.remote(
                prompt=prompt,
                duration=duration,
                width=width,
                height=height,
                fps=fps,
            )
            
            if result["success"]:
                return Response(
                    content=result["video_data"],
                    media_type="video/mp4",
                    headers={
                        "Content-Disposition": "attachment; filename=video.mp4",
                        "X-Generation-Time": str(result["generation_time"]),
                        "X-Duration": str(result["duration"]),
                        "X-FPS": str(result["fps"]),
                        "X-Resolution": f"{result['width']}x{result['height']}",
                    }
                )
            else:
                raise HTTPException(status_code=500, detail=result.get("error"))
                
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return web_app


# ============================================================================
# LOCAL TEST COMMAND
# ============================================================================

@app.local_entrypoint()
def test_generation():
    """
    Test video generation with sample prompt.
    Run with: modal run wan_video_api.py::test_generation
    """
    print("🚀 WAN Video Generator - Test Mode")
    print("=" * 80)
    print()
    
    try:
        generator = VideoGenerator()
        
        # Health check
        print("📋 Health Check")
        print("-" * 80)
        health = generator.health_check.remote()
        for k, v in health.items():
            print(f"  {k}: {v}")
        print()
        
        if not health.get("checkpoint_ready"):
            print("=" * 80)
            print("❌ Models not found in storage!")
            print()
            print("Please run:")
            print("  modal run wan_video_api.py::download_all_models")
            print("=" * 80)
            return
        
        # Generate test video
        print("=" * 80)
        print("🎬 Generating Test Video")
        print("=" * 80)
        print()
        
        result = generator.generate_video.remote(
            prompt="a beautiful sunset over the ocean, cinematic, 4k, golden hour",
            duration=3,
            width=512,
            height=512,
            fps=8,
        )
        
        if result["success"]:
            output_file = "test_output.mp4"
            with open(output_file, "wb") as f:
                f.write(result["video_data"])
            
            print()
            print("=" * 80)
            print("✅ TEST SUCCESSFUL")
            print("=" * 80)
            print(f"File: {output_file}")
            print(f"Size: {result['size_mb']:.2f} MB")
            print(f"Duration: {result['duration']:.1f}s")
            print(f"Frames: {result['frames']}")
            print(f"Resolution: {result['width']}x{result['height']}")
            print(f"Generation Time: {result['generation_time']:.1f}s")
            print("=" * 80)
        else:
            print()
            print("=" * 80)
            print(f"❌ TEST FAILED: {result.get('error')}")
            print("=" * 80)
        
    except Exception as e:
        print()
        print("=" * 80)
        print(f"❌ ERROR: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
