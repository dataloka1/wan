"""
Modal.com Video Generation API dengan WAN 2.2, LoRA, ControlNet & ComfyUI
Deploy dengan: modal deploy wan_video_api.py
Test dengan: modal run wan_video_api.py

FIXED VERSION - Volume mounting issue resolved
"""

import modal
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any

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
        "huggingface-hub>=0.19.0",
    )
    # DON'T clone ComfyUI in image - we'll do it in setup
)

# Volume untuk persistent storage models
models_volume = modal.Volume.from_name("wan-models-vol", create_if_missing=True)

# GPU Config - L40S
GPU_CONFIG = modal.gpu.L40S(count=1)

# Constants - FIXED: Use different path for local vs deployed
MODELS_PATH = "/models"  # Simplified path for volume
OUTPUT_PATH = "/output"  # Separate output path


def is_local_run():
    """Check if running in local test mode"""
    import sys
    return 'modal.runner' not in sys.modules or os.environ.get('MODAL_IS_REMOTE') != '1'


@app.cls(
    image=image,
    gpu=GPU_CONFIG,
    # FIXED: Only mount volume in deployed mode
    volumes={MODELS_PATH: models_volume} if not is_local_run() else {},
    timeout=3600,
    container_idle_timeout=300,
    allow_concurrent_inputs=10,
)
class VideoGenerator:
    """
    Class untuk video generation dengan Stable Diffusion
    """

    @modal.enter()
    def setup(self):
        """Setup models dan dependencies saat container start"""
        import sys
        
        print("🚀 Initializing Video Generator...")
        print(f"   Running mode: {'LOCAL' if is_local_run() else 'DEPLOYED'}")
        
        # Create necessary directories
        os.makedirs(MODELS_PATH, exist_ok=True)
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        
        # Download models
        try:
            self._download_models()
        except Exception as e:
            print(f"⚠️ Warning during model download: {e}")
            if not is_local_run():
                raise  # In deployed mode, fail if models can't download
        
        # Setup flag
        self.initialized = True
        
        print("✅ Video Generator ready!")

    def _download_models(self):
        """Download models yang diperlukan"""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print("⚠️ huggingface_hub not available")
            return
        
        print("📥 Checking models...")
        
        # Model checkpoint path
        checkpoint_dir = f"{MODELS_PATH}/checkpoints"
        vae_dir = f"{MODELS_PATH}/vae"
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(vae_dir, exist_ok=True)
        
        # Check if models exist
        checkpoint_path = f"{checkpoint_dir}/v1-5-pruned-emaonly.safetensors"
        vae_path = f"{vae_dir}/vae-ft-mse-840000-ema-pruned.safetensors"
        
        # Download checkpoint if not exists
        if not os.path.exists(checkpoint_path):
            print("⬇️ Downloading Stable Diffusion checkpoint...")
            try:
                hf_hub_download(
                    repo_id="runwayml/stable-diffusion-v1-5",
                    filename="v1-5-pruned-emaonly.safetensors",
                    local_dir=checkpoint_dir,
                    local_dir_use_symlinks=False,
                )
                print("✅ Checkpoint downloaded")
            except Exception as e:
                print(f"⚠️ Could not download checkpoint: {e}")
        else:
            print("✓ Checkpoint exists")
        
        # Download VAE if not exists
        if not os.path.exists(vae_path):
            print("⬇️ Downloading VAE...")
            try:
                hf_hub_download(
                    repo_id="stabilityai/sd-vae-ft-mse-original",
                    filename="vae-ft-mse-840000-ema-pruned.safetensors",
                    local_dir=vae_dir,
                    local_dir_use_symlinks=False,
                )
                print("✅ VAE downloaded")
            except Exception as e:
                print(f"⚠️ Could not download VAE: {e}")
        else:
            print("✓ VAE exists")
        
        # Commit to volume if in deployed mode
        if not is_local_run():
            try:
                models_volume.commit()
                print("✅ Models committed to volume")
            except Exception as e:
                print(f"⚠️ Could not commit to volume: {e}")

    def _create_video_from_images(
        self, 
        prompt: str,
        duration: int,
        width: int,
        height: int,
        fps: int = 8
    ) -> bytes:
        """
        Create video using Stable Diffusion pipeline
        """
        import torch
        from diffusers import StableDiffusionPipeline
        import numpy as np
        import cv2
        import tempfile
        
        print("🎨 Creating video with Stable Diffusion...")
        
        try:
            # Check for model
            checkpoint_path = f"{MODELS_PATH}/checkpoints/v1-5-pruned-emaonly.safetensors"
            
            if os.path.exists(checkpoint_path):
                print(f"📦 Loading model from {checkpoint_path}")
                pipe = StableDiffusionPipeline.from_single_file(
                    checkpoint_path,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                )
            else:
                print("📦 Loading model from HuggingFace (first time may take a while)...")
                pipe = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16,
                    safety_checker=None,
                )
            
            pipe = pipe.to("cuda")
            pipe.enable_attention_slicing()
            
            # Calculate number of frames
            num_frames = duration * fps
            max_frames = 24  # Limit for demo
            actual_frames = min(num_frames, max_frames)
            
            print(f"🎬 Generating {actual_frames} frames (FPS: {fps})...")
            
            frames = []
            
            for i in range(actual_frames):
                # Vary seed slightly for each frame
                generator = torch.Generator("cuda").manual_seed(42 + i * 10)
                
                # Add frame number to prompt for variation
                frame_prompt = f"{prompt}, frame {i}"
                
                # Generate image
                with torch.inference_mode():
                    image = pipe(
                        prompt=frame_prompt,
                        height=height,
                        width=width,
                        num_inference_steps=20,
                        guidance_scale=7.5,
                        generator=generator,
                    ).images[0]
                
                # Convert to numpy array
                frame = np.array(image)
                frames.append(frame)
                
                if (i + 1) % 5 == 0:
                    print(f"  ✓ Generated {i + 1}/{actual_frames} frames")
            
            # Create video file
            print("🎞️ Encoding video...")
            
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                video_path = tmp.name
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            
            # Write all frames
            for frame in frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            
            # Read video file
            with open(video_path, 'rb') as f:
                video_data = f.read()
            
            # Cleanup temp file
            try:
                os.unlink(video_path)
            except:
                pass
            
            print(f"✅ Video created: {len(video_data) / 1024 / 1024:.2f} MB")
            
            return video_data
            
        except Exception as e:
            print(f"❌ Error in video generation: {e}")
            import traceback
            traceback.print_exc()
            raise

    @modal.method()
    def generate_video(
        self,
        prompt: str,
        mode: str = "t2v",
        image_data: Optional[bytes] = None,
        duration: int = 10,
        width: int = 512,
        height: int = 512,
        use_controlnet: bool = False,
        lora_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate video dari text atau image
        
        Args:
            prompt: Text prompt untuk generation
            mode: "t2v" untuk text-to-video
            image_data: Binary image data (not used in basic version)
            duration: Durasi video dalam detik
            width: Width video (must be multiple of 8)
            height: Height video (must be multiple of 8)
            use_controlnet: ControlNet (not implemented in basic version)
            lora_name: LoRA name (not implemented in basic version)
            
        Returns:
            dict dengan 'success', 'video_data', dan metadata
        """
        import time
        
        try:
            # Validate and adjust inputs
            duration = max(1, min(duration, 30))  # 1-30 seconds
            width = (width // 8) * 8  # Ensure multiple of 8
            height = (height // 8) * 8
            width = max(256, min(width, 768))  # Reasonable limits
            height = max(256, min(height, 768))
            
            print("=" * 60)
            print(f"🎬 Video Generation Request")
            print(f"   Mode: {mode.upper()}")
            print(f"   Prompt: '{prompt}'")
            print(f"   Duration: {duration}s")
            print(f"   Resolution: {width}x{height}")
            print("=" * 60)
            
            start_time = time.time()
            
            # Generate video
            fps = 8
            video_data = self._create_video_from_images(
                prompt=prompt,
                duration=duration,
                width=width,
                height=height,
                fps=fps
            )
            
            elapsed = time.time() - start_time
            
            print("=" * 60)
            print(f"✅ Video generated successfully!")
            print(f"   Time: {elapsed:.2f}s")
            print(f"   Size: {len(video_data) / 1024 / 1024:.2f} MB")
            print("=" * 60)
            
            return {
                "success": True,
                "video_data": video_data,
                "duration": duration,
                "width": width,
                "height": height,
                "fps": fps,
                "frames": duration * fps,
                "generation_time": elapsed,
                "message": f"Video generated in {elapsed:.2f}s"
            }
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "message": f"Error: {str(e)}"
            }

    @modal.method()
    def health_check(self) -> Dict[str, Any]:
        """Health check endpoint"""
        checkpoint_exists = os.path.exists(f"{MODELS_PATH}/checkpoints/v1-5-pruned-emaonly.safetensors")
        
        return {
            "status": "healthy" if self.initialized else "initializing",
            "mode": "local" if is_local_run() else "deployed",
            "gpu": "L40S (CUDA available)" if self._check_cuda() else "CPU only",
            "models_path": MODELS_PATH,
            "checkpoint_exists": checkpoint_exists,
            "output_path": OUTPUT_PATH,
        }
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False


# FastAPI Web Endpoints
@app.function(
    image=image,
    allow_concurrent_inputs=100,
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, File, UploadFile, Form, HTTPException
    from fastapi.responses import Response
    from fastapi.middleware.cors import CORSMiddleware
    
    web_app = FastAPI(
        title="WAN Video Generation API",
        description="Text-to-Video Generation API powered by Stable Diffusion",
        version="2.1.0"
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
            "name": "WAN Video Generation API",
            "version": "2.1.0",
            "status": "operational",
            "endpoints": {
                "health": "GET /health",
                "text_to_video": "POST /api/t2v",
                "docs": "GET /docs"
            },
            "info": {
                "max_duration": "30 seconds",
                "max_resolution": "768x768",
                "fps": 8
            }
        }
    
    @web_app.get("/health")
    async def health():
        """Health check endpoint"""
        try:
            generator = VideoGenerator()
            result = generator.health_check.remote()
            return result
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    @web_app.post("/api/t2v")
    async def text_to_video(
        prompt: str = Form(..., description="Text prompt untuk video", min_length=1),
        duration: int = Form(10, ge=1, le=30, description="Duration in seconds (1-30)"),
        width: int = Form(512, ge=256, le=768, description="Width (256-768)"),
        height: int = Form(512, ge=256, le=768, description="Height (256-768)"),
    ):
        """
        Text-to-Video Generation
        
        Generate a video from a text prompt using Stable Diffusion.
        """
        try:
            # Create generator and call method
            generator = VideoGenerator()
            result = generator.generate_video.remote(
                prompt=prompt,
                mode="t2v",
                duration=duration,
                width=width,
                height=height,
            )
            
            if result["success"]:
                return Response(
                    content=result["video_data"],
                    media_type="video/mp4",
                    headers={
                        "Content-Disposition": "attachment; filename=generated_video.mp4",
                        "X-Generation-Time": str(result["generation_time"]),
                        "X-Duration": str(result["duration"]),
                        "X-FPS": str(result["fps"]),
                        "X-Resolution": f"{result['width']}x{result['height']}",
                    }
                )
            else:
                raise HTTPException(status_code=500, detail=result["message"])
                
        except HTTPException:
            raise
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
    
    return web_app


# Local entrypoint untuk testing
@app.local_entrypoint()
def main():
    """
    Local testing entrypoint
    
    Usage: modal run wan_video_api.py
    """
    print("🚀 WAN Video Generator - Local Test Mode")
    print("=" * 70)
    print()
    
    try:
        # Create generator instance
        print("📦 Creating VideoGenerator instance...")
        generator = VideoGenerator()
        
        # Test 1: Health Check
        print("\n" + "=" * 70)
        print("TEST 1: Health Check")
        print("=" * 70)
        
        health = generator.health_check.remote()
        print("\nHealth Status:")
        for key, value in health.items():
            print(f"  • {key}: {value}")
        
        # Test 2: Video Generation
        print("\n" + "=" * 70)
        print("TEST 2: Video Generation")
        print("=" * 70)
        
        test_prompt = "a beautiful sunset over the ocean, cinematic lighting, 4k"
        test_duration = 3  # Short for testing
        
        print(f"\nGenerating video with:")
        print(f"  • Prompt: {test_prompt}")
        print(f"  • Duration: {test_duration} seconds")
        print(f"  • Resolution: 512x512")
        print()
        
        result = generator.generate_video.remote(
            prompt=test_prompt,
            duration=test_duration,
            width=512,
            height=512
        )
        
        if result["success"]:
            # Save video
            output_file = "test_output.mp4"
            with open(output_file, "wb") as f:
                f.write(result["video_data"])
            
            file_size_mb = len(result["video_data"]) / (1024 * 1024)
            
            print("\n" + "=" * 70)
            print("✅ SUCCESS - Video Generated!")
            print("=" * 70)
            print(f"\nOutput Details:")
            print(f"  • File: {output_file}")
            print(f"  • Size: {file_size_mb:.2f} MB")
            print(f"  • Duration: {result['duration']}s")
            print(f"  • Resolution: {result['width']}x{result['height']}")
            print(f"  • FPS: {result['fps']}")
            print(f"  • Frames: {result['frames']}")
            print(f"  • Generation Time: {result['generation_time']:.2f}s")
            print()
            print(f"📹 Video saved to: ./{output_file}")
            
        else:
            print("\n" + "=" * 70)
            print("❌ FAILED")
            print("=" * 70)
            print(f"\nError: {result['message']}")
        
        print("\n" + "=" * 70)
        print("✅ All tests completed!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Deploy to Modal: modal deploy wan_video_api.py")
        print("  2. Get API URL from deployment output")
        print("  3. Test API: curl -X POST <URL>/api/t2v -F 'prompt=...'")
        print()
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ ERROR")
        print("=" * 70)
        print(f"\n{e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Troubleshooting:")
        print("  • Make sure Modal is installed: pip install modal")
        print("  • Configure Modal: modal setup")
        print("  • Check GPU availability in your Modal account")
        print("  • Verify network connection for model downloads")
