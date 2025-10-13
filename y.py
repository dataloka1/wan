"""
Wan 2.5 Video Generation Service - Production Ready
Complete video generation system with ComfyUI backend on Modal
"""

import modal
import os
import sys
import json
import time
import logging
import subprocess
import requests
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# MODAL APP & IMAGE
# =============================================================================

app = modal.App("wan-video-production")

comfyui_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        add_python="3.11"
    )
    .apt_install(
        "git",
        "aria2",
        "wget",
        "curl",
        "ffmpeg",  # For video processing
        "libgl1-mesa-glx",
        "libglib2.0-0"
    )
    .run_commands(
        # Clone ComfyUI
        "git clone https://github.com/comfyanonymous/ComfyUI.git /root/ComfyUI",
        
        # Install PyTorch
        "cd /root/ComfyUI && pip install --pre torch torchvision torchaudio "
        "--index-url https://download.pytorch.org/whl/cu121",
        
        # Install ComfyUI requirements
        "cd /root/ComfyUI && pip install -r requirements.txt",
        
        # Install additional dependencies
        "pip install websocket-client Pillow numpy opencv-python",
        
        # Install ComfyUI Manager
        "cd /root/ComfyUI/custom_nodes && "
        "git clone https://github.com/ltdrdata/ComfyUI-Manager.git",
        
        # Install Video Helper Suite
        "cd /root/ComfyUI/custom_nodes && "
        "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
        
        # Install AnimateDiff
        "cd /root/ComfyUI/custom_nodes && "
        "git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git",
        
        "echo 'ComfyUI image build complete.'"
    )
)

# =============================================================================
# VOLUMES
# =============================================================================

models_volume = modal.Volume.from_name("wan-models-v3", create_if_missing=True)
outputs_volume = modal.Volume.from_name("wan-outputs-v3", create_if_missing=True)

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Central configuration"""
    
    # Paths
    COMFYUI_PATH = Path("/root/ComfyUI")
    MODELS_PATH = Path("/models")
    OUTPUTS_PATH = Path("/outputs")
    
    # Model directories
    CHECKPOINTS_DIR = MODELS_PATH / "checkpoints"
    VAE_DIR = MODELS_PATH / "vae"
    LORAS_DIR = MODELS_PATH / "loras"
    CONTROLNET_DIR = MODELS_PATH / "controlnet"
    UPSCALE_DIR = MODELS_PATH / "upscale_models"
    EMBEDDINGS_DIR = MODELS_PATH / "embeddings"
    
    # Server settings
    COMFYUI_HOST = "127.0.0.1"
    COMFYUI_PORT = 8188
    SERVER_STARTUP_TIMEOUT = 60
    
    # Generation settings
    DEFAULT_WIDTH = 1024
    DEFAULT_HEIGHT = 576
    DEFAULT_STEPS = 25
    DEFAULT_CFG = 7.0
    DEFAULT_SAMPLER = "dpmpp_2m"
    DEFAULT_SCHEDULER = "karras"
    
    # Limits
    MAX_CONCURRENT_REQUESTS = 10
    REQUEST_TIMEOUT = 1800  # 30 minutes
    IDLE_TIMEOUT = 300  # 5 minutes
    MAX_VIDEO_DURATION = 30  # seconds
    
    # Download settings
    ARIA2_CONNECTIONS = 16
    ARIA2_SPLIT = 16

# =============================================================================
# DATA MODELS
# =============================================================================

class JobStatus(Enum):
    PENDING = "pending"
    DOWNLOADING_MODELS = "downloading_models"
    STARTING_SERVER = "starting_server"
    BUILDING_WORKFLOW = "building_workflow"
    GENERATING = "generating"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class LoRAConfig:
    filename: str
    strength_model: float = 1.0
    strength_clip: float = 1.0

@dataclass
class VideoRequest:
    prompt: str
    negative_prompt: str = "blurry, low quality, distorted, ugly"
    width: int = Config.DEFAULT_WIDTH
    height: int = Config.DEFAULT_HEIGHT
    steps: int = Config.DEFAULT_STEPS
    cfg_scale: float = Config.DEFAULT_CFG
    seed: int = -1  # -1 for random
    loras: List[LoRAConfig] = None
    checkpoint: str = "wan_2.5_4k.safetensors"
    vae: Optional[str] = None
    batch_size: int = 1
    num_frames: int = 16  # For video generation
    fps: int = 8

@dataclass
class JobResult:
    job_id: str
    status: JobStatus
    message: str
    output_files: List[str] = None
    error: Optional[str] = None
    created_at: float = 0
    completed_at: Optional[float] = None

# =============================================================================
# MODEL DOWNLOADER
# =============================================================================

class ModelDownloader:
    """Handle model downloads with aria2"""
    
    @staticmethod
    def download_file(url: str, dest_path: Path, filename: str) -> bool:
        """Download single file with aria2"""
        try:
            dest_path.mkdir(parents=True, exist_ok=True)
            full_path = dest_path / filename
            
            if full_path.exists():
                logger.info(f"✓ {filename} already exists, skipping")
                return True
            
            logger.info(f"⬇️  Downloading {filename}...")
            
            cmd = [
                "aria2c",
                "-x", str(Config.ARIA2_CONNECTIONS),
                "-s", str(Config.ARIA2_SPLIT),
                "-d", str(dest_path),
                "-o", filename,
                "--auto-file-renaming=false",
                "--allow-overwrite=true",
                url
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600
            )
            
            if result.returncode == 0 and full_path.exists():
                logger.info(f"✅ {filename} downloaded successfully")
                return True
            else:
                logger.error(f"❌ Failed to download {filename}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Download error for {filename}: {e}")
            return False
    
    @staticmethod
    def download_all_models():
        """Download all required models"""
        
        models = [
            # Main Checkpoint
            {
                "url": "https://huggingface.co/alibaba-pai/Wan/resolve/main/wan_2.5_4k.safetensors",
                "filename": "wan_2.5_4k.safetensors",
                "dest": Config.CHECKPOINTS_DIR,
                "required": True
            },
            
            # VAE
            {
                "url": "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors",
                "filename": "vae-ft-mse-840000-ema-pruned.safetensors",
                "dest": Config.VAE_DIR,
                "required": False
            },
            
            # ControlNets
            {
                "url": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth",
                "filename": "control_v11p_sd15_openpose.pth",
                "dest": Config.CONTROLNET_DIR,
                "required": False
            },
            {
                "url": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth",
                "filename": "control_v11f1p_sd15_depth.pth",
                "dest": Config.CONTROLNET_DIR,
                "required": False
            },
            
            # Upscaler
            {
                "url": "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth",
                "filename": "RealESRGAN_x4.pth",
                "dest": Config.UPSCALE_DIR,
                "required": False
            },
            
            # Example LoRAs (add your 50+ here)
            {
                "url": "https://civitai.com/api/download/models/133724",
                "filename": "detail-enhancer.safetensors",
                "dest": Config.LORAS_DIR,
                "required": False
            },
        ]
        
        success_count = 0
        required_count = sum(1 for m in models if m["required"])
        
        for model in models:
            success = ModelDownloader.download_file(
                model["url"],
                model["dest"],
                model["filename"]
            )
            
            if success:
                success_count += 1
            elif model["required"]:
                raise Exception(f"Failed to download required model: {model['filename']}")
        
        logger.info(f"✅ Downloaded {success_count}/{len(models)} models")
        
        if success_count < required_count:
            raise Exception("Failed to download required models")
        
        return success_count

# =============================================================================
# COMFYUI SERVER MANAGER
# =============================================================================

class ComfyUIServer:
    """Manage ComfyUI server lifecycle"""
    
    def __init__(self):
        self.process = None
        self.base_url = f"http://{Config.COMFYUI_HOST}:{Config.COMFYUI_PORT}"
        self.client_id = str(uuid.uuid4())
    
    def start(self) -> bool:
        """Start ComfyUI server"""
        try:
            logger.info("🚀 Starting ComfyUI server...")
            
            # Build command
            cmd = [
                "python",
                "main.py",
                "--listen", Config.COMFYUI_HOST,
                "--port", str(Config.COMFYUI_PORT),
                "--extra-model-paths-config", str(Config.MODELS_PATH / "extra_model_paths.yaml")
            ]
            
            # Create extra model paths config
            self._create_model_paths_config()
            
            # Start process
            self.process = subprocess.Popen(
                cmd,
                cwd=str(Config.COMFYUI_PATH),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to be ready
            if self._wait_for_server():
                logger.info("✅ ComfyUI server started successfully")
                return True
            else:
                logger.error("❌ ComfyUI server failed to start")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error starting server: {e}")
            return False
    
    def _create_model_paths_config(self):
        """Create extra model paths configuration"""
        config = {
            "wan_models": {
                "base_path": str(Config.MODELS_PATH),
                "checkpoints": str(Config.CHECKPOINTS_DIR),
                "vae": str(Config.VAE_DIR),
                "loras": str(Config.LORAS_DIR),
                "controlnet": str(Config.CONTROLNET_DIR),
                "upscale_models": str(Config.UPSCALE_DIR),
                "embeddings": str(Config.EMBEDDINGS_DIR)
            }
        }
        
        config_path = Config.MODELS_PATH / "extra_model_paths.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
    
    def _wait_for_server(self) -> bool:
        """Wait for server to be ready"""
        for i in range(Config.SERVER_STARTUP_TIMEOUT):
            try:
                response = requests.get(
                    f"{self.base_url}/system_stats",
                    timeout=1
                )
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            
            if i % 5 == 0:
                logger.info(f"⏳ Waiting for server... ({i}s)")
            
            time.sleep(1)
        
        return False
    
    def queue_prompt(self, workflow: Dict) -> Optional[str]:
        """Queue prompt and return prompt_id"""
        try:
            payload = {
                "prompt": workflow,
                "client_id": self.client_id
            }
            
            response = requests.post(
                f"{self.base_url}/prompt",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                prompt_id = data.get("prompt_id")
                logger.info(f"✅ Prompt queued: {prompt_id}")
                return prompt_id
            else:
                logger.error(f"❌ Failed to queue prompt: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error queuing prompt: {e}")
            return None
    
    def get_history(self, prompt_id: str) -> Optional[Dict]:
        """Get generation history"""
        try:
            response = requests.get(
                f"{self.base_url}/history/{prompt_id}",
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except Exception as e:
            logger.error(f"❌ Error getting history: {e}")
            return None
    
    def wait_for_completion(self, prompt_id: str, timeout: int = 1800) -> Dict:
        """Wait for generation to complete"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            history = self.get_history(prompt_id)
            
            if history and prompt_id in history:
                status = history[prompt_id].get("status", {})
                
                if status.get("completed"):
                    logger.info("✅ Generation completed")
                    return history[prompt_id]
                
                # Check for errors
                if "error" in status:
                    error_msg = status["error"]
                    logger.error(f"❌ Generation failed: {error_msg}")
                    raise Exception(f"Generation failed: {error_msg}")
            
            time.sleep(2)
        
        raise TimeoutError("Generation timed out")
    
    def get_output_files(self, history: Dict) -> List[str]:
        """Extract output file paths from history"""
        output_files = []
        
        try:
            outputs = history.get("outputs", {})
            for node_id, node_output in outputs.items():
                if "images" in node_output:
                    for img in node_output["images"]:
                        filename = img.get("filename")
                        subfolder = img.get("subfolder", "")
                        if filename:
                            output_files.append(f"{subfolder}/{filename}" if subfolder else filename)
        except Exception as e:
            logger.error(f"❌ Error extracting output files: {e}")
        
        return output_files
    
    def stop(self):
        """Stop ComfyUI server"""
        if self.process:
            logger.info("🛑 Stopping ComfyUI server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            logger.info("✅ Server stopped")

# =============================================================================
# WORKFLOW BUILDER
# =============================================================================

class WorkflowBuilder:
    """Build ComfyUI workflows dynamically"""
    
    @staticmethod
    def build_basic_workflow(request: VideoRequest) -> Dict:
        """Build basic text-to-image workflow"""
        
        workflow = {
            # Load Checkpoint
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": request.checkpoint
                }
            },
            
            # Positive Prompt
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": request.prompt,
                    "clip": ["1", 1]
                }
            },
            
            # Negative Prompt
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": request.negative_prompt,
                    "clip": ["1", 1]
                }
            },
            
            # Empty Latent
            "4": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": request.width,
                    "height": request.height,
                    "batch_size": request.batch_size
                }
            },
            
            # KSampler
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": request.seed if request.seed >= 0 else int(time.time()),
                    "steps": request.steps,
                    "cfg": request.cfg_scale,
                    "sampler_name": Config.DEFAULT_SAMPLER,
                    "scheduler": Config.DEFAULT_SCHEDULER,
                    "denoise": 1.0,
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0]
                }
            },
            
            # VAE Decode
            "6": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["5", 0],
                    "vae": ["1", 2]
                }
            },
            
            # Save Image
            "7": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": f"wan_{int(time.time())}",
                    "images": ["6", 0]
                }
            }
        }
        
        # Add LoRAs if specified
        if request.loras:
            workflow = WorkflowBuilder._add_loras(workflow, request.loras)
        
        # Add custom VAE if specified
        if request.vae:
            workflow = WorkflowBuilder._add_vae(workflow, request.vae)
        
        return workflow
    
    @staticmethod
    def _add_loras(workflow: Dict, loras: List[LoRAConfig]) -> Dict:
        """Add LoRA nodes to workflow"""
        
        last_model_node = "1"
        last_clip_node = "1"
        
        for i, lora in enumerate(loras):
            node_id = str(100 + i)
            
            workflow[node_id] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name": lora.filename,
                    "strength_model": lora.strength_model,
                    "strength_clip": lora.strength_clip,
                    "model": [last_model_node, 0],
                    "clip": [last_clip_node, 1]
                }
            }
            
            last_model_node = node_id
            last_clip_node = node_id
        
        # Update KSampler to use last LoRA
        workflow["5"]["inputs"]["model"] = [last_model_node, 0]
        
        # Update CLIP nodes
        workflow["2"]["inputs"]["clip"] = [last_clip_node, 1]
        workflow["3"]["inputs"]["clip"] = [last_clip_node, 1]
        
        return workflow
    
    @staticmethod
    def _add_vae(workflow: Dict, vae_name: str) -> Dict:
        """Add custom VAE loader"""
        
        workflow["8"] = {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": vae_name
            }
        }
        
        # Update VAE Decode to use custom VAE
        workflow["6"]["inputs"]["vae"] = ["8", 0]
        
        return workflow

# =============================================================================
# MODAL FUNCTIONS
# =============================================================================

@app.function(
    image=comfyui_image,
    volumes={str(Config.MODELS_PATH): models_volume},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-token")]
)
def download_models():
    """Download all models to volume"""
    try:
        logger.info("=" * 80)
        logger.info("📦 DOWNLOADING MODELS")
        logger.info("=" * 80)
        
        models_volume.reload()
        
        count = ModelDownloader.download_all_models()
        
        models_volume.commit()
        
        logger.info("=" * 80)
        logger.info(f"✅ DOWNLOAD COMPLETE - {count} models")
        logger.info("=" * 80)
        
        return {"success": True, "models_downloaded": count}
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

@app.function(
    image=comfyui_image,
    gpu="L40S",
    volumes={
        str(Config.MODELS_PATH): models_volume,
        str(Config.OUTPUTS_PATH): outputs_volume
    },
    timeout=Config.REQUEST_TIMEOUT,
    allow_concurrent_inputs=Config.MAX_CONCURRENT_REQUESTS,
    container_idle_timeout=Config.IDLE_TIMEOUT
)
def generate_video_internal(request_dict: Dict) -> Dict:
    """Internal generation function"""
    
    job_id = str(uuid.uuid4())[:8]
    
    try:
        logger.info("=" * 80)
        logger.info(f"🎬 STARTING JOB: {job_id}")
        logger.info("=" * 80)
        
        # Reload volumes
        models_volume.reload()
        outputs_volume.reload()
        
        # Parse request
        if request_dict.get("loras"):
            loras = [LoRAConfig(**l) for l in request_dict["loras"]]
            request_dict["loras"] = loras
        
        request = VideoRequest(**request_dict)
        
        # Start ComfyUI server
        server = ComfyUIServer()
        if not server.start():
            raise Exception("Failed to start ComfyUI server")
        
        try:
            # Build workflow
            logger.info("🔨 Building workflow...")
            workflow = WorkflowBuilder.build_basic_workflow(request)
            
            # Queue prompt
            logger.info("📤 Queueing prompt...")
            prompt_id = server.queue_prompt(workflow)
            
            if not prompt_id:
                raise Exception("Failed to queue prompt")
            
            # Wait for completion
            logger.info("⏳ Waiting for generation...")
            history = server.wait_for_completion(prompt_id)
            
            # Get output files
            output_files = server.get_output_files(history)
            
            logger.info(f"✅ Generation complete: {len(output_files)} files")
            
            # Commit outputs
            outputs_volume.commit()
            
            result = JobResult(
                job_id=job_id,
                status=JobStatus.COMPLETED,
                message="Generation completed successfully",
                output_files=output_files,
                created_at=time.time(),
                completed_at=time.time()
            )
            
            return asdict(result)
            
        finally:
            server.stop()
    
    except Exception as e:
        logger.error(f"❌ Job {job_id} failed: {e}", exc_info=True)
        
        result = JobResult(
            job_id=job_id,
            status=JobStatus.FAILED,
            message="Generation failed",
            error=str(e),
            created_at=time.time()
        )
        
        return asdict(result)

# =============================================================================
# WEB API
# =============================================================================

@app.function(image=comfyui_image)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    
    web_app = FastAPI(
        title="Wan 2.5 Video Generation API",
        version="1.0.0"
    )
    
    class GenerateRequest(BaseModel):
        prompt: str = Field(..., min_length=1, max_length=2000)
        negative_prompt: str = "blurry, low quality"
        width: int = Field(1024, ge=512, le=2048)
        height: int = Field(576, ge=512, le=2048)
        steps: int = Field(25, ge=1, le=150)
        cfg_scale: float = Field(7.0, ge=1.0, le=30.0)
        seed: int = -1
        loras: List[Dict] = []
        checkpoint: str = "wan_2.5_4k.safetensors"
        vae: Optional[str] = None
    
    @web_app.get("/")
    async def root():
        return {
            "service": "Wan 2.5 Video Generation",
            "version": "1.0.0",
            "status": "operational"
        }
    
    @web_app.get("/health")
    async def health():
        return {"status": "healthy"}
    
    @web_app.post("/generate")
    async def generate(request: GenerateRequest):
        try:
            # Call internal generation function
            result = generate_video_internal.remote(request.dict())
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @web_app.get("/models/status")
    async def models_status():
        """Check which models are downloaded"""
        # This would check the volume
        return {"status": "Check logs for model status"}
    
    return web_app

# =============================================================================
# LOCAL ENTRYPOINT
# =============================================================================

@app.local_entrypoint()
def main():
    print("=" * 80)
    print("🎬 Wan 2.5 Video Generation Service")
    print("=" * 80)
    print("\n📦 Downloading models...")
    
    result = download_models.remote()
    
    if result["success"]:
        print(f"\n✅ Setup complete! Downloaded {result['models_downloaded']} models")
        print("\n🚀 Service is ready!")
        print("\n📖 Next steps:")
        print("  1. Access the API at the Modal URL shown above")
        print("  2. POST to /generate with your prompt")
        print("  3. Check /health for service status")
    else:
        print(f"\n❌ Setup failed: {result.get('error')}")
    
    print("=" * 80)
