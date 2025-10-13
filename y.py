"""
Wan2.2 Video Generation Service - Refactored Production Version
Complete video generation system with ComfyUI backend on Modal

Key improvements:
- Better separation of concerns
- Enhanced error handling
- Improved configuration management
- More robust model downloading
- Better logging and monitoring
- Type hints throughout
- Async support where beneficial
"""

import modal
import json
import time
import logging
import subprocess
import requests
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from contextlib import contextmanager

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

def setup_logging() -> logging.Logger:
    """Configure structured logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# =============================================================================
# CONFIGURATION
# =============================================================================

class PathConfig:
    """Centralized path management"""
    COMFYUI_PATH = Path("/root/ComfyUI")
    MODELS_PATH = Path("/models")
    OUTPUTS_PATH = Path("/outputs")
    
    # Model subdirectories
    CHECKPOINTS_DIR = MODELS_PATH / "checkpoints"
    VAE_DIR = MODELS_PATH / "vae"
    LORAS_DIR = MODELS_PATH / "loras"
    CONTROLNET_DIR = MODELS_PATH / "controlnet"
    UPSCALE_DIR = MODELS_PATH / "upscale_models"
    EMBEDDINGS_DIR = MODELS_PATH / "embeddings"
    
    @classmethod
    def ensure_directories(cls):
        """Create all necessary directories"""
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if isinstance(attr, Path) and attr_name.endswith('_DIR'):
                attr.mkdir(parents=True, exist_ok=True)


class ServerConfig:
    """ComfyUI server configuration"""
    HOST = "127.0.0.1"
    PORT = 8188
    STARTUP_TIMEOUT = 60
    HEALTH_CHECK_INTERVAL = 1
    
    @property
    def base_url(self) -> str:
        return f"http://{self.HOST}:{self.PORT}"


class GenerationConfig:
    """Default generation parameters"""
    DEFAULT_WIDTH = 1280
    DEFAULT_HEIGHT = 720
    DEFAULT_STEPS = 25
    DEFAULT_CFG = 7.0
    DEFAULT_SAMPLER = "dpmpp_2m"
    DEFAULT_SCHEDULER = "karras"
    DEFAULT_FPS = 8
    DEFAULT_NUM_FRAMES = 16


class LimitsConfig:
    """Service limits and timeouts"""
    MAX_CONCURRENT_REQUESTS = 10
    REQUEST_TIMEOUT = 1800  # 30 minutes
    IDLE_TIMEOUT = 300  # 5 minutes
    MAX_VIDEO_DURATION = 30  # seconds
    MAX_PROMPT_LENGTH = 2000
    MAX_RESOLUTION = 2048
    MIN_RESOLUTION = 512


class DownloadConfig:
    """Download settings for aria2"""
    CONNECTIONS = 16
    SPLIT = 16
    TIMEOUT = 3600
    RETRY_COUNT = 3

# =============================================================================
# MODAL SETUP
# =============================================================================

app = modal.App("wan-video-production-v2")

# Optimized image with all dependencies
comfyui_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        add_python="3.11"
    )
    .apt_install(
        "git", "aria2", "wget", "curl", "ffmpeg",
        "libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6"
    )
    .run_commands(
        # Clone ComfyUI
        "git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /root/ComfyUI",
        
        # Install PyTorch with CUDA 12.1
        "cd /root/ComfyUI && pip install --pre torch torchvision torchaudio "
        "--index-url https://download.pytorch.org/whl/cu121",
        
        # Install ComfyUI requirements
        "cd /root/ComfyUI && pip install -r requirements.txt",
        
        # Install additional dependencies
        "pip install websocket-client Pillow numpy opencv-python aiohttp pyyaml",
        
        # Install custom nodes
        "cd /root/ComfyUI/custom_nodes && "
        "git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Manager.git && "
        "git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && "
        "git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git",
    )
)

# Persistent volumes
models_volume = modal.Volume.from_name("wan-models-v4", create_if_missing=True)
outputs_volume = modal.Volume.from_name("wan-outputs-v4", create_if_missing=True)

# =============================================================================
# DATA MODELS
# =============================================================================

class JobStatus(Enum):
    """Job lifecycle states"""
    PENDING = "pending"
    INITIALIZING = "initializing"
    DOWNLOADING_MODELS = "downloading_models"
    STARTING_SERVER = "starting_server"
    BUILDING_WORKFLOW = "building_workflow"
    GENERATING = "generating"
    POST_PROCESSING = "post_processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class LoRAConfig:
    """LoRA configuration"""
    filename: str
    strength_model: float = 1.0
    strength_clip: float = 1.0
    
    def __post_init__(self):
        """Validate LoRA parameters"""
        if not 0 <= self.strength_model <= 2:
            raise ValueError("LoRA strength_model must be between 0 and 2")
        if not 0 <= self.strength_clip <= 2:
            raise ValueError("LoRA strength_clip must be between 0 and 2")


@dataclass
class VideoRequest:
    """Video generation request parameters"""
    prompt: str
    negative_prompt: str = "blurry, low quality, distorted, ugly, watermark"
    width: int = GenerationConfig.DEFAULT_WIDTH
    height: int = GenerationConfig.DEFAULT_HEIGHT
    steps: int = GenerationConfig.DEFAULT_STEPS
    cfg_scale: float = GenerationConfig.DEFAULT_CFG
    seed: int = -1  # -1 for random
    sampler: str = GenerationConfig.DEFAULT_SAMPLER
    scheduler: str = GenerationConfig.DEFAULT_SCHEDULER
    loras: List[LoRAConfig] = field(default_factory=list)
    checkpoint: str = "wan_2.5_4k.safetensors"
    vae: Optional[str] = None
    batch_size: int = 1
    num_frames: int = GenerationConfig.DEFAULT_NUM_FRAMES
    fps: int = GenerationConfig.DEFAULT_FPS
    
    def __post_init__(self):
        """Validate request parameters"""
        if not self.prompt or len(self.prompt.strip()) == 0:
            raise ValueError("Prompt cannot be empty")
        
        if len(self.prompt) > LimitsConfig.MAX_PROMPT_LENGTH:
            raise ValueError(f"Prompt exceeds maximum length of {LimitsConfig.MAX_PROMPT_LENGTH}")
        
        if not (LimitsConfig.MIN_RESOLUTION <= self.width <= LimitsConfig.MAX_RESOLUTION):
            raise ValueError(f"Width must be between {LimitsConfig.MIN_RESOLUTION} and {LimitsConfig.MAX_RESOLUTION}")
        
        if not (LimitsConfig.MIN_RESOLUTION <= self.height <= LimitsConfig.MAX_RESOLUTION):
            raise ValueError(f"Height must be between {LimitsConfig.MIN_RESOLUTION} and {LimitsConfig.MAX_RESOLUTION}")
        
        if self.seed == -1:
            self.seed = int(time.time() * 1000) % (2**31)


@dataclass
class JobResult:
    """Generation job result"""
    job_id: str
    status: JobStatus
    message: str
    output_files: List[str] = field(default_factory=list)
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary with status as string"""
        result = asdict(self)
        result['status'] = self.status.value
        return result


@dataclass
class ModelDefinition:
    """Model download definition"""
    url: str
    filename: str
    dest_path: Path
    required: bool = False
    size_mb: Optional[int] = None
    checksum: Optional[str] = None


# =============================================================================
# MODEL REGISTRY
# =============================================================================

class ModelRegistry:
    """Central registry of all models"""
    
    @staticmethod
    def get_all_models() -> List[ModelDefinition]:
        """Get complete list of models to download"""
        return [
            # Main Wan2.2 Checkpoint
            ModelDefinition(
                url="https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B/resolve/main/diffusion_pytorch_model.safetensors",
                filename="wan_2.2_t2v_a14b.safetensors",
                dest_path=PathConfig.CHECKPOINTS_DIR,
                required=True,
                size_mb=54000
            ),
            
            # VAE
            ModelDefinition(
                url="https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors",
                filename="vae-ft-mse-840000-ema-pruned.safetensors",
                dest_path=PathConfig.VAE_DIR,
                required=False,
                size_mb=335
            ),
            
            # Text Encoder (T5)
            ModelDefinition(
                url="https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B/resolve/main/text_encoder/model.safetensors",
                filename="t5_encoder.safetensors",
                dest_path=PathConfig.EMBEDDINGS_DIR,
                required=True,
                size_mb=5000
            ),
            
            # ControlNet models
            ModelDefinition(
                url="https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth",
                filename="control_v11p_sd15_openpose.pth",
                dest_path=PathConfig.CONTROLNET_DIR,
                required=False
            ),
            
            # Upscaler
            ModelDefinition(
                url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                filename="RealESRGAN_x4plus.pth",
                dest_path=PathConfig.UPSCALE_DIR,
                required=False
            ),
        ]
    
    @staticmethod
    def get_required_models() -> List[ModelDefinition]:
        """Get only required models"""
        return [m for m in ModelRegistry.get_all_models() if m.required]
    
    @staticmethod
    def get_optional_models() -> List[ModelDefinition]:
        """Get only optional models"""
        return [m for m in ModelRegistry.get_all_models() if not m.required]


# =============================================================================
# MODEL DOWNLOADER
# =============================================================================

class ModelDownloader:
    """Handles model downloads with retry logic"""
    
    @staticmethod
    def download_file(model: ModelDefinition, retry: int = 0) -> bool:
        """Download a single file with retry logic"""
        try:
            model.dest_path.mkdir(parents=True, exist_ok=True)
            full_path = model.dest_path / model.filename
            
            # Check if already exists
            if full_path.exists():
                logger.info(f"✓ {model.filename} already exists")
                return True
            
            logger.info(f"⬇️  Downloading {model.filename}...")
            if model.size_mb:
                logger.info(f"   Expected size: {model.size_mb}MB")
            
            # Build aria2 command
            cmd = [
                "aria2c",
                "-x", str(DownloadConfig.CONNECTIONS),
                "-s", str(DownloadConfig.SPLIT),
                "-d", str(model.dest_path),
                "-o", model.filename,
                "--auto-file-renaming=false",
                "--allow-overwrite=true",
                "--max-tries=5",
                "--retry-wait=3",
                model.url
            ]
            
            # Execute download
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=DownloadConfig.TIMEOUT
            )
            
            # Verify download
            if result.returncode == 0 and full_path.exists():
                actual_size_mb = full_path.stat().st_size / (1024 * 1024)
                logger.info(f"✅ {model.filename} downloaded ({actual_size_mb:.1f}MB)")
                return True
            else:
                logger.error(f"❌ Download failed: {result.stderr}")
                
                # Retry logic
                if retry < DownloadConfig.RETRY_COUNT:
                    logger.info(f"🔄 Retrying ({retry + 1}/{DownloadConfig.RETRY_COUNT})...")
                    time.sleep(5 * (retry + 1))
                    return ModelDownloader.download_file(model, retry + 1)
                
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Download timeout for {model.filename}")
            if retry < DownloadConfig.RETRY_COUNT:
                return ModelDownloader.download_file(model, retry + 1)
            return False
            
        except Exception as e:
            logger.error(f"❌ Download error for {model.filename}: {e}")
            return False
    
    @staticmethod
    def download_all_models(required_only: bool = False) -> Tuple[int, int]:
        """
        Download all models
        Returns: (successful_count, total_count)
        """
        models = (ModelRegistry.get_required_models() if required_only 
                  else ModelRegistry.get_all_models())
        
        logger.info(f"📦 Downloading {len(models)} models...")
        
        successful = 0
        failed = []
        
        for model in models:
            if ModelDownloader.download_file(model):
                successful += 1
            else:
                failed.append(model.filename)
                if model.required:
                    raise Exception(f"Failed to download required model: {model.filename}")
        
        if failed:
            logger.warning(f"⚠️  Failed to download {len(failed)} optional models: {failed}")
        
        logger.info(f"✅ Successfully downloaded {successful}/{len(models)} models")
        
        return successful, len(models)


# =============================================================================
# COMFYUI SERVER MANAGER
# =============================================================================

class ComfyUIServer:
    """Manages ComfyUI server lifecycle"""
    
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.config = ServerConfig()
        self.client_id = str(uuid.uuid4())
        self.is_running = False
    
    def start(self) -> bool:
        """Start ComfyUI server"""
        try:
            logger.info("🚀 Starting ComfyUI server...")
            
            # Create model paths config
            self._create_model_config()
            
            # Build command
            cmd = [
                "python", "main.py",
                "--listen", self.config.HOST,
                "--port", str(self.config.PORT),
                "--extra-model-paths-config", str(PathConfig.MODELS_PATH / "extra_model_paths.yaml")
            ]
            
            # Start process
            self.process = subprocess.Popen(
                cmd,
                cwd=str(PathConfig.COMFYUI_PATH),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server
            if self._wait_for_ready():
                self.is_running = True
                logger.info("✅ ComfyUI server ready")
                return True
            else:
                logger.error("❌ Server failed to start")
                self._cleanup()
                return False
                
        except Exception as e:
            logger.error(f"❌ Error starting server: {e}")
            self._cleanup()
            return False
    
    def _create_model_config(self):
        """Create extra model paths configuration"""
        import yaml
        
        config = {
            "wan_models": {
                "base_path": str(PathConfig.MODELS_PATH),
                "checkpoints": str(PathConfig.CHECKPOINTS_DIR),
                "vae": str(PathConfig.VAE_DIR),
                "loras": str(PathConfig.LORAS_DIR),
                "controlnet": str(PathConfig.CONTROLNET_DIR),
                "upscale_models": str(PathConfig.UPSCALE_DIR),
                "embeddings": str(PathConfig.EMBEDDINGS_DIR)
            }
        }
        
        config_path = PathConfig.MODELS_PATH / "extra_model_paths.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
    
    def _wait_for_ready(self) -> bool:
        """Wait for server to be ready"""
        for i in range(ServerConfig.STARTUP_TIMEOUT):
            try:
                response = requests.get(
                    f"{self.config.base_url}/system_stats",
                    timeout=2
                )
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            
            if i % 10 == 0 and i > 0:
                logger.info(f"⏳ Waiting for server... ({i}s)")
            
            time.sleep(ServerConfig.HEALTH_CHECK_INTERVAL)
        
        return False
    
    def queue_prompt(self, workflow: Dict) -> Optional[str]:
        """Queue a prompt and return prompt_id"""
        try:
            payload = {
                "prompt": workflow,
                "client_id": self.client_id
            }
            
            response = requests.post(
                f"{self.config.base_url}/prompt",
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            data = response.json()
            prompt_id = data.get("prompt_id")
            
            logger.info(f"✅ Prompt queued: {prompt_id}")
            return prompt_id
            
        except Exception as e:
            logger.error(f"❌ Error queuing prompt: {e}")
            return None
    
    def wait_for_completion(self, prompt_id: str, timeout: int = 1800) -> Dict:
        """Wait for generation to complete"""
        start_time = time.time()
        last_log_time = start_time
        
        while time.time() - start_time < timeout:
            try:
                history = self._get_history(prompt_id)
                
                if history and prompt_id in history:
                    data = history[prompt_id]
                    status = data.get("status", {})
                    
                    # Check completion
                    if status.get("completed"):
                        logger.info("✅ Generation completed")
                        return data
                    
                    # Check for errors
                    if status.get("status_str") == "error":
                        error_msg = status.get("messages", ["Unknown error"])[0]
                        raise Exception(f"Generation failed: {error_msg}")
                    
                    # Progress logging
                    current_time = time.time()
                    if current_time - last_log_time > 30:
                        elapsed = int(current_time - start_time)
                        logger.info(f"⏳ Still generating... ({elapsed}s elapsed)")
                        last_log_time = current_time
                
                time.sleep(2)
                
            except Exception as e:
                if "Generation failed" in str(e):
                    raise
                logger.warning(f"Error checking status: {e}")
                time.sleep(5)
        
        raise TimeoutError(f"Generation timed out after {timeout}s")
    
    def _get_history(self, prompt_id: str) -> Optional[Dict]:
        """Get generation history"""
        try:
            response = requests.get(
                f"{self.config.base_url}/history/{prompt_id}",
                timeout=10
            )
            return response.json() if response.status_code == 200 else None
        except Exception:
            return None
    
    def extract_outputs(self, history: Dict) -> List[str]:
        """Extract output file paths from history"""
        outputs = []
        
        try:
            for node_id, node_output in history.get("outputs", {}).items():
                # Handle images
                if "images" in node_output:
                    for img in node_output["images"]:
                        filename = img.get("filename")
                        subfolder = img.get("subfolder", "")
                        if filename:
                            path = f"{subfolder}/{filename}" if subfolder else filename
                            outputs.append(path)
                
                # Handle videos
                if "videos" in node_output:
                    for vid in node_output["videos"]:
                        filename = vid.get("filename")
                        if filename:
                            outputs.append(filename)
        
        except Exception as e:
            logger.error(f"❌ Error extracting outputs: {e}")
        
        return outputs
    
    def stop(self):
        """Stop the server"""
        if self.process and self.is_running:
            logger.info("🛑 Stopping ComfyUI server...")
            self._cleanup()
            logger.info("✅ Server stopped")
    
    def _cleanup(self):
        """Clean up server process"""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None
        self.is_running = False
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()


# =============================================================================
# WORKFLOW BUILDER
# =============================================================================

class WorkflowBuilder:
    """Builds ComfyUI workflows dynamically"""
    
    @staticmethod
    def build_t2v_workflow(request: VideoRequest) -> Dict:
        """Build text-to-video workflow for Wan2.2"""
        
        node_counter = 1
        
        def next_node() -> str:
            nonlocal node_counter
            node_id = str(node_counter)
            node_counter += 1
            return node_id
        
        workflow = {}
        
        # Load Checkpoint
        checkpoint_node = next_node()
        workflow[checkpoint_node] = {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": request.checkpoint}
        }
        
        model_ref = [checkpoint_node, 0]
        clip_ref = [checkpoint_node, 1]
        vae_ref = [checkpoint_node, 2]
        
        # Add LoRAs
        if request.loras:
            for lora in request.loras:
                lora_node = next_node()
                workflow[lora_node] = {
                    "class_type": "LoraLoader",
                    "inputs": {
                        "lora_name": lora.filename,
                        "strength_model": lora.strength_model,
                        "strength_clip": lora.strength_clip,
                        "model": model_ref,
                        "clip": clip_ref
                    }
                }
                model_ref = [lora_node, 0]
                clip_ref = [lora_node, 1]
        
        # Custom VAE
        if request.vae:
            vae_node = next_node()
            workflow[vae_node] = {
                "class_type": "VAELoader",
                "inputs": {"vae_name": request.vae}
            }
            vae_ref = [vae_node, 0]
        
        # Positive prompt
        pos_node = next_node()
        workflow[pos_node] = {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": request.prompt,
                "clip": clip_ref
            }
        }
        
        # Negative prompt
        neg_node = next_node()
        workflow[neg_node] = {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": request.negative_prompt,
                "clip": clip_ref
            }
        }
        
        # Empty latent
        latent_node = next_node()
        workflow[latent_node] = {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": request.width,
                "height": request.height,
                "batch_size": request.num_frames
            }
        }
        
        # KSampler
        sampler_node = next_node()
        workflow[sampler_node] = {
            "class_type": "KSampler",
            "inputs": {
                "seed": request.seed,
                "steps": request.steps,
                "cfg": request.cfg_scale,
                "sampler_name": request.sampler,
                "scheduler": request.scheduler,
                "denoise": 1.0,
                "model": model_ref,
                "positive": [pos_node, 0],
                "negative": [neg_node, 0],
                "latent_image": [latent_node, 0]
            }
        }
        
        # VAE Decode
        decode_node = next_node()
        workflow[decode_node] = {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": [sampler_node, 0],
                "vae": vae_ref
            }
        }
        
        # Save as video
        save_node = next_node()
        workflow[save_node] = {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "frame_rate": request.fps,
                "loop_count": 0,
                "filename_prefix": f"wan_{int(time.time())}",
                "format": "video/h264-mp4",
                "images": [decode_node, 0]
            }
        }
        
        return workflow


# =============================================================================
# GENERATION SERVICE
# =============================================================================

class GenerationService:
    """High-level generation service"""
    
    @staticmethod
    def generate(request: VideoRequest) -> JobResult:
        """Execute complete generation pipeline"""
        
        job_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        try:
            logger.info("=" * 80)
            logger.info(f"🎬 JOB {job_id} STARTED")
            logger.info(f"📝 Prompt: {request.prompt[:100]}...")
            logger.info("=" * 80)
            
            # Use context manager for server
            with ComfyUIServer() as server:
                # Build workflow
                logger.info("🔨 Building workflow...")
                workflow = WorkflowBuilder.build_t2v_workflow(request)
                
                # Queue prompt
                logger.info("📤 Queueing prompt...")
                prompt_id = server.queue_prompt(workflow)
                
                if not prompt_id:
                    raise Exception("Failed to queue prompt")
                
                # Wait for completion
                logger.info("⏳ Generating video...")
                history = server.wait_for_completion(
                    prompt_id,
                    timeout=LimitsConfig.REQUEST_TIMEOUT
                )
                
                # Extract outputs
                outputs = server.extract_outputs(history)
                
                elapsed = time.time() - start_time
                logger.info(f"✅ Job completed in {elapsed:.1f}s")
                logger.info(f"📁 Generated {len(outputs)} files")
                
                return JobResult(
                    job_id=job_id,
                    status=JobStatus.COMPLETED,
                    message="Generation completed successfully",
                    output_files=outputs,
                    created_at=start_time,
                    completed_at=time.time(),
                    metadata={
                        "elapsed_seconds": elapsed,
                        "num_outputs": len(outputs),
                        "request": asdict(request)
                    }
                )
        
        except Exception as e:
            logger.error(f"❌ Job {job_id} failed: {e}", exc_info=True)
            
            return JobResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                message="Generation failed",
                error=str(e),
                created_at=start_time,
                completed_at=time.time()
            )


# =============================================================================
# MODAL FUNCTIONS
# =============================================================================

@app.function(
    image=comfyui_image,
    volumes={str(PathConfig.MODELS_PATH): models_volume},
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface-token", required=False)]
)
def download_models_task(required_only: bool = False):
    """Download models to persistent volume"""
    try:
        logger.info("=" * 80)
        logger.info("📦 MODEL DOWNLOAD TASK")
        logger.info("=" * 80)
        
        models_volume.reload()
        PathConfig.ensure_directories()
        
        successful, total = ModelDownloader.download_all_models(required_only)
        
        models_volume.commit()
        
        logger.info("=" * 80)
        logger.info(f"✅ DOWNLOAD COMPLETE - {count} models")
