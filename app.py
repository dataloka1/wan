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
import requests
from pathlib import Path
from typing import Dict, Optional, List, Union, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
import threading
import atexit
import tempfile
import signal
import sys
import queue
import asyncio
from datetime import datetime, timedelta

from modal import App, Image, Volume, Secret, asgi_app, enter, method, Queue
from fastapi import FastAPI, HTTPException, Body, BackgroundTasks
from pydantic import BaseModel, Field
import redis
import aiofiles

# Constants
COMFYUI_PATH = Path("/root/comfy/ComfyUI")
MODEL_PATH = Path("/root/comfy/ComfyUI/models")
CACHE_PATH = Path("/cache")

MIN_FILE_SIZE_KB = 500
MIN_LORA_SIZE_KB = 100
MAX_BASE64_SIZE = 100 * 1024 * 1024
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
SERVER_STARTUP_TIMEOUT = 120

# Dynamic timeouts (can be overridden by environment variables)
DEFAULT_GENERATION_TIMEOUT = int(os.getenv("GENERATION_TIMEOUT", "3600"))
DEFAULT_POLL_TIMEOUT = int(os.getenv("POLL_TIMEOUT", "600"))
DEFAULT_POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "3"))
DEFAULT_RETRY_COUNT = int(os.getenv("RETRY_COUNT", "3"))
DEFAULT_RETRY_DELAY = int(os.getenv("RETRY_DELAY", "5"))

# Progress reporting interval (seconds)
PROGRESS_REPORT_INTERVAL = int(os.getenv("PROGRESS_REPORT_INTERVAL", "30"))

# Scalability settings
MAX_CONCURRENT_REQUESTS = 100
MAX_CONTAINERS = 10
QUEUE_CHECK_INTERVAL = 5  # seconds

# Redis configuration for distributed queue
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_QUEUE_KEY = "comfyui:task_queue"
REDIS_STATUS_KEY = "comfyui:task_status"
REDIS_PROGRESS_KEY = "comfyui:task_progress"

class GenerationStatus(Enum):
    QUEUED = "queued"
    STARTED = "started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

@dataclass
class ProgressData:
    task_id: str
    status: GenerationStatus
    progress: float = 0.0
    current_step: int = 0
    total_steps: int = 0
    message: str = ""
    error: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    node_id: Optional[str] = None
    container_id: Optional[str] = None

@dataclass
class GenerationTask:
    task_id: str
    client_id: str
    prompt_id: Optional[str] = None
    status: GenerationStatus = GenerationStatus.QUEUED
    progress_data: Optional[ProgressData] = None
    webhook_url: Optional[str] = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    request_data: Optional[Dict] = None
    queue_time: Optional[float] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

# Model Registry
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

# Pydantic Models for API
class T2VRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 832
    height: int = 480
    num_frames: int = 121
    steps: int = 30
    cfg: float = 7.5
    seed: Optional[int] = None
    use_fast_mode: bool = False
    webhook_url: Optional[str] = None
    task_id: Optional[str] = None

class I2VRequest(BaseModel):
    image_base64: str
    prompt: str
    negative_prompt: str = ""
    width: int = 1280
    height: int = 704
    num_frames: int = 81
    steps: int = 20
    cfg: float = 3.5
    seed: Optional[int] = None
    use_fast_mode: bool = False
    webhook_url: Optional[str] = None
    task_id: Optional[str] = None

class AnimateRequest(BaseModel):
    reference_image_base64: str
    video_base64: str
    prompt: str
    negative_prompt: str = ""
    width: int = 640
    height: int = 640
    num_frames: int = 77
    steps: int = 6
    cfg: float = 1.0
    seed: Optional[int] = None
    use_fast_mode: bool = True
    webhook_url: Optional[str] = None
    task_id: Optional[str] = None

class CameraLoraRequest(BaseModel):
    image_base64: str
    prompt: str
    camera_motion: str = "ZoomIn"
    lora_strength: float = 1.0
    negative_prompt: str = ""
    width: int = 1280
    height: int = 704
    num_frames: int = 81
    steps: int = 20
    cfg: float = 3.5
    seed: Optional[int] = None
    webhook_url: Optional[str] = None
    task_id: Optional[str] = None

class CancelRequest(BaseModel):
    task_id: str

class QueueStatusResponse(BaseModel):
    queue_length: int
    active_tasks: int
    max_concurrent: int
    containers_running: int
    max_containers: int

# Utility Functions
def _validate_and_decode_base64(data: str, data_type: str = "image") -> str:
    """Validate and decode base64 data with improved error handling"""
    if not data:
        raise ValueError(f"{data_type} data is empty")
    
    if len(data) > MAX_BASE64_SIZE:
        raise ValueError(f"{data_type} data too large. Max {MAX_BASE64_SIZE/(1024*1024):.2f}MB")
    
    # Handle data URL format
    if data.startswith(f'data:{data_type}/') or data.startswith('data:audio/'):
        if ';base64,' not in data:
            raise ValueError(f"Invalid base64 {data_type} format")
        data = data.split(';base64,')[1]
    
    # Validate base64 format
    try:
        # Remove whitespace and fix padding
        data = ''.join(data.split())
        data += '=' * (-len(data) % 4)
        # Test decode
        base64.b64decode(data, validate=True)
    except Exception as e:
        raise ValueError(f"Invalid base64 encoding: {str(e)}")
    
    return data

def _save_base64_to_file(data_base64: str, temp_filename: str, data_type: str = "image") -> str:
    """Save base64 data to file with proper cleanup"""
    clean_b64 = _validate_and_decode_base64(data_base64, data_type)
    try:
        file_data = base64.b64decode(clean_b64)
        with open(temp_filename, "wb") as f:
            f.write(file_data)
        print(f"[{data_type.upper()}] Saved: {temp_filename} ({len(file_data)/1024:.2f} KB)")
        return temp_filename
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        raise ValueError(f"Failed to save {data_type}: {str(e)}")

def _download_model_safe(repo_id: str, filename: str, target_path: Path, model_type: str) -> bool:
    """Safely download model from HuggingFace with improved error handling"""
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
        
        # Create symlink
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists():
            target_path.unlink()
        
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
    """Setup ComfyUI models with improved error handling"""
    print("\n" + "="*80)
    print("MODEL DOWNLOAD STARTED")
    print("="*80)
    stats = {"total": 0, "success": 0, "failed": 0, "skipped": 0}
    
    for model_type, models in MODEL_REGISTRY.items():
        print(f"\n[{model_type.upper()}] Processing {len(models)} models...")
        target_dir = MODEL_PATH / model_type
        target_dir.mkdir(parents=True, exist_ok=True)
        
        for filename, source in models.items():
            stats["total"] += 1
            target_path = target_dir / filename
            
            # Check if already exists and valid
            if target_path.exists() and target_path.is_symlink():
                try:
                    file_size = target_path.stat().st_size / 1024
                    min_size = MIN_LORA_SIZE_KB if model_type == "loras" else MIN_FILE_SIZE_KB
                    
                    if file_size > min_size and target_path.resolve().exists():
                        print(f"[{model_type.upper()}] Already exists: {filename}")
                        stats["success"] += 1
                        stats["skipped"] += 1
                        continue
                    else:
                        print(f"[{model_type.upper()}] Invalid symlink, removing: {filename}")
                        target_path.unlink()
                except Exception as e:
                    print(f"[{model_type.upper()}] Error checking symlink: {filename} - {e}")
                    try:
                        target_path.unlink()
                    except:
                        pass
            
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
    print(f"MODEL DOWNLOAD COMPLETE: {stats['success']}/{stats['total']} successful, {stats['failed']} failed, {stats['skipped']} skipped")
    print("="*80 + "\n")
    
    if stats["failed"] > 0:
        print(f"WARNING: {stats['failed']} models failed to download. Some features may not work correctly.")

# Setup Modal Image and Volume with all dependencies
cache_volume = Volume.from_name("hf-hub-cache", create_if_missing=True)

comfy_image = (
    Image.debian_slim(python_version="3.12")
    .apt_install(
        "git", 
        "ffmpeg", 
        "libgl1-mesa-glx", 
        "libglib2.0-0", 
        "curl",
        "wget",
        "build-essential",
        "python3-dev"
    )
    # Install websocket-client first to fix the import error
    .pip_install(
        "websocket-client>=1.6.0",
        "huggingface_hub[hf_transfer]>=0.34.0,<1.0",
        "requests>=2.31.0",
        "aiofiles>=23.0.0",
        "redis>=4.5.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.5.0",
        "python-multipart>=0.0.6",
        force_build=True
    )
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
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTHONPATH": "/root/comfy",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
    })
    .run_function(
        setup_comfyui,
        volumes={str(CACHE_PATH): cache_volume},
    )
)

# FastAPI App
fastapi_app = FastAPI(
    title="ComfyUI Wan2.2 API",
    description="Scalable video generation API with queue management",
    version="2.0.0"
)

# Modal App with scalability settings
app = App(
    "comfyui-wan2-2-production",
    image=comfy_image
)

# Redis client for distributed queue
redis_client = None

def get_redis_client():
    """Get Redis client"""
    global redis_client
    if redis_client is None:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    return redis_client

# Distributed Queue Manager
class QueueManager:
    def __init__(self):
        self.redis = get_redis_client()
        self.local_queue = queue.Queue()
        self.active_tasks = {}
        self.max_concurrent = MAX_CONCURRENT_REQUESTS
        self._lock = threading.Lock()
    
    def enqueue_task(self, task_data: Dict) -> str:
        """Add task to distributed queue"""
        task_id = task_data.get('task_id') or str(uuid.uuid4())
        task_data['task_id'] = task_id
        task_data['queue_time'] = time.time()
        task_data['status'] = GenerationStatus.QUEUED.value
        
        # Add to Redis queue
        self.redis.lpush(REDIS_QUEUE_KEY, json.dumps(task_data))
        
        # Store task status
        self.redis.hset(REDIS_STATUS_KEY, task_id, json.dumps({
            'status': GenerationStatus.QUEUED.value,
            'queue_time': task_data['queue_time']
        }))
        
        print(f"[QUEUE] Enqueued task {task_id}")
        return task_id
    
    def dequeue_task(self) -> Optional[Dict]:
        """Get next task from queue"""
        try:
            # Try to get from Redis queue
            _, task_json = self.redis.brpop(REDIS_QUEUE_KEY, timeout=1)
            if task_json:
                task_data = json.loads(task_json)
                task_data['start_time'] = time.time()
                task_data['status'] = GenerationStatus.STARTED.value
                
                # Update status
                self.redis.hset(REDIS_STATUS_KEY, task_data['task_id'], json.dumps({
                    'status': GenerationStatus.STARTED.value,
                    'start_time': task_data['start_time']
                }))
                
                print(f"[QUEUE] Dequeued task {task_data['task_id']}")
                return task_data
        except:
            pass
        return None
    
    def update_task_status(self, task_id: str, status: GenerationStatus, **kwargs):
        """Update task status in Redis"""
        status_data = {
            'status': status.value,
            'timestamp': time.time()
        }
        status_data.update(kwargs)
        
        self.redis.hset(REDIS_STATUS_KEY, task_id, json.dumps(status_data))
        
        # If completed, remove from active tasks
        if status in [GenerationStatus.COMPLETED, GenerationStatus.FAILED, GenerationStatus.CANCELLED]:
            self.redis.hdel(REDIS_STATUS_KEY, task_id)
    
    def update_progress(self, task_id: str, progress_data: Dict):
        """Update task progress in Redis"""
        self.redis.hset(REDIS_PROGRESS_KEY, task_id, json.dumps(progress_data))
        
        # Clean up old progress data (keep for 1 hour)
        self.redis.expire(REDIS_PROGRESS_KEY, 3600)
    
    def get_queue_status(self) -> Dict:
        """Get current queue status"""
        queue_length = self.redis.llen(REDIS_QUEUE_KEY)
        active_tasks = len(self.redis.hgetall(REDIS_STATUS_KEY))
        
        return {
            'queue_length': queue_length,
            'active_tasks': active_tasks,
            'max_concurrent': self.max_concurrent,
            'containers_running': 1,  # This would be updated by container manager
            'max_containers': MAX_CONTAINERS
        }
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        # Check if task is in queue
        queue_tasks = self.redis.lrange(REDIS_QUEUE_KEY, 0, -1)
        for task_json in queue_tasks:
            task = json.loads(task_json)
            if task['task_id'] == task_id:
                # Remove from queue
                self.redis.lrem(REDIS_QUEUE_KEY, 1, task_json)
                self.update_task_status(task_id, GenerationStatus.CANCELLED)
                return True
        
        # Check if task is active
        status = self.redis.hget(REDIS_STATUS_KEY, task_id)
        if status:
            self.update_task_status(task_id, GenerationStatus.CANCELLED)
            return True
        
        return False

# Global queue manager
queue_manager = QueueManager()

# ComfyUI Class with scalability
@app.cls(
    image=comfy_image,
    gpu="L40S",
    volumes={str(CACHE_PATH): cache_volume},
    timeout=7200,
    min_containers=1,
    max_containers=MAX_CONTAINERS,
    allow_concurrent_inputs=MAX_CONCURRENT_REQUESTS,
    container_idle_timeout=300,  # 5 minutes
  # Keep 1 container warm
)
class ComfyUI:
    def __init__(self):
        self.proc = None
        self.active_tasks = {}  # task_id -> GenerationTask
        self.temp_files = []  # Track temporary files for cleanup
        self._lock = threading.Lock()
        self.container_id = str(uuid.uuid4())
        self.is_processing = False
        self.current_load = 0
        
        # Register cleanup handlers
        atexit.register(self._cleanup)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n[SHUTDOWN] Container {self.container_id} received signal {signum}, cleaning up...")
        self._cleanup()
        sys.exit(0)
    
    def _cleanup(self):
        """Clean up resources"""
        print(f"[CLEANUP] Container {self.container_id} starting cleanup...")
        
        # Cancel all active tasks
        with self._lock:
            for task_id, task in self.active_tasks.items():
                task.cancel_event.set()
                task.status = GenerationStatus.CANCELLED
                queue_manager.update_task_status(task_id, GenerationStatus.CANCELLED)
                print(f"[CLEANUP] Cancelled task: {task_id}")
        
        # Clean up temporary files
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"[CLEANUP] Removed temp file: {temp_file}")
            except Exception as e:
                print(f"[CLEANUP] Failed to remove {temp_file}: {e}")
        
        # Terminate ComfyUI process
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=10)
                print("[CLEANUP] ComfyUI process terminated")
            except:
                try:
                    self.proc.kill()
                    print("[CLEANUP] ComfyUI process killed")
                except:
                    pass
        
        print(f"[CLEANUP] Container {self.container_id} cleanup completed")

    @enter()
    def startup(self):
        """Start ComfyUI server with improved error handling"""
        print(f"[STARTUP] Container {self.container_id} starting up...")
        
        # Check if ComfyUI is already running
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{SERVER_PORT}/queue", timeout=5).read()
            print("[SERVER] ComfyUI already running")
            return
        except:
            pass
        
        cmd = ["comfy", "launch", "--", "--listen", SERVER_HOST, "--port", str(SERVER_PORT)]
        
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # Create new process group
            )
            print(f"[SERVER] Started with PID: {self.proc.pid}")
            
            # Wait for server to be ready
            for i in range(SERVER_STARTUP_TIMEOUT):
                if self.proc.poll() is not None:
                    stdout, stderr = self.proc.communicate()
                    raise RuntimeError(f"ComfyUI process exited early. stdout: {stdout.decode()}, stderr: {stderr.decode()}")
                
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{SERVER_PORT}/queue", timeout=5).read()
                    print(f"[SERVER] Ready after {i+1} seconds\n")
                    return
                except Exception:
                    if i % 10 == 0:
                        print(f"[SERVER] Waiting... ({i}/{SERVER_STARTUP_TIMEOUT}s)")
                    time.sleep(1)
            
            raise RuntimeError("ComfyUI failed to start within timeout")
            
        except Exception as e:
            print(f"[SERVER] Failed to start ComfyUI: {e}")
            if self.proc:
                try:
                    self.proc.kill()
                except:
                    pass
            raise

    def _create_task(self, task_id: str, webhook_url: Optional[str] = None, request_data: Optional[Dict] = None) -> GenerationTask:
        """Create a new generation task"""
        client_id = str(uuid.uuid4())
        task = GenerationTask(
            task_id=task_id,
            client_id=client_id,
            webhook_url=webhook_url,
            request_data=request_data,
            progress_data=ProgressData(
                task_id=task_id,
                status=GenerationStatus.QUEUED,
                container_id=self.container_id
            )
        )
        
        with self._lock:
            self.active_tasks[task_id] = task
            self.current_load += 1
        
        return task

    def _get_task(self, task_id: str) -> Optional[GenerationTask]:
        """Get active task by ID"""
        with self._lock:
            return self.active_tasks.get(task_id)

    def _remove_task(self, task_id: str):
        """Remove task from active tasks"""
        with self._lock:
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
                self.current_load = max(0, self.current_load - 1)

    def _send_webhook_with_retry(self, webhook_url: str, task_id: str, status: str, 
                                video_base64: str = None, error_detail: str = None, 
                                max_retries: int = DEFAULT_RETRY_COUNT):
        """Send webhook notification with retry mechanism"""
        if not webhook_url or not task_id:
            return False
            
        for attempt in range(max_retries):
            try:
                print(f"[WEBHOOK] Attempt {attempt + 1}/{max_retries}: Sending {status} result for {task_id}")
                
                if status == "success" and video_base64:
                    payload = {"status": "success", "video_base64": video_base64, "task_id": task_id}
                elif status == "progress":
                    task = self._get_task(task_id)
                    if task and task.progress_data:
                        payload = {
                            "status": "progress",
                            "task_id": task_id,
                            "progress": task.progress_data.progress,
                            "current_step": task.progress_data.current_step,
                            "total_steps": task.progress_data.total_steps,
                            "message": task.progress_data.message,
                            "generation_status": task.progress_data.status.value,
                            "container_id": self.container_id
                        }
                    else:
                        return False
                else:
                    payload = {"status": "error", "detail": error_detail or "Unknown error", "task_id": task_id}
                
                response = requests.post(webhook_url, json=payload, timeout=30)
                response.raise_for_status()
                
                print(f"[WEBHOOK] ✓ Successfully sent {status} result for {task_id}")
                return True
                
            except Exception as e:
                print(f"[WEBHOOK] ✗ Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(DEFAULT_RETRY_DELAY * (2 ** attempt))  # Exponential backoff
                else:
                    print(f"[WEBHOOK] ✗ All {max_retries} attempts failed for {task_id}")
                    return False

    def _update_progress(self, task_id: str, status: GenerationStatus, progress: float = None, 
                        message: str = None, error: str = None, node_id: str = None):
        """Update task progress and send webhook if needed"""
        task = self._get_task(task_id)
        if not task:
            return
        
        if task.progress_data:
            if status:
                task.progress_data.status = status
            if progress is not None:
                task.progress_data.progress = progress
            if message:
                task.progress_data.message = message
            if error:
                task.progress_data.error = error
            if node_id:
                task.progress_data.node_id = node_id
            
            # Update in Redis
            progress_dict = {
                "task_id": task_id,
                "status": status.value,
                "progress": progress,
                "message": message,
                "container_id": self.container_id
            }
            queue_manager.update_progress(task_id, progress_dict)
            
            # Update status in Redis
            queue_manager.update_task_status(task_id, status)
            
            # Send progress update
            if task.webhook_url:
                self._send_webhook_with_retry(task.webhook_url, task_id, "progress")

    def _queue_prompt(self, task: GenerationTask, prompt_workflow: dict) -> str:
        """Queue a prompt to ComfyUI with retry mechanism"""
        payload = {"prompt": prompt_workflow, "client_id": task.client_id}
        payload_json = json.dumps(payload)
        print(f"[QUEUE] Sending workflow ({len(payload_json)/1024:.2f} KB)")
        
        for attempt in range(DEFAULT_RETRY_COUNT):
            # Check if task was cancelled
            if task.cancel_event.is_set():
                raise RuntimeError("Task was cancelled")
            
            try:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{SERVER_PORT}/prompt",
                    data=payload_json.encode('utf-8'),
                    headers={'Content-Type': 'application/json'}
                )
                
                with urllib.request.urlopen(req, timeout=30) as response:
                    result = json.loads(response.read())
                    if 'error' in result:
                        error_detail = result.get('error', {})
                        print(f"[QUEUE] ERROR: {json.dumps(error_detail, indent=2)}")
                        raise RuntimeError(f"ComfyUI Error: {error_detail}")
                    if 'prompt_id' not in result:
                        raise RuntimeError(f"Invalid response: {result}")
                    
                    task.prompt_id = result['prompt_id']
                    print(f"[QUEUE] Prompt ID: {result['prompt_id']}")
                    return result['prompt_id']
                    
            except Exception as e:
                print(f"[QUEUE] Attempt {attempt + 1} failed: {str(e)}")
                if attempt < DEFAULT_RETRY_COUNT - 1:
                    time.sleep(DEFAULT_RETRY_DELAY)
                else:
                    raise RuntimeError(f"Failed to queue prompt after {DEFAULT_RETRY_COUNT} attempts: {str(e)}")

    def _get_history(self, prompt_id: str) -> dict:
        """Get history of a prompt with retry mechanism"""
        url = f"http://127.0.0.1:{SERVER_PORT}/history/{prompt_id}"
        
        for attempt in range(DEFAULT_RETRY_COUNT):
            try:
                with urllib.request.urlopen(url, timeout=10) as response:
                    return json.loads(response.read())
            except Exception as e:
                print(f"[HISTORY] Attempt {attempt + 1} failed: {str(e)}")
                if attempt < DEFAULT_RETRY_COUNT - 1:
                    time.sleep(DEFAULT_RETRY_DELAY)
                else:
                    raise RuntimeError(f"Failed to get history after {DEFAULT_RETRY_COUNT} attempts: {str(e)}")

    def _get_file(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        """Get file from ComfyUI with retry mechanism"""
        params = urllib.parse.urlencode({
            'filename': filename,
            'subfolder': subfolder,
            'type': folder_type
        })
        url = f"http://127.0.0.1:{SERVER_PORT}/view?{params}"
        
        for attempt in range(DEFAULT_RETRY_COUNT):
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    data = bytearray()
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        data.extend(chunk)
                    print(f"[FILE] Downloaded: {len(data)/1024/1024:.2f} MB")
                    return bytes(data)
            except Exception as e:
                print(f"[FILE] Attempt {attempt + 1} failed: {str(e)}")
                if attempt < DEFAULT_RETRY_COUNT - 1:
                    time.sleep(DEFAULT_RETRY_DELAY)
                else:
                    raise RuntimeError(f"Failed to get file after {DEFAULT_RETRY_COUNT} attempts: {str(e)}")

    def _get_video_from_websocket(self, task: GenerationTask, total_steps: int = 30) -> bytes:
        """Get video from ComfyUI via WebSocket with improved error handling"""
        ws_url = f"ws://127.0.0.1:{SERVER_PORT}/ws?clientId={task.client_id}"
        ws = None
        
        # Update progress
        self._update_progress(task.task_id, GenerationStatus.STARTED, 0.0, "Starting generation...")
        
        # Last progress report time
        last_progress_report = time.time()
        
        try:
            # Connect to WebSocket with retry
            for attempt in range(DEFAULT_RETRY_COUNT):
                if task.cancel_event.is_set():
                    raise RuntimeError("Task was cancelled")
                
                try:
                    ws = websocket.WebSocket()
                    ws.connect(ws_url, timeout=10)
                    ws.settimeout(60)
                    print("[WS] Connected")
                    break
                except Exception as e:
                    print(f"[WS] Connection attempt {attempt + 1} failed: {str(e)}")
                    if attempt < DEFAULT_RETRY_COUNT - 1:
                        time.sleep(DEFAULT_RETRY_DELAY)
                    else:
                        raise RuntimeError(f"WebSocket connection failed after {DEFAULT_RETRY_COUNT} attempts: {str(e)}")
            
            start_time = time.time()
            generation_done = False
            current_step = 0
            
            # Dynamic timeout based on total steps
            dynamic_timeout = min(DEFAULT_GENERATION_TIMEOUT, max(300, total_steps * 30))
            
            try:
                while time.time() - start_time < dynamic_timeout:
                    # Check for cancellation
                    if task.cancel_event.is_set():
                        self._update_progress(task.task_id, GenerationStatus.CANCELLED, 0.0, "Task was cancelled")
                        raise RuntimeError("Task was cancelled")
                    
                    try:
                        out = ws.recv()
                        if isinstance(out, str):
                            message = json.loads(out)
                            
                            if message.get('type') == 'progress':
                                data = message.get('data', {})
                                value = data.get('value', 0)
                                max_val = max(data.get('max', 1), 1)
                                pct = (value / max_val) * 100
                                current_step = value
                                
                                # Update progress
                                self._update_progress(
                                    task.task_id, 
                                    GenerationStatus.RUNNING, 
                                    pct, 
                                    f"Processing step {current_step}/{max_val}"
                                )
                                
                                print(f"[WS] Progress: {value}/{max_val} ({pct:.1f}%)")
                                
                                # Send progress update periodically
                                current_time = time.time()
                                if current_time - last_progress_report >= PROGRESS_REPORT_INTERVAL:
                                    self._send_webhook_with_retry(task.webhook_url, task.task_id, "progress")
                                    last_progress_report = current_time
                                
                            elif message.get('type') == 'executing':
                                data = message.get('data', {})
                                node = data.get('node')
                                if node is None:
                                    print("[WS] Execution complete signal received")
                                    generation_done = True
                                    self._update_progress(task.task_id, GenerationStatus.COMPLETED, 100.0, "Generation completed")
                                    break
                                else:
                                    print(f"[WS] Executing node: {node}")
                                    self._update_progress(task.task_id, None, None, f"Executing node: {node}", node_id=node)
                                    
                            elif message.get('type') == 'execution_error':
                                error_data = message.get('data', {})
                                print(f"[WS] Execution error: {error_data}")
                                self._update_progress(task.task_id, GenerationStatus.FAILED, 0.0, "Execution failed", str(error_data))
                                raise RuntimeError(f"ComfyUI execution error: {error_data}")
                                
                            elif message.get('type') == 'executed':
                                data = message.get('data', {})
                                node_id = data.get('node')
                                print(f"[WS] Node {node_id} completed")
                                
                    except websocket.WebSocketTimeoutException:
                        # Check if we should still be waiting
                        if time.time() - start_time >= dynamic_timeout:
                            break
                        continue
                        
            finally:
                if ws:
                    ws.close()
            
            if not generation_done:
                if time.time() - start_time >= dynamic_timeout:
                    self._update_progress(task.task_id, GenerationStatus.TIMEOUT, 0.0, f"Generation timeout after {dynamic_timeout}s")
                    raise TimeoutError(f"Generation timeout ({dynamic_timeout}s)")
                else:
                    self._update_progress(task.task_id, GenerationStatus.FAILED, 0.0, "Generation failed for unknown reason")
                    raise RuntimeError("Generation failed without completion signal")
            
            print("[POLL] Waiting 5 seconds for output finalization...")
            time.sleep(5)
            
            # More efficient polling with exponential backoff
            poll_timeout = DEFAULT_POLL_TIMEOUT
            poll_interval = DEFAULT_POLL_INTERVAL
            poll_start_time = time.time()
            poll_count = 0
            last_history_state = None
            consecutive_empty_results = 0
            
            while time.time() - poll_start_time < poll_timeout:
                # Check for cancellation
                if task.cancel_event.is_set():
                    raise RuntimeError("Task was cancelled")
                
                poll_count += 1
                elapsed = time.time() - poll_start_time
                
                # Adaptive polling interval
                if consecutive_empty_results > 3:
                    poll_interval = min(poll_interval * 1.5, 10)  # Increase interval up to 10s
                else:
                    poll_interval = max(poll_interval * 0.9, DEFAULT_POLL_INTERVAL)  # Decrease interval down to default
                
                print(f"[POLL] Attempt {poll_count}, elapsed: {elapsed:.1f}s, interval: {poll_interval:.1f}s")
                
                try:
                    history = self._get_history(task.prompt_id)
                    
                    if task.prompt_id not in history:
                        print(f"[POLL] Prompt ID not in history yet")
                        consecutive_empty_results += 1
                        time.sleep(poll_interval)
                        continue
                    
                    prompt_history = history[task.prompt_id]
                    last_history_state = prompt_history
                    
                    status = prompt_history.get('status', {})
                    status_str = status.get('status_str', 'unknown')
                    completed = status.get('completed', False)
                    
                    print(f"[POLL] Status: {status_str}, Completed: {completed}")
                    
                    if status_str == 'error':
                        error_info = status.get('messages', [])
                        self._update_progress(task.task_id, GenerationStatus.FAILED, 0.0, "Generation failed during polling", str(error_info))
                        raise RuntimeError(f"Generation failed: {error_info}")
                    
                    outputs = prompt_history.get('outputs', {})
                    
                    if not outputs:
                        print(f"[POLL] No outputs yet")
                        consecutive_empty_results += 1
                        time.sleep(poll_interval)
                        continue
                    
                    consecutive_empty_results = 0  # Reset counter
                    print(f"[POLL] Found {len(outputs)} output nodes: {list(outputs.keys())}")
                    
                    for node_id, node_output in outputs.items():
                        output_keys = list(node_output.keys())
                        print(f"[POLL] Node {node_id} has: {output_keys}")
                        
                        for media_type in ['videos', 'gifs', 'images']:
                            if media_type in node_output and node_output[media_type]:
                                items = node_output[media_type]
                                print(f"[POLL] Found {len(items)} {media_type} in node {node_id}")
                                
                                for item in items:
                                    filename = item.get('filename')
                                    subfolder = item.get('subfolder', '')
                                    file_type = item.get('type', 'output')
                                    
                                    print(f"[OUTPUT] Downloading: {filename} from {subfolder or 'root'}")
                                    
                                    try:
                                        data = self._get_file(filename, subfolder, file_type)
                                        if len(data) > 1024:
                                            print(f"[OUTPUT] Successfully got {len(data)/1024/1024:.2f} MB")
                                            
                                            # Final progress update
                                            self._update_progress(task.task_id, GenerationStatus.COMPLETED, 100.0, "Video ready for download")
                                            
                                            return data
                                        else:
                                            print(f"[OUTPUT] File too small, continuing search")
                                    except Exception as e:
                                        print(f"[OUTPUT] Failed to get file: {e}")
                    
                    print(f"[POLL] No valid output found, retrying...")
                    time.sleep(poll_interval)
                    
                except Exception as e:
                    print(f"[POLL] Error: {str(e)}")
                    consecutive_empty_results += 1
                    time.sleep(poll_interval)
            
            print(f"[ERROR] ===== POLLING TIMEOUT =====")
            print(f"[ERROR] Waited {poll_timeout}s across {poll_count} attempts")
            if last_history_state:
                print(f"[ERROR] Last known state:")
                print(json.dumps(last_history_state, indent=2))
            
            self._update_progress(task.task_id, GenerationStatus.TIMEOUT, 0.0, f"Polling timeout after {poll_timeout}s")
            raise ValueError(f"No video output found after {poll_timeout}s.")
            
        except Exception as e:
            self._update_progress(task.task_id, GenerationStatus.FAILED, 0.0, f"Error during generation: {str(e)}")
            raise

    def _copy_file_to_comfyui_input(self, base64_data: str, extension: str, data_type: str = "image") -> str:
        """Copy base64 data to ComfyUI input folder with proper cleanup"""
        # Create temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}")
        temp_file.close()
        
        try:
            _save_base64_to_file(base64_data, temp_file.name, data_type)
            
            # Copy to ComfyUI input
            input_dir = COMFYUI_PATH / "input"
            input_dir.mkdir(exist_ok=True)
            filename = f"{uuid.uuid4()}.{extension}"
            target_path = input_dir / filename
            
            shutil.copy2(temp_file.name, target_path)
            
            # Track for cleanup
            self.temp_files.append(temp_file.name)
            
            print(f"[FILE] Copied to ComfyUI input: {filename}")
            return filename
            
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)
            raise e

    # Workflow creation methods (unchanged)
    def _create_t2v_workflow(self, prompt: str, negative_prompt: str, width: int, height: int, 
                            num_frames: int, steps: int, cfg: float, seed: int, use_fast_mode: bool) -> dict:
        """Create T2V workflow"""
        if use_fast_mode:
            high_noise_model = "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
            low_noise_model = "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
            high_lora = "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors"
            low_lora = "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors"
            shift = 5.0
            steps = min(steps, 4)
        else:
            high_noise_model = "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
            low_noise_model = "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
            high_lora = None
            low_lora = None
            shift = 8.0
        
        workflow = {
            "71": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
            "73": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
            "72": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["71", 0]}},
            "89": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["71", 0]}},
            "74": {"class_type": "EmptyHunyuanLatentVideo", "inputs": {"width": width, "height": height, "length": num_frames, "batch_size": 1}},
        }
        
        if use_fast_mode:
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
        
        return workflow

    def _create_i2v_workflow(self, image_filename: str, prompt: str, negative_prompt: str, width: int, height: int, 
                            num_frames: int, steps: int, cfg: float, seed: int, use_fast_mode: bool) -> dict:
        """Create I2V workflow"""
        if use_fast_mode:
            high_noise_model = "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
            low_noise_model = "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
            high_lora = "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
            low_lora = "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"
            shift = 5.0
            steps = min(steps, 4)
        else:
            high_noise_model = "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
            low_noise_model = "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
            high_lora = None
            low_lora = None
            shift = 8.0
        
        workflow = {
            "38": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
            "39": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["38", 0]}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["38", 0]}},
            "62": {"class_type": "LoadImage", "inputs": {"image": image_filename}},
        }
        
        if use_fast_mode:
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
        
        return workflow

    def _create_animate_workflow(self, ref_image_filename: str, video_filename: str, prompt: str, negative_prompt: str, 
                                width: int, height: int, num_frames: int, steps: int, cfg: float, seed: int, use_fast_mode: bool) -> dict:
        """Create Animate workflow"""
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
        
        return workflow

    def _create_camera_lora_workflow(self, image_filename: str, prompt: str, negative_prompt: str, camera_motion: str, 
                                    lora_strength: float, width: int, height: int, num_frames: int, steps: int, cfg: float, seed: int) -> dict:
        """Create Camera LoRA workflow"""
        lora_name = f"v2_lora_{camera_motion}.safetensors"
        
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
        
        return workflow

    @method()
    def process_task(self, task_data: Dict):
        """Process a task from the queue"""
        task_id = task_data.get('task_id')
        task_type = task_data.get('type')
        webhook_url = task_data.get('webhook_url')
        
        print(f"[PROCESS] Container {self.container_id} processing task {task_id} ({task_type})")
        
        task = self._create_task(task_id, webhook_url, task_data)
        
        try:
            if task_type == 't2v':
                self._process_t2v(task, task_data)
            elif task_type == 'i2v':
                self._process_i2v(task, task_data)
            elif task_type == 'animate':
                self._process_animate(task, task_data)
            elif task_type == 'camera':
                self._process_camera(task, task_data)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
                
        except Exception as e:
            print(f"[ERROR] Task {task_id} failed: {e}")
            self._send_webhook_with_retry(webhook_url, task_id, "error", error_detail=str(e))
            queue_manager.update_task_status(task_id, GenerationStatus.FAILED, error=str(e))
        finally:
            self._remove_task(task_id)
            print(f"[PROCESS] Container {self.container_id} completed task {task_id}")

    def _process_t2v(self, task: GenerationTask, data: Dict):
        """Process T2V task"""
        prompt = data.get('prompt')
        negative_prompt = data.get('negative_prompt', '')
        width = data.get('width', 832)
        height = data.get('height', 480)
        num_frames = data.get('num_frames', 121)
        steps = data.get('steps', 30)
        cfg = data.get('cfg', 7.5)
        seed = data.get('seed') or int(time.time() * 1000000) % (2**32)
        use_fast_mode = data.get('use_fast_mode', False)
        
        workflow = self._create_t2v_workflow(
            prompt, negative_prompt, width, height, num_frames, steps, cfg, seed, use_fast_mode
        )
        
        self._queue_prompt(task, workflow)
        video_bytes = self._get_video_from_websocket(task, steps)
        video_base64 = base64.b64encode(video_bytes).decode('utf-8')
        
        self._send_webhook_with_retry(task.webhook_url, task.task_id, "success", video_base64=video_base64)
        queue_manager.update_task_status(task.task_id, GenerationStatus.COMPLETED)

    def _process_i2v(self, task: GenerationTask, data: Dict):
        """Process I2V task"""
        image_base64 = data.get('image_base64')
        prompt = data.get('prompt')
        negative_prompt = data.get('negative_prompt', '')
        width = data.get('width', 1280)
        height = data.get('height', 704)
        num_frames = data.get('num_frames', 81)
        steps = data.get('steps', 20)
        cfg = data.get('cfg', 3.5)
        seed = data.get('seed') or int(time.time() * 1000000) % (2**32)
        use_fast_mode = data.get('use_fast_mode', False)
        
        image_filename = self._copy_file_to_comfyui_input(image_base64, "png", "image")
        
        workflow = self._create_i2v_workflow(
            image_filename, prompt, negative_prompt, width, height, num_frames, steps, cfg, seed, use_fast_mode
        )
        
        self._queue_prompt(task, workflow)
        video_bytes = self._get_video_from_websocket(task, steps)
        video_base64 = base64.b64encode(video_bytes).decode('utf-8')
        
        self._send_webhook_with_retry(task.webhook_url, task.task_id, "success", video_base64=video_base64)
        queue_manager.update_task_status(task.task_id, GenerationStatus.COMPLETED)

    def _process_animate(self, task: GenerationTask, data: Dict):
        """Process Animate task"""
        reference_image_base64 = data.get('reference_image_base64')
        video_base64 = data.get('video_base64')
        prompt = data.get('prompt')
        negative_prompt = data.get('negative_prompt', '')
        width = data.get('width', 640)
        height = data.get('height', 640)
        num_frames = data.get('num_frames', 77)
        steps = data.get('steps', 6)
        cfg = data.get('cfg', 1.0)
        seed = data.get('seed') or int(time.time() * 1000000) % (2**32)
        use_fast_mode = data.get('use_fast_mode', True)
        
        ref_image_filename = self._copy_file_to_comfyui_input(reference_image_base64, "png", "image")
        video_filename = self._copy_file_to_comfyui_input(video_base64, "mp4", "video")
        
        workflow = self._create_animate_workflow(
            ref_image_filename, video_filename, prompt, negative_prompt, 
            width, height, num_frames, steps, cfg, seed, use_fast_mode
        )
        
        self._queue_prompt(task, workflow)
        video_bytes = self._get_video_from_websocket(task, steps)
        video_base64 = base64.b64encode(video_bytes).decode('utf-8')
        
        self._send_webhook_with_retry(task.webhook_url, task.task_id, "success", video_base64=video_base64)
        queue_manager.update_task_status(task.task_id, GenerationStatus.COMPLETED)

    def _process_camera(self, task: GenerationTask, data: Dict):
        """Process Camera LoRA task"""
        image_base64 = data.get('image_base64')
        prompt = data.get('prompt')
        camera_motion = data.get('camera_motion', 'ZoomIn')
        lora_strength = data.get('lora_strength', 1.0)
        negative_prompt = data.get('negative_prompt', '')
        width = data.get('width', 1280)
        height = data.get('height', 704)
        num_frames = data.get('num_frames', 81)
        steps = data.get('steps', 20)
        cfg = data.get('cfg', 3.5)
        seed = data.get('seed') or int(time.time() * 1000000) % (2**32)
        
        valid_motions = ["ZoomIn", "ZoomOut", "PanLeft", "PanRight", "TiltUp", "TiltDown", "RollingClockwise", "RollingAnticlockwise"]
        if camera_motion not in valid_motions:
            raise ValueError(f"Invalid camera motion. Must be one of: {valid_motions}")
        
        image_filename = self._copy_file_to_comfyui_input(image_base64, "png", "image")
        
        workflow = self._create_camera_lora_workflow(
            image_filename, prompt, negative_prompt, camera_motion, 
            lora_strength, width, height, num_frames, steps, cfg, seed
        )
        
        self._queue_prompt(task, workflow)
        video_bytes = self._get_video_from_websocket(task, steps)
        video_base64 = base64.b64encode(video_bytes).decode('utf-8')
        
        self._send_webhook_with_retry(task.webhook_url, task.task_id, "success", video_base64=video_base64)
        queue_manager.update_task_status(task.task_id, GenerationStatus.COMPLETED)

    @method()
    def cancel_generation(self, task_id: str) -> bool:
        """Cancel an active generation task"""
        task = self._get_task(task_id)
        if not task:
            return False
        
        task.cancel_event.set()
        task.status = GenerationStatus.CANCELLED
        self._update_progress(task_id, GenerationStatus.CANCELLED, 0.0, "Task cancelled")
        
        # Try to interrupt the prompt in ComfyUI
        if task.prompt_id:
            try:
                interrupt_url = f"http://127.0.0.1:{SERVER_PORT}/interrupt"
                requests.post(interrupt_url, timeout=5)
                print(f"[CANCEL] Sent interrupt for prompt {task.prompt_id}")
            except Exception as e:
                print(f"[CANCEL] Failed to interrupt prompt: {e}")
        
        return True

    @method()
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get status of an active task"""
        task = self._get_task(task_id)
        if not task:
            return None
        
        return {
            "task_id": task.task_id,
            "status": task.status.value,
            "progress": task.progress_data.progress if task.progress_data else 0.0,
            "message": task.progress_data.message if task.progress_data else "",
            "error": task.progress_data.error if task.progress_data else None,
            "start_time": task.progress_data.start_time if task.progress_data else None,
            "node_id": task.progress_data.node_id if task.progress_data else None,
            "container_id": self.container_id
        }

# Queue processor
@app.function(
    image=comfy_image,
    gpu="L40S",
    volumes={str(CACHE_PATH): cache_volume},
    timeout=7200,
    min_containers=1,
    max_containers=MAX_CONTAINERS,
    allow_concurrent_inputs=MAX_CONCURRENT_REQUESTS,
    container_idle_timeout=300,

)
class QueueProcessor:
    def __init__(self):
        self.comfyui = ComfyUI()
        self.running = True
    
    @enter()
    def startup(self):
        """Start the queue processor"""
        print("[PROCESSOR] Queue processor started")
        
        # Start processing loop in background
        threading.Thread(target=self._process_loop, daemon=True).start()
    
    def _process_loop(self):
        """Main processing loop"""
        while self.running:
            try:
                # Get next task from queue
                task_data = queue_manager.dequeue_task()
                if task_data:
                    # Process task
                    self.comfyui.process_task(task_data)
                else:
                    # No tasks, wait a bit
                    time.sleep(QUEUE_CHECK_INTERVAL)
            except Exception as e:
                print(f"[PROCESSOR] Error in processing loop: {e}")
                time.sleep(QUEUE_CHECK_INTERVAL)

# Create global queue processor
queue_processor = QueueProcessor()

# FastAPI Routes
@fastapi_app.post("/api/generate/t2v")
async def api_generate_t2v(request: T2VRequest):
    """API endpoint for text-to-video generation"""
    try:
        # Prepare task data
        task_data = {
            'type': 't2v',
            'task_id': request.task_id or str(uuid.uuid4()),
            'prompt': request.prompt,
            'negative_prompt': request.negative_prompt,
            'width': request.width,
            'height': request.height,
            'num_frames': request.num_frames,
            'steps': request.steps,
            'cfg': request.cfg,
            'seed': request.seed,
            'use_fast_mode': request.use_fast_mode,
            'webhook_url': request.webhook_url
        }
        
        # Add to queue
        task_id = queue_manager.enqueue_task(task_data)
        
        return {
            "status": "queued", 
            "message": "T2V generation queued",
            "task_id": task_id,
            "queue_position": queue_manager.get_queue_status()['queue_length']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@fastapi_app.post("/api/generate/i2v")
async def api_generate_i2v(request: I2VRequest):
    """API endpoint for image-to-video generation"""
    try:
        # Prepare task data
        task_data = {
            'type': 'i2v',
            'task_id': request.task_id or str(uuid.uuid4()),
            'image_base64': request.image_base64,
            'prompt': request.prompt,
            'negative_prompt': request.negative_prompt,
            'width': request.width,
            'height': request.height,
            'num_frames': request.num_frames,
            'steps': request.steps,
            'cfg': request.cfg,
            'seed': request.seed,
            'use_fast_mode': request.use_fast_mode,
            'webhook_url': request.webhook_url
        }
        
        # Add to queue
        task_id = queue_manager.enqueue_task(task_data)
        
        return {
            "status": "queued", 
            "message": "I2V generation queued",
            "task_id": task_id,
            "queue_position": queue_manager.get_queue_status()['queue_length']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@fastapi_app.post("/api/generate/animate")
async def api_generate_animate(request: AnimateRequest):
    """API endpoint for character animation"""
    try:
        # Prepare task data
        task_data = {
            'type': 'animate',
            'task_id': request.task_id or str(uuid.uuid4()),
            'reference_image_base64': request.reference_image_base64,
            'video_base64': request.video_base64,
            'prompt': request.prompt,
            'negative_prompt': request.negative_prompt,
            'width': request.width,
            'height': request.height,
            'num_frames': request.num_frames,
            'steps': request.steps,
            'cfg': request.cfg,
            'seed': request.seed,
            'use_fast_mode': request.use_fast_mode,
            'webhook_url': request.webhook_url
        }
        
        # Add to queue
        task_id = queue_manager.enqueue_task(task_data)
        
        return {
            "status": "queued", 
            "message": "Animation generation queued",
            "task_id": task_id,
            "queue_position": queue_manager.get_queue_status()['queue_length']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@fastapi_app.post("/api/generate/camera-lora")
async def api_apply_camera_lora(request: CameraLoraRequest):
    """API endpoint for camera motion LoRA"""
    try:
        # Prepare task data
        task_data = {
            'type': 'camera',
            'task_id': request.task_id or str(uuid.uuid4()),
            'image_base64': request.image_base64,
            'prompt': request.prompt,
            'camera_motion': request.camera_motion,
            'lora_strength': request.lora_strength,
            'negative_prompt': request.negative_prompt,
            'width': request.width,
            'height': request.height,
            'num_frames': request.num_frames,
            'steps': request.steps,
            'cfg': request.cfg,
            'seed': request.seed,
            'webhook_url': request.webhook_url
        }
        
        # Add to queue
        task_id = queue_manager.enqueue_task(task_data)
        
        return {
            "status": "queued", 
            "message": "Camera LoRA application queued",
            "task_id": task_id,
            "queue_position": queue_manager.get_queue_status()['queue_length']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@fastapi_app.post("/api/cancel")
async def api_cancel_generation(request: CancelRequest):
    """API endpoint to cancel a generation task"""
    try:
        success = queue_manager.cancel_task(request.task_id)
        if success:
            return {"status": "success", "message": f"Task {request.task_id} cancelled"}
        else:
            return {"status": "error", "message": f"Task {request.task_id} not found or already completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@fastapi_app.get("/api/status/{task_id}")
async def api_get_task_status(task_id: str):
    """API endpoint to get task status"""
    try:
        # Check Redis for task status
        redis = get_redis_client()
        status_data = redis.hget(REDIS_STATUS_KEY, task_id)
        
        if status_data:
            status = json.loads(status_data)
            return {
                "task_id": task_id,
                "status": status.get('status'),
                "timestamp": status.get('timestamp'),
                "queue_time": status.get('queue_time'),
                "start_time": status.get('start_time')
            }
        
        # Check progress data
        progress_data = redis.hget(REDIS_PROGRESS_KEY, task_id)
        if progress_data:
            return json.loads(progress_data)
        
        raise HTTPException(status_code=404, detail="Task not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@fastapi_app.get("/api/queue/status", response_model=QueueStatusResponse)
async def api_get_queue_status():
    """API endpoint to get queue status"""
    try:
        status = queue_manager.get_queue_status()
        return QueueStatusResponse(**status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@fastapi_app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ComfyUI Wan2.2 Production API",
        "version": "2.0.0",
        "max_concurrent": MAX_CONCURRENT_REQUESTS,
        "max_containers": MAX_CONTAINERS
    }

@fastapi_app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if ComfyUI is running
        urllib.request.urlopen(f"http://127.0.0.1:{SERVER_PORT}/queue", timeout=5).read()
        
        # Get queue status
        queue_status = queue_manager.get_queue_status()
        
        return {
            "status": "healthy",
            "comfyui": "running",
            "queue": queue_status
        }
    except:
        return {
            "status": "unhealthy",
            "comfyui": "not_running"
        }

# Modal ASGI App
# Ganti seluruh bagian akhir file dengan ini

@app.function()
@asgi_app()
def entrypoint():
    """Entry point for Modal ASGI app"""
    # Langsung jalankan FastAPI app
    return fastapi_app
