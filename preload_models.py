"""
Script untuk pre-download models ke Modal Volume
Usage: modal run preload_models.py
"""

import modal
from pathlib import Path

app = modal.App("preload-wan-models")

# Volume yang sama dengan app utama
models_volume = modal.Volume.from_name("wan-models-vol", create_if_missing=True)

# Image dengan dependencies untuk download
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface-hub>=0.19.0",
        "requests>=2.31.0",
    )
)


@app.function(
    image=image,
    volumes={"/models": models_volume},
    timeout=7200,  # 2 jam untuk download
    secrets=[modal.Secret.from_name("huggingface-secret", required=False)],
)
def download_all_models():
    """Download semua models yang diperlukan"""
    from huggingface_hub import hf_hub_download, HfFolder
    import os
    
    # Check HF token
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        HfFolder.save_token(hf_token)
        print("✓ HuggingFace token loaded")
    else:
        print("⚠️  No HF token found (some models may fail)")
    
    MODELS_PATH = "/models"
    
    # Definisi models yang akan didownload
    models_config = {
        "checkpoints": [
            {
                "repo": "stabilityai/stable-video-diffusion-img2vid-xt",
                "filename": "svd_xt.safetensors",
                "subfolder": None,
            },
        ],
        "animatediff_models": [
            {
                "repo": "guoyww/animatediff",
                "filename": "mm_sd_v15_v2.ckpt",
                "subfolder": None,
            },
        ],
        "controlnet": [
            {
                "repo": "lllyasviel/ControlNet-v1-1",
                "filename": "control_v11p_sd15_canny.pth",
                "subfolder": None,
            },
            {
                "repo": "lllyasviel/ControlNet-v1-1",
                "filename": "control_v11f1p_sd15_depth.pth",
                "subfolder": None,
            },
        ],
        "vae": [
            {
                "repo": "stabilityai/sd-vae-ft-mse-original",
                "filename": "vae-ft-mse-840000-ema-pruned.safetensors",
                "subfolder": None,
            },
        ],
    }
    
    print("=" * 60)
    print("📥 Starting model downloads to Modal Volume")
    print("=" * 60)
    
    total_downloaded = 0
    total_skipped = 0
    
    # Download setiap kategori
    for category, models in models_config.items():
        category_path = Path(MODELS_PATH) / category
        category_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📂 Category: {category}")
        print("-" * 60)
        
        for model_info in models:
            filename = model_info["filename"]
            target_path = category_path / filename
            
            # Check if already exists
            if target_path.exists():
                size_gb = target_path.stat().st_size / 1e9
                print(f"✓ {filename} (already exists, {size_gb:.2f} GB)")
                total_skipped += 1
                continue
            
            print(f"⬇️  Downloading: {filename}")
            print(f"    from {model_info['repo']}")
            
            try:
                # Download ke cache dulu
                downloaded_path = hf_hub_download(
                    repo_id=model_info["repo"],
                    filename=model_info["filename"],
                    subfolder=model_info.get("subfolder"),
                    cache_dir="/tmp/hf_cache",
                    resume_download=True,
                    local_files_only=False,
                )
                
                # Copy ke volume
                import shutil
                shutil.copy2(downloaded_path, target_path)
                
                size_gb = target_path.stat().st_size / 1e9
                print(f"✅ Downloaded: {filename} ({size_gb:.2f} GB)")
                total_downloaded += 1
                
            except Exception as e:
                print(f"❌ Failed to download {filename}")
                print(f"    Error: {str(e)}")
                continue
    
    # Commit volume
    print("\n" + "=" * 60)
    print("💾 Committing changes to volume...")
    models_volume.commit()
    
    # Summary
    print("=" * 60)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"✅ Downloaded: {total_downloaded} models")
    print(f"⏭️  Skipped: {total_skipped} models (already exist)")
    
    # Calculate total size
    total_size = 0
    file_count = 0
    for category in models_config.keys():
        category_path = Path(MODELS_PATH) / category
        if category_path.exists():
            for f in category_path.rglob("*"):
                if f.is_file():
                    total_size += f.stat().st_size
                    file_count += 1
    
    print(f"💾 Total size: {total_size / 1e9:.2f} GB")
    print(f"📁 Total files: {file_count}")
    print("=" * 60)
    print("✅ Models ready for use!")
    print("=" * 60)


@app.function(
    image=image,
    volumes={"/models": models_volume},
)
def list_models():
    """List semua models yang ada di volume"""
    import os
    from pathlib import Path
    
    MODELS_PATH = "/models"
    
    print("=" * 60)
    print("📋 MODELS IN VOLUME")
    print("=" * 60)
    
    categories = ["checkpoints", "animatediff_models", "controlnet", "vae", "loras"]
    
    total_size = 0
    total_files = 0
    
    for category in categories:
        category_path = Path(MODELS_PATH) / category
        
        if not category_path.exists():
            print(f"\n📂 {category}/")
            print("   (empty)")
            continue
        
        files = list(category_path.glob("*"))
        
        if not files:
            print(f"\n📂 {category}/")
            print("   (empty)")
            continue
        
        print(f"\n📂 {category}/")
        
        for file in sorted(files):
            if file.is_file():
                size_gb = file.stat().st_size / 1e9
                print(f"   ✓ {file.name} ({size_gb:.2f} GB)")
                total_size += file.stat().st_size
                total_files += 1
    
    print("\n" + "=" * 60)
    print(f"💾 Total: {total_files} files, {total_size / 1e9:.2f} GB")
    print("=" * 60)


@app.function(
    image=image,
    volumes={"/models": models_volume},
)
def clear_volume():
    """DANGER: Hapus semua models dari volume"""
    import shutil
    from pathlib import Path
    
    MODELS_PATH = "/models"
    
    print("⚠️  WARNING: This will delete ALL models!")
    print("=" * 60)
    
    categories = ["checkpoints", "animatediff_models", "controlnet", "vae", "loras"]
    
    for category in categories:
        category_path = Path(MODELS_PATH) / category
        if category_path.exists():
            print(f"🗑️  Deleting {category}/")
            shutil.rmtree(category_path)
    
    models_volume.commit()
    print("=" * 60)
    print("✅ Volume cleared!")


@app.local_entrypoint()
def main(action: str = "download"):
    """
    Main entrypoint
    
    Actions:
        download - Download all models (default)
        list - List models in volume
        clear - Clear all models (DANGER!)
    """
    if action == "download":
        print("🚀 Starting download process...")
        download_all_models.remote()
        print("\n✅ Done! Models are now in volume 'wan-models-vol'")
        
    elif action == "list":
        list_models.remote()
        
    elif action == "clear":
        response = input("⚠️  Are you sure? Type 'yes' to confirm: ")
        if response.lower() == "yes":
            clear_volume.remote()
        else:
            print("❌ Cancelled")
            
    else:
        print(f"❌ Unknown action: {action}")
        print("Available: download, list, clear")


# Quick commands untuk CLI
@app.function(image=image, volumes={"/models": models_volume})
def check_storage():
    """Check storage usage"""
    import subprocess
    result = subprocess.run(["du", "-sh", "/models"], capture_output=True, text=True)
    print(result.stdout)
    
    # List by category
    categories = ["checkpoints", "animatediff_models", "controlnet", "vae"]
    for cat in categories:
        result = subprocess.run(
            ["du", "-sh", f"/models/{cat}"], 
            capture_output=True, 
            text=True
        )
        print(f"{cat}: {result.stdout.strip()}")
