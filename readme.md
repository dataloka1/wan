# WAN 2.2 Complete Video API

Aplikasi production-ready untuk generasi video menggunakan semua model Wan 2.2 (14B) yang di-deploy di Modal. Mendukung 4 jenis model video generation dengan 100+ LoRAs, ControlNet, dan audio encoder.

## 🎯 Fitur Utama

- **4 Model Video Generation:**
  - **T2V (Text-to-Video)**: Buat video dari prompt teks
  - **I2V (Image-to-Video)**: Animasikan gambar menjadi video
  - **S2V (Story-to-Video)**: Buat video multi-scene dari cerita
  - **Animate**: Animasi advanced dengan kontrol motion

- **100+ LoRAs** dalam 9 kategori:
  - Artistic & Visual Styles
  - Concepts & Objects
  - Character & Clothing
  - Pose & Composition
  - Detail Enhancement
  - Environment
  - Lighting & Atmosphere
  - Utilities
  - Camera Motion

- **Spesifikasi Output:**
  - Resolusi: 1280x720 (HD)
  - Frame Rate: 24 FPS (customizable)
  - Durasi: 30 detik (customizable)
  - Format: MP4 (H.264)

## 📋 Requirements

- Akun [Modal](https://modal.com)
- Akun [Hugging Face](https://huggingface.co) dengan token API
- Python 3.10+
- Modal CLI

## 🚀 Instalasi & Setup

### 1. Install Modal CLI

```bash
pip install modal
```

### 2. Login ke Modal

```bash
modal token new
```

Ikuti instruksi di browser untuk autentikasi.

### 3. Setup Hugging Face Token

Buat secret untuk Hugging Face token:

```bash
modal secret create huggingface-token HUGGING_FACE_HUB_TOKEN=hf_xxxxxxxxxxxxxx
```

Dapatkan token dari: https://huggingface.co/settings/tokens

### 4. Download Models

Download semua model yang diperlukan (proses ini memakan waktu, ~60GB+):

```bash
modal run app.py::download_models
```

**Output yang diharapkan:**
```json
{
  "success": true,
  "summary": {
    "downloaded": 108,
    "cached": 0,
    "failed": 0,
    "total_gb": 62.45
  }
}
```

### 5. Deploy API

```bash
modal deploy app.py
```

Setelah deploy berhasil, Modal akan memberikan URL API Anda:
```
✓ Created objects.
├── 🔨 Created mount /app/app.py
├── 🔨 Created function download_models.
├── 🔨 Created class ComfyUI.
└── 🔨 Created function fastapi_app => https://your-username--comfyui-wan2-2-complete-api-fastapi-app.modal.run
```

## 📚 API Documentation

### Base URL

```
https://your-username--comfyui-wan2-2-complete-api-fastapi-app.modal.run
```

### Endpoints

#### 1. Root - Informasi API

```http
GET /
```

**Response:**
```json
{
  "service": "Wan 2.2 Complete Video API",
  "version": "4.0.0",
  "status": "production",
  "models": {
    "t2v": "Text-to-Video (14B)",
    "i2v": "Image-to-Video (14B)",
    "s2v": "Story-to-Video (14B)",
    "animate": "Animation (14B)"
  },
  "features": [
    "All 4 Wan 2.2 models",
    "100 Loras with 10 parallel workers",
    "Audio encoder support",
    "30s HD videos (1280x720 @ 24fps)",
    "ControlNet support",
    "Symlink-based model loading"
  ],
  "endpoints": {
    "generate": "POST /generate",
    "health": "GET /health",
    "loras": "GET /loras",
    "models": "GET /models"
  }
}
```

#### 2. Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": 4,
  "loras_available": 100
}
```

#### 3. List Available LoRAs

```http
GET /loras
```

**Response:**
```json
{
  "total_loras": 100,
  "categories": {
    "artistic_visual": {
      "count": 15,
      "loras": ["studio_ghibli.safetensors", "anime_lineart.safetensors", ...]
    },
    "camera_motion": {
      "count": 5,
      "loras": ["v2_lora_ZoomIn.safetensors", "v2_lora_PanLeft.safetensors", ...]
    }
  }
}
```

#### 4. List Available Models

```http
GET /models
```

**Response:**
```json
{
  "models": [
    {
      "name": "t2v",
      "full_name": "Wan 2.2 Text-to-Video (14B)",
      "description": "Generate videos from text prompts",
      "status": "Ready",
      "required_params": ["prompt"],
      "optional_params": ["negative_prompt", "steps", "cfg", "width", "height", "fps", "duration", "seed"],
      "example": {
        "model_type": "t2v",
        "prompt": "A cinematic shot of sunset over mountains",
        "duration": 30,
        "fps": 24
      }
    }
  ]
}
```

#### 5. Generate Video

```http
POST /generate
```

**Content-Type:** `application/json`

## 🎬 Cara Penggunaan API

### Model 1: Text-to-Video (T2V)

Generate video dari prompt teks.

**Required Parameters:**
- `model_type`: "t2v"
- `prompt`: Deskripsi video yang ingin dibuat

**Optional Parameters:**
- `negative_prompt`: Hal yang ingin dihindari (default: "low quality, blurry")
- `steps`: Jumlah sampling steps (default: 80)
- `cfg`: Classifier-free guidance scale (default: 7.5)
- `width`: Lebar video dalam pixel (default: 1280)
- `height`: Tinggi video dalam pixel (default: 720)
- `fps`: Frame per second (default: 24)
- `duration`: Durasi video dalam detik (default: 30)
- `seed`: Random seed untuk reproducibility (default: 123)

**Contoh Request dengan cURL:**

```bash
curl -X POST "https://your-url.modal.run/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "t2v",
    "prompt": "A majestic dragon flying over a medieval castle at sunset, cinematic lighting, epic fantasy",
    "negative_prompt": "low quality, blurry, distorted",
    "steps": 80,
    "cfg": 7.5,
    "width": 1280,
    "height": 720,
    "fps": 24,
    "duration": 30,
    "seed": 42
  }' \
  --output video_t2v.mp4
```

**Contoh Request dengan Python:**

```python
import requests
import json

url = "https://your-url.modal.run/generate"

payload = {
    "model_type": "t2v",
    "prompt": "A serene lake surrounded by mountains, morning mist, peaceful atmosphere",
    "negative_prompt": "low quality, blurry",
    "steps": 80,
    "cfg": 7.5,
    "width": 1280,
    "height": 720,
    "fps": 24,
    "duration": 30,
    "seed": 123
}

headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers)

if response.status_code == 200:
    with open("output_video.mp4", "wb") as f:
        f.write(response.content)
    print("Video berhasil dibuat!")
else:
    print(f"Error: {response.json()}")
```

**Contoh Request dengan JavaScript (Node.js):**

```javascript
const fs = require('fs');
const fetch = require('node-fetch');

const url = "https://your-url.modal.run/generate";

const payload = {
  model_type: "t2v",
  prompt: "A futuristic city with flying cars, neon lights, cyberpunk style",
  negative_prompt: "low quality, blurry",
  steps: 80,
  cfg: 7.5,
  width: 1280,
  height: 720,
  fps: 24,
  duration: 30,
  seed: 456
};

fetch(url, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(payload)
})
.then(response => response.buffer())
.then(buffer => {
  fs.writeFileSync('output_video.mp4', buffer);
  console.log('Video berhasil dibuat!');
})
.catch(error => console.error('Error:', error));
```

---

### Model 2: Image-to-Video (I2V)

Animasikan gambar statis menjadi video.

**Required Parameters:**
- `model_type`: "i2v"
- `image`: Gambar dalam format base64 (dengan atau tanpa prefix `data:image/...;base64,`)

**Optional Parameters:**
- `prompt`: Deskripsi gerakan/animasi (opsional)
- `negative_prompt`: Hal yang ingin dihindari (default: "low quality")
- `steps`: Jumlah sampling steps (default: 80)
- `cfg`: Classifier-free guidance scale (default: 7.5)
- `fps`: Frame per second (default: 24)
- `duration`: Durasi video dalam detik (default: 30)
- `seed`: Random seed (default: 123)

**Contoh Request dengan Python:**

```python
import requests
import base64

url = "https://your-url.modal.run/generate"

# Baca gambar dan convert ke base64
with open("input_image.png", "rb") as image_file:
    image_data = base64.b64encode(image_file.read()).decode('utf-8')

payload = {
    "model_type": "i2v",
    "image": f"data:image/png;base64,{image_data}",
    "prompt": "gentle camera movement, smooth animation",
    "negative_prompt": "low quality, distorted",
    "steps": 80,
    "cfg": 7.5,
    "fps": 24,
    "duration": 30,
    "seed": 789
}

response = requests.post(url, json=payload)

if response.status_code == 200:
    with open("output_i2v.mp4", "wb") as f:
        f.write(response.content)
    print("Video berhasil dibuat dari gambar!")
else:
    print(f"Error: {response.json()}")
```

**Contoh Request dengan cURL:**

```bash
# Pertama, convert gambar ke base64
BASE64_IMAGE=$(base64 -w 0 input_image.png)

curl -X POST "https://your-url.modal.run/generate" \
  -H "Content-Type: application/json" \
  -d "{
    \"model_type\": \"i2v\",
    \"image\": \"data:image/png;base64,$BASE64_IMAGE\",
    \"prompt\": \"smooth camera pan, cinematic movement\",
    \"fps\": 24,
    \"duration\": 30
  }" \
  --output video_i2v.mp4
```

---

### Model 3: Story-to-Video (S2V)

Buat video multi-scene dari cerita dengan beberapa scene.

**Required Parameters:**
- `model_type`: "s2v"
- `story`: Cerita dengan multiple scenes (pisahkan scene dengan newline `\n`)

**Optional Parameters:**
- `negative_prompt`: Hal yang ingin dihindari (default: "low quality, blurry")
- `steps`: Jumlah sampling steps (default: 80)
- `cfg`: Classifier-free guidance scale (default: 7.5)
- `width`: Lebar video (default: 1280)
- `height`: Tinggi video (default: 720)
- `fps`: Frame per second (default: 24)
- `duration`: Durasi video dalam detik (default: 30)
- `seed`: Random seed (default: 123)

**Contoh Request dengan Python:**

```python
import requests

url = "https://your-url.modal.run/generate"

story = """Scene 1: A brave knight stands at the castle gate, sword in hand.
Scene 2: The dragon emerges from the dark cave, breathing fire.
Scene 3: An epic battle begins between knight and dragon.
Scene 4: The knight defeats the dragon and saves the kingdom."""

payload = {
    "model_type": "s2v",
    "story": story,
    "negative_prompt": "low quality, blurry, distorted",
    "steps": 80,
    "cfg": 7.5,
    "width": 1280,
    "height": 720,
    "fps": 24,
    "duration": 30,
    "seed": 999
}

response = requests.post(url, json=payload)

if response.status_code == 200:
    with open("output_s2v.mp4", "wb") as f:
        f.write(response.content)
    print("Video story berhasil dibuat!")
else:
    print(f"Error: {response.json()}")
```

**Contoh Request dengan cURL:**

```bash
curl -X POST "https://your-url.modal.run/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "s2v",
    "story": "Scene 1: A spaceship lands on an alien planet.\nScene 2: Astronauts explore the strange landscape.\nScene 3: They discover ancient ruins.\nScene 4: Mysterious lights appear in the sky.",
    "steps": 80,
    "cfg": 7.5,
    "duration": 30
  }' \
  --output video_s2v.mp4
```

---

### Model 4: Animate

Model animasi advanced dengan kontrol motion. Bisa dengan atau tanpa input gambar.

**Required Parameters:**
- `model_type`: "animate"
- `prompt`: Deskripsi animasi yang ingin dibuat

**Optional Parameters:**
- `image`: Gambar input dalam base64 (opsional)
- `motion_type`: Jenis motion - "smooth", "dynamic", "zoom", "pan" (default: "smooth")
- `negative_prompt`: Hal yang ingin dihindari (default: "low quality")
- `steps`: Jumlah sampling steps (default: 80)
- `cfg`: Classifier-free guidance scale (default: 7.5)
- `width`: Lebar video (default: 1280)
- `height`: Tinggi video (default: 720)
- `fps`: Frame per second (default: 24)
- `duration`: Durasi video dalam detik (default: 30)
- `seed`: Random seed (default: 123)

**Contoh 1: Animate tanpa gambar input (Text-to-Animation)**

```python
import requests

url = "https://your-url.modal.run/generate"

payload = {
    "model_type": "animate",
    "prompt": "A magical fairy transforming with sparkles and glowing effects",
    "motion_type": "dynamic",
    "negative_prompt": "low quality, static",
    "steps": 80,
    "cfg": 7.5,
    "width": 1280,
    "height": 720,
    "fps": 24,
    "duration": 30,
    "seed": 555
}

response = requests.post(url, json=payload)

if response.status_code == 200:
    with open("output_animate.mp4", "wb") as f:
        f.write(response.content)
    print("Animasi berhasil dibuat!")
else:
    print(f"Error: {response.json()}")
```

**Contoh 2: Animate dengan gambar input (Image-to-Animation)**

```python
import requests
import base64

url = "https://your-url.modal.run/generate"

# Baca gambar
with open("character.png", "rb") as image_file:
    image_data = base64.b64encode(image_file.read()).decode('utf-8')

payload = {
    "model_type": "animate",
    "prompt": "Character running and jumping with dynamic motion",
    "image": f"data:image/png;base64,{image_data}",
    "motion_type": "dynamic",
    "negative_prompt": "static, low quality",
    "steps": 80,
    "cfg": 7.5,
    "fps": 24,
    "duration": 30,
    "seed": 777
}

response = requests.post(url, json=payload)

if response.status_code == 200:
    with open("output_animate_from_image.mp4", "wb") as f:
        f.write(response.content)
    print("Animasi dari gambar berhasil dibuat!")
else:
    print(f"Error: {response.json()}")
```

**Contoh dengan cURL:**

```bash
curl -X POST "https://your-url.modal.run/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "animate",
    "prompt": "Explosion with fire and smoke, dynamic camera shake",
    "motion_type": "dynamic",
    "steps": 80,
    "cfg": 7.5,
    "duration": 30
  }' \
  --output video_animate.mp4
```

---

## 🎨 Tips & Best Practices

### 1. Prompt Engineering

**Good Prompts:**
- Spesifik dan deskriptif
- Sertakan style/mood/lighting
- Gunakan istilah sinematik

**Contoh Prompt Bagus:**
```
"A majestic waterfall in a lush jungle, golden hour lighting, 
mist rising from the water, birds flying, cinematic composition, 
ultra detailed, 4k quality"
```

**Bad Prompts:**
```
"waterfall" (terlalu singkat)
"a video of something" (tidak spesifik)
```

### 2. Negative Prompts

Tambahkan hal-hal yang ingin dihindari:
```json
{
  "negative_prompt": "low quality, blurry, distorted, pixelated, bad anatomy, 
  watermark, text, signature, deformed"
}
```

### 3. Parameter Tuning

**Steps:**
- Low (20-40): Cepat tapi kualitas rendah
- Medium (50-80): Balance antara speed dan quality
- High (80-150): Kualitas terbaik tapi lambat

**CFG (Classifier-Free Guidance):**
- Low (3-5): Lebih creative, kurang sesuai prompt
- Medium (6-8): Balance
- High (9-15): Sangat sesuai prompt tapi bisa over-saturated

**Seed:**
- Gunakan seed yang sama untuk hasil konsisten
- Ubah seed untuk variasi

### 4. Durasi dan FPS

**Durasi:**
- Short (5-15s): Untuk GIF atau demo cepat
- Medium (20-30s): Standard, recommended
- Long (40-60s): Untuk video penuh, butuh waktu render lebih lama

**FPS:**
- 24 FPS: Cinematic (recommended)
- 30 FPS: Smooth motion
- 60 FPS: Ultra smooth (file size lebih besar)

### 5. Resolusi

**Preset Umum:**
- 1280x720 (HD): Balance antara quality dan speed
- 1920x1080 (Full HD): Kualitas tinggi, render lebih lama
- 854x480 (SD): Cepat, untuk draft/preview

### 6. Story-to-Video Tips

**Format Scene:**
```
Scene 1: [Deskripsi scene pertama dengan detail]
Scene 2: [Transisi dan scene kedua]
Scene 3: [Klimaks atau action]
Scene 4: [Ending atau resolution]
```

**Contoh Bagus:**
```
Scene 1: Wide shot of a peaceful village at dawn, smoke rising from chimneys.
Scene 2: Close-up of a hero preparing for journey, packing supplies.
Scene 3: The hero encounters a mystical creature in the forest.
Scene 4: Spectacular battle with magical effects and epic conclusion.
```

---

## 🛠️ Error Handling

### Common Errors

**1. Model Type Error**
```json
{
  "error": "Unsupported model_type: 'txt2vid'",
  "supported": ["t2v", "i2v", "s2v", "animate"]
}
```
**Solusi:** Gunakan model_type yang benar: "t2v", "i2v", "s2v", atau "animate"

**2. Missing Required Parameter**
```json
{
  "error": "'prompt' is required for T2V model"
}
```
**Solusi:** Pastikan semua parameter required ada dalam request

**3. Invalid Base64 Image**
```json
{
  "error": "Generation failed: Invalid image format"
}
```
**Solusi:** Pastikan gambar di-encode dengan benar ke base64

**4. Timeout Error**
```json
{
  "error": "Generation failed: Request timeout"
}
```
**Solusi:** Video generation membutuhkan waktu (~5-15 menit per video). Tingkatkan timeout di client Anda.

---

## 📊 Performance & Limits

### Waktu Generasi (Estimasi)

- **T2V**: 8-15 menit (tergantung steps & durasi)
- **I2V**: 10-18 menit
- **S2V**: 12-20 menit (tergantung jumlah scenes)
- **Animate**: 10-18 menit

### Resource Usage

- **GPU**: L40S (48GB VRAM)
- **Memory**: ~60GB untuk semua models
- **Storage**: ~62GB untuk models & cache

### Rate Limits

Modal secara default tidak membatasi request, namun:
- Hanya 1 generation per instance secara bersamaan
- Auto-scaling berdasarkan load
- Cold start: ~2-3 menit untuk warming up

---

## 🔧 Troubleshooting

### Issue: ComfyUI failed to start

**Symptoms:**
```
RuntimeError: ComfyUI failed to start after 60s
```

**Solutions:**
1. Check models sudah ter-download semua
2. Jalankan ulang: `modal run app.py::download_models`
3. Verify dengan: `modal volume ls comfyui-wan2-2-complete-volume`

### Issue: Critical models missing

**Symptoms:**
```
RuntimeError: Critical models missing! Run: modal run app.py::download_models
```

**Solutions:**
1. Pastikan Hugging Face token valid
2. Re-download models:
```bash
modal secret create huggingface-token HUGGING_FACE_HUB_TOKEN=hf_your_new_token
modal run app.py::download_models
```

### Issue: Video output is corrupted

**Possible Causes:**
- Steps terlalu rendah
- CFG terlalu tinggi/rendah
- Prompt conflict dengan negative prompt

**Solutions:**
- Gunakan steps minimal 50
- Set CFG antara 6-9
- Perbaiki prompt/negative prompt

---

## 📦 Example Projects

### Project 1: Batch Video Generator

```python
import requests
import json
import time

url = "https://your-url.modal.run/generate"

prompts = [
    "A serene beach at sunset",
    "A bustling futuristic city",
    "A magical forest with glowing mushrooms",
    "An underwater coral reef scene"
]

for i, prompt in enumerate(prompts):
    print(f"Generating video {i+1}/{len(prompts)}...")
    
    payload = {
        "model_type": "t2v",
        "prompt": prompt,
        "steps": 80,
        "duration": 30
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        filename = f"video_{i+1:03d}.mp4"
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"✓ Saved: {filename}")
    else:
        print(f"✗ Failed: {response.json()}")
    
    time.sleep(2)  # Rate limiting
```

### Project 2: Image Gallery to Video

```python
import requests
import base64
import os
import glob

url = "https://your-url.modal.run/generate"

# Get all images from folder
images = glob.glob("images/*.png") + glob.glob("images/*.jpg")

for img_path in images:
    print(f"Processing: {img_path}")
    
    with open(img_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    payload = {
        "model_type": "i2v",
        "image": f"data:image/png;base64,{img_base64}",
        "prompt": "smooth cinematic movement",
        "duration": 15,
        "fps": 30
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        output_name = os.path.basename(img_path).replace('.png', '.mp4').replace('.jpg', '.mp4')
        with open(f"videos/{output_name}", "wb") as f:
            f.write(response.content)
        print(f"✓ Created: videos/{output_name}")
```

### Project 3: Story Visualization

```python
import requests

url = "https://your-url.modal.run/generate"

# Multi-part story
stories = {
    "chapter1": """
Scene 1: A lone astronaut lands on Mars, red planet surface stretching endlessly.
Scene 2: Discovering ancient alien structures buried in the sand.
Scene 3: Inside the ruins, mysterious technology begins to activate.
Scene 4: A portal opens, revealing glimpses of another dimension.
""",
    "chapter2": """
Scene 1: The astronaut steps through the portal into a bioluminescent world.
Scene 2: Strange alien creatures observe from a distance, curious but cautious.
Scene 3: Communication begins through gestures and holographic projections.
Scene 4: A friendship forms across species and dimensions.
"""
}

for chapter, story in stories.items():
    print(f"Generating {chapter}...")
    
    payload = {
        "model_type": "s2v",
        "story": story,
        "steps": 100,
        "duration": 40,
        "seed": hash(chapter) % 10000
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        with open(f"{chapter}.mp4", "wb") as f:
            f.write(response.content)
        print(f"✓ Saved: {chapter}.mp4")
```

---

## 🚀 Advanced Usage

### Custom Workflow Integration

Jika Anda ingin memodifikasi workflow ComfyUI:

1. Edit fungsi `get_wan2_2_*_workflow()` di `app.py`
2. Tambahkan node baru atau ubah parameter
3. Re-deploy: `modal deploy app.py`

### Using LoRAs (Future Implementation)

LoRAs sudah ter-download namun belum terintegrasi dalam workflow saat ini. Untuk menggunakan:

1. Modifikasi workflow untuk include LoRA loader node
2. Tambahkan parameter `loras` di payload
3. Update workflow generation function

---

## 📝 API Response Examples

### Successful Response

```
HTTP/1.1 200 OK
Content-Type: video/mp4
Content-Disposition: attachment; filename=wan2_2_t2v_1234567890.mp4
Content-Length: 45678901

[Binary MP4 data]
```

### Error Responses

**400 Bad Request:**
```json
{
  "error": "'prompt' is required for T2V model"
}
```

**500 Internal Server Error:**
```json
{
  "error": "Generation failed: CUDA out of memory"
}
```

---

## 🔐 Security Best Practices

1. **Jangan share Modal URL publik** - Gunakan API Gateway atau authentication
2. **Rate limiting** - Implement di application layer
3. **Input validation** - Validate semua input di client side
4. **Cost monitoring** - Monitor Modal usage dashboard
5. **Token security** - Jangan commit HF token ke git

---

## 💰 Cost Estimation

Modal pricing (approximate):
- **GPU L40S**: ~$1.50/hour
- **Storage**: ~$0.15/GB/month
- **Network egress**: $0.10/GB

Typical costs per video:
- **Short generation (5-10 min)**: $0.15 - $0.25
- **Standard generation (15 min)**: $0.35 - $0.50
- **Long generation (30 min)**: $0.70 - $1.00

Monthly estimate (100 videos):
- **Compute**: $30-50
- **Storage**: $9 (for 60GB models)
- **Total**: ~$40-60/month

---

## 🆘 Support & Community

- **Modal Documentation**: https://modal.com/docs
- **ComfyUI GitHub**: https://github.com/comfyanonymous/ComfyUI
- **Wan Video**: https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged

---

## 📜 License

This project uses:
- Modal (Commercial license required for production)
- ComfyUI (GPL-3.0)
- Wan 2.2 Models (Check individual model licenses)

---

## 🎉 Changelog

### v4.0.0 (Current)
- ✅ All 4 Wan 2.2 models (T2V, I2V, S2V, Animate)
- ✅ 100+ LoRAs downloaded
- ✅ Audio encoder support
- ✅ ControlNet models
- ✅ Symlink-based efficient storage
- ✅ Parallel LoRA downloads (10 workers)
- ✅ Complete REST API
- ✅ FastAPI + CORS support

---

## 🤝 Contributing

Contributions
