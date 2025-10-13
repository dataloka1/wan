# 🎨 LoRA Factory - Production Ready

**AI-Powered LoRA Training Platform with Smart Image Collection & Manual Review**

[![Modal](https://img.shields.io/badge/Modal-Deploy-blue)](https://modal.com)
[![Python](https://img.shields.io/badge/Python-3.11-green)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## ✨ Features

### 🤖 AI-Powered Intelligence
- **Llama 3.2 3B Instruct** - Smart query generation & caption enhancement
- **BLIP2** - Automatic image captioning
- **Smart Fallback** - Works without LLM using optimized queries

### 🌐 Multi-Source Scraping
- **6 Sources Simultaneously**: DuckDuckGo, Pixabay, Pexels, Unsplash, Flickr, Wikimedia
- **Parallel Processing** - Fast collection with ThreadPoolExecutor
- **Deduplication** - MD5 hash-based duplicate removal

### 🖼️ Manual Review System
- **Interactive Gallery** - Approve/reject images before training
- **Thumbnail Preview** - Fast review with lazy loading
- **Batch Operations** - Approve all / Reject all
- **Statistics** - Real-time approval counter

### 🔄 Incremental Training
- **Continue Training** - Add new concepts to existing LoRAs
- **Checkpoint Management** - Resume from any epoch
- **Version Control** - Track training history

### 🎨 Modern UI
- **Responsive Design** - Works on desktop, tablet, mobile
- **Real-time Updates** - Live task progress monitoring
- **Dark Theme** - Easy on the eyes
- **Alpine.js** - Reactive without build step

### 🔒 Production Security
- **Input Sanitization** - XSS & injection protection
- **Path Traversal Prevention** - Secure file access
- **SSRF Protection** - IP range blocking
- **File Validation** - Magic number verification

---

## 🚀 Quick Start

### Prerequisites

1. **Modal Account** - [Sign up free](https://modal.com)
2. **Hugging Face Token** - [Get token](https://huggingface.co/settings/tokens)

### Installation

```bash
# Install Modal
pip install modal

# Authenticate
modal token new

# Add HF Token as secret
modal secret create huggingface-token HF_TOKEN=your_token_here
```

### Deployment

```bash
# Deploy to Modal
modal deploy lora_factory_production.py

# The app will:
# 1. Download Llama 3.2 3B (~6GB)
# 2. Download SDXL Base (~7GB)
# 3. Download Juggernaut XL (~7GB)
# 4. Start web server

# Access URL will be shown in terminal
```

---

## 📖 Usage Guide

### 1. Basic Training

```
Train Taylor Swift with 500 images
```

**What happens:**
1. AI generates 10 diverse search queries
2. Scrapes 6 sources simultaneously
3. Downloads & validates images
4. Shows review gallery
5. You approve best images
6. AI captions each image
7. Trains LoRA (10 epochs)
8. Generates test images

### 2. Multiple Concepts

```
Train anime character with spiky hair and blue eyes, 300 images each
```

Creates a LoRA with 2 concepts, 300 images per concept.

### 3. Continue Training

```
Continue training abc123de with red sneakers concept
```

Adds new concept to existing LoRA without retraining everything.

### 4. Advanced

```
Create LoRA for John Doe:
- Concept 1: Professional headshot, 400 images
- Concept 2: Casual outfit, 300 images  
- Concept 3: Smiling portrait, 200 images
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER INPUT                            │
│              "Train Taylor Swift with 500 images"           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    AI AGENT (Llama 3.2)                     │
│  • Parse request                                            │
│  • Generate search queries                                  │
│  • Enhance captions                                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              MULTI-SOURCE SCRAPER (6 sources)               │
│  DuckDuckGo │ Pixabay │ Pexels │ Unsplash │ Flickr │ Wiki  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  IMAGE DOWNLOADER                           │
│  • Parallel download (15 workers)                           │
│  • Validation (size, format, content)                       │
│  • Deduplication (MD5 hash)                                 │
│  • Thumbnail generation                                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   MANUAL REVIEW                             │
│  • User approves/rejects images                             │
│  • Minimum 10 images required                               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  PREPROCESSING                              │
│  • BLIP2 captioning                                         │
│  • AI caption enhancement                                   │
│  • Resize to 1024x1024                                      │
│  • Save as .jpg + .txt pairs                                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│               KOHYA TRAINING (A100 GPU)                     │
│  • SDXL LoRA training                                       │
│  • Network dim: 128, alpha: 64                              │
│  • 10 epochs, AdamW8bit                                     │
│  • Checkpoint every 2 epochs                                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              TEST GENERATION (A100 GPU)                     │
│  • Load Juggernaut XL + trained LoRA                        │
│  • Generate 6 test images                                   │
│  • Save to library                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Configuration

### Training Parameters

```python
RESOLUTION = 1024          # Training resolution
BATCH_SIZE = 1             # Batch size
EPOCHS = 10                # Training epochs
LEARNING_RATE = 1e-4       # Learning rate
NETWORK_DIM = 128          # LoRA dimension
NETWORK_ALPHA = 64         # LoRA alpha
```

### Image Collection

```python
MAX_IMAGE_COUNT = 1000     # Max images per concept
MIN_IMAGE_COUNT = 10       # Min images to start training
MIN_IMAGE_SIZE = 512       # Min image dimension
THUMBNAIL_SIZE = 300       # Thumbnail size
```

### GPU Requirements

| Function | GPU | Memory | Duration |
|----------|-----|--------|----------|
| LLM Download | A10G | 24GB | 10 min |
| Image Scraping | A10G | 24GB | 30 min |
| Preprocessing | A10G | 24GB | 20 min |
| Training | A100 | 40GB | 60-90 min |
| Testing | A100 | 40GB | 10 min |

---

## 🐛 Bug Fixes from Original

### ✅ Fixed Issues

1. **LLM Model**
   - ❌ GPT-OSS (not available, complex setup)
   - ✅ Llama 3.2 3B Instruct (free, easy, lightweight)

2. **Model Loading**
   - ❌ vLLM with Harmony (overcomplicated)
   - ✅ Transformers pipeline (standard, reliable)

3. **UI/UX**
   - ❌ Inline CSS, basic styling
   - ✅ Tailwind + Alpine.js (modern, reactive)

4. **Error Handling**
   - ❌ Basic try-catch
   - ✅ Comprehensive error handling with fallbacks

5. **Security**
   - ❌ Basic validation
   - ✅ Multi-layer security (sanitization, SSRF protection, path traversal)

6. **Performance**
   - ❌ Sequential operations
   - ✅ Parallel processing with ThreadPoolExecutor

---

## 📊 Model Comparison

| Model | Size | Purpose | Status |
|-------|------|---------|--------|
| Llama 3.2 3B | 6GB | Query generation, captions | **Optional** |
| BLIP2 2.7B | 5GB | Image captioning | **Required** |
| SDXL Base | 7GB | LoRA training base | **Required** |
| Juggernaut XL | 7GB | Test generation | **Required** |

**Total: ~25GB** (without LLM: ~19GB)

---

## 🔧 Troubleshooting

### Models Not Downloading

```bash
# Check Modal logs
modal app logs lora-factory-production

# Manually trigger download
modal run lora_factory_production.py::download_llm_model
modal run lora_factory_production.py::download_base_models
```

### Out of Memory

```python
# Reduce batch size in Config
BATCH_SIZE = 1  # Already minimal

# Or reduce resolution
RESOLUTION = 768  # Instead of 1024
```

### Scraping Returns Few Images

1. Check if sites are accessible from your region
2. LLM might help generate better queries
3. Try more specific concept descriptions

### Training Fails

1. Verify base model downloaded: `/models/sd_xl_base_1.0.safetensors`
2. Check dataset has enough images (min 10)
3. Review Modal logs for specific errors

---

## 🚢 Production Deployment

### Environment Variables

```bash
# Required
export MODAL_TOKEN_ID=your_token_id
export MODAL_TOKEN_SECRET=your_token_secret

# Optional (for custom model paths)
export HF_HOME=/custom/path
```

### Monitoring

```bash
# Watch logs in real-time
modal app logs lora-factory-production --follow

# Check function stats
modal app list

# View volume usage
modal volume list
```

### Cost Optimization

```python
# Use smaller GPU for non-training tasks
@app.function(gpu="T4")  # Instead of A10G
def scrape_images(...):
    pass

# Reduce timeout for quick operations
@app.function(timeout=300)  # 5 minutes
def check_models_status(...):
    pass
```

---

## 🎯 Roadmap

- [ ] Support for SD 1.5 LoRA training
- [ ] Flux.1 LoRA support
- [ ] Multi-user authentication
- [ ] Custom training parameters UI
- [ ] Automatic hyperparameter tuning
- [ ] Dataset augmentation options
- [ ] Training analytics dashboard
- [ ] API endpoints for programmatic access

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file

---

## 🙏 Credits

- **Kohya SS** - LoRA training scripts
- **Modal** - Serverless GPU platform
- **Hugging Face** - Model hosting
- **Llama 3.2** - Meta AI
- **SDXL** - Stability AI

---

## 💬 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/lora-factory/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/lora-factory/discussions)
- **Email**: support@yourproject.com

---

## 🌟 Star History

If this project helps you, please give it a ⭐️!

---

**Built with ❤️ for the AI community**
