# Prompt-Inpaint

English README. Chinese version: [Chinese (CN)](README_CN.md)

Automatically detect, segment, and remove objects from images using text prompts, then generate a clean background image.

**Core features:**
- Use **SAM 3** (Meta's Segment Anything 3) for one-shot text-prompted concept segmentation — detection and masks come from a single model
- Use **iopaint (LaMa)** to remove objects and inpaint the background
- Support **iterative mask expansion** to recover occluded parts of objects

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Arguments](#cli-arguments)
- [Configuration](#configuration)
- [Pipeline Flow](#pipeline-flow)
- [Batch Processing](#batch-processing)
- [Model Selection](#model-selection)
- [FAQ](#faq)

## Installation

### Requirements

- Python 3.10 - 3.11
- CUDA 12.x (recommended) or CPU
- ~8GB VRAM (with default models)

### Install with uv (recommended)

```bash
cd /path/to/grounded-segment-inpaint

# 1. Create venv
uv venv --python 3.11 .venv
source .venv/bin/activate

# 2. Install PyTorch (match your CUDA version)
uv pip install "torch>=2.7" "torchvision>=0.22" --index-url https://download.pytorch.org/whl/cu128

# 3. Install SAM3 + remaining dependencies
uv pip install --index-strategy unsafe-best-match \
    "sam3 @ git+https://github.com/facebookresearch/sam3.git" \
    einops huggingface_hub \
    "iopaint>=1.2.0" \
    "numpy<2.0" "opencv-python>=4.8.0" "pyyaml>=6.0" "requests>=2.31.0" "tqdm>=4.66.0" setuptools
```

### Install with pip

```bash
pip install "torch>=2.7" "torchvision>=0.22" --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### Authenticate with HuggingFace (required for SAM3 weights)

`facebook/sam3` is a gated model. Request access on the
[model page](https://huggingface.co/facebook/sam3) and then log in once:

```bash
huggingface-cli login
```

The first run downloads the SAM3 checkpoint (~3.4GB) and caches it under
`~/.cache/huggingface/hub`. Pre-existing weights at
`<repo>/checkpoints/sam3.pt` are used in priority.

### Verify installation

```bash
python -c "
import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
from sam3.model_builder import build_sam3_image_model; print('SAM3: OK')
import iopaint; print('iopaint: OK')
"
```

## Quick Start

### Basic usage

```bash
# Common usage: save individual masks + force output size 448x448
python main.py --image photo.jpg --resize-output --save-individual-masks --config configs/items.yml

# Save transparent cutouts (RGBA, cropped to object size)
python main.py --image photo.jpg --save-individual-transparent-masks --no-inpaint

# Use a config file
python main.py --image photo.jpg --config configs/items.yml

# Custom output directory
python main.py --image photo.jpg --output-dir outputs/demo_run

# Provide prompts via CLI (overrides config)
python main.py --image photo.jpg --prompts "cup" "knife" "robot arm"

# Segment only (skip inpaint)
python main.py --image photo.jpg --prompts "cup" --no-inpaint

# Save debug artifacts
python main.py --image photo.jpg --save-debug

# Force output size (default 448x448 when flag is set)
python main.py --image photo.jpg --prompts "cup" --resize-output

# Custom output size
python main.py --image photo.jpg --prompts "cup" --resize-output 640x480
```

**Config note:** If `--config` is not provided, the program tries to load `configs/items.yml` (if it exists). If not found, you must pass prompts via `--prompts`.

### Sample results

Input:

![sample input](assets/sample_input.png)

Combined mask:

![sample combined mask](assets/sample_combined_mask.png)

Cleaned background:

![sample clean background](assets/sample_clean_background.png)

Separated objects (RGB masks):

| | | |
| --- | --- | --- |
| cucumber<br>![cucumber](assets/cucumber_rgb.png) | banana<br>![banana](assets/banana_rgb.png) | corn<br>![corn](assets/corn_rgb.png) |
| sponge<br>![sponge](assets/sponge_rgb.png) | gripper<br>![gripper](assets/gripper_rgb.png) | computer mouse<br>![computer mouse](assets/computer_mouse_rgb.png) |

## CLI Arguments

| Argument | Type | Default | Description |
|------|------|--------|------|
| `--image`, `-i` | str | **required** | Input image path |
| `--output-dir`, `-o` | str | `outputs/<timestamp>` | Output directory |
| `--config`, `-c` | str | `configs/items.yml` | YAML config path |
| `--prompts`, `-p` | list | - | Prompt list (overrides config) |
| `--sam3-model` | str | `facebook/sam3` | SAM3 model id or local checkpoint path |
| `--device` | str | `cuda` | Device |
| `--sam3-threshold` | float | `0.5` | SAM3 detection confidence threshold |
| `--sam3-mask-threshold` | float | `0.5` | SAM3 mask binarization threshold |
| `--iou-threshold` | float | `0.5` | Mask dedup IoU threshold |
| `--inpaint-backend` | str | `iopaint` | Inpaint backend: `iopaint`/`opencv`/`none` |
| `--mask-dilate-pixels` | int | `12` | Mask dilation pixels (for inpaint) |
| `--no-inpaint` | flag | - | Skip background inpainting |
| `--save-debug` | flag | - | Save debug artifacts |
| `--save-individual-masks [0|1]` | int? | - | Save RGB masks (black background). By default excludes `robot arm`/`gripper`; pass `1` to include them |
| `--save-individual-transparent-masks [0|1]` | int? | - | Save transparent cutouts (RGBA, cropped to object size). By default excludes `robot arm`/`gripper`; pass `1` to include them |
| `--resize-output` | str | off | Force resize all outputs; when flag is set without value, default is `448x448` |

**Notes:**
- CLI args override config values
- If `--config` is not provided, it tries to load `configs/items.yml`; if missing, prompts must be passed via `--prompts`
- Example: if `items.yml` sets `sam3_threshold: 0.6`, the effective value will be 0.6 (not 0.5)
- `--save-individual-masks` and `--save-individual-transparent-masks` can be enabled together: each flag saves its own outputs; disable both to save neither

## Configuration

Config files are in YAML.

**Config files:**
- `configs/items.yml` - default config with common prompts

Example:

```yaml
# Prompt list
# Tip: put occluders first (e.g., robot arm)
prompts:
  # ========== Robot & Equipment ==========
  - "robot arm"
  - "gripper"

  # ========== Kitchen Utensils ==========
  - "pot"
  - "knife"
  - "cup"
  - "plate"

  # ========== Food Items ==========
  - "bread"
  - "apple"
  - "onion"

  # ========== Cleaning & Fabric ==========
  - "towel"
  - "rag"
  - "cloth"

settings:
  # SAM3 model: HuggingFace id or local checkpoint path
  sam3_model: facebook/sam3

  # SAM3 detection confidence (higher = stricter)
  sam3_threshold: 0.5

  # SAM3 mask binarization threshold
  sam3_mask_threshold: 0.5

  # Mask dedup threshold
  iou_threshold: 0.5

  # Containment merge threshold
  containment_overlap_ratio: 0.9

  # Contour overlap threshold
  contour_overlap_ratio: 0.3

  # Inpainting
  inpaint_backend: iopaint             # iopaint / opencv / none
  mask_dilate_pixels: 12               # mask dilation pixels

  # Debug
  save_debug: false

  # Output options
  save_individual_masks: false   # save RGB masks to masks/
  output_size: [448, 448]        # optional output size

  # Device
  device: cuda
```

Notes:
- If `output_size` is not set, outputs keep original size; otherwise all saved images are resized.
- Dedup rules: IoU over threshold merges masks; if small-mask overlap > `containment_overlap_ratio` and contour overlap > `contour_overlap_ratio`, they also merge.

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                           Input Image                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Initial Detection                                       │
│  - SAM3 takes each text prompt and returns masks + boxes + scores│
│    in a single forward pass per prompt (vision features cached)  │
│  - Deduplicate by IoU (merge overlapping masks)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: Iterative Mask Expansion                                │
│  For each object:                                                │
│  1. Remove it from current image (dilated mask + inpaint)         │
│  2. Re-detect on the inpainted image (only known labels)          │
│  3. If new mask is adjacent and same type:                        │
│     -> Expand that object's mask (limit: <= 3x original area)     │
│  4. Update current image and proceed                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: Final Inpainting                                        │
│  - Inpaint sequentially with expanded masks                       │
│  - Output clean background                                        │
└─────────────────────────────────────────────────────────────────┘
```

### Why iterative expansion?

When object A occludes object B:
1. Initial detection sees only the visible part of B
2. After removing A, hidden parts of B become visible
3. Re-detection finds the full extent of B
4. Expand B's mask to remove it completely

**Example:** robot arm occludes a towel
- Initial: towel mask is incomplete
- After removing arm: towel is fully visible
- Expansion: towel mask is completed
- Final: towel is fully removed

## Batch Processing

For datasets like Bridge V2, use batch scripts to process multiple sub-datasets.

### Usage

```bash
# Basic usage
python scripts/batch_process_datasets.py \
    --input-root /path/to/traj_group0 \
    --output-dir ./batch_outputs \
    --config configs/items.yml

# With resize and individual masks (exclude robot arm / gripper by default)
python scripts/batch_process_datasets.py \
    --input-root /path/to/traj_group0 \
    --output-dir ./batch_outputs \
    --config configs/items.yml \
    --resize-output 448x448 \
    --save-individual-masks

# Skip inpaint for speed (include robot arm / gripper)
python scripts/batch_process_datasets.py \
    --input-root /path/to/traj_group0 \
    --output-dir ./batch_outputs \
    --no-inpaint \
    --save-individual-masks 1
```

### Batch arguments

| Argument | Type | Default | Description |
|------|------|--------|------|
| `--input-root` | str | **required** | Root directory with sub-datasets |
| `--output-dir` | str | **required** | Output directory |
| `--config`, `-c` | str | `configs/items.yml` | Config file path |
| `--resize-output` | str | off | Resize outputs |
| `--save-individual-masks [0|1]` | int? | - | Save RGB masks; default excludes `robot arm`/`gripper`; pass `1` to include |
| `--no-inpaint` | flag | - | Skip inpainting |
| `--save-debug` | flag | - | Save debug artifacts |
| `--overwrite` | flag | - | Overwrite existing outputs |
| `--device` | str | `cuda` | Device |

### Output structure

```
batch_outputs/
├── dataset_001/
│   ├── input_image.png
│   ├── combined_mask.png
│   ├── clean_background.png
│   ├── report.json
│   ├── objects/
│   └── masks/              # if --save-individual-masks is enabled
├── dataset_002/
└── ...
```

**Notes:**
- Models are loaded once and reused for batch processing
- Skips datasets already processed unless `--overwrite` is set
- Processes the first image in each sub-dataset

## Model Selection

### SAM3

Currently only one public checkpoint is published: `facebook/sam3` (~0.9B params, 3.4GB).

**Offline/local checkpoint:** If `checkpoints/sam3.pt` exists, it is loaded locally;
otherwise the file is downloaded from Hugging Face on first run (gated; see Installation).

**VRAM reference (RTX 3090 24GB, RGB 448×448):**
- SAM3 inference: ~6–10 GB
- iopaint LaMa background inpainting: a few hundred MB

The model performs the equivalent of "Grounding DINO + SAM2" in a single forward pass —
text prompts produce masks, boxes, and scores directly.

## FAQ

### 1. Objects are not fully removed

**Possible causes:**
- SAM3 confidence threshold too high; some regions filtered
- Occluded parts not recovered by iterative expansion

**Fixes:**
- Lower threshold: `--sam3-threshold 0.3`
- Reorder prompts: put occluders first (so they get inpainted before nearby objects)
- Use `--save-debug` to inspect intermediate results

### 2. False positives

**Fixes:**
- Increase threshold: `--sam3-threshold 0.7`
- Use more specific prompts, e.g. "yellow knife" instead of "knife"
- Reduce prompt list size

### 3. Out of memory (OOM)

**Fixes:**
- Use CPU: `--device cpu` (much slower)
- Resize input images
- Close other GPU processes

### 4. Inpaint quality is poor

**Note:** iopaint (LaMa) can struggle with complex textures or large missing areas.

**Suggestions:**
- Ensure sufficient dilation (default 12px)
- For large areas, consider external retouching tools

### 5. `Cannot access gated repo for facebook/sam3`

**Cause:** You haven't been granted access on Hugging Face, or you aren't authenticated.

**Fixes:**
1. Visit https://huggingface.co/facebook/sam3 and request access
2. Run `huggingface-cli login` and paste a read token

## Acknowledgements

This project uses:
- [SAM 3](https://huggingface.co/facebook/sam3) - Meta's Segment Anything Model 3
- [iopaint](https://github.com/Sanster/IOPaint) - image inpainting

## License

MIT License
