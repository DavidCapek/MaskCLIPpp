# CPU-Only Setup Guide for MaskCLIPpp

This guide walks you through setting up MaskCLIPpp for CPU-only execution without CUDA.

## Quick Start

### 1. Create Conda Environment

```bash
conda create --name maskclippp-cpu python=3.10 -y
conda activate maskclippp-cpu
```

### 2. Install PyTorch (CPU-only)

```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
```

Verify PyTorch installation:
```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Expected output:
```
PyTorch: 2.3.1
CUDA available: False
```

### 3. Install Detectron2

```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### 4. Clone MaskCLIPpp

```bash
git clone https://github.com/HVision-NKU/MaskCLIPpp.git
cd MaskCLIPpp
```

### 5. Install Requirements

```bash
pip install -r requirements.txt
pip install git+https://github.com/cocodataset/panopticapi.git
```

### 6. Handle MSDeformAttn Extension

You have two options:

#### Option A: Skip Building (Recommended for CPU)
Simply skip the compilation step. The code will automatically use the pure PyTorch CPU fallback.

```bash
# No action needed - just continue to testing
```

#### Option B: Build CPU-Only Extension
If you prefer to build the extension (though it will still use CPU fallback):

```bash
cd maskclippp/segmentor/ops
FORCE_CPU=1 sh make.sh
cd ../../../
```

### 7. Test Your Installation

Run the provided test script:

```bash
python test_cpu_install.py
```

Expected output:
```
============================================================
  MaskCLIPpp CPU Installation Test
============================================================

============================================================
  Testing Basic Imports
============================================================
✓ PyTorch 2.3.1 imported successfully
  CUDA available: False
  CUDA version: N/A
  Device: cpu
✓ Detectron2 imported successfully

============================================================
  Testing MaskCLIPpp Imports
============================================================
✓ Deformable attention module imported
  CUDA extension available: False
  CPU fallback available: True
  Note: Using pure PyTorch CPU implementation (this is expected for CPU-only setup)
✓ MSDeformAttn module imported

============================================================
  Testing Device Operations
============================================================
Using device: cpu
✓ Basic tensor operations work on cpu
✓ MSDeformAttn forward pass works on cpu
  Input shape: torch.Size([1, 10, 256])
  Output shape: torch.Size([1, 10, 256])

============================================================
  Summary
============================================================
✓ All tests passed!

Your MaskCLIPpp installation is working correctly.
You can now run the model with --num-gpus 0 for CPU execution.
```

## Running MaskCLIPpp on CPU

### Training
```bash
python train_maskclippp.py \
    --config-file configs/your_config.yaml \
    --num-gpus 0 \
    --dist-url "auto" \
    --tag cpu_run
```

### Evaluation
```bash
# Use the evaluation scripts with num_gpus=0
source eval_all.sh
eval_ade150 $config $ckpt 0 $tag  # 0 means CPU-only
```

### Demo
```bash
# The demo automatically detects CPU and uses it
python demo/demo.py \
    --config-file configs/your_config.yaml \
    --input your_image.jpg \
    --output output/
```

## Troubleshooting

### Issue: "CUDA_HOME is None" when running make.sh

**Solution:** Use the `FORCE_CPU` flag:
```bash
FORCE_CPU=1 sh make.sh
```

Or simply skip building the extension - it's optional for CPU execution.

### Issue: Import error for MultiScaleDeformableAttention

**This is expected and normal.** The code automatically handles this and uses the CPU fallback. You should see a message like:
```
Note: Using pure PyTorch CPU implementation
```

### Issue: Slow performance on CPU

**This is expected.** Deep learning models are optimized for GPU. For CPU usage:
- Use smaller batch sizes (e.g., `--batch-size 1`)
- Use smaller model variants
- Consider CPU execution primarily for inference/evaluation
- Training on CPU will take significantly longer

### Issue: Out of memory on CPU

**Solution:** Reduce batch size and model size:
- Set smaller batch size in config
- Use smaller model variants (e.g., base instead of large)
- Close other applications to free up RAM

## Performance Expectations

| Task | GPU (CUDA) | CPU |
|------|-----------|-----|
| Training (1 epoch) | ~2-4 hours | ~24-48 hours |
| Inference (single image) | < 1 second | 5-10 seconds |
| Evaluation (dataset) | Minutes | Hours |

**Recommendation:** Use CPU for:
- Development and code testing
- Small-scale inference
- Environments without GPU access
- Educational purposes

Use GPU for:
- Model training
- Large-scale inference
- Production deployments

## Verifying CPU-Only Setup

To verify you're actually using CPU and not accidentally using CUDA:

```python
import torch
print("CUDA available:", torch.cuda.is_available())  # Should be False
print("Number of GPUs:", torch.cuda.device_count())  # Should be 0

# Test device selection in model
from maskclippp.maskclippp import MaskCLIPpp
# The model will automatically use CPU when CUDA is not available
```

## Additional Notes

### Why Two Options for MSDeformAttn?

The Multi-Scale Deformable Attention operation has two implementations:

1. **CUDA Extension** (GPU-only): Optimized CUDA kernels for best GPU performance
2. **Pure PyTorch** (CPU/GPU): Compatible implementation using standard PyTorch operations

For CPU execution, both will use the PyTorch implementation anyway, so building the extension is optional.

### Can I mix CPU and GPU?

No, the model runs entirely on one device (CPU or GPU). You specify this with the `--num-gpus` flag:
- `--num-gpus 0`: CPU-only
- `--num-gpus 1`: Single GPU
- `--num-gpus 2`: Two GPUs (distributed)

### Converting GPU checkpoint to CPU

If you have a checkpoint trained on GPU and want to use it on CPU:

```python
import torch

# Load checkpoint
checkpoint = torch.load('model_gpu.pth', map_location='cpu')

# Save for CPU
torch.save(checkpoint, 'model_cpu.pth')
```

Then use with `--num-gpus 0`.

## Getting Help

If you encounter issues:

1. Check that PyTorch is CPU-only: `python -c "import torch; print(torch.cuda.is_available())"`
2. Run the test script: `python test_cpu_install.py`
3. Check the error message carefully - most issues are related to dependencies
4. Consult the main [INSTALL.md](INSTALL.md) for general installation help

For CPU-specific questions, please open an issue on GitHub with:
- Output of `test_cpu_install.py`
- Your Python version: `python --version`
- Your PyTorch version and backend: `python -c "import torch; print(torch.__version__)"`
