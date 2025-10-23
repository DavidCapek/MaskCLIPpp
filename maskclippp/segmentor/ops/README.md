# MSDeformAttn - Multi-Scale Deformable Attention

This directory contains the implementation of Multi-Scale Deformable Attention, which can be compiled with CUDA support for GPU or used with CPU-only fallback.

## Building the Extension

### GPU (CUDA) Build

For GPU support with optimized CUDA kernels:

```bash
cd maskclippp/segmentor/ops
sh make.sh
```

Requirements:
- `CUDA_HOME` environment variable must be set
- CUDA toolkit installed (tested with CUDA 12.1)
- PyTorch with CUDA support

### CPU-Only Build

For CPU-only systems or if you don't want to compile CUDA kernels:

**Option 1: Build CPU-only extension**
```bash
cd maskclippp/segmentor/ops
FORCE_CPU=1 sh make.sh
```

**Option 2: Skip building entirely (recommended for CPU)**
Simply skip this compilation step. The code will automatically detect that the CUDA extension is not available and use the pure PyTorch CPU implementation as a fallback.

## How It Works

### CUDA Version (GPU)
When the CUDA extension is compiled and CUDA is available, the code uses optimized CUDA kernels for multi-scale deformable attention operations. This provides the best performance on GPU.

### CPU Fallback
When the CUDA extension is not available or CUDA is not detected, the code automatically falls back to a pure PyTorch implementation (`ms_deform_attn_core_pytorch`) that works on CPU. This fallback is:
- Fully functional and produces the same results
- Compatible with CPU-only PyTorch installations
- Slower than the CUDA version (expected for CPU execution)

### Automatic Detection

The code automatically detects which implementation to use:

```python
try:
    output = MSDeformAttnFunction.apply(...)  # Try CUDA version
except:
    output = ms_deform_attn_core_pytorch(...)  # Fall back to CPU
```

## Troubleshooting

### "CUDA_HOME is None" Error
If you see this error and want to use CPU:
- Set `FORCE_CPU=1` when running `make.sh`, OR
- Skip the compilation step entirely (the code will work without it)

### Import Error for MultiScaleDeformableAttention
This is expected and normal for CPU-only setups. The code handles this gracefully and uses the CPU fallback automatically.

## Performance Notes

- **GPU with CUDA**: Fastest performance, recommended for training
- **CPU with extension**: Similar performance to pure PyTorch fallback
- **CPU without extension**: Uses pure PyTorch implementation, slower but functional

For CPU usage, consider:
- Smaller batch sizes
- Inference/evaluation rather than training
- Smaller model variants
