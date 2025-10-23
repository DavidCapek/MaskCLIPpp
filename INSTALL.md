## Installation

### Requirements
- Linux with Python = 3.10
- PyTorch = 2.3.1 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  - **For GPU:** CUDA = 12.1 - Install from [pytorch.org](https://pytorch.org)
  - **For CPU:** Install CPU-only version from [pytorch.org](https://pytorch.org)
- Install Detectron2 from source: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
- `pip install -r requirements.txt`
- `pip install git+https://github.com/cocodataset/panopticapi.git`


### CUDA kernel for MSDeformAttn (Optional for GPU)
After preparing the required environment, run the following command to compile CUDA kernel for MSDeformAttn:

**For GPU (CUDA) support:**
`CUDA_HOME` must be defined and points to the directory of the installed CUDA toolkit.

```bash
cd maskclippp/segmentor/ops
sh make.sh
```

**For CPU-only (no CUDA):**
If you want to run on CPU without CUDA, set the `FORCE_CPU` environment variable:

```bash
cd maskclippp/segmentor/ops
FORCE_CPU=1 sh make.sh
```

Or simply skip this step entirely - the code will automatically use a pure PyTorch CPU fallback for deformable attention when the CUDA extension is not available.

### Example conda environment setup

**For GPU (CUDA 12.1):**
```bash
conda create --name maskclippp python=3.10 -y
conda activate maskclippp
# CUDA 12.1
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install git+https://github.com/cocodataset/panopticapi.git

git clone git@github.com:HVision-NKU/MaskCLIPpp.git
cd MaskCLIPpp
pip install -r requirements.txt
cd maskclippp/segmentor/ops
sh make.sh
cd ../../../
```

**For CPU-only:**
```bash
conda create --name maskclippp-cpu python=3.10 -y
conda activate maskclippp-cpu
# Install CPU-only PyTorch
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu

python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install git+https://github.com/cocodataset/panopticapi.git

git clone git@github.com:HVision-NKU/MaskCLIPpp.git
cd MaskCLIPpp
pip install -r requirements.txt

# Option 1: Build CPU-only extension (optional)
cd maskclippp/segmentor/ops
FORCE_CPU=1 sh make.sh
cd ../../../

# Option 2: Skip building extension entirely
# The code will automatically use pure PyTorch CPU fallback
```
