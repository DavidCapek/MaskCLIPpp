#!/usr/bin/env python3
"""
Test script to verify CPU-only installation of MaskCLIPpp.
This script checks that all critical components can be imported and work on CPU.

Usage:
    python test_cpu_install.py
"""

import sys
import os

def print_section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def test_basic_imports():
    """Test basic Python imports"""
    print_section("Testing Basic Imports")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} imported successfully")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        print(f"  CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
        print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    except ImportError as e:
        print(f"✗ Failed to import PyTorch: {e}")
        return False
    
    try:
        import detectron2
        print(f"✓ Detectron2 imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Detectron2: {e}")
        print("  Install with: python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'")
        return False
    
    return True

def test_maskclippp_imports():
    """Test MaskCLIPpp specific imports"""
    print_section("Testing MaskCLIPpp Imports")
    
    try:
        from maskclippp.segmentor.ops.functions.ms_deform_attn_func import (
            MSDA_AVAILABLE, 
            ms_deform_attn_core_pytorch
        )
        print(f"✓ Deformable attention module imported")
        print(f"  CUDA extension available: {MSDA_AVAILABLE}")
        print(f"  CPU fallback available: {ms_deform_attn_core_pytorch is not None}")
        
        if not MSDA_AVAILABLE:
            print("  Note: Using pure PyTorch CPU implementation (this is expected for CPU-only setup)")
    except ImportError as e:
        print(f"✗ Failed to import deformable attention: {e}")
        return False
    
    try:
        from maskclippp.segmentor.ops.modules.ms_deform_attn import MSDeformAttn
        print(f"✓ MSDeformAttn module imported")
    except ImportError as e:
        print(f"✗ Failed to import MSDeformAttn: {e}")
        return False
    
    return True

def test_device_operations():
    """Test that basic operations work on the available device"""
    print_section("Testing Device Operations")
    
    import torch
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Test basic tensor operations
        x = torch.randn(2, 3).to(device)
        y = torch.randn(3, 4).to(device)
        z = torch.matmul(x, y)
        print(f"✓ Basic tensor operations work on {device}")
        
        # Test with MSDeformAttn if available
        from maskclippp.segmentor.ops.modules.ms_deform_attn import MSDeformAttn
        
        batch_size = 1
        num_queries = 10
        d_model = 256
        n_levels = 4
        n_heads = 8
        n_points = 4
        
        # Create test inputs
        query = torch.randn(batch_size, num_queries, d_model).to(device)
        reference_points = torch.rand(batch_size, num_queries, n_levels, 2).to(device)
        
        H, W = 32, 32
        input_flatten = torch.randn(batch_size, H*W*n_levels, d_model).to(device)
        input_spatial_shapes = torch.tensor([[H, W] for _ in range(n_levels)], dtype=torch.long).to(device)
        input_level_start_index = torch.tensor([i * H * W for i in range(n_levels)], dtype=torch.long).to(device)
        
        # Test MSDeformAttn forward pass
        module = MSDeformAttn(d_model, n_levels, n_heads, n_points).to(device)
        output = module(query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index)
        
        print(f"✓ MSDeformAttn forward pass works on {device}")
        print(f"  Input shape: {query.shape}")
        print(f"  Output shape: {output.shape}")
        
    except Exception as e:
        print(f"✗ Device operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    print("=" * 60)
    print("  MaskCLIPpp CPU Installation Test")
    print("=" * 60)
    
    success = True
    
    # Run tests
    if not test_basic_imports():
        success = False
    
    if not test_maskclippp_imports():
        success = False
    
    if not test_device_operations():
        success = False
    
    print_section("Summary")
    if success:
        print("✓ All tests passed!")
        print("\nYour MaskCLIPpp installation is working correctly.")
        print("You can now run the model with --num-gpus 0 for CPU execution.")
    else:
        print("✗ Some tests failed")
        print("\nPlease check the error messages above and ensure:")
        print("1. PyTorch is installed correctly")
        print("2. Detectron2 is installed")
        print("3. All requirements are installed: pip install -r requirements.txt")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
