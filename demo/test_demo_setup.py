#!/usr/bin/env python3
"""
Diagnostic script to check demo dependencies and CPU compatibility.
Run this before running the demo to identify potential issues.

Usage:
    python test_demo_setup.py
"""

import sys
import os

def print_section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def test_basic_imports():
    """Test basic Python imports needed for demo"""
    print_section("Testing Basic Imports")
    
    success = True
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"✗ PyTorch not found: {e}")
        success = False
    
    try:
        import detectron2
        print(f"✓ Detectron2 imported")
    except ImportError as e:
        print(f"✗ Detectron2 not found: {e}")
        print("  Install with: python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'")
        success = False
    
    try:
        import gradio
        print(f"✓ Gradio {gradio.__version__}")
    except ImportError as e:
        print(f"✗ Gradio not found: {e}")
        print("  Install with: pip install gradio")
        success = False
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV not found: {e}")
        print("  Install with: pip install opencv-python")
        success = False
    
    return success

def test_maskclippp_imports():
    """Test MaskCLIPpp specific imports"""
    print_section("Testing MaskCLIPpp Imports")
    
    success = True
    
    # Add parent directory to path (same as demo does)
    sys.path.insert(1, os.path.join(sys.path[0], '..'))
    
    try:
        from maskclippp import add_maskformer2_config, add_maskclippp_config
        print(f"✓ MaskCLIPpp config imports successful")
    except ImportError as e:
        print(f"✗ Failed to import MaskCLIPpp configs: {e}")
        import traceback
        traceback.print_exc()
        success = False
        return success
    
    try:
        # Import VisualizationDemo from current directory (demo)
        demo_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, demo_dir)
        from predictor import VisualizationDemo
        print(f"✓ VisualizationDemo imported")
    except ImportError as e:
        print(f"✗ Failed to import VisualizationDemo: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    return success

def test_deformable_attention():
    """Test deformable attention CPU fallback"""
    print_section("Testing Deformable Attention")
    
    try:
        from maskclippp.segmentor.ops.functions.ms_deform_attn_func import (
            MSDA_AVAILABLE, 
            ms_deform_attn_core_pytorch
        )
        print(f"✓ Deformable attention module imported")
        print(f"  CUDA extension available: {MSDA_AVAILABLE}")
        print(f"  CPU fallback available: {ms_deform_attn_core_pytorch is not None}")
        
        if not MSDA_AVAILABLE:
            print("  ℹ Using pure PyTorch CPU implementation (expected for CPU-only setup)")
        
        return True
    except ImportError as e:
        print(f"✗ Failed to import deformable attention: {e}")
        return False

def test_config_loading():
    """Test configuration loading"""
    print_section("Testing Configuration Loading")
    
    try:
        from detectron2.config import get_cfg
        from detectron2.projects.deeplab import add_deeplab_config
        from maskclippp import add_maskformer2_config, add_maskclippp_config
        
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        add_maskclippp_config(cfg)
        cfg.RUN_DEMO = True
        
        print(f"✓ Configuration system working")
        print(f"  RUN_DEMO: {cfg.RUN_DEMO}")
        
        # Check if default config file exists
        default_config = "configs/coco-stuff/eva-clip-vit-l-14-336/maft-l/maskclippp_coco-stuff_eva-clip-vit-l-14-336_wtext_maft-l_ens.yaml"
        if os.path.exists(default_config):
            print(f"  ✓ Default config file found: {default_config}")
        else:
            print(f"  ⚠ Default config file not found: {default_config}")
            print(f"    You'll need to specify --config-file when running the demo")
        
        return True
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("  MaskCLIPpp Demo Setup Diagnostic")
    print("=" * 60)
    
    all_success = True
    
    if not test_basic_imports():
        all_success = False
    
    if not test_maskclippp_imports():
        all_success = False
    
    if not test_deformable_attention():
        all_success = False
    
    if not test_config_loading():
        all_success = False
    
    print_section("Summary")
    if all_success:
        print("✓ All tests passed!")
        print("\nYour demo setup is ready. You can now run:")
        print("  python demo/app.py")
        print("or")
        print("  python demo/demo.py --config-file <path-to-config> --input <image>")
    else:
        print("✗ Some tests failed")
        print("\nPlease fix the issues above before running the demo.")
        print("Common issues:")
        print("1. Missing dependencies - install with: pip install -r requirements.txt")
        print("2. Detectron2 not installed - follow INSTALL.md")
        print("3. MaskCLIPpp not in path - run from repository root")
    
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main())
