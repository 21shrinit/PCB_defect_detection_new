#!/usr/bin/env python3
"""
Comprehensive verification script for custom attention modules (ECA, CBAM, CoordAtt).
Implements the verification checklist from Gemini's research findings.
"""

import torch
import sys
import os
import numpy as np
from pathlib import Path

# Add the ultralytics directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
ultralytics_path = os.path.join(current_dir, '..', 'ultralytics')
sys.path.insert(0, ultralytics_path)
sys.path.insert(0, os.path.join(current_dir, '..'))

def test_attention_modules():
    """Test individual attention modules - Unit Test Passed."""
    print("🧪 Testing Attention Modules (Unit Test)...")
    
    try:
        # Import attention modules
        from ultralytics.nn.modules.attention import ECA, CBAM, CoordAtt
        
        # Test ECA
        print("  📍 Testing ECA...")
        eca = ECA(c1=64)
        x = torch.randn(1, 64, 80, 80)
        y = eca(x)
        assert y.shape == x.shape, f"ECA output shape mismatch: {y.shape} vs {x.shape}"
        
        # Verify attention weights are normalized (0-1 range)
        attention_weights = eca.avg_pool(x)
        attention_weights = eca.conv(attention_weights.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        attention_weights = eca.sigmoid(attention_weights)
        assert torch.all(attention_weights >= 0) and torch.all(attention_weights <= 1), "ECA attention weights not normalized"
        print("    ✅ ECA: PASS (shapes + normalization)")
        
        # Test CBAM
        print("  📍 Testing CBAM...")
        cbam = CBAM(c1=64)
        x = torch.randn(1, 64, 80, 80)
        y = cbam(x)
        assert y.shape == x.shape, f"CBAM output shape mismatch: {y.shape} vs {x.shape}"
        
        # Verify CBAM attention weights
        channel_att = cbam.channel_attention(x)
        spatial_att = cbam.spatial_attention(x)
        assert torch.all(channel_att >= 0) and torch.all(channel_att <= 1), "CBAM channel attention not normalized"
        assert torch.all(spatial_att >= 0) and torch.all(spatial_att <= 1), "CBAM spatial attention not normalized"
        print("    ✅ CBAM: PASS (shapes + normalization)")
        
        # Test CoordAtt
        print("  📍 Testing CoordAtt...")
        coordatt = CoordAtt(inp=64, oup=64)
        x = torch.randn(1, 64, 80, 80)
        y = coordatt(x)
        assert y.shape == x.shape, f"CoordAtt output shape mismatch: {y.shape} vs {x.shape}"
        
        print("    ✅ CoordAtt: PASS (shapes)")
        
        print("  🎉 All attention modules passed unit tests!")
        return True
        
    except Exception as e:
        print(f"  ❌ Attention module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gradient_flow():
    """Test gradient flow through attention modules - Gradient Flow Confirmed."""
    print("\n🌊 Testing Gradient Flow...")
    
    try:
        from ultralytics.nn.modules.attention import ECA, CBAM, CoordAtt
        
        # Test ECA gradients
        print("  📍 Testing ECA gradients...")
        eca = ECA(c1=64)
        x = torch.randn(1, 64, 32, 32, requires_grad=True)
        y = eca(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None, "ECA gradients not flowing"
        assert x.grad.norm().item() > 0, "ECA gradients are zero"
        print("    ✅ ECA gradients: PASS")
        
        # Test CBAM gradients
        print("  📍 Testing CBAM gradients...")
        cbam = CBAM(c1=64)
        x = torch.randn(1, 64, 32, 32, requires_grad=True)
        y = cbam(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None, "CBAM gradients not flowing"
        assert x.grad.norm().item() > 0, "CBAM gradients are zero"
        print("    ✅ CBAM gradients: PASS")
        
        # Test CoordAtt gradients
        print("  📍 Testing CoordAtt gradients...")
        coordatt = CoordAtt(inp=64, oup=64)
        x = torch.randn(1, 64, 32, 32, requires_grad=True)
        y = coordatt(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None, "CoordAtt gradients not flowing"
        assert x.grad.norm().item() > 0, "CoordAtt gradients are zero"
        print("    ✅ CoordAtt gradients: PASS")
        
        print("  🎉 All gradient flow tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_attention_weight_normalization():
    """Test attention weight normalization - Attention Weights Normalized."""
    print("\n⚖️  Testing Attention Weight Normalization...")
    
    try:
        from ultralytics.nn.modules.attention import ECA, CBAM, CoordAtt
        
        # Test ECA weight normalization
        print("  📍 Testing ECA weight normalization...")
        eca = ECA(c1=64)
        x = torch.randn(1, 64, 32, 32)
        y = eca(x)
        # Extract attention weights
        att_weights = eca.avg_pool(x)
        att_weights = eca.conv(att_weights.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        att_weights = eca.sigmoid(att_weights)
        assert torch.all(att_weights >= 0) and torch.all(att_weights <= 1), "ECA weights not in [0,1] range"
        print(f"    ✅ ECA weights: min={att_weights.min().item():.4f}, max={att_weights.max().item():.4f}")
        
        # Test CBAM weight normalization
        print("  📍 Testing CBAM weight normalization...")
        cbam = CBAM(c1=64)
        x = torch.randn(1, 64, 32, 32)
        y = cbam(x)
        # Extract attention weights
        channel_weights = cbam.channel_attention(x)
        spatial_weights = cbam.spatial_attention(x)
        assert torch.all(channel_weights >= 0) and torch.all(channel_weights <= 1), "CBAM channel weights not in [0,1] range"
        assert torch.all(spatial_weights >= 0) and torch.all(spatial_weights <= 1), "CBAM spatial weights not in [0,1] range"
        print(f"    ✅ CBAM channel weights: min={channel_weights.min().item():.4f}, max={channel_weights.max().item():.4f}")
        print(f"    ✅ CBAM spatial weights: min={spatial_weights.min().item():.4f}, max={spatial_weights.max().item():.4f}")
        
        print("  🎉 All attention weight normalization tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Attention weight normalization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_attention_map_visualization():
    """Test attention map generation for visualization - Visualize Attention Map."""
    print("\n👁️  Testing Attention Map Visualization...")
    
    try:
        from ultralytics.nn.modules.attention import ECA, CBAM, CoordAtt
        
        # Create a simple test image with known structure
        x = torch.randn(1, 64, 32, 32)
        
        # Test ECA attention map
        print("  📍 Testing ECA attention map...")
        eca = ECA(c1=64)
        eca.eval()
        with torch.no_grad():
            y = eca(x)
            # Extract attention weights
            att_weights = eca.avg_pool(x)
            att_weights = eca.conv(att_weights.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            att_weights = eca.sigmoid(att_weights)
            # Reshape for visualization
            att_map = att_weights.squeeze().cpu().numpy()
        print(f"    ✅ ECA attention map shape: {att_map.shape}")
        print(f"    ✅ ECA attention map stats: min={att_map.min():.4f}, max={att_map.max():.4f}, mean={att_map.mean():.4f}")
        
        # Test CBAM attention maps
        print("  📍 Testing CBAM attention maps...")
        cbam = CBAM(c1=64)
        cbam.eval()
        with torch.no_grad():
            y = cbam(x)
            # Extract attention weights
            channel_weights = cbam.channel_attention(x)
            spatial_weights = cbam.spatial_attention(x)
            # Reshape for visualization
            channel_map = channel_weights.squeeze().cpu().numpy()
            spatial_map = spatial_weights.squeeze().cpu().numpy()
        print(f"    ✅ CBAM channel map shape: {channel_map.shape}")
        print(f"    ✅ CBAM spatial map shape: {spatial_map.shape}")
        print(f"    ✅ CBAM channel stats: min={channel_map.min():.4f}, max={channel_map.max():.4f}, mean={channel_map.mean():.4f}")
        print(f"    ✅ CBAM spatial stats: min={spatial_map.min():.4f}, max={spatial_map.max():.4f}, mean={spatial_map.mean():.4f}")
        
        print("  🎉 All attention map visualization tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Attention map visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all verification tests based on Gemini's research checklist."""
    print("🎯 Comprehensive Attention Module Verification Suite")
    print("=" * 60)
    print("Based on Gemini's research findings and best practices")
    print("=" * 60)
    
    tests = [
        ("Attention Modules (Unit Test)", test_attention_modules),
        ("Gradient Flow Confirmed", test_gradient_flow),
        ("Attention Weights Normalized", test_attention_weight_normalization),
        ("Visualize Attention Map", test_attention_map_visualization),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Verification Checklist Results:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} verification checks passed")
    
    if passed == total:
        print("🎉 All verification checks passed! Attention modules are ready for training.")
        print("\n🚀 Next Steps:")
        print("  1. Proceed with attention-enhanced model training")
        print("  2. Monitor attention weights during training")
        print("  3. Visualize attention maps for validation images")
        print("  4. Compare performance with baseline models")
        return True
    else:
        print("⚠️  Some verification checks failed. Please fix the issues above.")
        print("\n🔧 Required Fixes:")
        print("  1. Address failed verification checks")
        print("  2. Re-run verification suite")
        print("  3. Ensure all modules pass before training")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
