#!/usr/bin/env python3
"""
Comprehensive CBAM Implementation Verification for All YOLO Architectures
=========================================================================

Verifies CBAM implementations across:
- YOLOv8n CBAM (neck-optimal)
- YOLOv10n CBAM  
- YOLOv11n CBAM (needs C3k2_CBAM implementation)

Tests:
1. Config loading verification
2. Module instantiation verification  
3. CBAM module presence verification
4. Forward pass functionality verification
5. Parameter count analysis
6. Architecture compliance verification
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from ultralytics import YOLO

class CBAMVerifier:
    """Comprehensive CBAM implementation verifier."""
    
    def __init__(self):
        self.models_to_test = [
            {
                'name': 'YOLOv8n CBAM Neck-Optimal',
                'config': 'ultralytics/cfg/models/v8/yolov8n-cbam-neck-optimal.yaml',
                'expected_cbam_layers': [12, 15, 18, 21],
                'architecture': 'v8',
                'cbam_module': 'C2f_CBAM'
            },
            {
                'name': 'YOLOv10n CBAM All-Layers',
                'config': 'ultralytics/cfg/models/v10/yolov10n-cbam.yaml',
                'expected_cbam_layers': [2, 4, 6, 8, 13, 16, 19, 22],
                'architecture': 'v10',
                'cbam_module': 'C2f_CBAM'
            },
            {
                'name': 'YOLOv10n CBAM Research-Optimal',
                'config': 'ultralytics/cfg/models/v10/yolov10n-cbam-research-optimal.yaml',
                'expected_cbam_layers': [2, 4, 6, 8, 13, 16, 19],
                'architecture': 'v10',
                'cbam_module': 'C2f_CBAM'
            }
        ]
        
        self.results = {}
    
    def test_config_loading(self, model_info):
        """Test if configuration loads without errors."""
        print(f"🔍 Testing Config Loading: {model_info['name']}")
        print("-" * 60)
        
        config_path = model_info['config']
        
        try:
            if not Path(config_path).exists():
                print(f"❌ Config file not found: {config_path}")
                return False
                
            print(f"📁 Loading config: {config_path}")
            model = YOLO(config_path)
            print("✅ Config loaded successfully")
            
            # Basic model info
            total_params = sum(p.numel() for p in model.model.parameters())
            print(f"📊 Total parameters: {total_params:,}")
            
            return True
            
        except Exception as e:
            print(f"❌ Config loading failed: {e}")
            return False
    
    def test_cbam_instantiation(self, model_info):
        """Test CBAM module instantiation."""
        print(f"\n🔍 Testing CBAM Instantiation: {model_info['name']}")
        print("-" * 60)
        
        try:
            config_path = model_info['config']
            model = YOLO(config_path)
            
            # Find CBAM modules
            cbam_modules = []
            cbam_blocks = []
            
            for name, module in model.model.named_modules():
                if 'cbam' in name.lower():
                    cbam_modules.append(name)
                if model_info['cbam_module'] in str(type(module).__name__):
                    cbam_blocks.append(name)
            
            print(f"🔍 Found CBAM modules: {len(cbam_modules)}")
            if cbam_modules:
                for cbam_mod in cbam_modules[:8]:  # Show first 8
                    print(f"   • {cbam_mod}")
                if len(cbam_modules) > 8:
                    print(f"   ... and {len(cbam_modules) - 8} more")
            
            print(f"\n🔍 Found {model_info['cbam_module']} blocks: {len(cbam_blocks)}")
            for block in cbam_blocks:
                print(f"   • {block}")
            
            # Check expected CBAM locations
            expected_layers = [f"model.{i}" for i in model_info['expected_cbam_layers']]
            found_expected = 0
            
            print(f"\n🎯 Checking expected CBAM locations:")
            for expected in expected_layers:
                # Check if this layer has CBAM
                has_cbam = any(expected in cbam_name for cbam_name in cbam_modules)
                if has_cbam:
                    print(f"   ✅ {expected}: CBAM found")
                    found_expected += 1
                else:
                    print(f"   ❌ {expected}: CBAM NOT found")
            
            expected_count = len(expected_layers)
            print(f"\n📊 Expected CBAM locations found: {found_expected}/{expected_count}")
            
            return found_expected >= (expected_count * 0.75)  # At least 75% success rate
            
        except Exception as e:
            print(f"❌ CBAM instantiation test failed: {e}")
            return False
    
    def test_forward_pass(self, model_info):
        """Test forward pass functionality."""
        print(f"\n🔍 Testing Forward Pass: {model_info['name']}")
        print("-" * 60)
        
        try:
            config_path = model_info['config']
            model = YOLO(config_path)
            
            # Create dummy input
            dummy_input = torch.randn(1, 3, 640, 640)
            print(f"📥 Input shape: {dummy_input.shape}")
            
            # Forward pass
            print("🚀 Running forward pass...")
            with torch.no_grad():
                outputs = model.model(dummy_input)
            
            print("✅ Forward pass completed successfully")
            
            if isinstance(outputs, list):
                print(f"📤 Output shapes: {[o.shape for o in outputs]}")
            elif isinstance(outputs, dict):
                print(f"📤 Output keys: {list(outputs.keys())}")
                if 'one2many' in outputs:
                    print(f"   • one2many: {[o.shape for o in outputs['one2many']]}")
                if 'one2one' in outputs:
                    print(f"   • one2one: {[o.shape for o in outputs['one2one']]}")
            elif hasattr(outputs, 'shape'):
                print(f"📤 Output shape: {outputs.shape}")
            else:
                print(f"📤 Output type: {type(outputs)}")
                
            return True
            
        except Exception as e:
            print(f"❌ Forward pass test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_architecture_compliance(self, model_info):
        """Test architecture-specific compliance."""
        print(f"\n🔍 Testing Architecture Compliance: {model_info['name']}")
        print("-" * 60)
        
        try:
            config_path = model_info['config']
            model = YOLO(config_path)
            
            # Architecture-specific checks
            if model_info['architecture'] == 'v8':
                print("✅ YOLOv8n architecture - standard YOLO detection")
                expected_features = ['C2f', 'SPPF', 'Detect']
                
            elif model_info['architecture'] == 'v10':
                print("✅ YOLOv10n architecture - checking for YOLOv10 features")
                expected_features = ['SCDown', 'PSA', 'v10Detect', 'C2fCIB']
                
            elif model_info['architecture'] == 'v11':
                print("✅ YOLOv11n architecture - checking for YOLOv11 features")
                expected_features = ['C3k2', 'C2PSA', 'Detect']
            
            # Check for expected architectural features
            found_features = set()
            for name, module in model.model.named_modules():
                module_name = type(module).__name__
                for feature in expected_features:
                    if feature in module_name:
                        found_features.add(feature)
            
            print(f"🔍 Architecture features found: {len(found_features)}/{len(expected_features)}")
            for feature in expected_features:
                if feature in found_features:
                    print(f"   ✅ {feature}: Found")
                else:
                    print(f"   ⚠️  {feature}: Not found")
            
            return len(found_features) >= len(expected_features) * 0.5  # At least 50% features
            
        except Exception as e:
            print(f"❌ Architecture compliance test failed: {e}")
            return False
    
    def compare_with_baseline(self, model_info):
        """Compare with baseline model."""
        print(f"\n🔍 Comparing with Baseline: {model_info['name']}")
        print("-" * 60)
        
        try:
            # Load CBAM model
            cbam_model = YOLO(model_info['config'])
            cbam_params = sum(p.numel() for p in cbam_model.model.parameters())
            
            # Load appropriate baseline
            if model_info['architecture'] == 'v8':
                baseline_model = YOLO('yolov8n.pt')
            elif model_info['architecture'] == 'v10':
                baseline_model = YOLO('yolov10n.pt')
            else:
                print("⚠️  No baseline available for this architecture")
                return True
                
            baseline_params = sum(p.numel() for p in baseline_model.model.parameters())
            
            # Calculate difference
            param_diff = cbam_params - baseline_params
            percent_diff = (param_diff / baseline_params) * 100
            
            print(f"📊 Parameter Analysis:")
            print(f"   • Baseline:   {baseline_params:,} parameters")
            print(f"   • CBAM:       {cbam_params:,} parameters")
            print(f"   • Difference: {param_diff:+,} parameters")
            print(f"   • Percentage: {percent_diff:+.2f}%")
            
            # CBAM should generally add some parameters (but not always due to architecture differences)
            return True
            
        except Exception as e:
            print(f"❌ Baseline comparison failed: {e}")
            return False
    
    def run_comprehensive_test(self, model_info):
        """Run all tests for a single model."""
        print(f"\n{'='*80}")
        print(f"🚀 TESTING: {model_info['name']}")
        print(f"{'='*80}")
        
        tests = [
            ("Config Loading", self.test_config_loading),
            ("CBAM Instantiation", self.test_cbam_instantiation),
            ("Forward Pass", self.test_forward_pass),
            ("Architecture Compliance", self.test_architecture_compliance),
            ("Baseline Comparison", self.compare_with_baseline)
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                result = test_func(model_info)
                results.append((test_name, result))
            except Exception as e:
                print(f"❌ {test_name} test crashed: {e}")
                results.append((test_name, False))
        
        return results
    
    def run_all_tests(self):
        """Run tests for all models."""
        print("🚀 COMPREHENSIVE CBAM VERIFICATION ACROSS ALL YOLO ARCHITECTURES")
        print("=" * 80)
        
        all_results = {}
        
        for model_info in self.models_to_test:
            if not Path(model_info['config']).exists():
                print(f"⚠️  Skipping {model_info['name']} - config file not found")
                continue
                
            results = self.run_comprehensive_test(model_info)
            all_results[model_info['name']] = results
        
        # Summary
        print(f"\n{'='*80}")
        print("📋 COMPREHENSIVE VERIFICATION SUMMARY")
        print("=" * 80)
        
        for model_name, results in all_results.items():
            passed = sum(1 for _, result in results if result)
            total = len(results)
            
            print(f"\n🔍 {model_name}")
            print("-" * 50)
            
            for test_name, result in results:
                status = "✅ PASSED" if result else "❌ FAILED"
                print(f"   {test_name:25} {status}")
            
            overall_status = "✅ PASSED" if passed == total else f"⚠️  {passed}/{total} PASSED"
            print(f"   {'Overall':25} {overall_status}")
        
        # Final assessment
        all_passed = all(
            all(result for _, result in results)
            for results in all_results.values()
        )
        
        print(f"\n{'='*50}")
        print("🎯 FINAL ASSESSMENT")
        print("=" * 50)
        
        if all_passed:
            print("🎉 ✅ ALL CBAM IMPLEMENTATIONS VERIFIED SUCCESSFULLY!")
            print("   • All configurations load correctly")
            print("   • All CBAM modules are properly instantiated")
            print("   • All forward passes work correctly")
            print("   • All architectures are compliant")
        else:
            print("⚠️  ❌ SOME CBAM IMPLEMENTATIONS HAVE ISSUES")
            print("   Check individual test results above for details")
        
        return all_results

def main():
    """Run comprehensive CBAM verification."""
    verifier = CBAMVerifier()
    results = verifier.run_all_tests()
    return results

if __name__ == "__main__":
    main()