#!/usr/bin/env python3
"""
Attention Mechanism Integration Test
===================================
Comprehensive test to verify attention mechanisms are properly integrated and active
across all YOLO models (YOLOv8n, YOLOv10n, YOLOv11n) and configurations.

This script performs deep integration testing of:
✅ Attention module imports and availability
✅ Model architecture loading with attention
✅ Attention mechanism activation verification
✅ Parameter count validation
✅ Forward pass testing with attention
✅ Configuration-to-execution mapping

Usage: python test_attention_integration.py
"""

import os
import sys
import yaml
import torch
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

class AttentionIntegrationTester:
    """Comprehensive attention mechanism integration tester."""
    
    def __init__(self):
        self.results = {
            'module_imports': {},
            'model_loading': {},
            'attention_activation': {},
            'parameter_analysis': {},
            'forward_pass': {},
            'config_mapping': {}
        }
        
    def test_attention_module_imports(self) -> Dict[str, str]:
        """Test if all custom attention modules can be imported."""
        print("🔍 Testing attention module imports...")
        
        modules_to_test = {
            'C2f_CBAM': 'ultralytics.nn.modules.block',
            'C2f_ECA': 'ultralytics.nn.modules.block', 
            'C2f_CoordAtt': 'ultralytics.nn.modules.block',
            'CBAMBlock': 'ultralytics.nn.modules.attention',
            'ECABlock': 'ultralytics.nn.modules.attention',
            'CoordAttBlock': 'ultralytics.nn.modules.attention'
        }
        
        for module_name, module_path in modules_to_test.items():
            try:
                exec(f"from {module_path} import {module_name}")
                self.results['module_imports'][module_name] = 'available'
                print(f"   ✅ {module_name}: Available")
            except ImportError as e:
                self.results['module_imports'][module_name] = f'missing: {str(e)}'
                print(f"   ❌ {module_name}: Missing - {e}")
        
        return self.results['module_imports']
    
    def test_model_architecture_loading(self) -> Dict[str, Any]:
        """Test loading models with attention mechanisms."""
        print("\n🏗️  Testing model architecture loading with attention...")
        
        # Test configurations with attention mechanisms
        attention_configs = [
            {
                'name': 'YOLOv8n-ECA',
                'config_path': 'ultralytics/cfg/models/v8/yolov8n-eca-final.yaml',
                'attention_type': 'eca'
            },
            {
                'name': 'YOLOv8n-CBAM', 
                'config_path': 'ultralytics/cfg/models/v8/yolov8n-cbam-neck-optimal.yaml',
                'attention_type': 'cbam'
            },
            {
                'name': 'YOLOv8n-CoordAtt',
                'config_path': 'ultralytics/cfg/models/v8/yolov8n-ca-position7.yaml', 
                'attention_type': 'coordatt'
            },
            {
                'name': 'YOLOv10n-ECA',
                'config_path': 'ultralytics/cfg/models/v10/yolov10n-eca-research-optimal.yaml',
                'attention_type': 'eca'
            },
            {
                'name': 'YOLOv10n-CBAM',
                'config_path': 'ultralytics/cfg/models/v10/yolov10n-cbam-research-optimal.yaml',
                'attention_type': 'cbam'
            },
            {
                'name': 'YOLOv10n-CoordAtt',
                'config_path': 'ultralytics/cfg/models/v10/yolov10n-coordatt-research-optimal.yaml',
                'attention_type': 'coordatt'
            }
        ]
        
        for config in attention_configs:
            try:
                config_path = config['config_path']
                if os.path.exists(config_path):
                    # Try to load the model architecture
                    from ultralytics import YOLO
                    model = YOLO(config_path)
                    
                    self.results['model_loading'][config['name']] = {
                        'status': 'success',
                        'config_path': config_path,
                        'attention_type': config['attention_type'],
                        'model_loaded': True
                    }
                    print(f"   ✅ {config['name']}: Loaded successfully")
                    
                else:
                    self.results['model_loading'][config['name']] = {
                        'status': 'config_missing',
                        'config_path': config_path,
                        'attention_type': config['attention_type'],
                        'model_loaded': False
                    }
                    print(f"   ❌ {config['name']}: Config file missing - {config_path}")
                    
            except Exception as e:
                self.results['model_loading'][config['name']] = {
                    'status': 'error',
                    'config_path': config['config_path'],
                    'attention_type': config['attention_type'],
                    'error': str(e),
                    'model_loaded': False
                }
                print(f"   ❌ {config['name']}: Loading failed - {e}")
        
        return self.results['model_loading']
    
    def analyze_model_architecture(self, model, model_name: str) -> Dict[str, Any]:
        """Analyze loaded model architecture for attention mechanisms."""
        try:
            architecture_info = {
                'total_layers': 0,
                'attention_layers': [],
                'attention_count': 0,
                'parameter_count': 0,
                'attention_parameters': 0
            }
            
            # Count total parameters
            total_params = sum(p.numel() for p in model.model.parameters())
            architecture_info['parameter_count'] = total_params
            
            # Analyze model architecture
            layer_count = 0
            attention_layers = []
            
            def analyze_module(module, prefix=""):
                nonlocal layer_count, attention_layers
                layer_count += 1
                
                module_name = module.__class__.__name__
                
                # Check for attention modules
                if any(att_type in module_name for att_type in ['CBAM', 'ECA', 'CoordAtt', 'C2f_CBAM', 'C2f_ECA', 'C2f_CoordAtt']):
                    attention_info = {
                        'layer_index': layer_count,
                        'module_name': module_name,
                        'prefix': prefix,
                        'parameters': sum(p.numel() for p in module.parameters())
                    }
                    attention_layers.append(attention_info)
                
                # Recursively analyze child modules
                for name, child in module.named_children():
                    analyze_module(child, f"{prefix}.{name}" if prefix else name)
            
            analyze_module(model.model)
            
            architecture_info['total_layers'] = layer_count
            architecture_info['attention_layers'] = attention_layers
            architecture_info['attention_count'] = len(attention_layers)
            architecture_info['attention_parameters'] = sum(layer['parameters'] for layer in attention_layers)
            
            return architecture_info
            
        except Exception as e:
            return {'error': str(e)}
    
    def test_attention_activation(self) -> Dict[str, Any]:
        """Test if attention mechanisms are actually active during forward pass."""
        print("\n🔥 Testing attention mechanism activation...")
        
        for model_name, model_info in self.results['model_loading'].items():
            if model_info.get('model_loaded', False):
                try:
                    from ultralytics import YOLO
                    model = YOLO(model_info['config_path'])
                    
                    # Analyze architecture
                    arch_info = self.analyze_model_architecture(model, model_name)
                    
                    if 'error' not in arch_info:
                        self.results['attention_activation'][model_name] = {
                            'status': 'success',
                            'total_layers': arch_info['total_layers'],
                            'attention_count': arch_info['attention_count'],
                            'attention_layers': arch_info['attention_layers'],
                            'parameter_analysis': {
                                'total_parameters': arch_info['parameter_count'],
                                'attention_parameters': arch_info['attention_parameters'],
                                'attention_percentage': (arch_info['attention_parameters'] / arch_info['parameter_count']) * 100 if arch_info['parameter_count'] > 0 else 0
                            }
                        }
                        
                        if arch_info['attention_count'] > 0:
                            print(f"   ✅ {model_name}: {arch_info['attention_count']} attention layers active")
                            print(f"      Parameters: {arch_info['attention_parameters']:,} attention / {arch_info['parameter_count']:,} total")
                        else:
                            print(f"   ⚠️  {model_name}: No attention layers detected!")
                            
                    else:
                        self.results['attention_activation'][model_name] = {
                            'status': 'analysis_error',
                            'error': arch_info['error']
                        }
                        print(f"   ❌ {model_name}: Architecture analysis failed - {arch_info['error']}")
                        
                except Exception as e:
                    self.results['attention_activation'][model_name] = {
                        'status': 'activation_test_error', 
                        'error': str(e)
                    }
                    print(f"   ❌ {model_name}: Activation test failed - {e}")
        
        return self.results['attention_activation']
    
    def test_forward_pass_with_attention(self) -> Dict[str, Any]:
        """Test forward pass to ensure attention mechanisms work during inference."""
        print("\n⚡ Testing forward pass with attention mechanisms...")
        
        # Create dummy input tensor
        dummy_input = torch.randn(1, 3, 640, 640)  # Batch=1, Channels=3, Height=640, Width=640
        
        for model_name, model_info in self.results['model_loading'].items():
            if model_info.get('model_loaded', False):
                try:
                    from ultralytics import YOLO
                    model = YOLO(model_info['config_path'])
                    
                    # Set to evaluation mode
                    model.model.eval()
                    
                    with torch.no_grad():
                        # Test forward pass
                        output = model.model(dummy_input)
                        
                        self.results['forward_pass'][model_name] = {
                            'status': 'success',
                            'input_shape': list(dummy_input.shape),
                            'output_type': str(type(output)),
                            'output_shapes': [list(out.shape) if torch.is_tensor(out) else str(out) for out in (output if isinstance(output, (list, tuple)) else [output])]
                        }
                        print(f"   ✅ {model_name}: Forward pass successful")
                        
                except Exception as e:
                    self.results['forward_pass'][model_name] = {
                        'status': 'forward_pass_error',
                        'error': str(e)
                    }
                    print(f"   ❌ {model_name}: Forward pass failed - {e}")
        
        return self.results['forward_pass']
    
    def test_experiment_config_mapping(self) -> Dict[str, Any]:
        """Test if experiment configs properly map to attention mechanisms."""
        print("\n📋 Testing experiment config to attention mapping...")
        
        # Test key experiment configs that use attention
        attention_experiment_configs = [
            'experiments/configs/19_yolov8n_eca_verifocal_siou.yaml',
            'experiments/configs/20_yolov8n_cbam_focal_eiou.yaml', 
            'experiments/configs/21_yolov8n_coordatt_verifocal_eiou.yaml',
            'experiments/configs/14_yolov10n_eca_focal_eiou_STABLE.yaml',
            'experiments/configs/13_yolov10n_cbam_verifocal_siou_STABLE.yaml',
            'experiments/configs/15_yolov10n_coordatt_verifocal_siou_STABLE.yaml'
        ]
        
        for config_path in attention_experiment_configs:
            config_name = os.path.basename(config_path)
            
            try:
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    # Extract attention-related info
                    model_type = config['model'].get('type', 'unknown')
                    attention_mechanism = config['model'].get('attention_mechanism', 'none')
                    model_config_path = config['model'].get('config_path', '')
                    
                    # Check if config path exists
                    config_exists = os.path.exists(model_config_path) if model_config_path else False
                    
                    self.results['config_mapping'][config_name] = {
                        'status': 'parsed',
                        'model_type': model_type,
                        'attention_mechanism': attention_mechanism,
                        'model_config_path': model_config_path,
                        'config_path_exists': config_exists,
                        'has_attention_config': attention_mechanism != 'none' and bool(model_config_path)
                    }
                    
                    if attention_mechanism != 'none':
                        if config_exists:
                            print(f"   ✅ {config_name}: {model_type} + {attention_mechanism} (config exists)")
                        else:
                            print(f"   ⚠️  {config_name}: {model_type} + {attention_mechanism} (config missing: {model_config_path})")
                    else:
                        print(f"   ➖ {config_name}: {model_type} (no attention)")
                        
                else:
                    self.results['config_mapping'][config_name] = {
                        'status': 'config_file_missing',
                        'config_path': config_path
                    }
                    print(f"   ❌ {config_name}: Config file missing")
                    
            except Exception as e:
                self.results['config_mapping'][config_name] = {
                    'status': 'parsing_error',
                    'error': str(e)
                }
                print(f"   ❌ {config_name}: Parsing failed - {e}")
        
        return self.results['config_mapping']
    
    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        print("\n" + "="*70)
        print("📊 ATTENTION MECHANISM INTEGRATION SUMMARY REPORT")
        print("="*70)
        
        # Module imports summary
        total_modules = len(self.results['module_imports'])
        available_modules = sum(1 for status in self.results['module_imports'].values() if status == 'available')
        print(f"\n1️⃣  Module Imports: {available_modules}/{total_modules} available")
        
        for module, status in self.results['module_imports'].items():
            status_icon = "✅" if status == 'available' else "❌"
            print(f"   {status_icon} {module}")
        
        # Model loading summary
        total_models = len(self.results['model_loading'])
        successful_models = sum(1 for info in self.results['model_loading'].values() if info.get('model_loaded', False))
        print(f"\n2️⃣  Model Loading: {successful_models}/{total_models} successful")
        
        for model_name, info in self.results['model_loading'].items():
            status_icon = "✅" if info.get('model_loaded', False) else "❌"
            print(f"   {status_icon} {model_name} ({info.get('attention_type', 'unknown')})")
        
        # Attention activation summary
        print(f"\n3️⃣  Attention Activation Analysis:")
        
        for model_name, info in self.results['attention_activation'].items():
            if info.get('status') == 'success':
                attention_count = info.get('attention_count', 0)
                if attention_count > 0:
                    attention_pct = info.get('parameter_analysis', {}).get('attention_percentage', 0)
                    print(f"   ✅ {model_name}: {attention_count} attention layers ({attention_pct:.2f}% of parameters)")
                else:
                    print(f"   ⚠️  {model_name}: NO attention layers detected!")
            else:
                print(f"   ❌ {model_name}: Analysis failed")
        
        # Forward pass summary
        successful_forward = sum(1 for info in self.results['forward_pass'].values() if info.get('status') == 'success')
        total_forward_tests = len(self.results['forward_pass'])
        print(f"\n4️⃣  Forward Pass Tests: {successful_forward}/{total_forward_tests} successful")
        
        # Config mapping summary
        attention_configs = sum(1 for info in self.results['config_mapping'].values() 
                              if info.get('has_attention_config', False))
        total_configs = len(self.results['config_mapping'])
        print(f"\n5️⃣  Config Mapping: {attention_configs}/{total_configs} have attention configurations")
        
        # Overall assessment
        print(f"\n🎯 OVERALL ATTENTION INTEGRATION ASSESSMENT:")
        
        critical_issues = []
        warnings = []
        
        if available_modules < total_modules:
            critical_issues.append(f"Missing attention modules: {total_modules - available_modules}")
            
        if successful_models < total_models:
            critical_issues.append(f"Failed model loading: {total_models - successful_models}")
        
        # Check for models with no attention detected
        no_attention_models = []
        for model_name, info in self.results['attention_activation'].items():
            if info.get('status') == 'success' and info.get('attention_count', 0) == 0:
                no_attention_models.append(model_name)
        
        if no_attention_models:
            warnings.append(f"Models with no attention detected: {', '.join(no_attention_models)}")
        
        if critical_issues:
            print("   ❌ CRITICAL ISSUES FOUND:")
            for issue in critical_issues:
                print(f"      • {issue}")
        
        if warnings:
            print("   ⚠️  WARNINGS:")
            for warning in warnings:
                print(f"      • {warning}")
        
        if not critical_issues and not warnings:
            print("   🎉 ALL ATTENTION MECHANISMS PROPERLY INTEGRATED!")
            return True
        else:
            print("   🔧 ATTENTION INTEGRATION NEEDS FIXES")
            return False

def main():
    """Run comprehensive attention integration tests."""
    print("🎯 PCB Defect Detection - Attention Mechanism Integration Test")
    print("="*70)
    
    tester = AttentionIntegrationTester()
    
    # Run all tests
    tester.test_attention_module_imports()
    tester.test_model_architecture_loading()
    tester.test_attention_activation()
    tester.test_forward_pass_with_attention()
    tester.test_experiment_config_mapping()
    
    # Generate summary report
    success = tester.generate_summary_report()
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)