#!/usr/bin/env python3
"""
Comprehensive Configuration Validation Script
=============================================

Validates all experiment configuration files to ensure they use the
FIXED loss function and attention mechanism integrations correctly.

Checks:
✅ Model config paths exist and are correct
✅ Loss function configurations are valid
✅ Attention mechanism references are accurate
✅ Required sections are present
✅ Parameter structures are correct
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

class ConfigValidator:
    """Validates experiment configuration files for correctness."""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.validated_count = 0
        
        # Valid loss types (based on our implementation)
        self.valid_loss_types = [
            'standard', 'ciou', 'siou', 'eiou', 'giou',
            'focal', 'varifocal',
            'focal_ciou', 'focal_siou', 'focal_eiou', 'focal_giou',
            'verifocal_ciou', 'verifocal_siou', 'verifocal_eiou', 'verifocal_giou'
        ]
        
        # Valid attention mechanisms
        self.valid_attention = ['none', 'eca', 'cbam', 'coordatt', 'eca_enhanced']
        
        # Valid model types
        self.valid_models = ['yolov8n', 'yolov8s', 'yolov10n', 'yolov10s', 'yolo11n', 'yolo11s']

    def validate_model_config_path(self, config_path: str, model_config: Dict) -> List[str]:
        """Validate model configuration path exists."""
        issues = []
        
        if 'config_path' in model_config:
            model_path = PROJECT_ROOT / model_config['config_path']
            if not model_path.exists():
                issues.append(f"Model config path does not exist: {model_config['config_path']}")
            else:
                # Check if it's the correct attention mechanism file
                attention = model_config.get('attention_mechanism', 'none')
                filename = model_path.name
                
                if attention == 'eca' and 'eca' not in filename:
                    issues.append(f"ECA attention specified but model path doesn't contain 'eca': {filename}")
                elif attention == 'cbam' and 'cbam' not in filename:
                    issues.append(f"CBAM attention specified but model path doesn't contain 'cbam': {filename}")
                elif attention == 'coordatt' and ('ca' not in filename and 'coordatt' not in filename):
                    issues.append(f"CoordAtt attention specified but model path doesn't contain 'ca' or 'coordatt': {filename}")
        
        return issues

    def validate_loss_configuration(self, loss_config: Dict) -> List[str]:
        """Validate loss function configuration."""
        issues = []
        
        if 'type' in loss_config:
            loss_type = loss_config['type']
            if loss_type not in self.valid_loss_types:
                issues.append(f"Invalid loss type: {loss_type}. Valid types: {self.valid_loss_types}")
        
        # Check for proper weight configurations
        for weight in ['box_weight', 'cls_weight', 'dfl_weight']:
            if weight in loss_config:
                try:
                    float(loss_config[weight])
                except (ValueError, TypeError):
                    issues.append(f"Invalid {weight}: {loss_config[weight]} (must be numeric)")
        
        return issues

    def validate_training_configuration(self, training_config: Dict) -> List[str]:
        """Validate training configuration structure."""
        issues = []
        warnings = []
        
        # Check for required fields
        required_fields = ['epochs', 'batch', 'imgsz']
        for field in required_fields:
            if field not in training_config:
                issues.append(f"Missing required training field: {field}")
        
        # Check for dataset path (two possible structures)
        has_data_path = False
        if 'dataset' in training_config and 'path' in training_config['dataset']:
            has_data_path = True
        elif 'data' in training_config and 'path' in training_config['data']:
            has_data_path = True
        
        if not has_data_path:
            issues.append("Missing dataset path. Expected training.dataset.path or data.path")
        
        # Validate loss configuration if present
        if 'loss' in training_config:
            issues.extend(self.validate_loss_configuration(training_config['loss']))
        
        return issues, warnings

    def validate_config_structure(self, config: Dict, config_path: str) -> Tuple[List[str], List[str]]:
        """Validate overall configuration structure."""
        issues = []
        warnings = []
        
        # Check required top-level sections
        required_sections = ['experiment', 'model', 'training']
        for section in required_sections:
            if section not in config:
                issues.append(f"Missing required section: {section}")
        
        # Validate model section
        if 'model' in config:
            model_config = config['model']
            
            # Check model type
            if 'type' in model_config:
                model_type = model_config['type']
                if model_type not in self.valid_models:
                    warnings.append(f"Unusual model type: {model_type}")
            
            # Check attention mechanism
            if 'attention_mechanism' in model_config:
                attention = model_config['attention_mechanism']
                if attention not in self.valid_attention:
                    issues.append(f"Invalid attention mechanism: {attention}. Valid: {self.valid_attention}")
            
            # Validate model config path
            issues.extend(self.validate_model_config_path(config_path, model_config))
        
        # Validate training section
        if 'training' in config:
            train_issues, train_warnings = self.validate_training_configuration(config['training'])
            issues.extend(train_issues)
            warnings.extend(train_warnings)
        
        return issues, warnings

    def validate_file(self, config_path: Path) -> Dict:
        """Validate a single configuration file."""
        result = {
            'path': str(config_path),
            'valid': True,
            'issues': [],
            'warnings': []
        }
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if not config:
                result['valid'] = False
                result['issues'].append("Empty or invalid YAML file")
                return result
            
            # Validate structure
            issues, warnings = self.validate_config_structure(config, str(config_path))
            result['issues'].extend(issues)
            result['warnings'].extend(warnings)
            
            if issues:
                result['valid'] = False
            
            self.validated_count += 1
            
        except yaml.YAMLError as e:
            result['valid'] = False
            result['issues'].append(f"YAML parsing error: {e}")
        except Exception as e:
            result['valid'] = False
            result['issues'].append(f"Validation error: {e}")
        
        return result

    def validate_all_configs(self) -> Dict:
        """Validate all configuration files."""
        config_dir = PROJECT_ROOT / "experiments" / "configs"
        
        # Find all YAML config files
        config_files = []
        for pattern in ["*.yaml", "**/*.yaml"]:
            config_files.extend(config_dir.glob(pattern))
        
        # Filter out dataset configs and templates
        experiment_configs = [
            f for f in config_files 
            if not any(skip in str(f) for skip in ['datasets', 'templates', 'models'])
        ]
        
        results = {
            'total_configs': len(experiment_configs),
            'valid_configs': 0,
            'invalid_configs': 0,
            'configs_with_warnings': 0,
            'results': []
        }
        
        print(f"VALIDATING {len(experiment_configs)} EXPERIMENT CONFIGURATIONS")
        print("=" * 70)
        
        for config_path in sorted(experiment_configs):
            result = self.validate_file(config_path)
            results['results'].append(result)
            
            if result['valid']:
                results['valid_configs'] += 1
                status = "VALID"
            else:
                results['invalid_configs'] += 1
                status = "INVALID"
            
            if result['warnings']:
                results['configs_with_warnings'] += 1
            
            # Print compact status
            relative_path = config_path.relative_to(PROJECT_ROOT)
            print(f"{status:10} | {relative_path}")
            
            # Print issues and warnings
            for issue in result['issues']:
                print(f"           ERROR: {issue}")
            for warning in result['warnings']:
                print(f"           WARNING: {warning}")
        
        return results

    def generate_report(self, results: Dict):
        """Generate comprehensive validation report."""
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        
        total = results['total_configs']
        valid = results['valid_configs']
        invalid = results['invalid_configs']
        with_warnings = results['configs_with_warnings']
        
        print(f"Total Configurations: {total}")
        print(f"Valid Configurations: {valid} ({valid/total*100:.1f}%)")
        print(f"Invalid Configurations: {invalid} ({invalid/total*100:.1f}%)")
        print(f"Configurations with Warnings: {with_warnings} ({with_warnings/total*100:.1f}%)")
        
        if invalid == 0:
            print("\nALL CONFIGURATIONS ARE VALID!")
            print("Ready to run experiments with confidence!")
        else:
            print(f"\n{invalid} CONFIGURATIONS NEED ATTENTION")
            print("Fix issues before running experiments")
        
        # Detailed issue breakdown
        if invalid > 0:
            print("\nISSUES TO FIX:")
            issue_categories = {}
            
            for result in results['results']:
                if not result['valid']:
                    for issue in result['issues']:
                        category = issue.split(':')[0] if ':' in issue else issue.split('.')[0]
                        issue_categories[category] = issue_categories.get(category, 0) + 1
            
            for category, count in sorted(issue_categories.items()):
                print(f"   • {category}: {count} configs")
        
        return invalid == 0


def main():
    """Main validation function."""
    validator = ConfigValidator()
    results = validator.validate_all_configs()
    success = validator.generate_report(results)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)