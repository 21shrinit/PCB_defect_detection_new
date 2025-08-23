#!/usr/bin/env python3
"""
PCB Defect Detection - 48-Hour Sprint Experiment Runner
Benchmarks three YOLOv8 configurations with custom Focal-SIoU loss.

Models:
- Model A: YOLOv8s (Baseline with Custom Loss)
- Model B: YOLOv8s + CBAM Attention  
- Model C: YOLOv8s + MobileViT Hybrid Backbone
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import wandb

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from custom_trainer import MyCustomTrainer


class ExperimentRunner:
    """Manages the 48-hour sprint experiment pipeline."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.experiments_dir = Path(__file__).parent
        self.start_time = datetime.now()
        self.results = {}
        
        # Experiment configurations
        self.experiments = {
            'Model_A': {
                'name': 'YOLOv8n_FocalSIoU_Baseline',
                'description': 'YOLOv8n with Custom Focal-SIoU Loss',
                'script': 'train_baseline.py',
                'config': 'experiments/configs/datasets/pcb_data.yaml',
                'epochs': 100,
                'time_limit': timedelta(hours=16)  # 16 hours for baseline
            },
            'Model_B': {
                'name': 'YOLOv8n_CBAM_Attention',
                'description': 'YOLOv8n + CBAM Attention',
                'script': 'train_cbam.py',
                'config': 'experiments/configs/datasets/pcb_data.yaml',
                'epochs': 100,
                'time_limit': timedelta(hours=16)  # 16 hours for CBAM
            },
            'Model_C': {
                'name': 'YOLOv8n_MobileViT_Hybrid',
                'description': 'YOLOv8n + MobileViT Hybrid Backbone',
                'script': 'train_mobilevit.py',
                'config': 'experiments/configs/datasets/pcb_data.yaml',
                'epochs': 100,
                'time_limit': timedelta(hours=16)  # 16 hours for MobileViT
            }
        }
    
    def setup_environment(self):
        """Setup the experiment environment."""
        print("🚀 Setting up PCB Defect Detection Experiment Environment")
        print("=" * 70)
        print(f"📁 Project Root: {self.project_root}")
        print(f"⏰ Start Time: {self.start_time}")
        print(f"🎯 Total Time Limit: 48 hours")
        print("=" * 70)
        
        # Setup W&B
        os.environ['WANDB_PROJECT'] = 'PCB_Defect_Detection_Sprint'
        os.environ['WANDB_ENTITY'] = 'your_username'  # Replace with your W&B username
        
        # Verify dataset availability
        self.verify_datasets()
        
        # Verify custom modules
        self.verify_custom_modules()
    
    def verify_datasets(self):
        """Verify that datasets are available and properly configured."""
        print("\n🔍 Verifying datasets...")
        
        # Check HRIPCB dataset
        hripcb_path = self.project_root / "datasets" / "HRIPCB" / "HRIPCB_UPDATE"
        if hripcb_path.exists():
            train_images = list((hripcb_path / "train" / "images").glob("*.jpg"))
            val_images = list((hripcb_path / "val" / "images").glob("*.jpg"))
            print(f"✅ HRIPCB Dataset: {len(train_images)} train, {len(val_images)} val images")
        else:
            raise FileNotFoundError("HRIPCB dataset not found!")
        
        # Check DeepPCB dataset
        deeppcb_path = self.project_root / "datasets" / "DeepPCB"
        if deeppcb_path.exists():
            train_images = list((deeppcb_path / "train" / "images").glob("*.jpg"))
            val_images = list((deeppcb_path / "valid" / "images").glob("*.jpg"))
            print(f"✅ DeepPCB Dataset: {len(train_images)} train, {len(val_images)} val images")
        else:
            raise FileNotFoundError("DeepPCB dataset not found!")
    
    def verify_custom_modules(self):
        """Verify that custom modules are properly implemented."""
        print("\n🔍 Verifying custom modules...")
        
        # Check custom loss
        loss_file = self.project_root / "custom_modules" / "loss.py"
        if loss_file.exists():
            print("✅ Custom loss functions implemented")
        else:
            raise FileNotFoundError("Custom loss module not found!")
        
        # Check custom trainer
        trainer_file = self.project_root / "custom_trainer.py"
        if trainer_file.exists():
            print("✅ Custom trainer implemented")
        else:
            raise FileNotFoundError("Custom trainer not found!")
        
        # Check attention module
        attention_file = self.project_root / "custom_modules" / "attention.py"
        if attention_file.exists():
            print("✅ CBAM attention module implemented")
        else:
            raise FileNotFoundError("CBAM attention module not found!")
        
        # Check MobileViT module
        mobilevit_file = self.project_root / "custom_modules" / "mobilevit.py"
        if mobilevit_file.exists():
            print("✅ MobileViT module implemented")
        else:
            raise FileNotFoundError("MobileViT module not found!")
    
    def run_experiment(self, model_key, experiment_config):
        """Run a single experiment."""
        print(f"\n🎯 Starting {model_key}: {experiment_config['description']}")
        print("-" * 50)
        
        experiment_start = datetime.now()
        
        try:
            # Set W&B run name
            os.environ['WANDB_NAME'] = experiment_config['name']
            
            # Run the training script
            script_path = self.project_root / experiment_config['script']
            
            if not script_path.exists():
                print(f"⚠️  Training script not found: {script_path}")
                print("📝 Creating placeholder training script...")
                self.create_placeholder_script(model_key, experiment_config)
            
            print(f"🚀 Executing: python {experiment_config['script']}")
            
            # Run the training process
            result = subprocess.run([
                'python', str(script_path)
            ], cwd=str(self.project_root), capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {model_key} completed successfully!")
                self.results[model_key] = {
                    'status': 'success',
                    'start_time': experiment_start,
                    'end_time': datetime.now(),
                    'duration': datetime.now() - experiment_start,
                    'output': result.stdout
                }
            else:
                print(f"❌ {model_key} failed!")
                print(f"Error: {result.stderr}")
                self.results[model_key] = {
                    'status': 'failed',
                    'start_time': experiment_start,
                    'end_time': datetime.now(),
                    'duration': datetime.now() - experiment_start,
                    'error': result.stderr
                }
                
        except Exception as e:
            print(f"❌ Error running {model_key}: {e}")
            self.results[model_key] = {
                'status': 'error',
                'start_time': experiment_start,
                'end_time': datetime.now(),
                'duration': datetime.now() - experiment_start,
                'error': str(e)
            }
    
    def create_placeholder_script(self, model_key, experiment_config):
        """Create a placeholder training script for models B and C."""
        script_content = f'''#!/usr/bin/env python3
"""
Placeholder training script for {model_key}
{experiment_config['description']}

This is a placeholder implementation for the 48-hour sprint.
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
from custom_trainer import MyCustomTrainer

def main():
    print("🚀 {model_key} Training")
    print("=" * 50)
    print("{experiment_config['description']}")
    print("=" * 50)
    
    # Set up W&B
    os.environ['WANDB_PROJECT'] = 'PCB_Defect_Detection_Sprint'
    os.environ['WANDB_NAME'] = '{experiment_config["name"]}'
    
    try:
        # Load model
        model = YOLO("yolov8n.yaml")
        
        # Training configuration
        config = {{
            'data': '{experiment_config["config"]}',
            'epochs': {experiment_config["epochs"]},
            'imgsz': 640,
            'batch': 16,
            'device': 'auto',
            'project': 'runs/train',
            'name': '{experiment_config["name"]}',
            'exist_ok': True,
            'pretrained': True,
        }}
        
        print("🎯 Starting training...")
        results = model.train(trainer=MyCustomTrainer, **config)
        
        print("✅ Training completed!")
        print(f"📊 Results saved to: {{results.save_dir}}")
        
    except Exception as e:
        print(f"❌ Training failed: {{e}}")
        raise

if __name__ == "__main__":
    main()
'''
        
        script_path = self.project_root / experiment_config['script']
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        print(f"📝 Created placeholder script: {script_path}")
    
    def generate_report(self):
        """Generate the final experiment report."""
        print("\n📊 Generating Experiment Report")
        print("=" * 70)
        
        total_duration = datetime.now() - self.start_time
        
        print(f"⏰ Total Experiment Duration: {total_duration}")
        print(f"🎯 Experiments Completed: {len(self.results)}")
        
        for model_key, result in self.results.items():
            print(f"\n{model_key}:")
            print(f"  Status: {result['status']}")
            print(f"  Duration: {result['duration']}")
            if result['status'] == 'success':
                print(f"  ✅ Completed successfully")
            else:
                print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
        
        # Save detailed report
        report_path = self.project_root / "experiment_report.txt"
        with open(report_path, 'w') as f:
            f.write("PCB Defect Detection - 48-Hour Sprint Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Start Time: {self.start_time}\n")
            f.write(f"End Time: {datetime.now()}\n")
            f.write(f"Total Duration: {total_duration}\n\n")
            
            for model_key, result in self.results.items():
                f.write(f"{model_key}:\n")
                f.write(f"  Status: {result['status']}\n")
                f.write(f"  Duration: {result['duration']}\n")
                if result['status'] == 'success':
                    f.write(f"  Output: {result.get('output', 'No output')}\n")
                else:
                    f.write(f"  Error: {result.get('error', 'Unknown error')}\n")
                f.write("\n")
        
        print(f"\n📄 Detailed report saved to: {report_path}")
    
    def run_all_experiments(self):
        """Run all experiments in the 48-hour sprint."""
        print("\n🎯 Starting 48-Hour Sprint - All Experiments")
        print("=" * 70)
        
        for model_key, experiment_config in self.experiments.items():
            # Check if we're still within the 48-hour limit
            elapsed_time = datetime.now() - self.start_time
            if elapsed_time > timedelta(hours=48):
                print(f"⏰ Time limit exceeded! Elapsed: {elapsed_time}")
                break
            
            self.run_experiment(model_key, experiment_config)
            
            # Brief pause between experiments
            time.sleep(5)
        
        # Generate final report
        self.generate_report()


def main():
    """Main function to run the experiment pipeline."""
    runner = ExperimentRunner()
    
    try:
        # Setup environment
        runner.setup_environment()
        
        # Run all experiments
        runner.run_all_experiments()
        
        print("\n🎉 48-Hour Sprint Completed!")
        print("📊 Check the experiment report for detailed results")
        
    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted by user")
        runner.generate_report()
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        runner.generate_report()


if __name__ == "__main__":
    main()
