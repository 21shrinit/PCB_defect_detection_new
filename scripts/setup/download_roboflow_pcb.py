#!/usr/bin/env python3
"""
Download PCB Defects Dataset from Roboflow using Roboflow API
- Supports Colab Secrets and environment variables for Roboflow API key
- Also supports passing --api-key via CLI
- Downloads from: https://universe.roboflow.com/rahul-jhj03/pcb-defects-dataset/dataset/2
"""

import os
import sys
import argparse
from pathlib import Path

# Try to import Colab userdata (optional)
try:
	from google.colab import userdata  # type: ignore
	IN_COLAB = True
except Exception:
	userdata = None
	IN_COLAB = False

from roboflow import Roboflow

def setup_environment():
	"""Setup the download environment."""
	# Add current directory to Python path
	sys.path.append(str(Path(__file__).parent))
	
	# Create datasets directory if it doesn't exist
	datasets_dir = Path("datasets")
	datasets_dir.mkdir(exist_ok=True)
	
	print("🚀 Setting up PCB Defects Dataset download environment...")
	print(f"📁 Working directory: {Path.cwd()}")
	print(f"📦 Datasets directory: {datasets_dir.absolute()}")

def get_roboflow_api_key(cli_api_key: str | None = None) -> str | None:
	"""Return Roboflow API key from CLI, Colab secrets or env variables."""
	if cli_api_key:
		print("🔐 Using Roboflow API key from CLI argument")
		return cli_api_key
	api_key = None
	if userdata is not None:
		try:
			api_key = userdata.get('ROBOFLOW_API_KEY')
			if api_key:
				print('🔐 Loaded Roboflow API key from Colab user secrets')
		except Exception:
			api_key = None
	if not api_key:
		api_key = os.environ.get('ROBOFLOW_API_KEY') or os.environ.get('ROBOFLOW_APIKEY')
		if api_key:
			print('🔐 Loaded Roboflow API key from environment variables')
	return api_key

def download_pcb_defects_dataset(cli_api_key: str | None = None):
	"""Download PCB Defects Dataset from Roboflow."""
	try:
		print("📥 Downloading PCB Defects Dataset from Roboflow...")
		print("🔗 Source: https://universe.roboflow.com/rahul-jhj03/pcb-defects-dataset/dataset/2")
		
		api_key = get_roboflow_api_key(cli_api_key)
		if not api_key:
			print("❌ Roboflow API key not found. Set ROBOFLOW_API_KEY in Colab secrets or env, or pass --api-key.")
			print("💡 To get your API key:")
			print("   1. Go to https://roboflow.com/")
			print("   2. Sign up/login")
			print("   3. Go to Settings > Roboflow API")
			print("   4. Copy your Private API Key")
			return False
		
		# Initialize Roboflow
		rf = Roboflow(api_key=api_key)
		
		# Get project and version from new Roboflow link
		# URL: https://universe.roboflow.com/rahul-jhj03/pcb-defects-dataset/dataset/2
		print("🔄 Connecting to Roboflow workspace...")
		project = rf.workspace("rahul-jhj03").project("pcb-defects-dataset")
		version = project.version(2)
		
		# Download dataset in YOLOv8 format
		print("🔄 Downloading dataset (this may take a few minutes)...")
		dataset = version.download("yolov8")
		
		print("✅ PCB Defects Dataset downloaded successfully!")
		print(f"📁 Dataset location: {Path.cwd()}")
		
		# Verify the download
		verify_dataset_structure()
		
		return True
		
	except Exception as e:
		print(f"❌ Error downloading PCB Defects Dataset: {e}")
		print("💡 Make sure you have a valid ROBOFLOW_API_KEY and internet connection")
		print("💡 Check if the dataset URL is correct: https://universe.roboflow.com/rahul-jhj03/pcb-defects-dataset/dataset/2")
		return False

def verify_dataset_structure():
	"""Verify the downloaded dataset structure."""
	try:
		print("🔍 Verifying dataset structure...")
		
		# Check for PCB-Defects-Dataset-2 directory (expected name based on Roboflow convention)
		possible_dirs = ["PCB-Defects-Dataset-2", "pcb-defects-dataset-2", "pcb-defects-dataset"]
		deeppcb_dir = None
		
		for dir_name in possible_dirs:
			test_dir = Path(dir_name)
			if test_dir.exists():
				deeppcb_dir = test_dir
				break
		
		if not deeppcb_dir:
			print("⚠️  Dataset directory not found. Looking for any downloaded directories...")
			# List all directories to help user identify the correct one
			current_dirs = [d for d in Path.cwd().iterdir() if d.is_dir()]
			print(f"📁 Available directories: {[d.name for d in current_dirs]}")
			return False
		
		print(f"✅ Found dataset directory: {deeppcb_dir}")
		
		# Check for train/val/test splits
		splits = ['train', 'valid', 'test']
		total_images = 0
		total_labels = 0
		
		for split in splits:
			split_dir = deeppcb_dir / split
			if split_dir.exists():
				images_dir = split_dir / 'images'
				labels_dir = split_dir / 'labels'
				
				if images_dir.exists() and labels_dir.exists():
					image_count = len(list(images_dir.glob('*.jpg'))) + len(list(images_dir.glob('*.png')))
					label_count = len(list(labels_dir.glob('*.txt')))
					total_images += image_count
					total_labels += label_count
					print(f"✅ {split}: {image_count} images, {label_count} labels")
				else:
					print(f"⚠️  {split}: Missing images or labels directory")
			else:
				print(f"⚠️  {split}: Directory not found")
		
		# Check data.yaml
		data_yaml = deeppcb_dir / 'data.yaml'
		if data_yaml.exists():
			print(f"✅ data.yaml found: {data_yaml}")
			
			# Read and display basic info from data.yaml
			try:
				import yaml
				with open(data_yaml, 'r') as f:
					data_config = yaml.safe_load(f)
				
				print(f"📊 Dataset Info:")
				print(f"   Classes: {data_config.get('nc', 'Unknown')} classes")
				print(f"   Names: {data_config.get('names', 'Not specified')}")
				
			except Exception as e:
				print(f"⚠️  Could not read data.yaml content: {e}")
		else:
			print("⚠️  data.yaml not found")
		
		print(f"📊 Total: {total_images} images, {total_labels} labels")
		print("🎉 PCB Defects Dataset verification completed!")
		
		# Generate configuration template
		generate_config_template(deeppcb_dir)
		
		return True
		
	except Exception as e:
		print(f"❌ Error verifying dataset: {e}")
		return False

def generate_config_template(dataset_dir: Path):
	"""Generate a configuration template for the downloaded dataset."""
	try:
		config_template = f"""
# Generated configuration for PCB Defects Dataset
# Dataset path: {dataset_dir.absolute()}

data:
  path: "{dataset_dir.absolute() / 'data.yaml'}"
  
# Training configuration example:
training:
  epochs: 100
  batch: 32
  imgsz: 640
  device: "0"
  
# For domain adaptation from HRIPCB:
domain_adaptation:
  source_weights: "path/to/hripcb_best.pt"
  target_data: "{dataset_dir.absolute() / 'data.yaml'}"
  fine_tune_epochs: 30
"""
		
		config_file = Path("roboflow_pcb_config_template.yaml")
		with open(config_file, 'w') as f:
			f.write(config_template)
		
		print(f"📝 Configuration template saved: {config_file}")
		
	except Exception as e:
		print(f"⚠️  Could not generate config template: {e}")

def main():
	"""Main download function."""
	print("🎯 Roboflow PCB Defects Dataset Download")
	print("🔗 https://universe.roboflow.com/rahul-jhj03/pcb-defects-dataset/dataset/2")
	print("=" * 70)
	
	# Parse CLI args
	parser = argparse.ArgumentParser(
		description="Download PCB Defects Dataset from Roboflow",
		epilog="""
Examples:
  python download_roboflow_pcb.py
  python download_roboflow_pcb.py --api-key your_roboflow_api_key_here
  
Get your API key from: https://roboflow.com/settings/api
		""",
		formatter_class=argparse.RawDescriptionHelpFormatter
	)
	parser.add_argument('--api-key', dest='api_key', default=None, 
					   help='Roboflow API key override')
	args = parser.parse_args()
	
	# Setup environment
	setup_environment()
	
	# Download dataset
	success = download_pcb_defects_dataset(cli_api_key=args.api_key)
	
	if success:
		print("\n🎉 PCB Defects Dataset setup completed successfully!")
		print("📝 Next steps:")
		print("1. Review the dataset structure above")
		print("2. Check the generated configuration template")
		print("3. Update your training configuration files accordingly")
		print("4. For domain adaptation, use:")
		print("   python run_domain_analysis_deeppcb.py --weights <hripcb_model> --data-yaml <dataset>/data.yaml")
	else:
		print("\n❌ PCB Defects Dataset setup failed!")
		print("💡 Please check your Roboflow API key and internet connection")
		print("💡 Make sure you can access: https://universe.roboflow.com/rahul-jhj03/pcb-defects-dataset/dataset/2")

if __name__ == "__main__":
	main()