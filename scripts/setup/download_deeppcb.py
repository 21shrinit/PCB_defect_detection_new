#!/usr/bin/env python3
"""
Download DeepPCB dataset using Roboflow API
- Supports Colab Secrets and environment variables for Roboflow API key
- Also supports passing --api-key via CLI
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
	
	print("🚀 Setting up DeepPCB download environment...")
	print(f"📁 Working dire1ctory: {Path.cwd()}")
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

def download_deeppcb_dataset(cli_api_key: str | None = None):
	"""Download DeepPCB dataset from Roboflow."""
	try:
		print("📥 Downloading DeepPCB dataset...")
		
		api_key = get_roboflow_api_key(cli_api_key)
		if not api_key:
			print("❌ Roboflow API key not found. Set ROBOFLOW_API_KEY in Colab secrets or env, or pass --api-key.")
			return False
		
		# Initialize Roboflow
		rf = Roboflow(api_key=api_key)
		
		# Get project and version
		project = rf.workspace("tack-hwa-wong-zak5u").project("deeppcb-4dhir")
		version = project.version(1)
		
		# Download dataset in YOLOv8 format
		print("🔄 Downloading dataset (this may take a few minutes)...")
		_ = version.download("yolov8")
		
		print("✅ DeepPCB dataset downloaded successfully!")
		print(f"📁 Dataset location: {Path.cwd()}")
		
		# Verify the download
		verify_dataset_structure()
		
		return True
		
	except Exception as e:
		print(f"❌ Error downloading DeepPCB dataset: {e}")
		print("💡 Make sure you have a valid ROBOFLOW_API_KEY in Colab secrets or env")
		return False

def verify_dataset_structure():
	"""Verify the downloaded dataset structure."""
	try:
		print("🔍 Verifying dataset structure...")
		
		# Check for DeepPCB-1 directory
		deeppcb_dir = Path("DeepPCB-1")
		if not deeppcb_dir.exists():
			print("⚠️  DeepPCB-1 directory not found")
			return False
		
		# Check for train/val/test splits
		splits = ['train', 'valid', 'test']
		for split in splits:
			split_dir = deeppcb_dir / split
			if split_dir.exists():
				images_dir = split_dir / 'images'
				labels_dir = split_dir / 'labels'
				
				if images_dir.exists() and labels_dir.exists():
					image_count = len(list(images_dir.glob('*.jpg')))
					label_count = len(list(labels_dir.glob('*.txt')))
					print(f"✅ {split}: {image_count} images, {label_count} labels")
				else:
					print(f"⚠️  {split}: Missing images or labels directory")
			else:
				print(f"⚠️  {split}: Directory not found")
		
		# Check data.yaml
		data_yaml = deeppcb_dir / 'data.yaml'
		if data_yaml.exists():
			print(f"✅ data.yaml found: {data_yaml}")
		else:
			print("⚠️  data.yaml not found")
		
		print("🎉 DeepPCB dataset verification completed!")
		return True
		
	except Exception as e:
		print(f"❌ Error verifying dataset: {e}")
		return False

def main():
	"""Main download function."""
	print("🎯 DeepPCB Dataset Download")
	print("=" * 50)
	
	# Parse CLI args
	parser = argparse.ArgumentParser()
	parser.add_argument('--api-key', dest='api_key', default=None, help='Roboflow API key override')
	args = parser.parse_args()
	
	# Setup environment
	setup_environment()
	
	# Download dataset
	success = download_deeppcb_dataset(cli_api_key=args.api_key)
	
	if success:
		print("\n🎉 DeepPCB dataset setup completed successfully!")
		print("📝 Next steps:")
		print("1. Review the dataset structure above")
		print("2. Check if images and annotations are properly organized")
		print("3. Update your training configuration files accordingly")
		print("4. Start training with: python train_baseline.py")
	else:
		print("\n❌ DeepPCB dataset setup failed!")
		print("💡 Please check your Roboflow API key and try again")

if __name__ == "__main__":
	main()
