#!/usr/bin/env python3
"""
Setup script for Weights & Biases integration with YOLO.
This script enables W&B logging by default for all YOLO operations.
- Supports Colab Secrets and environment variables for W&B API key
"""

import os
import sys
from pathlib import Path

# Add project root to path for local ultralytics import
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Go up from scripts/setup/
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO, SETTINGS

# Try to import Colab userdata (optional)
try:
	from google.colab import userdata  # type: ignore
	IN_COLAB = True
except Exception:
	userdata = None
	IN_COLAB = False


def get_wandb_api_key() -> str | None:
	api_key = None
	if userdata is not None:
		try:
			api_key = userdata.get('WANDB_API_KEY')
		except Exception:
			api_key = None
	if not api_key:
		api_key = os.environ.get('WANDB_API_KEY')
	return api_key


def setup_wandb_integration():
	"""
	Enable Weights & Biases integration by default for YOLO and login if API key provided.
	"""
	# Get the current settings
	settings = SETTINGS
	
	# Enable W&B integration
	settings.update({
		'wandb': True,  # Enable W&B logging
	})
	
	# Optional: login with API key if present
	api_key = get_wandb_api_key()
	if api_key:
		try:
			import wandb
			os.environ['WANDB_API_KEY'] = api_key
			wandb.login(key=api_key)
			print('✅ W&B login successful via API key')
		except Exception as e:
			print(f'⚠️  W&B login skipped: {e}')
	else:
		print('ℹ️  No W&B API key found; proceeding with default settings (you can wandb.login() manually).')
	
	print("✅ Weights & Biases integration enabled!")
	print(f"📊 W&B logging: {settings['wandb']}")
	print(f"💾 Runs directory: {settings['runs_dir']}")
	print(f"📁 Datasets directory: {settings['datasets_dir']}")
	
	return settings


def test_yolo_import():
	"""
	Test that YOLO can be imported and W&B settings are applied.
	"""
	try:
		# Import YOLO to verify it works
		model = YOLO('yolov8n.pt')  # Load a small model for testing
		print("✅ YOLO import successful!")
		
		# Get model name safely
		model_name = getattr(model, 'name', 'yolov8n.pt')
		print(f"📦 Model loaded: {model_name}")
		
		# Check if W&B is enabled in settings
		if SETTINGS['wandb']:
			print("✅ W&B integration confirmed in settings!")
		else:
			print("⚠️  W&B integration not enabled in settings")
			
		return True
		
	except Exception as e:
		print(f"❌ Error during YOLO import: {e}")
		return False


if __name__ == "__main__":
	print("🚀 Setting up Weights & Biases integration with YOLO...")
	print("=" * 50)
	
	# Setup W&B integration
	settings = setup_wandb_integration()
	
	print("\n🧪 Testing YOLO import and W&B integration...")
	print("-" * 50)
	
	# Test the setup
	success = test_yolo_import()
	
	if success:
		print("\n🎉 Setup completed successfully!")
		print("\n📝 Next steps:")
		print("1. Make sure you're logged into W&B: wandb login (if not done above)")
		print("2. Your YOLO training runs will now automatically log to W&B")
		print("3. Check your W&B dashboard for experiment tracking")
		print(f"4. Your runs will be saved to: {settings['runs_dir']}")
	else:
		print("\n❌ Setup failed. Please check the error messages above.")
		print("Make sure you have installed all required packages:")
		print("pip install ultralytics wandb")
