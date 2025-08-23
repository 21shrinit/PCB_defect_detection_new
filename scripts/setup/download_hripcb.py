#!/usr/bin/env python3
"""
Download HRIPCB dataset from Kaggle.
This script downloads the HRIPCB dataset and extracts it to the datasets/HRIPCB directory.
- Supports Colab Secrets and environment variables for Kaggle credentials
"""

import os
import json
import zipfile
import shutil
from pathlib import Path

# Try to import Colab userdata (optional)
try:
	from google.colab import userdata  # type: ignore
	IN_COLAB = True
except Exception:
	userdata = None
	IN_COLAB = False


def ensure_kaggle_credentials() -> bool:
	"""Ensure Kaggle credentials are available and write kaggle.json if needed.
	Order of precedence:
	1) Colab userdata secrets (KAGGLE_USERNAME, KAGGLE_KEY)
	2) Environment variables (KAGGLE_USERNAME, KAGGLE_KEY)
	3) Existing /root/.kaggle/kaggle.json or ~/.kaggle/kaggle.json or /root/.config/kaggle/kaggle.json
	"""
	# Check if kaggle.json already exists
	candidate_paths = [
		Path('/root/.kaggle/kaggle.json'),
		Path.home() / '.kaggle' / 'kaggle.json',
		Path('/root/.config/kaggle/kaggle.json'),
	]
	for p in candidate_paths:
		if p.exists():
			print(f"✅ Found existing Kaggle credentials at: {p}")
			# Ensure env points to directory containing kaggle.json
			os.environ['KAGGLE_CONFIG_DIR'] = str(p.parent)
			return True

	# Try Colab userdata
	username = None
	key = None
	if userdata is not None:
		try:
			username = userdata.get('KAGGLE_USERNAME')
			key = userdata.get('KAGGLE_KEY')
			if username and key:
				print('🔐 Loaded Kaggle credentials from Colab user secrets')
		except Exception:
			pass

	# Fallback to environment variables
	if not (username and key):
		username = os.environ.get('KAGGLE_USERNAME')
		key = os.environ.get('KAGGLE_KEY')
		if username and key:
			print('🔐 Loaded Kaggle credentials from environment variables')

	if not (username and key):
		print('❌ Kaggle credentials not found. Please set KAGGLE_USERNAME and KAGGLE_KEY in Colab user secrets or env.')
		print('   Colab: left panel > Key icon > Add user secret > names KAGGLE_USERNAME, KAGGLE_KEY')
		return False

	# Export to env for kaggle package (some versions read env on import)
	os.environ['KAGGLE_USERNAME'] = username
	os.environ['KAGGLE_KEY'] = key

	# Write kaggle.json to both standard locations
	written = []
	for cred_dir in [Path.home() / '.kaggle', Path('/root/.config/kaggle')]:
		try:
			cred_dir.mkdir(parents=True, exist_ok=True)
			cred_path = cred_dir / 'kaggle.json'
			with cred_path.open('w') as f:
				json.dump({'username': username, 'key': key}, f)
			os.chmod(cred_path, 0o600)
			written.append(str(cred_path))
		except Exception as e:
			print(f"⚠️  Could not write credentials to {cred_dir}: {e}")

	if written:
		# Point kaggle to the first successful path
		os.environ['KAGGLE_CONFIG_DIR'] = str(Path(written[0]).parent)
		print("✅ Wrote Kaggle credentials to:")
		for p in written:
			print(f"   - {p}")
		return True
	else:
		print('❌ Failed to write kaggle.json to standard locations')
		return False


def setup_directories():
	"""Create necessary directories if they don't exist."""
	datasets_dir = Path("datasets")
	hripcb_dir = datasets_dir / "HRIPCB"
	
	# Create directories
	datasets_dir.mkdir(exist_ok=True)
	hripcb_dir.mkdir(exist_ok=True)
	
	print(f"📁 Created directory: {datasets_dir}")
	print(f"📁 Created directory: {hripcb_dir}")
	
	return hripcb_dir


def download_dataset():
	"""Download the HRIPCB dataset from Kaggle."""
	try:
		print("🔐 Ensuring Kaggle credentials...")
		if not ensure_kaggle_credentials():
			return False

		# Import Kaggle API only after credentials are in place
		from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
		print("🔐 Authenticating with Kaggle...")
		api = KaggleApi()
		api.authenticate()
		print("✅ Kaggle authentication successful!")
		
		print("📥 Downloading HRIPCB dataset...")
		api.dataset_download_files(
			dataset="youssefhassan12/hripcb-dataset",
			path=".",
			unzip=False
		)
		print("✅ Dataset download completed!")
		
		return True
		
	except Exception as e:
		print(f"❌ Error downloading dataset: {e}")
		print("💡 Make sure Colab secrets or env variables are set: KAGGLE_USERNAME, KAGGLE_KEY")
		return False


def find_downloaded_zip():
	"""Find the downloaded zip file."""
	current_dir = Path(".")
	zip_files = list(current_dir.glob("*.zip"))
	
	if not zip_files:
		print("❌ No zip files found in current directory")
		return None
	
	# Look for HRIPCB related zip files
	hripcb_zips = [f for f in zip_files if "hripcb" in f.name.lower()]
	
	if hripcb_zips:
		return hripcb_zips[0]
	else:
		# Return the first zip file if no specific HRIPCB zip found
		return zip_files[0]


def extract_dataset(zip_file_path, target_dir):
	"""Extract the dataset to the target directory."""
	try:
		print(f"📦 Extracting {zip_file_path.name} to {target_dir}...")
		
		with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
			zip_ref.extractall(target_dir)
		
		print("✅ Dataset extraction completed!")
		return True
		
	except Exception as e:
		print(f"❌ Error extracting dataset: {e}")
		return False


def cleanup_files(zip_file_path):
	"""Remove the downloaded zip file."""
	try:
		print(f"🧹 Cleaning up {zip_file_path.name}...")
		zip_file_path.unlink()
		print("✅ Cleanup completed!")
		return True
		
	except Exception as e:
		print(f"⚠️  Warning: Could not remove {zip_file_path.name}: {e}")
		return False


def list_extracted_contents(target_dir):
	"""List the contents of the extracted dataset."""
	try:
		print(f"\n📋 Contents of {target_dir}:")
		print("-" * 50)
		
		for item in sorted(target_dir.rglob("*")):
			if item.is_file():
				rel_path = item.relative_to(target_dir)
				print(f"📄 {rel_path}")
			elif item.is_dir():
				rel_path = item.relative_to(target_dir)
				print(f"📁 {rel_path}/")
		
		print("-" * 50)
		
	except Exception as e:
		print(f"⚠️  Could not list contents: {e}")


def main():
	"""Main function to download and extract the HRIPCB dataset."""
	print("🚀 HRIPCB Dataset Downloader")
	print("=" * 50)
	
	# Setup directories
	target_dir = setup_directories()
	
	# Download dataset
	if not download_dataset():
		print("❌ Failed to download dataset. Exiting.")
		return
	
	# Find downloaded zip file
	zip_file = find_downloaded_zip()
	if not zip_file:
		print("❌ No zip file found. Exiting.")
		return
	
	print(f"📦 Found zip file: {zip_file.name}")
	
	# Extract dataset
	if not extract_dataset(zip_file, target_dir):
		print("❌ Failed to extract dataset. Exiting.")
		return
	
	# List extracted contents
	list_extracted_contents(target_dir)
	
	# Cleanup
	cleanup_files(zip_file)
	
	print("\n🎉 HRIPCB dataset setup completed successfully!")
	print(f"📁 Dataset location: {target_dir.absolute()}")
	print("\n📝 Next steps:")
	print("1. Review the dataset structure above")
	print("2. Check if images and annotations are properly organized")
	print("3. Update your training configuration files accordingly")

if __name__ == "__main__":
	main()
