#!/usr/bin/env python3
"""
Missing Functionality to Add to the Jupyter Notebook
===================================================

These functions need to be added to complete the requirements:
1. GFLOPs calculation for efficiency metrics
2. Media retrieval function for Ultralytics logged artifacts
"""

import wandb
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64

def calculate_gflops(model_type, image_size=640):
    """
    Calculate approximate GFLOPs for different YOLO models.
    
    Args:
        model_type (str): Model type (yolov8n, yolov8s, yolov10s)
        image_size (int): Input image size
        
    Returns:
        float: Estimated GFLOPs
    """
    # Approximate GFLOPs for different models at 640px
    # These are estimated values - real values would need model profiling
    base_gflops = {
        'yolov8n': 8.7,
        'yolov8s': 28.6, 
        'yolov10s': 24.5,  # Estimated based on YOLOv10s architecture
    }
    
    # Scale GFLOPs based on image size (quadratic relationship)
    size_factor = (image_size / 640) ** 2
    
    if model_type.lower() in base_gflops:
        return base_gflops[model_type.lower()] * size_factor
    else:
        return None


def retrieve_ultralytics_media(run, media_types=['confusion_matrix', 'PR_curve', 'F1_curve']):
    """
    Retrieve and display media logged automatically by Ultralytics integration.
    
    Args:
        run: WandB run object
        media_types (list): Types of media to retrieve
        
    Returns:
        dict: Dictionary of retrieved media objects
    """
    media_artifacts = {}
    
    try:
        # Get run files
        files = run.files()
        
        print(f"📊 Retrieving media for run: {run.name}")
        
        for file in files:
            file_name = file.name.lower()
            
            # Look for confusion matrix
            if 'confusion_matrix' in media_types and 'confusion' in file_name and file_name.endswith('.png'):
                try:
                    # Download the file
                    file_content = file.download(replace=True)
                    media_artifacts['confusion_matrix'] = file_content.name
                    print(f"   ✅ Found confusion matrix: {file.name}")
                except Exception as e:
                    print(f"   ⚠️  Error downloading confusion matrix: {e}")
            
            # Look for PR curve
            if 'PR_curve' in media_types and ('pr_curve' in file_name or 'precision_recall' in file_name) and file_name.endswith('.png'):
                try:
                    file_content = file.download(replace=True)
                    media_artifacts['PR_curve'] = file_content.name
                    print(f"   ✅ Found PR curve: {file.name}")
                except Exception as e:
                    print(f"   ⚠️  Error downloading PR curve: {e}")
            
            # Look for F1 curve
            if 'F1_curve' in media_types and 'f1' in file_name and 'curve' in file_name and file_name.endswith('.png'):
                try:
                    file_content = file.download(replace=True)
                    media_artifacts['F1_curve'] = file_content.name
                    print(f"   ✅ Found F1 curve: {file.name}")
                except Exception as e:
                    print(f"   ⚠️  Error downloading F1 curve: {e}")
        
        # Also check for media in run history/summary
        if hasattr(run, 'summary'):
            summary = run.summary
            
            # Look for wandb.Image objects in summary
            for key, value in summary.items():
                if isinstance(value, dict) and '_type' in value and value['_type'] == 'image-file':
                    print(f"   📸 Found image in summary: {key}")
                    media_artifacts[key] = value
        
        # Check logged media in run history
        try:
            history = run.scan_history()
            for row in history:
                for key, value in row.items():
                    if isinstance(value, dict) and '_type' in value and value['_type'] == 'image-file':
                        if key not in media_artifacts:
                            print(f"   📸 Found image in history: {key}")
                            media_artifacts[key] = value
                        break  # Only need to find it once
        except Exception as e:
            print(f"   ⚠️  Could not scan history: {e}")
            
    except Exception as e:
        print(f"   ❌ Error retrieving media: {e}")
    
    return media_artifacts


def display_ultralytics_media(media_artifacts, figsize=(15, 10)):
    """
    Display retrieved Ultralytics media artifacts.
    
    Args:
        media_artifacts (dict): Dictionary of media artifacts from retrieve_ultralytics_media
        figsize (tuple): Figure size for matplotlib
    """
    if not media_artifacts:
        print("⚠️  No media artifacts to display")
        return
    
    # Filter out image files that exist locally
    image_files = {k: v for k, v in media_artifacts.items() 
                   if isinstance(v, str) and v.endswith('.png')}
    
    if not image_files:
        print("⚠️  No displayable image files found")
        return
    
    # Create subplots
    n_images = len(image_files)
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_images == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, list) else [axes]
    else:
        axes = axes.flatten()
    
    for idx, (name, file_path) in enumerate(image_files.items()):
        try:
            # Load and display image
            img = Image.open(file_path)
            axes[idx].imshow(img)
            axes[idx].set_title(name.replace('_', ' ').title())
            axes[idx].axis('off')
        except Exception as e:
            axes[idx].text(0.5, 0.5, f"Error loading\n{name}", 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(f"Error: {name}")
            axes[idx].axis('off')
    
    # Hide empty subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def demonstrate_media_retrieval(df, n_runs=3):
    """
    Demonstrate how to retrieve and display media from multiple runs.
    
    Args:
        df: DataFrame with run data
        n_runs (int): Number of runs to demonstrate with
    """
    print("🎨 DEMONSTRATING ULTRALYTICS MEDIA RETRIEVAL")
    print("=" * 60)
    
    if len(df) == 0:
        print("⚠️  No runs available for demonstration")
        return
    
    # Initialize WandB API
    api = wandb.Api()
    
    # Get project name from global variable or config
    project_name = "pcb-defect-systematic-study"  # Update as needed
    
    # Select top performing runs for demonstration
    demo_runs = df.head(n_runs) if 'map50' not in df.columns else df.nlargest(n_runs, 'map50')
    
    print(f"📊 Demonstrating media retrieval for top {len(demo_runs)} runs:")
    
    for idx, (_, run_data) in enumerate(demo_runs.iterrows()):
        print(f"\n🔍 Run {idx+1}: {run_data['run_name']}")
        
        try:
            # Get the actual WandB run object
            run = api.run(f"{project_name}/{run_data['run_id']}")
            
            # Retrieve media artifacts
            media_artifacts = retrieve_ultralytics_media(run)
            
            # Display the artifacts
            if media_artifacts:
                print(f"   📸 Displaying {len(media_artifacts)} media artifacts:")
                display_ultralytics_media(media_artifacts)
            else:
                print(f"   ⚠️  No media artifacts found for this run")
                
        except Exception as e:
            print(f"   ❌ Error processing run: {e}")
    
    print(f"\n✅ Media retrieval demonstration complete")


# Function to add GFLOPs to existing dataframe
def add_gflops_to_dataframe(df):
    """
    Add GFLOPs calculation to existing dataframe.
    
    Args:
        df: Existing dataframe from fetch_wandb_runs
        
    Returns:
        pd.DataFrame: Updated dataframe with GFLOPs column
    """
    if 'model_type' in df.columns and 'image_size' in df.columns:
        df['gflops'] = df.apply(
            lambda row: calculate_gflops(row['model_type'], row.get('image_size', 640)), 
            axis=1
        )
        print("✅ Added GFLOPs calculation to dataframe")
    else:
        print("⚠️  Cannot calculate GFLOPs - missing model_type or image_size columns")
    
    return df


# Updated summary table function with GFLOPs
def create_enhanced_comparison_table(df):
    """
    Create enhanced comparison table with all required metrics including GFLOPs.
    """
    print("📋 ENHANCED COMPARISON TABLE (with GFLOPs)")
    print("="*60)
    
    if len(df) == 0:
        print("⚠️  No data available for comparison")
        return
    
    # Add GFLOPs if not present
    if 'gflops' not in df.columns:
        df = add_gflops_to_dataframe(df)
    
    # Select enhanced columns for comparison
    comparison_columns = [
        'model_variant', 
        # Accuracy metrics
        'map50', 'map50_95', 'precision', 'recall', 'f1_score',
        # Efficiency metrics  
        'inference_time_ms', 'fps', 'gflops', 'total_parameters',
        # Training metrics
        'training_time_hours', 'model_size_mb'
    ]
    
    # Filter available columns
    available_columns = [col for col in comparison_columns if col in df.columns]
    
    # Create comparison table
    comparison_df = df[available_columns].copy()
    
    # Round numeric columns
    numeric_columns = comparison_df.select_dtypes(include=['number']).columns
    comparison_df[numeric_columns] = comparison_df[numeric_columns].round(4)
    
    # Sort by mAP@0.5 if available
    if 'map50' in comparison_df.columns:
        comparison_df = comparison_df.sort_values('map50', ascending=False)
    
    print("📊 FINAL ACCURACY & EFFICIENCY METRICS:")
    print("Accuracy: mAP@0.5:0.95, Precision, Recall, F1")
    print("Efficiency: Inference Latency (ms), FPS, GFLOPs, Parameters")
    print("-" * 60)
    print(comparison_df.to_string(index=False))
    
    return comparison_df


if __name__ == "__main__":
    print("🔧 Missing functionality implementations ready to be added to the Jupyter notebook")
    print("\nTo complete the requirements, add these functions to the notebook:")
    print("1. calculate_gflops() - for GFLOPs efficiency metric")
    print("2. retrieve_ultralytics_media() - to fetch confusion matrix, PR curves, etc.")  
    print("3. display_ultralytics_media() - to show the retrieved media")
    print("4. demonstrate_media_retrieval() - to demonstrate the media retrieval capability")
    print("5. create_enhanced_comparison_table() - enhanced table with GFLOPs")