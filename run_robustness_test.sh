#!/bin/bash
# PCB Defect Detection Robustness Evaluation Script

echo "🔧 Starting PCB Defect Detection Robustness Evaluation..."

# Set paths
TEST_IMAGES_DIR="/content/drive/MyDrive/PCB_defect_detection_new/datasets/PCB--Defects-DATASET-2/test/images"
DATA_CONFIG="experiments/configs/datasets/roboflow_pcb_data.yaml"
RESULTS_DIR="robustness_results"

# Create results directory
mkdir -p $RESULTS_DIR

echo "📊 Testing model robustness against:"
echo "  - Gaussian Noise (light/medium/heavy)"
echo "  - Gaussian Blur (light/medium/heavy)"  
echo "  - Motion Blur (light/medium/heavy)"
echo "  - Low Light Conditions (light/medium/heavy)"

# Run robustness evaluation
python scripts/robustness_evaluation.py \
    --test_images $TEST_IMAGES_DIR \
    --data_config $DATA_CONFIG \
    --results_dir $RESULTS_DIR

echo "✅ Robustness evaluation completed!"
echo "📈 Results saved to: $RESULTS_DIR/"
echo "📊 Summary report: $RESULTS_DIR/robustness_summary.md"