# PCB Defect Detection with YOLOv8 and Attention Mechanisms

A comprehensive implementation of PCB (Printed Circuit Board) defect detection using YOLOv8 with various attention mechanisms including ECA (Efficient Channel Attention), CoordAtt (Coordinate Attention), and MobileViT attention.

## 🚀 Features

- **Multiple Attention Mechanisms**: ECA, CoordAtt, and MobileViT attention implementations
- **YOLOv8 Integration**: Built on Ultralytics YOLOv8 framework
- **Custom Loss Functions**: Optimized loss functions for defect detection
- **Comprehensive Training**: Support for multiple datasets and training configurations
- **Domain Adaptation**: Tools for adapting models across different PCB datasets
- **Analysis Tools**: Comprehensive benchmarking and analysis capabilities

## 📁 Project Structure

```
PCB_defect/
├── custom_modules/          # Custom attention and loss modules
├── src/                     # Source code
├── scripts/                 # Utility scripts
├── examples/                # Example notebooks and usage
├── tests/                   # Test files
├── docs/                    # Documentation
├── docker/                  # Docker configurations
└── experiments/             # Experiment configurations
```

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <your-new-repo-url>
cd PCB_defect
```

2. **Create virtual environment**:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 🎯 Quick Start

### Basic Training
```bash
python train_unified.py --config experiments/configs/04_yolov8n_eca_standard.yaml
```

### Domain Adaptation
```bash
python run_simple_domain_adaptation.py --source_dataset source --target_dataset target
```

### Analysis and Benchmarking
```bash
python analysis_comprehensive_benchmark_fix.py
```

## 🔧 Configuration

The project uses YAML configuration files located in `experiments/configs/`. Key configurations include:

- **Model Architecture**: YOLOv8 variants with attention mechanisms
- **Training Parameters**: Learning rates, batch sizes, epochs
- **Dataset Settings**: Paths and augmentation parameters
- **Loss Functions**: Custom loss configurations

## 📊 Available Attention Mechanisms

1. **ECA (Efficient Channel Attention)**: Lightweight channel attention
2. **CoordAtt (Coordinate Attention)**: Spatial coordinate attention
3. **MobileViT Attention**: Vision transformer attention for mobile devices

## 🧪 Experiments

Run systematic experiments using:
```bash
python experiments/run_experiments.py
```

## 📈 Results and Analysis

- Comprehensive benchmarking results
- Performance comparisons across attention mechanisms
- Domain adaptation analysis
- Training stability improvements

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Inspired by research on attention mechanisms for computer vision
- Community contributions and feedback

## 📚 Documentation

For detailed documentation, see the [docs/](docs/) folder or visit our documentation site.

## 🔍 Citation

If you use this work in your research, please cite:

```bibtex
@software{pcb_defect_detection,
  title={PCB Defect Detection with YOLOv8 and Attention Mechanisms},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/pcb-defect-detection}
}
```

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the documentation
- Review example notebooks

---

**Note**: This repository contains only the essential source code and documentation. Large files like model weights, datasets, and experiment results are excluded via `.gitignore` to maintain a clean and manageable repository size.
