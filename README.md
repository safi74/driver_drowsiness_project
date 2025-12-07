# Driver Drowsiness Detection Using Deep Learning

A comprehensive deep learning project comparing **Custom CNN** and **ResNet50 Transfer Learning** approaches for detecting driver drowsiness from facial images. This repository includes complete data preprocessing, model training, evaluation, and an interactive Streamlit web application.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Project Overview

Driver drowsiness is a leading cause of road accidents worldwide. This project implements and compares two deep learning approaches for automated drowsiness detection using facial image analysis:

- **Custom CNN**: Built from scratch with 456K parameters
- **ResNet50 Transfer Learning**: Pre-trained on ImageNet with two-phase fine-tuning

### Key Results

| Model | Accuracy | ROC AUC | False Negatives | Training Epochs |
|-------|----------|---------|-----------------|-----------------|
| **Custom CNN** | 89.66% | 0.9712 | 219 | 30 |
| **ResNet50** | **94.69%** | **0.9922** | **118** (-46%) | 15 |

✅ **ResNet50 achieves 5% higher accuracy with 46% fewer critical errors (missed drowsy states)**

## 📁 Project Structure

```
├── 1-data.ipynb                    # Data preprocessing & partitioning
├── 2-cnn.ipynb                     # Custom CNN training
├── 3-transfer-learning.ipynb       # ResNet50 transfer learning
├── 4-metrics.ipynb                 # Model evaluation & metrics
├── app.py                          # Streamlit web application
├── models/
│   ├── best_cnn.h5                # Trained CNN model
│   ├── resnet50_final.h5          # Trained ResNet50 model
│   ├── *_confusion_matrix.png     # Confusion matrices
│   └── *_roc_curve.png            # ROC curves
├── architecture_diagrams/          # Model architecture visualizations
├── dataset/                        # Processed dataset (train/val/test)
├── research_paper.md               # Full research paper
├── research_paper_short.md         # Condensed 3-4 page version
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 📊 Dataset

**Source**: Facial images dataset for drowsiness detection

**Statistics**:
- Total images: 66,521
- Drowsy class: 36,030 (54.2%)
- Non-drowsy class: 30,491 (45.8%)
- Image size: 224 × 224 pixels (RGB)

**Partitioning** (70/20/10 split):
- Training: 46,564 images
- Validation: 13,304 images
- Test: 6,653 images

**Preprocessing**:
- RGB conversion & resizing to 224×224
- Data augmentation (flip, rotation, zoom, translation, contrast)
- Class weights: {0: 0.7, 1: 1.3} to emphasize drowsy detection

## 🏗️ Model Architectures

### Custom CNN
```
Input (224×224×3)
    ↓
Data Augmentation
    ↓
4× Conv Blocks (32→64→128→256)
    ↓ (each: Conv2D → BatchNorm → ReLU → MaxPool)
Global Average Pooling
    ↓
Dense(256) + Dropout(0.5)
    ↓
Dense(1, sigmoid)
```
**Parameters**: 456,385 (all trainable)

### ResNet50 Transfer Learning
```
Input (224×224×3)
    ↓
Data Augmentation
    ↓
ResNet50 (ImageNet pre-trained)
    ↓
Global Average Pooling (2048 features)
    ↓
Dense(256) + Dropout(0.5)
    ↓
Dense(1, sigmoid)
```
**Parameters**: 24.1M total (525K trainable initially)

**Two-Phase Training**:
1. **Phase 1 (Frozen)**: Train classification head only, LR=1e-3, 5-10 epochs
2. **Phase 2 (Fine-tuning)**: Unfreeze top layers, LR=1e-5, 10-20 epochs

## 🚀 Installation & Usage

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- CUDA-compatible GPU (optional, for faster training)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd CV
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Prepare dataset**
   - Place raw images in `train_data/drowsy` and `train_data/notdrowsy`
   - Run `1-data.ipynb` to preprocess and partition

### Training Models

**Train Custom CNN**:
```bash
jupyter notebook 2-cnn.ipynb
# Or run cells programmatically
```

**Train ResNet50**:
```bash
jupyter notebook 3-transfer-learning.ipynb
```

### Evaluate Models
```bash
jupyter notebook 4-metrics.ipynb
```

### Run Web Application
```bash
streamlit run app.py
```
Access at `http://localhost:8501`

**Features**:
- Upload driver face image (PNG/JPG)
- Select model: CNN, ResNet50, or Both
- View prediction: Drowsy / Not Drowsy
- See confidence percentages
- Inference logging to CSV

## 📈 Performance Metrics

### Custom CNN Results

| Metric | Drowsy (Class 0) | Non-Drowsy (Class 1) |
|--------|------------------|----------------------|
| Precision | 93.47% | 85.79% |
| Recall | 86.98% | 92.82% |
| F1-Score | 90.11% | 89.17% |

**Confusion Matrix**: [[3134, 469], [219, 2831]]

### ResNet50 Results

| Metric | Drowsy (Class 0) | Non-Drowsy (Class 1) |
|--------|------------------|----------------------|
| Precision | 96.62% | 92.58% |
| Recall | 93.48% | 96.13% |
| F1-Score | 95.02% | 94.32% |

**Confusion Matrix**: [[3368, 235], [118, 2932]]

### Key Improvements with Transfer Learning
- ✅ **54% reduction** in false negatives (critical for safety)
- ✅ **50% reduction** in false positives (better user experience)
- ✅ **2.1% higher AUC** (0.9922 vs 0.9712)
- ✅ **50% faster training** (15 vs 30 epochs)

## 📑 Research Paper

This project includes a comprehensive research paper documenting the methodology, experiments, and results:

- **Full version**: `research_paper.md` (8-10 pages)
- **Short version**: `research_paper_short.md` (3-4 pages)

**Sections**:
- Abstract & Keywords
- Introduction & Related Work
- Dataset Description & Preprocessing
- Model Architectures & Training Strategies
- Results & Analysis (Quantitative & Qualitative)
- Conclusion & Future Work
- References

## 🖼️ Visualizations

The project includes several visualizations in `architecture_diagrams/`:
- `cnn_block_diagram.png` - Custom CNN architecture
- `resnet50_block_diagram.png` - ResNet50 architecture
- `architecture_comparison.png` - Side-by-side comparison
- `two_phase_training.png` - Training strategy diagram

Model evaluation visualizations in `models/`:
- Confusion matrices (heatmaps)
- ROC curves with AUC scores
- Training/validation learning curves

## 🔬 Methodology Highlights

### Data Augmentation
- **Random flip**: Horizontal (probability=0.5)
- **Random rotation**: ±10 degrees
- **Random zoom**: ±5%
- **Random translation**: ±3% (horizontal & vertical)
- **Random contrast**: ±8%

### Training Configuration
- **Optimizer**: Adam
- **Loss**: Binary cross-entropy
- **Metrics**: Accuracy, Precision, Recall, AUC
- **Callbacks**: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
- **Class weights**: Manual adjustment for better drowsy detection

### Evaluation Protocol
- Stratified test set (10% of data)
- Multiple metrics: Accuracy, Precision, Recall, F1, AUC
- Confusion matrix analysis
- ROC curve comparison
- Error analysis (false positives vs false negatives)

## 🎯 Use Cases

This system can be deployed for:
- Real-time in-vehicle drowsiness monitoring
- Fleet management safety systems
- Driver training and assessment
- Research and development in automotive safety

## ⚠️ Limitations & Future Work

### Current Limitations
- Static image analysis (no temporal information)
- Performance may vary with extreme lighting conditions
- Dataset specific to certain demographics

### Future Enhancements
1. **Temporal Modeling**: Implement LSTM/3D CNN for video sequence analysis
2. **Multi-modal Fusion**: Combine facial analysis with head pose estimation
3. **Edge Deployment**: Optimize for Raspberry Pi, Jetson devices
4. **Dataset Expansion**: Include diverse demographics and conditions
5. **Real-time Optimization**: Model quantization and pruning
6. **Explainability**: Add Grad-CAM visualizations

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or collaborations, please open an issue in this repository.

## 🙏 Acknowledgments

- TensorFlow & Keras teams for excellent deep learning frameworks
- ResNet50 pre-trained weights from ImageNet
- Streamlit for the interactive web framework
- Research community for drowsiness detection datasets and methodologies

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{driver_drowsiness_detection_2025,
  title={Driver Drowsiness Detection Using Deep Learning: A Comparative Study},
  author={[Your Name]},
  year={2025},
  howpublished={\url{https://github.com/[your-username]/[repo-name]}}
}
```

---

**⚠️ Disclaimer**: This is a research/educational prototype and should not be used as the sole basis for safety-critical driving decisions. Always drive responsibly and take appropriate breaks when feeling drowsy.
