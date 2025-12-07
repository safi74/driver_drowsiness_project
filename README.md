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


### Web Application

**Features**:
- Upload driver face image (PNG/JPG)
- Select model: CNN, ResNet50, or Both
- View prediction: Drowsy / Not Drowsy
- See confidence percentages
- Inference logging to CSV


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

```

---

**⚠️ Disclaimer**: This is a research/educational prototype and should not be used as the sole basis for safety-critical driving decisions. Always drive responsibly and take appropriate breaks when feeling drowsy.
