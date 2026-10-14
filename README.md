markdown
# 🩺 Kidney Disease Classification using Deep Learning

A Deep Learning project for classifying kidney CT scans into 4 categories: **Normal, Cyst, Tumor, Stone**.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.2-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-94.42%25-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📊 Dataset

- **Source**: [CT Kidney Dataset on Kaggle](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone)
- **Size**: 12,446 CT images (~1.66 GB)
- **Classes**: 4 (Normal, Cyst, Tumor, Stone)
- **Split**: 80% training / 20% validation

### Class Distribution
| Class | Images |
|---|---|
| Normal | 5,077 |
| Cyst | 3,709 |
| Tumor | 2,283 |
| Stone | 1,377 |

---

## 🧠 Model Architecture

- **Base Model**: EfficientNetB0 (pretrained on ImageNet)
- **Technique**: Transfer Learning + Fine-tuning
- **Preprocessing**: CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Inference**: Ensemble + Test-Time Augmentation (TTA)

---

## 📈 Results

### Kaggle Test Set (20%)
| Metric | Value |
|---|---|
| **Accuracy** | **94.42%** |
| Macro F1-Score | 92.89% |
| Weighted F1-Score | 94.44% |

### Per-Class Performance
| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Cyst | 0.9092 | 0.9959 | 0.9506 |
| Normal | 0.9927 | 0.9427 | 0.9671 |
| Stone | 0.8535 | 0.8859 | 0.8694 |
| Tumor | 0.9566 | 0.9018 | 0.9284 |

---

## 📦 Libraries & Versions

### Core Deep Learning
| Library | Version | Purpose |
|---|---|---|
| `tensorflow` | 2.16.2 | Deep Learning framework |
| `keras` | 3.12.4 | Neural Network API |
| `numpy` | 1.26.4 | Numerical computations |

### Data Processing & Visualization
| Library | Version | Purpose |
|---|---|---|
| `pandas` | 2.1.4 | Data manipulation |
| `matplotlib` | 3.8.2 | Plotting and visualization |
| `seaborn` | 0.13.0 | Statistical visualizations |
| `scikit-learn` | 1.3.2 | Metrics and evaluation |
| `scipy` | (latest) | Scientific computing |

### Image Processing
| Library | Version | Purpose |
|---|---|---|
| `Pillow` | 10.2.0 | Image I/O |
| `opencv-python-headless` | 4.9.0.80 | Image preprocessing (CLAHE) |
| `scikit-image` | 0.22.0 | Advanced image processing |

### Web Interface
| Library | Version | Purpose |
|---|---|---|
| `gradio` | 4.19.2 | Interactive web interface |
| `huggingface_hub` | 0.20.3 | HF Spaces integration |

---

## 🛠️ Techniques Used

- ✅ **Transfer Learning** with EfficientNetB0
- ✅ **Class Weight Balancing** (for imbalanced data)
- ✅ **Test-Time Augmentation** (10 augmentations per image)
- ✅ **Ensemble Learning** (v2 + v3 models)
- ✅ **CLAHE Preprocessing** (standard for medical CT)
- ✅ **Data Augmentation** (flip, rotation, zoom, brightness, contrast)

---

## 📁 Project Structure

```

kidney-disease-classification/
├── app.py                          # Gradio web interface
├── train_with_mlflow.py            # Training experiments log
├── experiments_log.md              # Detailed experiments documentation
├── confusion_matrix.png            # Model evaluation
├── training_curves.png             # Training history
├── requirements.txt                # Dependencies
├── README.md                       # This file
└── .gitignore                      # Git exclusions

```

---

## ⚙️ Installation & Usage

### 1. Clone Repository
```bash
git clone https://github.com/Tasneem-Nashwan/kidney-disease-classification.git
cd kidney-disease-classification
```

2. Create Environment

```bash
conda create -n kidney python=3.10 -y
conda activate kidney
```

3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install tensorflow==2.16.2 keras==3.12.4 numpy==1.26.4 pandas==2.1.4 matplotlib==3.8.2 seaborn==0.13.0 scikit-learn==1.3.2 Pillow==10.2.0 opencv-python-headless==4.9.0.80 scikit-image==0.22.0 gradio==4.19.2 huggingface_hub==0.20.3 scipy
```

4. Download Dataset (Optional - for retraining)

```bash
kaggle datasets download -d nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone
unzip ct-kidney-dataset-normal-cyst-tumor-and-stone.zip -d data/
```

5. Run the Web Interface

```bash
python app.py
```

Then open http://127.0.0.1:7860 in your browser.

---

🔬 Limitations & Future Work

Domain Shift Challenge

While the model achieves 94.42% accuracy on the Kaggle dataset, accuracy decreases on external images from different CT scanners. This is a well-known challenge in medical AI called Domain Shift.

Mitigation Applied

· ✅ CLAHE preprocessing
· ✅ Test-Time Augmentation (TTA)
· ✅ Ensemble Learning

Future Improvements

· Train on diverse multi-center datasets
· Apply Domain Adaptation techniques
· Use larger architectures (EfficientNetB7, Vision Transformers)
· Deploy on Hugging Face Spaces for permanent access

---

📚 References

· EfficientNet Paper (Tan & Le, 2019)
· CT Kidney Dataset - Kaggle
· TensorFlow Documentation
· Gradio Documentation

---

👩‍💻 Author

Tasneem Nashwan
AI Student — Level 3
GitHub

---

📄 License

This project is licensed under the MIT License.
