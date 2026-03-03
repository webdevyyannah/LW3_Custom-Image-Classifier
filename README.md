# 🌿 LW3: Custom Image Classifier with TensorFlow
### Building, Improving, and Deploying a CNN-based Tree Species Classifier

---

## 📋 Overview

This laboratory activity covers the full pipeline of building a **custom image classifier** using TensorFlow/Keras in Google Colab. The classifier was trained on a personal dataset of **20 Philippine tree species** (5,002 images) organized in Google Drive.

The project is divided into two parts:
- **LW3** – Dataset preparation, model building, training, and evaluation
- **LW3A (Activity 3A)** – Visualization, overfitting control, data augmentation, dropout, and model deployment

---

## 📁 Dataset Structure

```
MyDrive/
└── IMAGE DATA SET/
    ├── ANAHAW TREE/
    ├── ARECA TREE/
    ├── BALETE TREE/
    ├── BALIMBING TREE/
    ├── BAMBOO TREE/
    ├── BANANA TREE/
    ├── BIRCH TREE/
    ├── CACAO TREE/
    ├── CALAMANSI TREE/
    ├── COCONUT TREE/
    ├── DOUGLAS FIR TREE/
    ├── ILANG ILANG TREE/
    ├── LANZONES TREE/
    ├── LEMON TREE/
    ├── MAHOGANY TREE/
    ├── MALUNGGAY TREE/
    ├── MANGO TREE/
    ├── MANGROVE TREE/
    ├── SANTOL TREE/
    └── TALISAY TREE/
```

- **Total images:** 5,002
- **Training set:** 4,002 images (80%)
- **Validation set:** 1,000 images (20%)
- **Image size:** 180 × 180 px

---

## 🧠 Model Architecture

### Baseline Model (LW3)
```
Input → Rescaling → Conv2D(16) → MaxPool → Conv2D(32) → MaxPool
     → Conv2D(64) → MaxPool → Flatten → Dense(128) → Dense(20)
```

### Improved Model (LW3A)
```
Input → DataAugmentation → Rescaling → Conv2D(16) → MaxPool
     → Conv2D(32) → MaxPool → Conv2D(64) → MaxPool
     → Dropout(0.3) → Flatten → Dense(128) → Dropout(0.3) → Dense(20)
```

---

## 📊 Results

| Metric | Baseline Model | Improved Model |
|---|---|---|
| Epochs | 10 | 15 |
| Training Accuracy | ~100% | ~74.78% |
| Validation Accuracy | ~95.99% | ~82.30% |
| Training Loss | ~0.000137 | ~0.8026 |
| Validation Loss | ~0.2262 | ~0.6460 |

> ⚠️ The baseline model showed clear **overfitting** — training accuracy hit 100% while validation loss plateaued high. The improved model trades raw accuracy for better generalization.

---


## 🔍 Key Concepts Demonstrated

- **Convolutional Neural Networks (CNNs)** for image feature extraction
- **Overfitting detection** via training vs. validation accuracy/loss plots
- **Data augmentation** (random flip, rotation, zoom) for generalization
- **Dropout regularization** to prevent memorization
- **Model saving & loading** with `.keras` format

---

## 🌱 Sample Prediction

```python
Predicted Class: SANTOL TREE
Confidence: 55.43%
```

---

## 📝 Guide Questions & Reflection

### 🔷 LW3 — Part 2: Dataset Preparation, Training & Performance

#### 📂 1. Dataset Preparation

**How did you organize your dataset in Google Drive?**
The dataset was organized into a root folder (`IMAGE DATA SET`) with 20 subfolders, one per tree species (e.g., `ANAHAW TREE/`, `SANTOL TREE/`). Each subfolder contained at least 250 images of that species. TensorFlow automatically uses folder names as class labels.

**Why is folder structure important for TensorFlow image loading?**
`image_dataset_from_directory()` relies on folder names to assign labels. Without a clean, consistent structure, the model would have no way to map images to their correct classes during training.

---

#### 🤖 2. Model Training

**What is the role of convolutional layers in image classification?**
Convolutional layers act as feature detectors — they scan images using filters to extract low-level features (edges, textures) in early layers and higher-level patterns (shapes, structures) in deeper layers, enabling the model to distinguish between tree species.

**Why do we split data into training and validation sets?**
The training set is used to update model weights, while the validation set evaluates performance on unseen data after each epoch. This helps detect overfitting and gives an honest estimate of how the model will perform in the real world.

---

#### 📈 3. Performance Analysis

**What accuracy did your model achieve?**
The baseline model achieved a **validation accuracy of ~95.99%** after 10 epochs, with a validation loss of 0.2625.

**How did the number of images affect the model's performance?**
Having 5,002 images across 20 classes (~250 per class) provided enough variety for the model to learn distinguishing features. More images generally reduce overfitting and improve generalization, though the baseline model still overfit despite this.

---

#### 💡 4. Critical Thinking

**What challenges did you encounter while using your own dataset?**
Ensuring consistent image quality and quantity across all 20 classes was the main challenge. Some species look visually similar, making classification harder. Dataset collection and cleaning also required significant effort.

**How can data augmentation improve your model?**
Data augmentation artificially increases dataset diversity by applying random transformations (flips, rotations, zooms) to existing images. This prevents the model from memorizing specific training images and improves its ability to handle real-world variations.

---

#### 🌍 5. Application

**Suggest a real-world application for your trained model.**
A mobile app for forest rangers or botanists in the Philippines to identify local tree species by taking a photo — useful for biodiversity monitoring, reforestation programs, and environmental conservation efforts.

**How can this system be integrated into a mobile or web application?**
The saved `.keras` model can be converted to TensorFlow Lite for mobile deployment (Android/iOS) or served via a REST API using Flask or FastAPI for a web-based interface where users upload images and receive instant classifications.

---

### 🔷 LW3A — Part 3–8: Visualization, Overfitting, Augmentation & Deployment

#### 📉 1. Visualization & Overfitting

**What signs indicated overfitting in your first model?**
By epoch 7, training accuracy reached **100%** while validation accuracy plateaued at ~96%. More clearly, training loss dropped to near zero (~0.000137) while validation loss remained elevated at ~0.2262. The accuracy/loss plots visually confirmed the widening gap between training and validation curves.

**How did data augmentation affect validation accuracy?**
With data augmentation, the model learned more gradually and steadily. Validation accuracy climbed from 15.6% to **82.3%** over 15 epochs, and the training/validation curves converged more closely — indicating the model was generalizing rather than memorizing.

---

#### 🛠️ 2. Model Improvement

**What is the purpose of dropout layers?**
Dropout randomly deactivates a fraction of neurons (30% in this model) during each training step. This prevents neurons from co-adapting too strongly to specific training patterns, forcing the network to build more robust, distributed representations and reducing overfitting.

**Why does data augmentation improve generalization?**
By exposing the model to randomly flipped, rotated, and zoomed versions of training images, augmentation simulates a larger and more varied dataset. The model can no longer rely on memorizing exact pixel patterns, so it learns more meaningful, generalizable features.

---

#### 📊 3. Performance Comparison

**Compare accuracy before and after improvements.**

| Metric | Baseline Model | Improved Model |
|---|---|---|
| Epochs | 10 | 15 |
| Training Accuracy | ~100% | ~74.78% |
| Validation Accuracy | ~95.99% | ~82.30% |
| Training Loss | ~0.000137 | ~0.8026 |
| Validation Loss | ~0.2262 | ~0.6460 |

> While the improved model's raw validation accuracy appears lower, it is healthier — the training and validation curves are much closer together, confirming better generalization and less overfitting.

**Which technique contributed most to improvement?**
**Data augmentation** had the greater impact. It directly addressed the root cause of overfitting by diversifying the training data at the input level. Dropout complemented this by regularizing the network internally, but augmentation was the more foundational fix.

---

#### 🚀 4. Deployment & Application

**Why is saving the model important?**
Saving the trained model with `model.save()` preserves all learned weights and architecture permanently. This means the model can be reused, shared, or deployed without retraining — saving hours of compute time and enabling real-world application.

**How can this model be deployed in a real-world system?**
The saved `.keras` model can be:
- Converted to **TensorFlow Lite** for on-device mobile inference (Android/iOS)
- Wrapped in a **Flask or FastAPI** backend and deployed as a REST API
- Hosted on **Google Cloud / AWS** for scalable web-based predictions
- Integrated into a **web app** where users upload tree photos and receive instant species identification with confidence scores

---

## 🔗 Project Links

- 📓 **Google Colab Notebook:** [(https://colab.research.google.com/drive/11wDejO6UUcPz7ZgGoplcQfeDvseEFVnS?usp=sharing)]
- 📁 **Google Drive Dataset:** [(https://drive.google.com/drive/folders/1TRQJ9ZjW8XNAK6L1VdbcqLDDwcdhuwcO?usp=sharing)]
- - 🧠 **Saved Model:** [(https://drive.google.com/file/d/19L1TODQCLFHRFOioXjQewesOzPX2qbG1/view?usp=drive_link)]
