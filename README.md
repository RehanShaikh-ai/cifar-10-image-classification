# 🖼️ CIFAR-10 Image Classification  

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-DL-red?logo=keras)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-yellow?logo=plotly)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-green?logo=scikitlearn)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview
This project builds an **image classification system** using the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).  
We compare two approaches:
- A **Custom Convolutional Neural Network (CNN)** with Dropout, BatchNorm, and Data Augmentation.  
- A **Pretrained Model (ResNet50)** fine-tuned on CIFAR-10.  

---

## 🗂 Dataset
**Source:** [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)  
**Structure:**
- 60,000 color images (32x32 pixels, RGB)
- 50,000 training images + 10,000 testing images
- **10 categories**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck  

---

## 🛠 Tech Stack
- **Python 3.x**
- **TensorFlow / Keras** → Deep learning models
- **NumPy** → Numerical operations
- **Matplotlib / Seaborn** → Visualization
- **scikit-learn** → Evaluation metrics
- **Jupyter Notebook** → Development environment

---

## 📍 Project Steps
1. **Data Loading & Preprocessing**
   - Normalize pixel values
   - Apply Data Augmentation  

2. **Model 1: Custom CNN**
   - Convolution + MaxPooling layers
   - Dropout + Batch Normalization
   - Dense layers for classification  

3. **Model 2: Transfer Learning**
   - Load pretrained **ResNet50**
   - Replace final layers for CIFAR-10 classes
   - Fine-tune selected layers  

4. **Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - Confusion Matrix
   - Training vs Validation curves  

5. **Visualization**
   - Example predictions (correct vs incorrect)
   - Confusion matrix heatmap  

---

## 🚀 How to Run
```bash
# Clone the repository
git clone https://github.com/RehanShaikh-ai/cifar-10-image-classification.git
cd cifar-10-image-classification

# Install dependencies
pip install -r requirements.txt

# Run Jupyter Notebook
jupyter main.ipynb
```

---

## 👤 **Author**  
**Rehan Abdul Gani Shaikh**  
_Aspiring Data Scientist | B.Tech Student_  

🔗 **Connect with me:** [LinkedIn](https://www.linkedin.com/in/rehan-shaikh-68153a246)  
📬 **Email:** rehansk.3107@gmail.com  
