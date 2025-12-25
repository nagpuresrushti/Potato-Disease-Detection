# 🥔 Potato Disease Detection Using CNN

This project aims to detect **potato plant diseases** using a **Convolutional Neural Network (CNN)**. By analyzing images of potato leaves, the model classifies them into healthy or diseased categories, helping in early disease identification.

---

## 📌 Project Description

Potato plants are affected by diseases like **Early Blight** and **Late Blight**, which can severely reduce crop yield.
This project uses **deep learning and image processing** to automatically identify potato leaf diseases from images.

---

## 🎯 Objectives

* Detect potato leaf diseases using image classification
* Build a CNN model for accurate prediction
* Reduce manual disease identification effort
* Assist farmers with early disease detection

---

## 🧠 Disease Classes

* **Healthy**
* **Early Blight**
* **Late Blight**

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* OpenCV
* Matplotlib
* Jupyter Notebook

---

## 📂 Project Structure

```
Potato-Disease-Detection/
│
├── dataset/
│   ├── Healthy/
│   ├── Early_Blight/
│   └── Late_Blight/
│
├── model/
│   └── cnn_model.h5
│
├── potato_disease_detection.ipynb
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

* Potato leaf images collected from the **PlantVillage dataset**
* Images are labeled and split into training and testing sets

---

## ⚙️ Installation Steps

1. Clone the repository

```bash
git clone https://github.com/your-username/potato-disease-detection.git
cd potato-disease-detection
```

2. Create a virtual environment (optional)

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Model Training

* CNN architecture with convolution, pooling, and dense layers
* Optimizer: Adam
* Loss Function: Categorical Crossentropy

Training is performed using the Jupyter Notebook provided.

---

## 🔍 Prediction

The trained model predicts the disease class of a potato leaf image and displays the result.

---

## 📈 Results

* High accuracy on validation data
* Successfully classifies potato leaf diseases
* Visual performance analysis using accuracy and loss graphs

---

## 🚀 Future Scope

* Use transfer learning (ResNet, VGG, MobileNet)
* Deploy using Flask or Streamlit
* Expand dataset with more diseases
* Mobile application integration

---
