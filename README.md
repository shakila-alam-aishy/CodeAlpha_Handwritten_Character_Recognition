# 🧠 AI-Powered Handwritten Digit Recognition System

A deep learning-based handwritten digit recognition system built using PyTorch and CNN.
The project uses the MNIST dataset to classify handwritten digits with high accuracy and includes an interactive Gradio web application for real-time predictions.

---

## 🚀 Features

* Handwritten digit recognition (0–9)
* Convolutional Neural Network (CNN)
* Real-time prediction using Gradio
* PyTorch-based deep learning model
* Interactive web interface
* Achieved **99.11% accuracy** on the MNIST dataset

---

## 🛠️ Tech Stack

* Python
* PyTorch
* Torchvision
* Gradio
* NumPy
* PIL

---

## 📂 Dataset

This project uses the **MNIST Handwritten Digits Dataset**.

* 70,000 grayscale digit images
* 28×28 image size
* Digits from 0 to 9

Dataset is automatically downloaded using `torchvision.datasets`.

---

## 🧠 Model Architecture

The CNN model consists of:

* 2 Convolutional Layers
* ReLU Activation
* MaxPooling Layers
* Fully Connected Layers

---

## 🎯 Model Performance

| Metric        | Result     |
| ------------- | ---------- |
| Test Accuracy | **99.11%** |
| Dataset       | MNIST      |
| Framework     | PyTorch    |

---

## 📁 Project Structure

```bash
CodeAlpha_Handwritten_Character_Recognition/
│
├── dataset/
├── model/
│   └── mnist_cnn.pth
│
├── app/
│   └── app.py
│
├── train.py
├── requirements.txt
├── README.md
```

---

## ▶️ How to Run the Project

### 1️⃣ Clone Repository

```bash
git clone https://github.com/shakila-alam-aishy/CodeAlpha_Handwritten_Character_Recognition.git
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train the Model

```bash
python train.py
```

### 4️⃣ Run the Gradio App

```bash
cd app
python app.py
```

---

## 🌐 Web Interface

The Gradio app allows users to:

* Upload handwritten digit images
* Draw digits
* Get real-time predictions

---


## 👩‍💻 Developed By

**Shakila Alam Aishy**

---

## 📌 Internship Project

This project was developed as part of the Machine Learning Internship at CodeAlpha.
