
# 🌿 Plant Disease Detection using Deep Learning

A deep learning project that detects plant type and disease from leaf images using PyTorch and a multi-task ResNet18 model, deployed with Streamlit.

---

## 🚀 Project Features
- Classifies plant type 🌱  
- Detects plant disease 🦠  
- Built with PyTorch + ResNet18  
- Web app using Streamlit  
- Image preprocessing and augmentation  

---

## 🧠 Model Architecture
- Backbone: ResNet18 (pretrained)
- Feature extraction layer: 512 → 256
- Two output heads:
  - Plant classification head 🌱
  - Disease classification head 🦠

---

## 📂 Dataset
The dataset contains images of plant leaves categorized as:

- Apple, Tomato, Corn, Grape, etc.
- Each plant has different diseases + healthy class

---

## 📁 Project Structure


plant-disease-detection/
│
├── app/
│ └── planet_app.py # Streamlit app
│
├── model/
│ ├── planet_model.py # Model architecture
│ └── model.pth # Trained weights
│
├── training/
│ └── train.py # Training script
│
├── requirements.txt # Dependencies
├── .gitignore
└── README.md

------

## 🚀 Demo

![Demo](Demo.gif)


## ⚙️ Installation

```bash
pip install -r requirements.txt

