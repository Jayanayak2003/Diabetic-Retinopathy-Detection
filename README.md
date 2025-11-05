# 👁️ Diabetic Retinopathy Detection Using Deep Learning (ImageNet vs ResNet)

This project focuses on **automated diabetic retinopathy (DR) detection** using deep learning.  
Two powerful models — **ResNet** and **ImageNet (transfer learning)** — were compared for classifying retinal fundus images.  
After experimentation, the **ImageNet-based model achieved 96% accuracy**, outperforming ResNet’s 92%.  

---

## 🧠 Overview

**Diabetic Retinopathy (DR)** is a diabetes complication that affects the eyes and can lead to blindness.  
This project aims to develop an automated system that classifies the severity of DR using deep learning,  
helping in **early detection and prevention of vision loss**.

---

## 🚀 Features

- 🧩 Comparison between **ResNet** and **ImageNet transfer learning** models  
- 🩺 **ImageNet model achieved 96% accuracy** — selected as the final classifier  
- 🧠 Multi-class classification of diabetic retinopathy (5 stages)  
- 🧾 **Flask-based web app** for uploading and predicting images  
- 📊 Visualization of training accuracy, loss, and confusion matrix  
- 💾 Trained model saved as `imagenet_dr_model.pt` for deployment  

---

## 🏗️ Architecture & Workflow

1. **Dataset** — [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection)  
2. **Preprocessing** — Image resizing (224×224), normalization, and data augmentation  
3. **Model Training**  
   - **ResNet**: baseline model (92% accuracy)  
   - **ImageNet (Transfer Learning)**: fine-tuned for DR classification (96% accuracy ✅)
4. **Evaluation** — accuracy, loss, F1-score, confusion matrix  
5. **Deployment** — Flask app with simple web interface  

---

## 📊 Model Comparison

| Model | Training Accuracy | Validation Accuracy | Remarks |
|--------|-------------------|---------------------|----------|
| ResNet | 92% | **84%** | Strong baseline |
| ImageNet (Transfer Learning) | **96%** | **90%** | ✅ Best performing model |

> The ImageNet fine-tuned model showed excellent generalization and stability across all DR stages.

---

## 🛠️ Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend** | HTML5, CSS3, Bootstrap |
| **Backend** | Python (Flask) |
| **Deep Learning Framework** | PyTorch |
| **Model Architectures** | ResNet, ImageNet |
| **Dataset** | APTOS 2019 Blindness Detection |
| **IDE/Tools** | Google Colab, VS Code |
| **Version Control** | Git & GitHub |

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Jayanayak2003/Diabetic-Retinopathy-Detection.git
cd Diabetic-Retinopathy-Detection
