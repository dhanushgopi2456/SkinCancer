# 🧬 Skin Cancer Detection Using CNN

This project is a **Machine Learning/Deep Learning-based system** for detecting skin cancer using image classification. It leverages **Convolutional Neural Networks (CNNs)** to classify skin lesions as either benign or malignant. The goal is to assist in early diagnosis using computer vision techniques.

## 🧪 Tech Stack

- **Language**: Python
- **Libraries**:
  - TensorFlow / Keras
  - NumPy
  - Matplotlib
  - OpenCV
- **Dataset**: Custom dataset (ISIC or any other skin lesion image dataset)
- **IDE**: Jupyter Notebook / VS Code

## ⚙️ How It Works

1. **Dataset Preparation**
   - Images of skin lesions (categorized into benign and malignant) are preprocessed and resized.

2. **Model Building**
   - A CNN model is built using Keras with multiple convolutional, pooling, and dense layers.

3. **Training**
   - Model is trained using labeled images with validation and test splits.

4. **Evaluation**
   - Accuracy, loss, confusion matrix, and predictions are used to evaluate the model.

5. **Prediction**
   - Model can classify new input images as benign or malignant.

## 🚀 Features

- Image pre-processing (resizing, normalization)
- CNN model architecture
- Model evaluation and accuracy plotting
- Predicts class of new skin lesion images
- Visualizations of training/validation metrics

## 📁 Project Structure

SkinCancer/
├── dataset/
│ ├── train/
│ ├── test/
│ └── validation/
├── skin_cancer_detection.ipynb
├── model/
│ └── trained_model.h5
├── images/
│ └── sample_prediction.png
├── README.md
└── requirements.txt


## 📊 Sample Output

> Include charts such as:
- Accuracy/Loss curves
- Confusion matrix
- Predicted image samples

## 📦 Installation

1. **Clone the Repository**
```bash
git clone https://github.com/dhanushgopi2456/SkinCancer
cd SkinCancer

Future Improvements
Integrate Flask or Streamlit for Web UI

Use pre-trained models (ResNet, InceptionV3)

Include more classes (e.g., melanoma, nevus)

Mobile integration for real-time diagnosis

🙋‍♂️ Developed By
Dhanush Gopi Kavala
AI-ML Intern | B.Tech Student | Deep Learning Enthusiast
