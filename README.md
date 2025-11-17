# Deep Learning Coursework: Adversarial Attacks & Vision Transformers (ViT)

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg) 
![Course](https://img.shields.io/badge/Course-Deep%20Learning-blueviolet.svg)
![University](https://img.shields.io/badge/University-University%20of%20Tehran-red.svg)

This repository contains the solution for **Assignment 1** of the Deep Learning course at the University of Tehran, focusing on image classification, adversarial attacks, and defensive techniques.

The project explores the robustness of **ResNet** models against noise and contrasts it with the performance of **Vision Transformers (ViT)**. A significant part of the work involves implementing adversarial attacks (like FGSM) and evaluating defensive methods, specifically adversarial training.

<img width="1050" height="502" alt="image" src="https://github.com/user-attachments/assets/b4666153-b21c-402d-ac6a-69a7ec91c0e2" />

---

## 🚀 Project Goals

This assignment was designed to provide hands-on experience with:
* Implementing and training standard models like **ResNet** on image datasets.
* Evaluating model robustness against simple perturbations like **Gaussian Noise**.
* Understanding and implementing **Adversarial Attacks** to exploit model vulnerabilities.
* Applying **Defensive Techniques** (e.g., Adversarial Training) to build more robust models.
* Fine-tuning and training **Vision Transformers (ViT)** and comparing their behavior to CNNs.

---

## 📂 Repository Structure

* `/Q1.ipynb`: The main Jupyter Notebook containing all the code, training loops, attack implementations, and visualizations.
* `/Q1.pdf`: The detailed Persian report (گزارش کار) explaining the theory, methodology, and results.
* `README.md`: This file.

---

## 📊 Key Findings & Visualizations

We analyzed the models' performance not just on accuracy, but on *why* they make certain decisions, especially under attack.

### 1. ResNet Robustness to Noise
We first established a baseline by training a ResNet model. We found that adding simple Gaussian noise significantly degraded performance, highlighting the sensitivity of standard models.

<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/9025938d-8d34-4498-9073-0e7ef883734d" />
<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/28bda7ca-6828-487e-957e-756849f66deb" />


### 2. Vision Transformer (ViT) Performance
We then trained two ViT models: one fine-tuned from pre-trained weights and one trained from scratch. The fine-tuned model achieved superior results, demonstrating the power of transfer learning.

> **[Image Placeholder]**
<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/983b7fca-c6b2-47f5-9a9e-3a4ac02b1251" />
<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/faaf8f10-8052-42ae-aaed-ac91a565bc9a" />


### 3. Adversarial Attacks & Defense
This was the core of the project. We observed that standard models are extremely vulnerable to adversarial attacks, even when invisible to the human eye.

Our key result, shown through Grad-CAM, is that **Adversarial Training** fundamentally changes *how* the model "sees" an image.

* **Standard Model (ViT-Finetuned):** Focuses on small, high-frequency textures (e.g., a few specific petals). This is a "brittle" strategy.
* **Defended Model (ViT-Finetuned-Adv):** Learns to look at the overall, holistic shape of the object (e.g., the entire cluster of flowers). This is a much more robust and human-like strategy.

> **[Image Placeholder]**
<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/30823834-9b3f-4372-8d36-360eed9a8323" />
<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/36ead93f-58e9-4af9-b590-c88c0ccea96c" />


## 🛠️ Getting Started

To run this project locally, ensure you have the necessary libraries.

### Prerequisites

* Python 3.9+
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* Tqdm

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[YourUsername]/[Your-Repo-Name].git
    cd [Your-Repo-Name]
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    *(I am assuming these based on your `Q1.ipynb` imports)*
    ```bash
    pip install torch torchvision numpy matplotlib tqdm jupyter
    ```

### Usage

All the code is contained within the Jupyter Notebook:
```bash
jupyter notebook Q1.ipynb
```
You can run the cells sequentially to reproduce the training, attacks, and visualizations.

---



## 🙏 Acknowledgements

* **Course:** Deep Learning (Neural Networks) - University of Tehran
* **Authors:**
    * Ali Ghorbani Bargani (810103209)
    * Mobin Tirafkan (810103091)

---

## 📜 License

This project is licensed under the **MIT License**.
