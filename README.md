# 🌾 Rice Leaf Disease Classifier: AI for Healthier Crops 🚀  

**Empowering Farmers with Deep Learning for Early Disease Detection**  

## 📌 Project Overview  
Rice feeds half the world's population, but diseases like **Bacterial Leaf Blight, Brown Spot, and Leaf Smut** threaten global food security. This project leverages **computer vision** and a **custom AlexNet model** to accurately classify rice leaf diseases—even with a small dataset of just 120 images. Designed for **real-world farming applications**, it helps farmers detect diseases early, reduce crop losses, and boost yields sustainably.  

🔬 **Key Innovations:**  
✅ **Small Dataset Mastery** – Achieves high accuracy with only 120 images using smart augmentation.  
✅ **Custom AlexNet + BatchNorm** – Optimized for stability and fast convergence.  
✅ **Farmer-Friendly AI** – Scalable for mobile deployment in agricultural fields.  

---

## 🌟 Why This Project Matters  

### 🌍 **Real-World Impact**  
- **Prevents Crop Losses:** Early detection saves up to **20-30% of rice yields**.  
- **Supports Small Farmers:** Low-cost AI solution for developing regions.  
- **Sustainable Agriculture:** Reduces excessive pesticide use by **targeted treatment**.  

### 🧠 **Technical Excellence**  
- **Advanced Augmentation:** Simulates real field conditions with **random crops, flips, rotations, and color shifts**.  
- **Robust Validation:** **5-fold cross-validation** ensures reliability.  
- **Interpretable AI:** Visualizations (confusion matrix, ROC curves) explain model decisions.  

---

## 📊 **Dataset & Preprocessing**  

### 🌿 **Rice Leaf Diseases Dataset (120 Images)**  
| **Class**                | **Samples** | **Example Use Case** |  
|---------------------------|-------------|-----------------------|  
| Bacterial Leaf Blight     | 40          | Detects water-soaked lesions |  
| Brown Spot                | 40          | Identifies fungal infections |  
| Leaf Smut                 | 40          | Spots black spore masses |  

### 🔧 **Preprocessing Pipeline**  
| **Stage**       | **Training**                          | **Testing**                     |  
|-----------------|---------------------------------------|---------------------------------|  
| **Resizing**    | 256x256 → Random Crop (224x224)       | Direct Resize (224x224)         |  
| **Augmentation**| Flips, Rotations, Color Jitter        | None                            |  
| **Normalization**| Mean=[0.485, 0.456, 0.406], Std=[0.229, 0.224, 0.225] | Same as Training |  

---

## 🛠️ **Model Architecture (Custom AlexNet)**  

```python
class CustomAlexNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # Layer 1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(64),  # Added BatchNorm for stability
            # ... (4 more convolutional layers)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.6),  # Combat overfitting
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
```
**Key Improvements Over Vanilla AlexNet:**  
- **Batch Normalization** → Faster convergence  
- **Higher Dropout (0.6)** → Better generalization on small data  
- **Optimized Kernel Sizes** → Balances accuracy & speed  

---

## 📈 **Performance & Results**  

| **Metric**          | **Expected** | **Achieved (Post-Training)** |  
|----------------------|-------------|-----------------------------|  
| **Accuracy**         | 80-90%      | **87.5%** ✅                |  
| **F1-Score (Avg)**   | 0.80-0.85   | **0.83**                    |  
| **AUC (Avg)**        | 0.90        | **0.91**                    |  

### 📊 **Visual Insights**  
🎯 **Confusion Matrix**  
<img width="319" alt="image" src="https://github.com/user-attachments/assets/18845d8c-2192-4d2e-9a99-7fce2be3e6cc" />


📉 **ROC Curves (Per-Class)**  
<img width="402" alt="image" src="https://github.com/user-attachments/assets/5507d713-f4c5-42cd-bf8b-19dafa6d8d4a" />


---

## 🚀 **Getting Started**  

### 1️⃣ **Installation**  
```bash
git clone https://github.com/yourusername/rice-leaf-disease-classifier.git
cd rice-leaf-disease-classifier
pip install -r requirements.txt  # Python 3.11+, PyTorch 2.0
```

### 2️⃣ **Run in Google Colab**  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/rice-leaf-disease-classifier/blob/main/idpprojectb3.ipynb)  

### 3️⃣ **Train & Evaluate**  
```python
python train.py --epochs 50 --lr 0.001 --batch_size 16
```

---

## 🌱 **Future Roadmap**  

🔹 **Expand Dataset** – Crowdsource farmer-submitted images for diversity.  
🔹 **Mobile App** – Build a **React Native app** for field diagnostics.  
🔹 **Transfer Learning** – Test ResNet50/EfficientNet for comparison.  
🔹 **IoT Integration** – Link with drone imagery for large-scale monitoring.  

---

## 🤝 **Contribute & Support**  

**We welcome:**  
- 👩‍🌾 **Farmers** to share real-world leaf images.  
- 👩‍💻 **Developers** to improve model architecture.  
- 📊 **Data Scientists** to enhance visual analytics.  

**How to Contribute:**  
1. Fork the repo → `git checkout -b feature/new-augmentation`  
2. Submit a PR with clear documentation.  

---
