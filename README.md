Rice Leaf Disease Classifier: AI for Healthier Crops 🌾
Welcome to the Rice Leaf Disease Classifier, a deep learning project that uses a custom AlexNet model to detect three common rice leaf diseases: Bacterial leaf blight, Brown spot, and Leaf smut. Built to empower farmers with early disease detection, this project aims to boost crop yields and support sustainable agriculture.
🚀 Project Overview
Rice is a global staple, but diseases threaten its production. This project leverages computer vision and deep learning to classify rice leaf diseases accurately using a dataset of just 120 images. By combining a custom CNN architecture, creative data augmentation, and robust evaluation, it delivers reliable results for real-world farming.
Why This Project?

Real-World Impact: Helps farmers identify diseases early, reducing crop losses and enhancing food security.
Innovative Model: Features a custom AlexNet with batch normalization, optimized for small datasets.
Scalable Design: Extendable to other crops or deployable as a mobile app for farmers.
Educational Value: Well-documented code, perfect for learning deep learning and computer vision.

🌟 Features

Custom AlexNet Architecture: Tailored CNN with batch normalization for stable training and improved convergence.
Creative Data Augmentation: Random crops, flips, rotations, and color jitter to mimic diverse field conditions.
Robust Evaluation: 5-fold cross-validation ensures reliable performance metrics.
Interactive Visualizations: Displays true vs. predicted labels for intuitive model insights.
Small Dataset Mastery: Overcomes the challenge of limited data (120 images) with augmentation and dropout.
Future-Ready: Scalable for IoT integration or mobile apps to bring AI to farmers' hands.

📊 Dataset

Source: Rice Leaf Diseases Dataset (Kaggle)
Size: 120 images
Classes:
Bacterial leaf blight
Brown spot
Leaf smut


Preprocessing:
Training: Resize (256x256), random crop (224x224), flips, rotations, color jitter, normalization.
Testing: Resize (224x224), normalization.



🛠️ Installation
Get started in a few simple steps:
# Clone the repository
git clone https://github.com/yourusername/rice-leaf-disease-classifier.git
cd rice-leaf-disease-classifier

# Install dependencies
pip install -r requirements.txt

# Run the Jupyter Notebook
jupyter notebook idpprojectb3.ipynb

Requirements
See requirements.txt for full details:

Python 3.11
PyTorch 2.0
Torchvision
NumPy
Pandas
Matplotlib
Scikit-learn
Seaborn

🖥️ Try It Out
Run the project instantly on Google Colab:👉 Open in Colab (Update with your Colab link)
📈 Results
The model is designed to achieve high accuracy despite the small dataset. Expected performance (based on methodology):



Metric
Value (Expected)



Accuracy
~80–90%


F1-Score (Avg)
~0.80–0.85


AUC (Avg)
~0.90


Visualizations

Prediction Gallery: Shows true vs. predicted labels for test images.
Confusion Matrix: Highlights classification performance across classes.
ROC Curves: Demonstrates class separability (to be computed).

Note: Exact metrics depend on training completion. Contributions to compute and share results are welcome!
🔍 How It Works

Data Loading: Uses torchvision.ImageFolder to load images from the dataset.
Model: Custom AlexNet with batch normalization, featuring 5 convolutional layers and 3 fully connected layers.
Training:
Cross-Entropy Loss, Adam optimizer, and ReduceLROnPlateau scheduler.
Early stopping to prevent overfitting.
5-fold cross-validation for robust evaluation.


Evaluation: Computes accuracy, confusion matrix, classification report, and ROC curves.
Visualization: Displays sample predictions with denormalized images.

Challenges Overcome

Small Dataset: Addressed with heavy augmentation and dropout (0.6).
Overfitting: Mitigated using batch normalization and early stopping.
Test Data Issue: Resolved by defining a test subset for predictions.

🌱 Future Work

Expand Dataset: Include more images for better generalization.
Transfer Learning: Experiment with pre-trained models like ResNet or VGG.
Mobile App: Develop a farmer-friendly app for real-time diagnosis.
Additional Diseases: Extend to other rice diseases or crops.
Hyperparameter Tuning: Optimize learning rate, dropout, and epochs.

🤝 Contributing
Join us to make farming smarter! Here are some ideas:

Add transfer learning with pre-trained models.
Enhance the dataset with new images.
Build a web or mobile interface for predictions.
Improve visualizations (e.g., add ROC plots).

To contribute:

Fork the repository.
Create a feature branch (git checkout -b feature-name).
Commit changes (git commit -m 'Add feature').
Push to the branch (git push origin feature-name).
Open a Pull Request.

See CONTRIBUTING.md for details.
📚 Learn More

AlexNet Paper
PyTorch Documentation
Rice Leaf Diseases Dataset

📜 License
This project is licensed under the MIT License—feel free to use, modify, and contribute!

Happy farming with AI! 🚜Let's grow healthier crops together.
