# - Practical Applications of Machine Learning  

This repository contains the complete solution for **Assignment 1** of the Data Science / Machine Learning course.  
All tasks have been implemented in **Google Colab** and the notebook is linked below.  

The assignment demonstrates the application of different machine learning techniques to real-world datasets.  
The goal is not only to achieve good performance but also to understand **how different models work, why results differ, and how to select the right model for the right task**.  

---

## 🔗 Google Colab Notebook
[👉 Click here to open the full Colab Notebook](https://colab.research.google.com/drive/1J5u-dJJTlRJahA6oatOi08W-wUQpoTnJ#scrollTo=V8VqUsT_e03J)

---

## 📂 Contents

### Part 2: Digit Classification (Logistic Regression)  
- Dataset: `sklearn.datasets.load_digits()`  
- Implemented **Logistic Regression** as a baseline for handwritten digit classification.  
- Logistic Regression is a simple linear model that provides a starting point for performance comparison.  
- Evaluation Metrics: Accuracy, Precision, Recall, plus confusion matrix for error analysis.  
- **Table 1: Model configuration and evaluation for digit classification**

---

### Part 3: SVM Implementation  
- Dataset: `sklearn.datasets.load_digits()`  
- Implemented **SVM with kernels (linear, rbf, poly)** to explore different decision boundaries.  
- SVM, especially with the **RBF kernel**, can capture **non-linear relationships** in digit images, leading to improved accuracy compared to Logistic Regression.  
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score.  
- Confusion Matrices for each kernel.  
- **Table 2: Performance of different SVM kernels on digit recognition**

---

### Part 4: Performance Differences and Model Selection Insights  
- Explained why Logistic Regression and SVM produce different results.  
- Logistic Regression is simple and efficient but limited to linear separations.  
- SVM with RBF kernel is more flexible, handling complex patterns in handwriting more effectively.  
- Provided practical guidelines on choosing models for image classification tasks based on accuracy vs resource trade-offs.  

---

### Part 5: PCA on Wine Quality Dataset  
- Dataset: `winequality-red.csv`  
- Applied **Principal Component Analysis (PCA)** to reduce dataset dimensionality and identify latent variables.  
- PCA highlighted key features such as **alcohol, acidity, and sulfur compounds** as important contributors to wine quality.  
- Outputs: Scree Plot, Cumulative Variance, 2D & 3D PCA Visualization, Heatmap of feature loadings.  
- **Table 3: Top Principal Components and Variance Explained**

---

### Part 6: Non-Linear PCA  
- Implemented **Kernel PCA (RBF kernel)** to detect curved and complex structures in data that regular PCA cannot capture.  
- Compared results with linear PCA to demonstrate advantages on non-linear datasets.  
- Trade-off: Kernel PCA provides better separation but has higher reconstruction error.  
- Visualized differences in explained variance and data structure discovery.  

---

### Part 7: Iris Classification  
- Dataset: `sklearn.datasets.load_iris()`  
- Implemented three models: **KNN, Logistic Regression, Decision Tree**.  
- All models achieved over **97% accuracy**, with KNN achieving perfect classification.  
- Discovered that **petal measurements (length and width)** are the strongest predictors, while sepals are less useful.  
- Demonstrates how even simple models can perform very well on well-structured datasets.  

---

### Part 8: Reflection on ML Concepts  
- Learned about **reinforcement learning** as a new paradigm: learning by trial and error with rewards and penalties.  
- Understood the importance of **data preprocessing** (cleaning, scaling, handling missing values) for building reliable models.  
- Clarified the concepts of **overfitting vs underfitting**, and techniques like regularization and dropout to improve generalization.  
- Machine learning workflow now feels like a clear roadmap: Problem Understanding → Data Preparation → Model Training → Evaluation.  

---

### Part 9: Google Colab Submission  
- All parts combined into a single, structured Colab notebook.  
- Notebook is **properly annotated**, **free of runtime errors**, and includes all code: data preprocessing, model training, visualizations, and evaluations.  
- Notebook link provided above.  

---

### Part 10: Written Communication  
- Report and README prepared with **clear structure, professional formatting, and simple explanations**.  
- Focused on making both technical results and concepts easy to understand.  
- Demonstrates good written communication by balancing technical depth with readability.  

---

## 📌 How to Run
1. Open the Colab link above  
2. Run all cells sequentially (`Runtime → Run All`)  
3. All plots, metrics, and tables will be generated automatically  

---

## ✅ Requirements
The code uses standard Python libraries:  
- `numpy`  
- `pandas`  
- `matplotlib`  
- `seaborn`  
- `scikit-learn`  

All are available by default in Google Colab.  

---

## 📜 License
This project is for **educational purposes only**.  
