# Blog posts on my Machine Learning Projects

This repository contains my machine learning projects. Each blog post is based on a  project I did. In each blog postI explain the problem, the dataset, and the methods I used. The projects cover both technical implementations (like coding algorithms from scratch) and applied work (like using ML to study bias and fairness).

---

## Projects

### 1. Classifying Palmer Penguins

* **Goal:** Predict penguin species (Adélie, Chinstrap, Gentoo) using physical measurements.
* **Methods:** Logistic regression with feature selection and visualization.
* **Results:** Achieved 99% training accuracy and 100% test accuracy.
* **Key Idea:** Simple biological traits like flipper length and island location can be powerful predictors.

---

### 2. Design and Impact of Automated Decision Systems

* **Goal:** Build a credit-risk prediction system and study its impact on different groups.
* **Methods:** Binary decision functions, profit optimization, visualization of loan data.
* **Key Idea:** Automated systems can maximize profit but may unintentionally disadvantage certain groups (e.g., renters, younger borrowers).

---

### 3. Dissecting Racial Bias in Healthcare Algorithms

* **Goal:** Replicate Obermeyer et al. (2019) and analyze racial bias in healthcare cost prediction.
* **Methods:** Ridge regression with polynomial features, visualizations of risk vs. illness.
* **Findings:** Black patients incur \~81% of the costs of White patients with similar illness burden, meaning cost-based risk scores underestimate Black patients’ care needs.

---

### 4. Dark Web Product Classification

* **Goal:** Classify listings on the Agora darknet marketplace into categories (Drugs, Services, Weapons, etc.) using only metadata (price, rating, vendor, shipping).
* **Methods:** Logistic regression, random forest, and a custom PyTorch implementation of multiclass logistic regression.
* **Results:** Random forest achieved **94% accuracy**.
* **Key Idea:** Even without text data, metadata is strong enough to identify illicit products.

---

### 5. Implementing Logistic Regression from Scratch

* **Goal:** Build logistic regression using PyTorch.
* **Experiments:**

  * Gradient descent (with and without momentum).
  * Overfitting experiments.
  * Real-world dataset (Campus Placement Prediction).
* **Key Idea:** Writing logistic regression from scratch helped me understand optimization and convergence better.

---

### 6. Implementing Perceptron from Scratch

* **Goal:** Code the perceptron algorithm in PyTorch.
* **Experiments:**

  * Classify 2D linearly separable data.
  * Train on higher-dimensional data.
* **Key Idea:** The perceptron can perfectly separate linearly separable data, but fails on non-separable cases.

---

### 7. Sparse Kernel Machines

* **Goal:** Implement kernelized logistic regression with sparsity.
* **Methods:** RBF kernel, ℓ1 regularization for sparsity.
* **Experiments:** Showed how λ controls sparsity and how γ affects nonlinear decision boundaries.
* **Key Idea:** Kernel methods capture nonlinear patterns, while sparsity keeps the model efficient.

---

### 8. Overfitting, Overparameterization, and Double Descent

* **Goal:** Explore double descent using custom linear regression with random features.
* **Experiments:** Synthetic nonlinear data + image corruption prediction.
* **Key Idea:** Increasing model complexity past interpolation can actually improve generalization (double descent).

---

### 9. Limits of Quantitative Approaches to Bias and Fairness

* **Goal:** Reflect on Arvind Narayanan’s critique of fairness metrics.
* **Key Idea:** Numbers can both reveal and hide discrimination; quantitative analysis must be paired with social context.

---

## How to Use This Repo

* Each project has its own folder with code and notes.
* You can read the full blog posts for more detailed explanations using this [ML Project Blogs](https://pbabu-github.github.io/ML-Projects/)
  
---

## Values

These projects are not just technical exercises. They also explore the **social impact of machine learning**—bias, fairness, and ethics. My goal is to learn how to build models responsibly and understand their consequences.

---
