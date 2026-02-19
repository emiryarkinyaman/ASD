# Machine Learning Approach To Early Autism Diagnosis Using ATR-FTIR Spectroscopy

![Status](https://img.shields.io/badge/Status-Research%20Archived-purple)
![Affiliation](https://img.shields.io/badge/Affiliation-Hacettepe%20University-crimson)

-------

## ⚠️ Important Disclaimer: Data Privacy & Reproducibility

**Please Note:** The dataset included in the `Data/` directory consists of **synthetic/dummy data** generated solely for the purpose of demonstrating the code pipeline and algorithmic approach. Due to strict privacy protocols and intellectual property rights associated with the original clinical study, the actual ATR-FTIR spectroscopy data from urine samples cannot be shared publicly.

**Result Discrepancy:** The performance metrics (Accuracy, F1-Score) you will observe when running this code with dummy data **may not match** the high accuracy (>90%) reported in the study.

**Purpose:** This repository is intended to showcase the **Machine Learning Architecture, Preprocessing Pipeline, and Evaluation Methodology** used in the research.

-------

## 🔬 Project Overview

This project explores the application of machine learning algorithms for the early diagnosis of autism spectrum disorder (ASD) based on **Attenuated Total Reflection Fourier Transform Infrared (ATR-FTIR)** spectroscopy data.

The primary objective was to design a robust classification pipeline to handle high-dimensional spectral data derived from urine samples, aiming to improve non-invasive early detection outcomes.

## 🛠️ Tech Stack & Methodology

The project utilizes a comprehensive Data Science stack to process spectral signals:

* **Language:** Python 3.14.3
* **Primary Libraries:** `scikit-learn`, `pandas`, `numpy`, `matplotlib`
* **Algorithms Implemented:**
    * SVM
    * kNN
    * Logistic Regression
    * Random Forest Classifier
    * Decision Trees
    * Linear & Quadratic Discriminant Analysis
    * Naive Bayes

## 🤝 Workflow
* **Data Ingestion:** Automated loading of spectral data from Excel formats.
* **Preprocessing:** Feature scaling using `StandardScaler` to normalize spectral intensities.
* **Hyperparameter Tuning (`optimizer.ipynb`):** Using `GridSearchCV` to find optimal kernels and regularization parameters for the top-performing models.
* **Model Benchmarking (`main.py`):** Simultaneous training and evaluation of 8+ algorithms using K-Fold Cross Validation.

## 🚀 Key Results (Based On Private Dataset)

Using the proprietary dataset collected during the study, the developed machine learning pipeline achieved significant milestones:

* **Accuracy:** Achieved **>90% accuracy** in distinguishing ASD cases from control groups.
* **Optimization:** Hyperparameter tuning significantly reduced False Negatives, crucial for medical diagnosis.
* **Best Performing Model:** Support Vector Machines (SVM) showed superior performance in spectral classification.

## 📂 Repository Structure

```plaintext
/ASD
|-- /Data
|-- /Plots
|-- /Primary Study
|-- /Results
|-- /Supportive Studies
|-- LICENSE
|-- README.md
|-- main.py
|-- optimizer.ipynb
|-- requirements.txt
```

## 💻 Usage & Exploration

Since this is a research archive, the code is provided for educational and portfolio review purposes.

## 📚 Acknowledgments

This project builds upon the foundational research conducted by Neslihan Sarigul, Leyla Bozatli, Ilhan Kurultak, and Filiz Korkmaz, detailed in "Using urine FTIR spectra to screen autism spectrum disorder" (2023).

Special thanks to the authors for their contributions to the field.

## 🔒 License & Copyright

© 2022-2026 Emir Yarkın Yaman.

**Strictly Prohibited:** Commercial use, redistribution, or modification of this code for production systems without prior written consent.

**Allowed:** You may view and fork this repository to review the code structure and methodology.

This project is published for portfolio and academic review purposes only. For inquiries regarding the methodology or collaboration, please contact.
