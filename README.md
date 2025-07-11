# 🌼 Iris Flower Classification – From EDA to Explainable ML  

> **EDA → Baseline & Tuned Models → Cross-Validated Benchmarking →  
> Decision-Boundary Visuals → Model Explainability (Coefficients & SHAP) → Learning-Curve Diagnostics**

---

## Table of Contents
1. [Project Motivation](#motivation)  
2. [Dataset](#dataset)  
3. [Exploratory Data Analysis](#eda)  
4. [Modeling Pipeline](#modeling)  
5. [Cross-Validation & Model Comparison](#cv)  
6. [2-D Decision Boundaries](#boundaries)  
7. [Model Interpretability](#interpretability)  
8. [Learning-Curve Diagnostics](#learningcurve)  
9. [Key Findings](#findings)  
10. [How to Run](#run)  
11. [Project Structure](#structure)  
12. [Author](#author)

---

<a name="motivation"></a>
## 1  Project Motivation
* Demonstrate an **end-to-end ML workflow** on a classic dataset.  
* Compare multiple algorithms, hyper-parameter tuning & cross-validation.  

---

<a name="dataset"></a>
## 2. Dataset
| Property | Details |
|----------|---------|
| Source   | UCI - Fisher’s Iris (builtin in `sklearn.datasets.load_iris()`) |
| Size     | 150 rows × 4 numeric features + target species |
| Classes  | *Iris-setosa*, *Iris-versicolor*, *Iris-virginica* |
| License  | Public domain |

---

<a name="eda"></a>
## 3. Exploratory Data Analysis
<div align="center">
  <img src="assets/histograms.png" width="80%">
</div>

* **Distribution check** – stacked histograms reveal clear petal-length separation ⬆️.  
* **Species trends** – scatter-reg plots show linear relations (e.g. sepal L vs W).  
* **Correlation heatmap** confirms petal length & width are the most correlated (ρ≈0.96).  
* No missing values or outliers → safe to model without heavy cleaning.

---

<a name="modeling"></a>
## 4. Modeling Pipeline
Algorithms tested (with GridSearchCV 5-fold):

* Logistic Regression (`C`, `penalty`)  
* Decision Tree (`max_depth`)  
* Random Forest (`n_estimators`)  
* SVM-RBF (`C`, γ)  
* k-Nearest Neighbors (`k`)

---

<a name="cv"></a>
## 5  Cross-Validation & Model Comparison
| Model | Accuracy (µ±σ) | Precision µ | Recall µ | F1 µ |
|-------|---------------|-------------|----------|------|
| **QDA / LDA / KNN / Logistic / SVM** | **1.00 ± 0.00** | 1.00 | 1.00 | 1.00 |
| Random Forest | 0.96 ± 0.02 | 0.96 | 0.96 | 0.96 |
| … | … | … | … | … |

<details><summary>Why near-perfect scores?</summary>

*Setosa is linearly separable; the other two classes separate on petal features, so classical models converge to ≈100 % when properly tuned.*
</details>

![CV barplot](assets/cv_barplot.png)

---

<a name="boundaries"></a>
## 6  2-D Decision Boundaries
![PCA decision regions](assets/pca_boundaries.png)

* Logistic Regression ⇒ single linear split.  
* Decision Tree / Random Forest ⇒ blocky axis-aligned cuts.  
* SVM (RBF) ⇒ smooth curved boundary hugging clusters.  
* k-NN ⇒ jagged islands (risk of over-fitting).

---

<a name="interpretability"></a>
## 7  Model Interpretability
| Logistic Regression | Decision Tree | Random Forest |
|---------------------|--------------|---------------|
| ![](assets/lr_coef.png) | ![](assets/dt_imp.png) | ![](assets/rf_imp.png) |

**SHAP summaries** (all models) confirm *petal length* & *petal width* are the strongest features across the board.

---

<a name="learningcurve"></a>
## 8  Learning-Curve Diagnostics
![Learning curve](assets/learning_curve.png)

* Curves converge ≈ 40 samples ⇒ dataset is ample.  
* Minimal train–CV gap ⇒ low variance; adding data likely yields diminishing returns.

---

<a name="findings"></a>
## 9  Key Findings
1. ≥ 99 % accuracy achievable with simple linear models.  
2. Petal measurements carry nearly all discriminative power.  
3. Choose model by **simplicity vs flexibility vs explainability**:  
   *Logistic (simple + interpretable) ←→ SVM (flexible) ←→ RF (ensemble robustness).*

---

<a name="run"></a>
## 10  How to Run
git clone https://github.com/<your-user>/iris-flower-classification.git
cd iris-flower-classification
conda env create -f environment.yml       # or  pip install -r requirements.txt
jupyter notebook iris_flower.ipynb

---

<a name="structure"></a>
## 11 Project Structure
├── iris_flower.ipynb            # main notebook
├── iris-flower-dataset/         # raw CSV (optional)
├── assets/                      # exported figures used in README
├── environment.yml              # Conda environment spec
└── README.md

---

<a name="author"></a>
## 12 Author & License
Laylo Karimova – 2025
