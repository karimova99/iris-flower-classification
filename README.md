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

---

<a name="motivation"></a>
## 1  Project Motivation
* Demonstrate an **end-to-end ML workflow** on a classic dataset.  
* Compare multiple algorithms, hyper-parameter tuning & cross-validation.  

---

<a name="dataset"></a>
## 2. Dataset
Source: UCI - Fisher’s Iris (builtin in `sklearn.datasets.load_iris()`)<br>
Size: 150 rows × 4 numeric features + target species<br>
Classes: *Iris-setosa*, *Iris-versicolor*, *Iris-virginica*<br>

---

<a name="eda"></a>
## 3. Exploratory Data Analysis
We explored the Iris dataset to verify data quality, uncover class–feature patterns, and decide which variables should drive modelling. Key findings:
* No missing values or extreme outliers, so minimal preprocessing is required.  
* Petal length and petal width are highly correlated (ρ ≈ 0.96) and provide the clearest species separation.  
* Sepal length vs sepal width shows weaker linear relationships that still add some discriminative power.  
* *Setosa* is linearly separable; only modest model complexity is needed to distinguish *versicolor* from *virginica*. 

---

<a name="modeling"></a>
## 4. Modeling Pipeline
Algorithms tested (with GridSearchCV 5-fold):

* Logistic Regression (`C`, `penalty`)  
* Decision Tree (`max_depth`)  
* Random Forest (`n_estimators`)  
* SVM-RBF (`C`, γ)  
* K-Nearest Neighbors (`k`)

---

<a name="cv"></a>
## 5  Cross-Validation & Model Comparison
| Model                        |   Accuracy |   Precision |   Recall |   F1 Score |
|------------------------------|-----------------|------------------|---------------|-----------------|
| Logistic Regression|          1 |            1 |      1 |             1 |
| Decision Tree   |          1 |            1 |      1 |             1 |
| Random Forest  |          1 |            1 |      1 |             1 |
| SVM-RBF    |          1 |            1 |      1 |             1 |
| K-Nearest Neighbors         |          1 |            1 |      1 |             1 |
| …                            |        …   |        …     |   …    |            …  |

<details><summary>Why near-perfect scores?</summary>

*Setosa is linearly separable; the other two classes separate on petal features, so classical models converge to ≈100 % when properly tuned.*
</details>


---

<a name="boundaries"></a>
## 6  2-D Decision Boundaries

* Logistic Regression ⇒ single linear split.  
* Decision Tree / Random Forest ⇒ blocky axis-aligned cuts.  
* SVM (RBF) ⇒ smooth curved boundary hugging clusters.  
* KNeighbors ⇒ jagged islands (risk of over-fitting).

---

<a name="interpretability"></a>
## 7  Model Interpretability
Petal-length and petal-width dominate the decision process — they receive the largest absolute coefficients in Logistic Regression, the highest Gini/entropy gains in the tree-based models, and the largest mean decrease in accuracy under permutation for SVM/KNN.

SHAP value clouds reinforce the point: moving along the petal-length or petal-width axes drives the prediction score far more than changes in sepal dimensions; setosa appears as a tight low-value cluster, while versicolor and virginica separate along increasing petal size.

Sepal-width offers minimal signal (very small coefficients / importances) and is mostly useful as a secondary tie-breaker once petal information is considered.

---

<a name="learningcurve"></a>
## 8  Learning-Curve Diagnostics

Rapid convergence – both training and cross-validation accuracy climb above 95 % with only ≈ 20–30 samples; beyond ~40 samples the curves flatten, indicating that the model has already learned all meaningful patterns.

Small train–validation gap – the two curves sit almost on top of each other, which means the classifier generalises well and is not over-fitting despite the tiny dataset.

Diminishing returns – adding the remaining data points yields only marginal (≤ 1 %) accuracy gains; more data would not materially improve performance unless it introduced additional class variability or noise.

Practical implication – the Iris problem is “easy”: a modest, interpretable model trained on a small subset already reaches ceiling performance, so effort is better spent on model explainability or deployment rather than gathering more samples.

---

<a name="findings"></a>
## 9  Key Findings
1. ≥ 99 % accuracy achievable with simple linear models.  
2. Petal measurements carry nearly all discriminative power.  
3. Choose model by **simplicity vs flexibility vs explainability**:  
   *Logistic (simple + interpretable) ←→ SVM (flexible) ←→ RF (ensemble robustness).*

---

## Author
Laylo Karimova
