# Forest Cover Type Classification

## Overview

This project tackles the challenge of **predicting forest cover types** based on cartographic and environmental features.
The dataset originates from the UCI *Covertype* repository and represents real-world forest data from the Roosevelt National Forest in Colorado, USA.

The problem is framed as a **multi-class classification task** with seven possible classes (different forest types).
Our approach combines **exploratory analysis, feature preprocessing, baseline modeling, and hyperparameter optimization** to achieve strong predictive performance.

---

## Data at a Glance

The dataset contains **54 features**:

* **Continuous variables** such as elevation, slope, aspect, and various distance metrics to hydrology, roads, and fire points.
* **Binary categorical indicators** for wilderness area (4 dummies) and soil type (40 dummies).
* **Target variable**: `Cover_Type` (integer labels 1–7).

The data is **imbalanced**, with certain forest types more prevalent than others.

---

## Approach

### Understanding the Data

We began by **visualizing distributions** of continuous features and exploring **correlation patterns** to identify redundancies.
For example, elevation strongly correlates with horizontal distance to fire points, hinting at possible redundancy but also potential interaction effects.

### Preparing the Features

* **Scaling**: Continuous features were standardized to zero mean and unit variance to help gradient-boosting models converge more efficiently.
* **Preserving Binary Indicators**: Dummy variables for soil type and wilderness area were kept in their original 0/1 form.
* **Stratified Train-Test Split**: Ensured class proportions remained consistent across training and test sets.

### First Models: Establishing a Baseline

Two algorithms were chosen for the baseline:

1. **Random Forest Classifier** – Known for handling high-dimensional tabular data well without scaling issues.
2. **XGBoost Classifier** – A gradient boosting framework optimized for speed and performance.

We compared their performance to see which was more promising for tuning.

### Making XGBoost Better: Hyperparameter Optimization

While XGBoost performed competitively out-of-the-box, we turned to **Optuna** for systematic hyperparameter search.
Instead of manually guessing parameters, Optuna:

* Explored learning rate, tree depth, subsampling ratios, and regularization terms.
* Used **5-fold stratified cross-validation** to ensure fair evaluation.
* Applied **pruning** to stop unpromising trials early.
* Leveraged **GPU acceleration** for faster searches.

The optimization aimed for **maximum average validation accuracy**, not just a single lucky split.

### Evaluating the Final Model

The tuned XGBoost model was retrained on the full training set and evaluated on the held-out test set.
Key metrics included:

* **Accuracy** – overall correctness
* **Confusion Matrix** – where the model confuses certain forest types
* **Per-class F1-scores** – highlighting minority class performance

---

## Key Insights from the Results

* **Elevation** consistently emerged as the strongest predictor, followed by horizontal distance to roadways and soil type indicators.
* The tuned XGBoost model outperformed both the default XGBoost and Random Forest in accuracy and per-class balance.
* Minority classes (like Class 4 and Class 7) benefited most from hyperparameter tuning.

---

## Next Steps

1. **Engineer new features** from combinations (e.g., elevation × slope) to capture more complex geographic interactions.
2. **Address class imbalance** with oversampling or class-weight adjustments.
3. **Model ensembling** – blending tuned XGBoost, LightGBM, and Random Forest to push accuracy higher.
4. **Explainability tools** like SHAP to understand why the model makes certain predictions.


