# 🏥 Insurance Charge Prediction
### Linear Regression vs Random Forest — A Complete ML Comparison Study

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Models](https://img.shields.io/badge/Models-2-purple)

---

## 📌 Project Overview

This project predicts **medical insurance charges** for individuals based on demographic and health features. Two machine learning models were built, evaluated, and compared:

- **Model 1** — Linear Regression (baseline)
- **Model 2** — Random Forest Regressor (with feature engineering + hyperparameter tuning)

A key finding: engineering a single interaction feature (`smoker × bmi`) became **82% of the model's predictive signal** — demonstrating that domain knowledge + feature engineering outperforms model complexity alone.

---

## 📂 Project Structure

```
Insurance-Charge-Prediction/
│
├── insurance.csv          # Dataset (1338 rows, 7 features)
├── Model1.ipynb           # Linear Regression — baseline model
├── Model2.ipynb           # Random Forest — tuned model with feature engineering
└── README.md
```

---

## 📊 Dataset

**Source:** [Kaggle — Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)

| Feature | Type | Description |
|---|---|---|
| `age` | int | Age of the individual |
| `sex` | object | Gender (male / female) |
| `bmi` | float | Body Mass Index |
| `children` | int | Number of dependents |
| `smoker` | object | Smoker status (yes / no) |
| `region` | object | US region (northeast / northwest / southeast / southwest) |
| `charges` | float | 🎯 **Target** — Medical insurance charges (USD) |

**Shape:** 1338 rows × 7 columns | **No missing values**

---

## 🔧 Preprocessing Steps

Both models use the same preprocessing pipeline:

1. **Label Encoding** — Binary columns (`sex`, `smoker`) mapped to 0/1
2. **One-Hot Encoding** — `region` column expanded with `drop_first=True` to avoid multicollinearity
3. **Bool → Int conversion** — OHE output columns cast to integer
4. **Train/Test Split** — 80% train, 20% test, `random_state=42`

```python
df['sex']    = df['sex'].map({'female': 1, 'male': 0})
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
df = pd.get_dummies(df, columns=['region'], drop_first=True)
```

---

## 🤖 Model 1 — Linear Regression

### Approach
Baseline model trained directly on preprocessed features. No feature engineering applied.

### Feature Coefficients

| Feature | Coefficient |
|---|---|
| `smoker` | +23,077 |
| `region_southeast` | −839 |
| `region_southwest` | −659 |
| `children` | +533 |
| `region_northwest` | −392 |
| `bmi` | +319 |
| `age` | +248 |
| `sex` | +102 |

**Key insight:** `smoker` dominates with a coefficient of +23,077 — being a smoker adds ~$23K to predicted charges.

### Results

| Metric | Train | Test |
|---|---|---|
| R² Score | 0.7299 | 0.8069 |
| Adjusted R² | — | 0.8058 |
| MAE | — | $4,177 |
| RMSE | — | $5,956 |
| Train-Test Gap | — | −0.077 |

### Observations
- R² of 0.807 is decent for a baseline model
- Residual distribution is **right-skewed** — long tail up to +$25K errors
- Actual vs Predicted scatter shows **two distinct clusters** (smokers vs non-smokers) that a linear model struggles to separate
- Train R² < Test R² (unusual) — suggests test split was slightly favorable

### Graphs 
<img width="1187" height="490" alt="download" src="https://github.com/user-attachments/assets/837305df-36bb-4f6a-88b3-fb849437b1b1" /><br>
<img width="847" height="699" alt="download" src="https://github.com/user-attachments/assets/d0a75401-7cfb-45bf-8efb-28cca0662e8f" />


---

## 🌲 Model 2 — Random Forest + Feature Engineering

### Key Addition: Interaction Feature

```python
df['smoker_bmi'] = df['smoker'] * df['bmi']
```

Smokers with high BMI have disproportionately high charges — this relationship is **non-linear** and was missed by Model 1. Creating an explicit interaction term allows both models to capture it.

### Hyperparameter Tuning

Default Random Forest showed a train-test gap of 0.10 (overfitting). Fixed with:

```python
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,        # prevents trees from growing too deep
    min_samples_leaf=4,  # ignores small noisy patterns
    random_state=42
)
```

### Feature Importance

| Feature | Importance |
|---|---|
| `smoker_bmi` | **82.1%** 🔥 |
| `age` | 12.3% |
| `bmi` | 2.8% |
| `children` | 1.2% |
| `smoker` | 0.8% |
| `sex` | 0.3% |
| `region_*` | < 0.6% each |

> One engineered feature accounts for **82% of the model's predictive signal.**

### Results

| Metric | Train | Test |
|---|---|---|
| R² Score | 0.9089 | **0.8735** |
| Adjusted R² | — | 0.8691 |
| MAE | — | **$2,509** |
| RMSE | — | **$4,431** |
| Train-Test Gap | — | **0.035** ✅ |

### 5-Fold Cross Validation

| Fold | R² Score |
|---|---|
| Fold 1 | 0.8728 |
| Fold 2 | 0.7968 |
| Fold 3 | 0.8920 |
| Fold 4 | 0.8398 |
| Fold 5 | 0.8604 |
| **Mean** | **0.8524** |
| **Std Dev** | **0.0325** |

Low standard deviation (0.03) confirms the model is **stable across different data splits** — not just a lucky test set.

### Graphs 
<img width="1390" height="490" alt="download" src="https://github.com/user-attachments/assets/889b45e8-cf85-4f36-aea0-4a029f6d92ac" />
<img width="571" height="432" alt="download" src="https://github.com/user-attachments/assets/4999ffa8-e66b-4bef-97dc-df3e0b57d8ac" />

---

## 📈 Model Comparison — M1 vs M2

| Metric | Model 1 (Linear Regression) | Model 2 (Random Forest) | Improvement |
|---|---|---|---|
| R² Score | 0.8653 | **0.8735** | +0.8 pts ✅ |
| MAE | $2,757 | **$2,509** | −9.0% ✅ |
| RMSE | $4,574 | **$4,431** | −3.1% ✅ |
| Overfitting | Train < Test (unusual) | Gap = 0.035 | ✅ |
| Cross Validation | ❌ Not done | Mean R² = 0.852 | ✅ |
| Feature Engineering | ❌ Raw features | `smoker_bmi` added | ✅ |

### Visual Comparison

**Actual vs Predicted:**
- Model 1: Two visible clusters, points spread away from the ideal line
- Model 2: Tighter clustering, points hug the diagonal more closely

**Residual Distribution:**
- Model 1: Right-skewed tail extending to +$25K
- Model 2: More symmetric, tighter distribution around zero

---

## 💡 Key Learnings

1. **Feature engineering > model complexity** — One interaction feature (`smoker_bmi`) improved MAE by 9% and became 82% of the signal
2. **Always cross-validate** — A single train/test split can be misleading (Model 1 had test R² > train R²)
3. **Hyperparameter tuning matters** — Default RF had 0.10 gap (overfitting); tuned RF brought it down to 0.035
4. **Residual analysis is essential** — Model 1's skewed residuals revealed a non-linearity that guided Model 2's design
5. **Report both MAE and RMSE** — MAE shows average error, RMSE penalises large errors more heavily

---

## 🚀 Tech Stack

| Tool | Usage |
|---|---|
| Python 3.10 | Core language |
| Pandas | Data loading, preprocessing |
| NumPy | Numerical operations |
| Scikit-Learn | Model training, evaluation, cross-validation |
| Matplotlib | Plotting |
| Seaborn | Visualisation |
| Jupyter Notebook | Development environment |

---

## 👤 Author

**Vikas Ajay Vishwakarma**
BSc IT Student | Aspiring Data Scientist
🔗 [GitHub](https://github.com/codewithvikas96-ui) | 💼 [Linkedin](https://www.linkedin.com/in/vikas-vishwakarma-62959a387)
