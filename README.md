# STAT3013.Q21_Group05

# Prediction of E-Commerce Repurchase Behavior Using Deep Learning and Gradient Boosting Models

**Course:** STAT3013 — University of Information Technology, VNU-HCM  
**Group:** STAT3013.Q21_Group05  
**Members:** Bui Huu Tai (24521543) · Nguyen Bao Chinh (24520225) · Nguyen Khoi Nguyen (24521191)

---

## Project Overview

This project compares three predictive models for e-commerce customer repurchase (conversion) behavior on a digital marketing campaign dataset of **8,000 records and 20 features**:

| Model | ROC-AUC | PR-AUC | F1 Score |
|---|---|---|---|
| LightGBM | **0.8070** | **0.9498** | 0.8678 |
| Embedding + Attention Neural Network | 0.7779 | 0.9469 | **0.9437** |
| Deep Residual MLP | 0.7494 | 0.9404 | 0.8600 |
| Logistic Regression (baseline) | 0.6843 | — | — |

Feature selection uses the **Chi-Square test** (categorical) and **two-sample t-test** (numerical) at α = 0.05, retaining 10 out of 17 candidate features.

---

## Repository Structure

```
STAT3013.Q21_Group05/
├── dataset/              # Raw and processed dataset files
├── models/               # Model implementation scripts
│   ├── deep_residual_mlp.py
│   ├── embedding_attention_mlp.py
│   └── lightgbm_model.py
├── notebooks/            # Jupyter notebooks for EDA, training, evaluation
├── results/              # Output figures, confusion matrices, ROC/PR curves
├── LICENSE
└── README.md
```

---

## Dataset

- **Source:** Kaggle — [Predict Conversion in Digital Marketing Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predict-conversion-in-digital-marketing-dataset)
- **Size:** 8,000 customer records, 20 features
- **Target:** `Conversion` (1 = purchased, 0 = did not purchase)
- **Class distribution:** 87.65% positive / 12.35% negative

> Download the dataset from Kaggle and place it in the `dataset/` folder before running.

---

## Environment Requirements

**Python version:** 3.9+

Install dependencies:

```bash
pip install -r requirements.txt
```

**Key libraries:**

```
numpy
pandas
scikit-learn
tensorflow >= 2.10
keras
lightgbm
matplotlib
seaborn
scipy
jupyter
```

Or create a conda environment:

```bash
conda create -n ecommerce-pred python=3.9
conda activate ecommerce-pred
pip install numpy pandas scikit-learn tensorflow lightgbm matplotlib seaborn scipy jupyter
```

---

## Run Instructions

### 1. Clone the repository

```bash
git clone https://github.com/24521543/STAT3013.Q21_Group05.git
cd STAT3013.Q21_Group05
```

### 2. Download the dataset

Download from [Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/predict-conversion-in-digital-marketing-dataset) and place the CSV file in `dataset/`.

### 3. Run notebooks

```bash
jupyter notebook
```

Open notebooks in the `notebooks/` folder in order:
1. `01_EDA_and_feature_selection.ipynb` — Exploratory analysis + Chi-Square/t-test feature selection
2. `02_deep_residual_mlp.ipynb` — Train and evaluate Deep Residual MLP
3. `03_embedding_attention.ipynb` — Train and evaluate Embedding + Attention model
4. `04_lightgbm.ipynb` — Train and evaluate LightGBM
5. `05_comparison.ipynb` — Compare all models, plot ROC/PR curves

### 4. Run scripts directly

```bash
python models/lightgbm_model.py
python models/deep_residual_mlp.py
python models/embedding_attention_mlp.py
```

---

## Results

All models are evaluated on a common **20% held-out test set (1,600 records)** with stratified sampling to preserve the 87.65%/12.35% class distribution.

Key findings:
- **LightGBM** achieves the best ROC-AUC (0.8070) and most balanced error across both classes
- **Embedding + Attention** achieves the highest F1 (0.9437) and positive-class recall (98.72%) — best for minimizing missed converters
- **Deep Residual MLP** ranks third but benefits from skip connections preventing vanishing gradients
- Top predictors: `TimeOnSite`, `ClickThroughRate`, `PagesPerVisit`, `LoyaltyPoints`, `AdSpend`

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

We thank the instructors and teaching assistants of STAT3013 at the University of Information Technology, Vietnam National University Ho Chi Minh City, for their guidance throughout this project.
