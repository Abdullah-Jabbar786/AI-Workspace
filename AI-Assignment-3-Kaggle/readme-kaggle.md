# Irrigation Need Prediction — Kaggle Competition

A machine learning pipeline built for a Kaggle classification competition to predict irrigation needs for crops. Achieved **0.96743 accuracy**, ranking in the **top 1200 on the leaderboard**.

## Results

| Metric | Score |
|---|---|
| Kaggle Accuracy | **0.96743** |
| Leaderboard Rank | **~1197** |
| Best Model | **LightGBM** |

## Models Compared

| Model | Type |
|---|---|
| Decision Tree | Supervised |
| Naive Bayes | Supervised |
| Logistic Regression | Supervised |
| Random Forest | Ensemble |
| XGBoost | Ensemble |
| LightGBM | Ensemble |
| K-Means | Unsupervised (used as classifier) |

## Pipeline Overview

1. **Feature Engineering** — Created interaction features (`Temp_Humidity`, `Moisture_Rainfall`, `pH_Conductivity`)
2. **Encoding** — Label encoded all categorical columns
3. **Scaling** — StandardScaler applied to all features
4. **Training** — 80/20 train-val split with stratification
5. **Evaluation** — Accuracy + confusion matrices for all models
6. **Cross Validation** — 5-Fold Stratified CV + LOOCV on subset
7. **Submission** — Final predictions generated using LightGBM

## Project Structure

```
Assignment-3-Kaggle/
├── assignment3.py        # Full ML pipeline
├── submission.csv        # Final Kaggle submission
├── cm_Decision_Tree.png
├── cm_Naive_Bayes.png
├── cm_Logistic_Regression.png
├── cm_Random_Forest.png
├── cm_XGBoost.png
├── cm_LightGBM.png
├── cm_KMeans.png
└── AI-A3-Report.pdf      # Full assignment report
```

> **Note:** `train.csv` and `test.csv` are not included — dataset is sourced from Kaggle.

## How to Run

1. Download `train.csv` and `test.csv` from the Kaggle competition page
2. Place them in the same directory as `assignment3.py`
3. Install dependencies
   ```bash
   pip install pandas numpy scikit-learn matplotlib xgboost lightgbm
   ```
4. Run the script
   ```bash
   python assignment3.py
   ```
   Outputs: `submission.csv` + confusion matrix PNGs

## Tech Stack

- **Language:** Python
- **Libraries:** scikit-learn, XGBoost, LightGBM, pandas, NumPy, matplotlib
