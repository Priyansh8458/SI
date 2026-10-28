# Statistics & Machine Learning — Complete Project

A comprehensive Jupyter Notebook project applying **18 statistical and machine-learning techniques** to the Medical Cost Personal Dataset.

## Dataset

**Medical Cost Personal Dataset** (insurance.csv)
- **Source:** [GitHub — Machine Learning with R datasets](https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv)
- **Rows:** 1,338 | **Columns:** 7 (expanded to 10+ via feature engineering)
- **Variables:** age, sex, bmi, children, smoker, region, charges

## How to Run

### 1. Launch the Streamlit Web Application
```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit dashboard
streamlit run app.py
```

### 2. Launch Jupyter Notebook
```bash
jupyter notebook project.ipynb
```

Or run the notebook non-interactively:
```bash
python -m nbconvert --to notebook --execute project.ipynb --output project_executed.ipynb
```

## Techniques Covered

| # | Topic | Key Methods |
|---|-------|------------|
| 1 | **Introduction** | Dataset description, column guide, objectives |
| 2 | **Data Cleaning** | Missing values, duplicates, dtype checks, IQR & Z-score outlier detection |
| 3 | **Preprocessing** | Label/One-Hot encoding, StandardScaler/MinMaxScaler, feature engineering, train-test split |
| 4 | **EDA** | Mean, median, mode, variance, std, skewness, kurtosis; histograms, boxplots, count plots |
| 5 | **Probability Distributions** | Normal, Binomial, Poisson, Exponential fits; Q-Q plots; Shapiro-Wilk & KS tests; probability computation |
| 6 | **Random Sampling** | Simple random, systematic, stratified, cluster; comparison with population stats |
| 7 | **Central Limit Theorem** | 1,000 sample means at n = 5, 30, 100 from a skewed column; SE verification |
| 8 | **Confidence Intervals** | 90/95/99% CIs for a mean (t- and z-based) and a proportion; width vs level/sample-size plots |
| 9 | **Hypothesis Testing Framework** | Structured H₀/H₁ template for all tests |
| 10 | **One-Sample Test** | One-sample t-test for BMI |
| 11 | **Z-Tests** | One-sample, two-sample, and proportion z-tests |
| 12 | **T-Tests** | One-sample, independent two-sample (Levene + Welch), paired t-test |
| 13 | **Chi-Square** | Test of independence (contingency table, Cramér's V), goodness-of-fit |
| 14 | **Correlation & Covariance** | Covariance matrix, Pearson & Spearman correlation, heatmaps, scatter/pair plots |
| 15 | **Two-Way ANOVA** | smoker × sex on charges; interaction effect, assumption checks, interaction plot, Tukey HSD |
| 16 | **Regression** | Simple & multiple linear regression; R², RMSE, MAE, VIF, residual diagnostics; sklearn comparison |
| 17 | **Classification** | Logistic Regression, KNN, Decision Tree, Random Forest; accuracy, precision, recall, F1, ROC-AUC, cross-validation |
| 18 | **Conclusion** | Key findings, best model, limitations, future work |

## Reproducibility

- `random_state = 42` is used everywhere.
- `α = 0.05` for all hypothesis tests unless stated otherwise.

## Files

| File | Description |
|------|-------------|
| `app.py` | Interactive Streamlit Web UI (Prediction + EDA + Stats) |
| `project.ipynb` | Fully executed Jupyter Notebook with all 18 sections |
| `insurance.csv` | The dataset (auto-downloaded in the notebook if missing) |
| `requirements.txt` | Python package dependencies |
| `README.md` | This file |

## License

The dataset is sourced from the book *Machine Learning with R* by Brett Lantz and is publicly available for educational use.
