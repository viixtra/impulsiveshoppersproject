# 🛒 Predicting Impulsive Online Purchase Behavior

> **Business Question:** Can we predict which online shoppers are likely to make impulsive purchases — and what drives them?

---

## Overview

Impulse buying accounts for a significant share of e-commerce revenue, yet it remains difficult to predict and even harder to influence strategically. This project investigates impulsive purchasing behavior from two angles: **what people do** (behavioral data) and **who people are** (psychological data).

Using three real-world datasets and four machine learning models, we built and evaluated classifiers to identify impulsive buyers — then translated model performance into actionable business recommendations.

---

## The Datasets

| Dataset | Source | Goal | Size |
|---|---|---|---|
| Consumer Shopping Behavior | Kaggle | Predict behavioral impulsivity from purchase patterns | ~3,900 records |
| Mendeley E-Paylater Survey | Mendeley | Predict psychological impulsivity from Likert-scale survey data | 306 respondents |
| Vietnamese TikTok Shopping | Kaggle | Predict platform-driven impulsivity from social commerce behavior | ~350 respondents |

> **Note:** Datasets can additionally be found by searching in the links below.
> - [Consumer Shopping Behavior — Kaggle](https://www.kaggle.com/)
> - [Mendeley Dataset](https://data.mendeley.com/)
> - [TikTok Impulse Buying Dataset — Kaggle](https://www.kaggle.com/)

---

## Methodology

### Target Variable Engineering
Since no dataset included an explicit "impulsive buyer" flag, I engineered custom target variables for each:

- **Behavioral dataset:** A shopper was labeled impulsive if they shopped monthly or more (or had above-average prior purchases) AND used a discount or promo code — capturing frequency + incentive-triggered buying
- **Mendeley dataset:** Impulsive Buying Behavior score computed as the mean of IBB1–IBB4 (Likert items); threshold set at ≥ 3.5
- **TikTok dataset:** Online Impulse Buying score computed as the mean of OIB1–OIB3; threshold set at ≥ 3.5

### Feature Engineering
- Converted categorical purchase frequency strings to numeric monthly equivalents
- Aggregated Likert-scale survey items into psychological constructs (Self-Control, Happiness, Social Influence, Promotion, Normative Evaluation) to reduce multicollinearity
- One-hot encoded categorical variables; dropped data leakage columns

### Models Used
- K-Nearest Neighbors (KNN)
- Logistic Regression (Full, Forward Selection, Backward Selection)
- Decision Tree
- Random Forest

All models were tuned using `GridSearchCV` with 5-fold cross-validation, optimizing for **F1 score** (chosen because false negatives — missing an impulsive buyer — carry the highest business cost).

---

## Key Results

### Behavioral Dataset (strongest performer)
| Model | F1 (Test) | Accuracy | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression (Forward) | 0.75 | 0.83 | 0.76 | 0.74 |
| Decision Tree (Tuned) | 0.78 | 0.87 | 0.98 | 0.65 |
| Random Forest (Tuned) | 0.78 | 0.87 | 0.97 | 0.65 |
| KNN (Tuned) | 0.69 | 0.82 | 0.82 | 0.59 |

### Psychological Dataset (Mendeley — harder to predict)
| Model | F1 (Test) | Accuracy |
|---|---|---|
| Logistic Regression (Backward) | 0.55 | 0.79 |
| Decision Tree (Tuned) | 0.52 | 0.76 |
| KNN (Tuned) | 0.46 | 0.73 |

---

## What We Found

**Behavior tells you *who* to target. Psychology tells you *how* to influence them.**

1. **Behavioral data is highly predictive.** Purchase frequency, subscription status, and promo code usage are the strongest signals of impulsive buying. Repeat, subscribed customers are the most impulse-prone group.

2. **Psychological data is harder to model.** Self-reported survey responses are noisy — impulsivity is often unconscious and situational. Promotion sensitivity and low self-control were the most consistent psychological predictors.

3. **Two impulsive buyer types emerged:**
   - **Predictable Impulsives** — frequent buyers, promo-reactive, subscription-heavy. Easy to target.
   - **Situational Impulsives** — irregular buyers triggered by mood or context. Timing-dependent, harder to classify.

---

## Business Recommendations

- **Personalize promotions** for high-frequency, subscribed customers — they're the most reliably impulsive segment
- **Use flash sales and limited-time offers** to activate situational impulsives at the right moment
- **Combine behavioral and psychological signals** for richer customer segmentation — behavior identifies who, psychology explains why

---

## Files

```
📁 impulsive-shoppers-analysis
├── project_datasets_overview.py   # Data loading, target variable engineering, preprocessing
├── knn_behavior.py                # KNN modeling on Consumer Behavior dataset
├── knn_mendely.py                 # KNN modeling on Mendeley dataset
└── README.md
```

> Additional model files (Logistic Regression, Decision Tree, Random Forest) coming soon.

---

## Tools & Libraries

`Python` `pandas` `scikit-learn` `matplotlib` `seaborn` `GridSearchCV`

---

## Authors

Amber Ramirez, Team Member 2, Team Member 3 — MISM3515 Data Mining, Northeastern University
