# Predictive Modeling and Analysis of Airbnb Listing Prices Using Machine Learning

Course project for DAI 310: Machine Learning, Abdullah Al-Salem University.

## Overview

This project builds an end-to-end machine learning pipeline to predict Airbnb listing prices in New York City, using publicly available data from Inside Airbnb. The goal is to identify the key drivers of pricing and evaluate which modeling approach best captures the complex, non-linear relationships between listing features and price.

## Dataset

- Source: [Inside Airbnb](http://insideairbnb.com/) — New York City listings
- Features: location (latitude/longitude), room type, minimum nights, number of reviews, availability, host information
- Engineered features: demand score, availability ratio, host professionalism, distance to city center

## Methodology

1. **Data Cleaning** — handled invalid values (e.g. zero prices), missing value analysis, feature transformation, and removal of irrelevant identifiers
2. **Feature Engineering** — created demand score, availability ratio, host professionalism flag, and geographic distance-to-center feature; applied log transformation to the target variable (price) to correct skew
3. **Exploratory Data Analysis** — histograms, boxplots, scatter plots, correlation heatmap, and skewness/kurtosis analysis
4. **Modeling** — trained and compared three regression models:
   - Linear Regression (baseline)
   - Decision Tree Regressor
   - Random Forest Regressor (tuned via GridSearchCV, 5-fold cross-validation)
5. **Clustering** — applied K-Means to segment listings into behavioral groups based on pricing, availability, and demand

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 0.348 | 0.482 | 0.518 |
| Decision Tree | 0.332 | 0.467 | 0.547 |
| **Random Forest (tuned)** | **0.314** | **0.442** | **0.595** |

The tuned Random Forest model achieved the strongest performance. Feature importance analysis showed that **room type** and **location** (distance to city center, latitude/longitude) were the dominant predictors of price, while demand-related features (reviews, availability) played a smaller role.

K-Means clustering (k=3) revealed distinct listing segments: low-availability/selectively-rented listings, high-activity/high-demand listings, and long-term/premium listings with high minimum-stay requirements.

## Tech Stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn

## Limitations

- Dataset is a single snapshot and does not capture seasonal or temporal pricing trends
- Amenities, property condition, and host ratings were not available and may explain additional price variation
- Model accuracy decreases for extreme/high-priced listings (regression-to-the-mean effect)

## Author

Nourah Al-Sehali
