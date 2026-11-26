# Operational Delay & Turnaround Analysis (Python)

[← Back to Home](./index.html)

## Project Overview
Statistical analysis and feature engineering pipeline built on **10,470+ commercial flights** to evaluate departure block-off deviations (`Blockoffdiff`), passenger load impact, and hub congestion patterns.

### Key Techniques Applied
* **Data Cleaning & Skewness Normalization:** Yeo-Johnson power transformations and logarithmic scaling to handle extreme operational outliers.
* **Inferential Validation:** One-way ANOVA and 95% confidence intervals (t-Student) to validate route and origin impact prior to modeling.
* **Feature Matrix Preparation:** One-Hot Encoding across aircraft registrations, departure/arrival stations, and 60-minute schedule blocks.
