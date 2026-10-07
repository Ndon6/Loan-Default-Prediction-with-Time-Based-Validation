 Loan Default Prediction with Time-Based Validation

 Project Overview

I built a machine learning model to predict whether a Lending Club loan
would become a bad loan at the time of approval.

Rather than using a random train/test split, I used a time-based split:
the model was trained on loans issued from 2012–2014 and tested on 2015
loans. This better reflects how a model would be used in production.

 Key Questions

- Can loan default risk be predicted using only information available at approval?
- How much does data leakage inflate model performance?
- How well does the model generalize to future loans?
- Can predicted probabilities be converted into a practical approve/reject decision?

 Dataset

- Source: Lending Club loan data
- 36-month loans
- Period: 2012–2015
- Total loans: 589,315
- Training period: 2012–2014
- Test period: 2015

 Methodology

1. Data cleaning and preparation
2. Target creation (`bad_loan`)
3. Leakage investigation
4. Time-based train/test split
5. Logistic Regression
6. Gradient Boosting
7. Probability calibration
8. Cost-based threshold selection
9. Business-cost comparison

 Results

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Logistic Regression | 0.689 | 0.263 |
| Gradient Boosting | 0.692 | 0.270 |

The 2015 default rate was 14.9%, giving a no-skill PR-AUC baseline
of approximately 0.149.

The Gradient Boosting model achieved a PR-AUC of 0.270.

 Leakage Investigation

Including post-outcome variables produced a PR-AUC of approximately
0.98.

After removing these variables, performance dropped to approximately
0.27.

This demonstrated how strongly data leakage can inflate model performance.

 Calibration

The raw model underestimated 2015 risk:

- Mean predicted probability: 10.9%
- Actual default rate: 14.9%

Platt scaling improved the mean prediction to 12.7% and slightly
improved the Brier score from 0.1218 to 0.1206.

 Cost-Based Decision

A threshold of 0.167 was selected using illustrative costs:

- False positive: ₦1,000
- False negative: ₦5,000

| Strategy | Wrong Rejections | Missed Defaults | Total Cost |
|---|---:|---:|---:|
| Model (0.167) | 50,260 | 24,064 | **₦170.6M** |
| Approve Everyone | 0 | 42,048 | ₦210.2M |
| Reject Everyone | 240,805 | 0 | ₦240.8M |

Using these invented costs, the model reduced estimated cost by about
₦39.6M (approximately 19%) compared with approving everyone.

 Limitations

- The dataset contains only loans that Lending Club approved.
- Default rates changed between the training and test periods.
- Only one time-based split was used.
- The costs used for threshold selection are illustrative.
- Further hyperparameter tuning was not performed.

 Future Improvements

- Test additional time periods
- Perform cross-validation using rolling time windows
- Improve categorical feature encoding
- Explore additional models
- Recalibrate probabilities using more recent data
- Test the model on a more representative applicant population

 Notebook
The complete analysis, code, visualizations, and results are available
in the Jupyter notebook in this repository.
