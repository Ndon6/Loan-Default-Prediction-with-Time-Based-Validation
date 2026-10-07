LOAN DEFAULT PREDICTION WITH TIME-BASED VALIDATION

I built a model that predicts whether a Lending Club loan will go bad at the moment it is approved, then checked it the way it would be used in practice: trained on the past, tested on the future, checked for leakage and calibration, and turned the probabilities into an approve/reject decision using costs.

The decisions and changes behind each step are written at the top of each step in the notebook.

Data: Lending Club loan data on Kaggle. I used 36-month loans issued 2012-2015 (589,315 loans), trained on 2012-2014 and tested on 2015.

SUMMARY
Task: predict bad_loan (Charged Off or Default) using only information known at approval.
Honest performance: PR-AUC about 0.27 against a no-skill baseline of 0.149, so roughly 1.8x better than guessing.
Leakage: two columns recorded after the outcome pushed PR-AUC to 0.98. Removing them dropped it to 0.27.
Calibration: the model underestimated 2015 risk. Platt scaling helped partly.
Decision: a cost-based threshold cut total cost by about 19% compared with approving everyone (invented costs).

RESULTS

Step	   Result
Leakage check:  PR-AUC 0.98 with leaked columns, 0.27 without. recoveries and total_pymnt carried nearly all the importance
Time split:	Train 2012-2014 (13.2% bad, 306,462 loans), test 2015 (14.9% bad, 282,853 loans)
Logistic regression:	ROC-AUC 0.689, PR-AUC 0.263
Gradient boosting:	ROC-AUC 0.692, PR-AUC 0.270 (a near tie with logistic regression)
Calibration (2015):	Mean predicted 10.9% raw vs. 14.9% actual. Platt: 12.7%, Brier 0.1218 to 0.1206
Cost threshold:	0.167 from costs of 1,000 (false positive) and 5,000 (false negative) naira

Strategy on 2015 loans	  Wrongly rejected	 Missed defaults	 Total cost (naira)
Model at threshold 0.167	 50,260	            24,064	          170.6M
Approve everyone	          0	                42,048	          210.2M
Reject everyone          	 240,805	          0	                240.8M

LIMITATIONS
Selection bias: outcomes exist only for loans Lending Club approved, so the model may not hold for the full applicant pool.
Drift: the bad rate rose from 13.2% to 14.9%. Calibration learned on 2014 cannot fully correct 2015, so recalibrating on recent data would be needed in practice.
One split, no uncertainty: I did not measure how much the scores vary, so small differences (such as logistic regression vs. gradient boosting) should not be over-read.
Invented costs: the savings figure is illustrative. Real costs would change the threshold.
Not done: hyperparameter tuning, a more careful encoding of categories for gradient boosting, and capping revol_bal.
