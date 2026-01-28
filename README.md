Mini Project II – Loan Approval Prediction (A-Upgrade)

Problem Overview:
Predict whether a loan application is approved (Loan_Status=Y) or rejected (Loan_Status=N).

Dataset:
- 614 rows, 13 columns
- Class distribution: ~68.7% approved, ~31.3% rejected (moderate imbalance)

A-Upgrade Improvements:
- Added missingness indicator flags (e.g., Credit_History_missing) to reduce imputation bias risk.
- Replaced SMOTE with SMOTENC (categorical-aware oversampling) to avoid unrealistic synthetic applicants.
- Added threshold tuning using business costs (FP cost=5× FN cost).

Final Test Results (20% stratified split):
- LogReg + class_weight (missing indicators): Precision=0.859, Recall=0.929, F1=0.893, ROC-AUC=0.860
- LogReg + SMOTENC (cv=3, best C=0.01): Precision=0.870, Recall=0.706, F1=0.779, ROC-AUC=0.826
- LogReg + SMOTENC (threshold tuned @ 0.48, FP cost=5×): Precision=0.893, Recall=0.882, F1=0.888, ROC-AUC=0.826

Best Model Notes:
- Best by F1 in this run: LogReg + class_weight (missing indicators) (F1=0.893).
- Cost-optimized operating point (FP cost=5×): threshold=0.48, confusion matrix TN=29, FP=9, FN=10, TP=75.

How to Run:
Open in Google Colab (free) and run all cells top-to-bottom.

Team Contributions:
All work completed by Vibhor.
