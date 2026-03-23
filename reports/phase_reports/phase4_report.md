# Phase 4 — Model Evaluation Report

## SVM + TF-IDF
- Accuracy: 0.4972
- F1 Macro: 0.4763

               precision    recall  f1-score   support

Very Negative       0.58      0.55      0.57      1468
     Negative       0.53      0.22      0.32      1412
      Neutral       0.50      0.37      0.43      1859
     Positive       0.45      0.57      0.50      2067
Very Positive       0.49      0.68      0.57      2095

     accuracy                           0.50      8901
    macro avg       0.51      0.48      0.48      8901
 weighted avg       0.50      0.50      0.48      8901


## XGBoost + TF-IDF
- Accuracy: 0.5210
- F1 Macro: 0.5060

               precision    recall  f1-score   support

Very Negative       0.57      0.53      0.55      1468
     Negative       0.45      0.35      0.40      1412
      Neutral       0.45      0.36      0.40      1859
     Positive       0.49      0.66      0.56      2067
Very Positive       0.60      0.64      0.62      2095

     accuracy                           0.52      8901
    macro avg       0.51      0.51      0.51      8901
 weighted avg       0.52      0.52      0.51      8901

