# Final Model Comparison

| Model                 |   accuracy |   f1_macro |   Inference (ms/sample) |
|:----------------------|-----------:|-----------:|------------------------:|
| SVM + TF-IDF          |      0.542 |     0.5216 |                     0.2 |
| XGBoost + TF-IDF      |      0.526 |     0.5156 |                     0.3 |
| DistilBERT fine-tuned |      0.625 |     0.6069 |                   123.5 |

_Generated automatically by scripts/generate_final_report.py_
