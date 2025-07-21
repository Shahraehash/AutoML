# Glossary: Report Column Definitions

| Column | Definition |
| --- | --- |
| key | A unique ID representing the model identified. This key can be broken apart by `__` to identify the separate pipeline elements and finally which rank it received within the search (e.g., grid or random). For multi-class OvR models, the key includes `_ovr_class_X` suffix where X is the class index. |
| class_type | Identifies the classification type: 'binary', 'multiclass', or 'ovr' (One-vs-Rest for specific class analysis) |
| class_index | For multi-class models, indicates which class this row represents (0, 1, 2, etc.). NULL for binary classification or overall multi-class metrics |
| class_label | Human-readable class label. Shows custom class labels if provided, otherwise displays "Class X" format |
| scaler | Identifies the scaler used within the pipeline |
| feature_selector | Identifies the feature selector used within the pipeline |
| algorithm | Identifies the algorithm used within the pipeline |
| searcher | Identifies the hyper parameter search method used within the pipeline |
| scorer | Identifies the scoring method used to assess models within the search |
| accuracy | The models accuracy against the generalization dataset. For multi-class, this represents overall accuracy or class-specific accuracy for OvR models |
| acc_95_ci | The models 95% CI for accuracy against the generalization dataset (reported as an array representing lower and upper bounds respectively) |
| mcc | Matthews correlation coefficient which measures the quality of the classification assessed against the generalization dataset. For multi-class, represents macro-averaged MCC or class-specific MCC for OvR |
| avg_sn_sp | The models average sensitivity and specificity against the generalization dataset. For multi-class, represents macro-averaged values or class-specific values for OvR |
| roc_auc | The models ROC AUC score against the generalization dataset. For multi-class, shows macro-averaged ROC AUC or class-specific ROC AUC for OvR analysis |
| roc_auc_95_ci | The models 95% CI for ROC AUC against the generalization dataset (reported as an array representing lower and upper bounds respectively) |
| f1 | The models F1 score against the generalization dataset. For multi-class, shows macro-averaged F1 or class-specific F1 for OvR |
| sensitivity | The models sensitivity against the generalization dataset. For multi-class, shows macro-averaged sensitivity or class-specific sensitivity for OvR |
| sn_95_ci | The models 95% CI for sensitivity against the generalization dataset (reported as an array representing lower and upper bounds respectively) |
| specificity | The models specificity against the generalization dataset. For multi-class, shows macro-averaged specificity or class-specific specificity for OvR |
| sp_95_ci | The models 95% CI for specificity against the generalization dataset (reported as an array representing lower and upper bounds respectively) |
| prevalence | The models prevalence against the generalization dataset. For multi-class OvR, shows the prevalence of the specific class vs all others |
| pr_95_ci | The models 95% CI for prevalence against the generalization dataset (reported as an array representing lower and upper bounds respectively) |
| ppv | The models positive predictive value against the generalization dataset. For multi-class OvR, shows class-specific PPV |
| ppv_95_ci | The models 95% CI for positive predictive value against the generalization dataset (reported as an array representing lower and upper bounds respectively) |
| npv | The models negative predictive value against the generalization dataset. For multi-class OvR, shows class-specific NPV |
| npv_95_ci | The models 95% CI for negative predictive value against the generalization dataset (reported as an array representing lower and upper bounds respectively) |
| tn | The number of true negatives identified by the model against the generalization dataset. For multi-class OvR, represents true negatives for the specific class vs others |
| tp | The number of true positives identified by the model against the generalization dataset. For multi-class OvR, represents true positives for the specific class |
| fn | The number of false negatives identified by the model against the generalization dataset. For multi-class OvR, represents false negatives for the specific class |
| fp | The number of false positives identified by the model against the generalization dataset. For multi-class OvR, represents false positives for the specific class |
| selected_features | The features selected by the feature selector of the pipeline |
| feature_scores | The score or importance of each feature |
| best_params | The hyper parameters found by the pipeline's search for the model identified |
| test_fpr | An array representing the false positive rate at various threshold values (used to plot an ROC AUC curve) for the training dataset. For multi-class OvR, specific to the individual class |
| test_tpr | An array representing the true positive rate at various threshold values (used to plot an ROC AUC curve) for the training dataset. For multi-class OvR, specific to the individual class |
| training_roc_auc | The models ROC AUC score against the training dataset using a train/test split with cross validation. For multi-class, shows overall or class-specific training ROC AUC |
| roc_delta | The absolute value of the difference between the generalization ROC AUC score and the training ROC AUC score. Indicates potential overfitting |
| generalization_fpr | An array representing the false positive rate at various threshold values (used to plot an ROC AUC curve) for the generalization dataset. For multi-class OvR, specific to the individual class |
| generalization_tpr | An array representing the true positive rate at various threshold values (used to plot an ROC AUC curve) for the generalization dataset. For multi-class OvR, specific to the individual class |
| brier_score | The models Brier score against the generalization dataset. For multi-class OvR, represents calibration quality for the specific class |
| fop | An array representing the fraction of positives at various threshold values (used to plot a reliability curve) for the generalization dataset. For multi-class OvR, specific to the individual class |
| mpv | An array representing the mean predicted probability at various threshold values (used to plot a reliability curve) for the generalization dataset. For multi-class OvR, specific to the individual class |
| precision | An array representing the precision (aka positive predictive value) at various threshold values (used to plot a precision recall curve) for the generalization dataset. For multi-class OvR, specific to the individual class |
| recall | An array representing the recall (aka sensitivity) at various threshold values (used to plot a precision recall curve) for the generalization dataset. For multi-class OvR, specific to the individual class |

## Notes for Multi-class Classification

### Overall vs Class-Specific Results
- **Overall multi-class results**: Show macro-averaged metrics across all classes
- **OvR (One-vs-Rest) results**: Show performance for individual classes treated as binary problems against all other classes combined
- **Class filtering**: Results can be filtered by class_index to view specific class performance

### Interpreting Multi-class Metrics
- **Macro-averaging**: Metrics are averaged across all classes, giving equal weight to each class regardless of class frequency
- **Class imbalance**: Consider class prevalence when interpreting results, especially for imbalanced datasets
- **OvR interpretation**: OvR results help understand how well the model distinguishes each individual class from all others

### Custom Class Labels
- **class_label column**: Shows human-readable class names if custom labels were provided during data upload
- **Numerical mapping**: class_index still shows the underlying numerical representation (0, 1, 2, etc.)
- **Consistency**: Custom labels are used consistently throughout exports and visualizations