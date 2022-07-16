# Glossary: Report Column Definitions

| Column | Definition |
| --- | --- |
| key | A unique ID representing the model identified. This key can be broken apart by `__` to identify the separate pipeline elements and finally which rank it received within the search (e.g., grid or random) |
| scaler | Identifies the scaler used within the pipeline |
| feature_selector | Identifies the feature selector used within the pipeline |
| algorithm | Identifies the algorithm used within the pipeline |
| searcher | Identifies the hyper parameter search method used within the pipeline |
| scorer | Identifies the scoring method used to assess models within the search |
| accuracy | The models accuracy against the generalization dataset |
| acc_95_ci | The models 95% CI for accuracy against the generalization dataset (reported as an array representing lower and upper bounds respectively) |
| mcc | Matthews correlation coefficient or the mean square contingency coefficient which measures the quality of the binary classification assessed against the generalization dataset |
| avg_sn_sp | The models average sensitivity and specificity against the generalization dataset |
| roc_auc | The models ROC AUC score against the generalization dataset |
| roc_auc_95_ci | The models 95% CI for ROC AUC against the generalization dataset (reported as an array representing lower and upper bounds respectively) |
| f1 | The models F1 score against the generalization dataset |
| sensitivity | The models sensitivity against the generalization dataset |
| sn_95_ci | The models 95% CI for sensitivity against the generalization dataset (reported as an array representing lower and upper bounds respectively) |
| specificity | The models specificity against the generalization dataset |
| sp_95_ci | The models 95% CI for specificity against the generalization dataset (reported as an array representing lower and upper bounds respectively) |
| prevalence | The models prevalence against the generalization dataset |
| pr_95_ci | The models 95% CI for prevalence against the generalization dataset (reported as an array representing lower and upper bounds respectively) |
| ppv | The models positive predictive value against the generalization dataset |
| ppv_95_ci | The models 95% CI for positive predictive value against the generalization dataset (reported as an array representing lower and upper bounds respectively) |
| npv | The models negative predictive value against the generalization dataset |
| npv_95_ci | The models 95% CI for negative predictive value against the generalization dataset (reported as an array representing lower and upper bounds respectively) |
| tn | The number of true negatives identified by the model against the generalization dataset |
| tp | The number of true positives identified by the model against the generalization dataset |
| fn | The number of false negatives identified by the model against the generalization dataset |
| fp | The number of false positives identified by the model against the generalization dataset |
| selected_features | The features selected by the feature selector of the pipeline |
| feature_scores | The score or importance of each feature |
| best_params | The hyper parameters found by the pipeline's search for the model identified |
| test_fpr | An array representing the false positive rate at various threshold values (used to plot an ROC AUC curve) for the training dataset |
| test_tpr | An array representing the true positive rate at various threshold values (used to plot an ROC AUC curve) for the training dataset |
| training_roc_auc | The models ROC AUC score against the training dataset using a train/test split with cross validation |
| roc_delta | The absolute value of the difference between the generalization ROC AUC score and the training ROC AUC score |
| generalization_fpr | An array representing the false positive rate at various threshold values (used to plot an ROC AUC curve) for the generalization dataset |
| generalization_tpr | An array representing the true positive rate at various threshold values (used to plot an ROC AUC curve) for the generalization dataset |
| brier_score | The models Brier score against the generalization dataset |
| fop | An array representing the fraction of positives at various threshold values (used to plot a reliability curve) for the generalization dataset |
| mpv | An array representing the mean predicated probability at various threshold values (used to plot a reliability curve) for the generalization dataset |
| precision | An array representing the precision (aka positive predictive value) at various threshold values (used to plot a precision recall curve) for the generalization dataset |
| recall | An array representing the recall (aka sensitivity) at various threshold values (used to plot a precision recall curve) for the generalization dataset |
