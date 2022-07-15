from math import sqrt, log, exp
from numpy import isnan, std
from scipy.stats import beta, t

def clopper_pearson(x, n, alpha=0.95):
    """Calculate the Clopper-Pearson confidence interval for a sample."""
    lower = beta.ppf((1 - alpha) / 2, x, n - x + 1)
    upper = beta.ppf(1 - ((1 - alpha) / 2), x + 1, n - x)
    if isnan(lower):
        lower = 0

    if isnan(upper):
        upper = 1

    return [round(lower, 4), round(upper, 4)]

def roc_auc_ci(auc, tpr, alpha=0.95):
    """Calculate the confidence interval for the ROC AUC."""
    t_score = t.ppf((1 + alpha) / 2., len(tpr))

    lower = auc - t_score * (std(tpr) / sqrt(len(tpr)))
    upper = auc + t_score * (std(tpr) / sqrt(len(tpr)))

    lower = 0 if lower < 0 else lower
    upper = 1 if upper > 1 else upper

    return [round(lower, 4), round(upper, 4)]

def ppv_ci(sensitivity, specificity, positive_count, negative_count, prevalence):
    """Calculates the confidence interval for the positive predictive value"""

    ppv = (sensitivity * prevalence) / (sensitivity * prevalence + (1 - specificity) * (1 - prevalence))
    logit = log(ppv / (1 - ppv))
    var_logit = ((1 - sensitivity) / sensitivity) * (1 / positive_count) + (specificity / (1 - specificity)) * (1 / negative_count)
    logit_lower = logit - 1.96 * sqrt(var_logit)
    logit_upper = logit + 1.96 * sqrt(var_logit)
    return [round(exp(logit_lower) / (1 + exp(logit_lower)), 4), round(exp(logit_upper) / (1 + exp(logit_upper)), 4)]

def npv_ci(sensitivity, specificity, positive_count, negative_count, prevalence):
    """Calculates the confidence interval for the negative predictive value"""

    npv = (specificity * (1 - prevalence)) / ((1 - sensitivity) * prevalence + (specificity * (1 - prevalence)))
    logit = log(npv / (1 - npv))
    var_logit = (sensitivity / (1 - sensitivity)) * (1 / positive_count) + ((1 - specificity) / specificity) * (1 / negative_count)
    logit_lower = logit - 1.96 * sqrt(var_logit)
    logit_upper = logit + 1.96 * sqrt(var_logit)
    return [round(exp(logit_lower) / (1 + exp(logit_lower)), 4), round(exp(logit_upper) / (1 + exp(logit_upper)), 4)]
