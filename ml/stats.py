from math import sqrt
from numpy import isnan, std
from scipy.stats import beta, t

def clopper_pearson(x, n, alpha=0.95):
    lower = beta.ppf((1 - alpha) / 2, x, n - x + 1)
    upper = beta.ppf(1 - ((1 - alpha) / 2), x + 1, n - x)
    if isnan(lower):
        lower = 0

    if isnan(upper):
        upper = 1

    return [round(lower, 4), round(upper, 4)]

def roc_auc_ci(auc, tpr, alpha=0.95):
    t_score = t.ppf((1 + alpha) / 2., len(tpr))

    lower = auc - t_score * (std(tpr) / sqrt(len(tpr)))
    upper = auc + t_score * (std(tpr) / sqrt(len(tpr)))

    lower = 0 if lower < 0 else lower
    upper = 1 if upper > 1 else upper

    return [round(lower, 4), round(upper, 4)]
