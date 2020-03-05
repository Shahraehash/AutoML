from numpy import isnan
from scipy.stats import beta

def clopper_pearson(x, n, alpha=0.95):
    lower = beta.ppf((1 - alpha) / 2, x, n - x + 1)
    upper = beta.ppf(1 - ((1 - alpha) / 2), x + 1, n - x)
    if isnan(lower):
        lower = 0

    if isnan(upper):
        upper = 1

    return (lower, upper)
