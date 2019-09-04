# Dependencies
from sklearn.preprocessing import StandardScaler

scalers = {
    'none': '',
    'std': StandardScaler()
}

scalerNames = {
    'none': 'no scaling',
    'std': 'standard scaler'
}