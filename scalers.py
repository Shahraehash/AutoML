# Dependencies
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scalers = {
    'none': '',
    'minmax': MinMaxScaler(),
    'std': StandardScaler()
}

scalerNames = {
    'none': 'no scaling',
    'minmax': 'min max scaler',
    'std': 'standard scaler'
}