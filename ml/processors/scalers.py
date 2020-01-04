"""
All scalers used
"""

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

SCALERS = {
    'none': '',
    'minmax': MinMaxScaler(),
    'robust': RobustScaler(),
    'std': StandardScaler()
}

SCALER_NAMES = {
    'none': 'no scaling',
    'minmax': 'min max scaler',
    'robust': 'robust scaler',
    'std': 'standard scaler'
}
