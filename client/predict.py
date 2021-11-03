"""
Use the exported pipeline to predict against new data
"""

import pandas
from joblib import load

## Configuration
DATA_FILE = 'input.csv'
PIPELINE_FILE = 'pipeline.joblib'
THRESHOLD = 0.5
# End configuration

## --------------------------------
## DO NOT MODIFY BELOW THIS LINE
## --------------------------------

# Import the CSV file
data = pandas.read_csv(DATA_FILE)

# Load the exported pipeline/model
pipeline = load(PIPELINE_FILE)

# Probability of being positive
probability = pipeline.predict_proba(data)[:, 1]

# Prediction based on the defined threshold
prediction = (probability >= THRESHOLD).astype(int)

# Round the probabilities to 4 decimal places
probability = [round(i, 4) for i in probability]

# Invert the probabilities when the prediction is negative so it reflects the correct probability
probability = [1 - i if i < THRESHOLD else i for i in probability]

# Add the predictions and probabilities to the dataframe
data['prediction'] = prediction
data['probability'] = probability

# Export the results
data.to_csv('output.csv', index=False)
