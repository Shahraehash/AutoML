# Predictive Model Guide

This package contains the exported model along with the files needed to predict against that model using Python code.

## Contents

- `predict.py`: The primary script which loads the model and predicts against it
- `input.csv`: A blank spreadsheet indicating the features or columns the model expects (and their positions)
- `requirements.txt`: The Python dependencies to properly run the model
- `README.md`: This help file

## Setup Requirements

Please ensure all Python requirements are installed using:

```sh
pip install -r requirements.txt
```

## Configuration

The configuration is contained within `predict.py` towards the top of the file. The following configuration items are available:

- `DATA_FILE`: This is the CSV file which contains the data the model needs
- `PIPELINE_FILE`: The exported joblib file containing the model itself
- `THRESHOLD`: The threshold at which to consider a result positive (pre-populated when exported)

Please do not edit anything below the configuration as this may result in the script no longer working.

## Data Input

When the model is executed, data is read in from the `DATA_FILE`. In order to predict on new data, please edit this CSV and add the required data.

## Run Model

To run the model, open a Terminal and execute the following command:

```sh
# Change directory to the extracted folder
cd model

# Runs the model and outputs to output.csv
python predict.py
```

This will result in a new file in the folder called `output.csv` which will have two columns appended to the end for both the prediction and probability of that prediction.
