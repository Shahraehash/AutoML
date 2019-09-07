# AutoML

Supervised and unsupervised learning using an exhaustive search of ideal
pre-processing (if any), algorithms, and hyper-parameters with feature engineering.

## Data

Currently data is expected to be in the `data` directory and named:
`train.csv`: The complete training set which will be split into a train and test set.
`test.csv`: A secondary data set which will be used independent of model generation
    and will be used to determine the generalizability of the model.

## Running

To run the program simply execute the below command:

```sh
python app.py | tee report.txt
```

This will execute the program and send the output to both the terminal and
the file `report.txt`.