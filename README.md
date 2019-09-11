# AutoML

Supervised and unsupervised learning using an exhaustive search of ideal
pre-processing (if any), algorithms, and hyper-parameters with feature engineering.

This program is currently tuned towards binary classification such as those seen
in medicine (eg. disease or no disease).

## Data

Currently data is expected to be in the `data` directory and named:
`train.csv`: The complete training set which will be split into a train and test set.
`test.csv`: A secondary data set which will be used independent of model generation
    and will be used to determine the generalizability of the model.

## Running

To run the program simply execute the below command:

```sh
python app.py train.csv test.csv | tee report.txt
```

This will execute the program and send the output to both the terminal and
the file `report.txt`. It will also output `report.csv` which contains the summary
of all models generated.

If you do not pass a train and test spreadsheet, sample data contained within
`sample-data` will be used.

The `train.csv` is will contain your full training data set which should be
balanced 50/50 for binary classifications. If the data set is not balanced
a warning will be shown.

The `test.csv` is a secondary data set which reflects the prevalence of disease
being classified (eg. not balanced). This data set is not used to train the models
in any way and is only used to calculate the generalizability of the models
generated.
