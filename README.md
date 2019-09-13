# AutoML

Supervised and unsupervised learning using an exhaustive search of ideal
pre-processing (if any), algorithms, and hyper-parameters with feature engineering.

This program is currently tuned towards binary classification such as those seen
in medicine (eg. disease or no disease).

## Data

Currently data is expected to be in CSV format in two files:

`train.csv`: The complete training set which will be split into a train and test set.
&nbsp;&nbsp;&nbsp;&nbsp;This data set should be balanced 50/50. If the data set is not balanced a warning
&nbsp;&nbsp;&nbsp;&nbsp;will be shown.

`test.csv`: A secondary data set which will be used independent of model generation
&nbsp;&nbsp;&nbsp;&nbsp;and will be used to determine the generalizability of the model. This data set
&nbsp;&nbsp;&nbsp;&nbsp;reflects the prevalence of disease being classified (eg. not balanced).

## Running

To run the program simply execute the below command:

```sh
python cli.py train.csv test.csv | tee report.txt
```

This will execute the program and send the output to both the terminal and
the file `report.txt`. It will also output `report.csv` which contains the summary
of all models generated.

If you do not pass a train and test spreadsheet, sample data contained within
`sample-data` will be used.

## Web Service

Running the application as a service with an HTTP API and Angular SPA front end.
The following steps need to be completed before the server can be launched:

```sh
cd ui
npm install
npm run build
cd ..
python server.py
```

The `npm` steps only need to be run once or if a change to the Angular SPA are
done.
