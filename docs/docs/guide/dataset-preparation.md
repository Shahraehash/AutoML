# Dataset Preparation

Data can come in all shapes and sizes however in order for your data to be used with MILO it needs to first be compatible and second structured in a specific way. This guide will walk you through those steps to quickly prepare your data.

## Types of compatible data

MILO uses numerical data only and therefore is not compatible with data such as images, audio, video, graphic, textual or other non-numeric data formats. In some instances, invalid formats can be converted into a numeric representation allowing MILO to function properly. For example, if you have textual data such as "Low", "Medium", and "High" then you could encode these into 1, 2 and 3 respectively.

Additionally, your data might be spread across several databases or spreadsheets and will need to be joined or flattened into a single table. The result will look something similar to the below spreadsheet:

![Sample Data](./images/sample-data.png)

::: tip
Notice the first row contains the column headers which allows your model to have named inputs
:::

## Defining the model target

Once you have gathered your data, you need to add one additional column which represents the target of your model. MILO only allows for binary classification based models meaning everything under the target is valued at either 0 or 1 representing states like negative or positive, not-present or present, etc.

![Sample Data with Target](./images/sample-data-with-target.png)

::: tip
Notice the last column now contains a new column and the values below it represent a negative value for the listed row.
:::

## Assessing data completeness

When exporting data in bulk, often times you will find gaps for some rows in various columns and those values maybe represented in various ways. Gaps can be blank, NaN, Null, or other representations. MILO will remove all rows which do not contain 100% of the column data. The number of rows removed versus used will be visible after uploading the dataset and discussed in a later chapter.

::: warning
Some datasets may have other invalid values such as numerical representations (eg. -1) and these cases will not be removed by MILO and must be handled prior to upload.
:::

## Creating the two required datasets

One of MILO's strengths is the generalization step which tests a model against a secondary dataset which helps avoid common pitfalls of model building such as over-fitting. This is because the second (generalization) dataset is not used during the training phase whereas the first (training) dataset is split internally into a train/test split and is influenced by it's own data.

The training dataset should typically be balanced between negative and positive cases of your binary classifier. The generalization dataset should be more reflective of the target prevalence.
