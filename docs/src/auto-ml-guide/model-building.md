# Step 3: "Train"

The "Train" page is the brain of MILO-ML and gives rise to the large number of ML pipelines (combinations of the various ML elements and steps shown below, such as algorithms, scalers, feature selectors, searchers and scorers) that ultimately allows MILO-ML to build the thousands of ML models of interest.

![Training Page](./images/image17.png)
![Training Step](./images/image18.png)

Before a new run can begin, some options must be configured which will be explained in this section. Keep in mind the default configuration is to enable all options and is the recommended approach. Removing any of the pre-selected options will reduce the chance of finding one's best model. Although it will speed up the run since less ML pipelines (i.e., combinations of each algorithm with their respective scaler, feature selector, hyperparameter searcher and scorer) are constructed.

![Pipeline Elements](./images/pipeline-elements.png)

For additional information on each configuration option please reference the following review article which highlights how each algorithm works: [Artificial Intelligence and Machine Learning in Pathology: The Present Landscape of Supervised Methods].
<https://journals.sagepub.com/doi/pdf/10.1177/2374289519873088>).

## Multi-class Training Enhancements

MILO-ML automatically detects multi-class problems and applies appropriate configurations:

### Automatic Multi-class Detection
- **Binary Detection**: 2 unique classes in target column
- **Multi-class Detection**: 3+ unique classes in target column
- **Algorithm Adaptation**: Algorithms automatically configured for the detected classification type
- **Metric Selection**: Appropriate metrics selected (macro-averaging for multi-class)

### Enhanced Algorithm Support
All algorithms in MILO-ML support both binary and multi-class classification:
- **Logistic Regression**: Uses multinomial approach for multi-class
- **SVM**: Employs One-vs-Rest strategy automatically
- **Random Forest**: Native multi-class support
- **Neural Networks**: Output layer adapted to number of classes
- **XGBoost**: Configured with appropriate objective function
- **Naive Bayes**: Natural multi-class capability
- **K-Nearest Neighbors**: Distance-based multi-class classification

## One-vs-Rest (OvR) Optimization

For multi-class problems, MILO-ML offers an advanced feature for enhanced per-class analysis:

### OvR Re-optimization Option
**Location**: Advanced Training Options → "Re-optimize OvR Models"

**Default Setting**: Disabled (Efficient Mode)
- Uses main multi-class model for OvR analysis
- Fast training, lower resource usage
- Good for exploration and balanced datasets

**When Enabled**: Re-optimization Mode
- Creates dedicated binary classifiers for each class
- Each class gets individually optimized hyperparameters
- Higher potential accuracy per class
- Increased training time (N times longer for N classes)

### Configuration Guidance

**Enable OvR Re-optimization When**:
- Individual class performance is critical
- Dataset has significant class imbalance
- Deploying class-specific models separately
- Maximum accuracy is required per class
- Computational resources are available

**Use Default (Efficient) When**:
- Rapid prototyping and exploration
- Resource constraints exist
- Classes are well-balanced
- Overall performance is satisfactory

## Shuffle

Each time the data is split internally, we can choose to shuffle the data to ensure the order of the data is not influencing the model. The default is to have this option checked however it is configurable and can be unchecked if you choose to do so.

**Multi-class Enhancement**: Shuffling now uses stratified sampling to maintain class proportions across splits, ensuring each class is represented in both training and validation sets.

![Cross Validation Options](./images/cross-validation-options.png)

## Advanced Multi-class Training Options

### Stratified Cross-Validation
- **Automatic**: Applied to all multi-class problems
- **Class Preservation**: Ensures each fold contains samples from all classes
- **Balance Maintenance**: Maintains original class proportions in each fold

### Class Imbalance Handling
- **Detection**: Automatic identification of imbalanced classes
- **Warnings**: Alerts when class imbalance might affect results
- **Recommendations**: Suggestions for handling severe imbalances

### Memory Management for Multi-class
- **Enhanced Memory Management**: Improved handling of larger model sets
- **Progressive Training**: Models trained incrementally to manage memory
- **Automatic Cleanup**: Unused models removed from memory during training

### Training Time Estimates
- **Standard Training**: Base time estimate for efficient mode
- **OvR Re-optimization**: Additional time estimate (typically N×base time)
- **Progress Tracking**: Enhanced progress indicators for multi-class training

For detailed information about OvR optimization strategies, see the [One-vs-Rest Optimization Guide](./ovr-optimization-guide.md).

This page allows one to visualize the model performances (based on their Generalization dataset assessment) and to fine-tune or deploy the models if needed. MILO-ML now supports both binary and multi-class classification results with enhanced visualization and analysis capabilities.

![Generalization Step](./images/image19.png)

Once a run is completed, you will automatically be taken to this results page (see below). Here you can view all pipelines run and their statistical performance against your generalization dataset.

## Overview

In the image below you will see 3 graphs at the top and a table below them. The graphs represent details for the selected model (highlighted in grey) within the table. As you select new models, the graphs will update with their corresponding models and their associated performance measures.

For **multi-class models**, additional visualization options are available to analyze class-specific performance using One-vs-Rest (OvR) approaches.

![Results](./images/image20.png)

## Classification Type Detection

MILO-ML automatically detects whether your problem is:
- **Binary Classification** (2 classes): Traditional binary metrics and visualizations
- **Multi-class Classification** (3+ classes): Enhanced metrics with macro-averaging and class-specific analysis

## Performance Metrics by Classification Type

### Binary Classification Metrics
- **Accuracy**: Overall prediction accuracy
- **Sensitivity (Recall)**: True positive rate
- **Specificity**: True negative rate
- **Precision (PPV)**: Positive predictive value
- **NPV**: Negative predictive value
- **ROC AUC**: Area under the ROC curve
- **F1 Score**: Harmonic mean of precision and recall

### Multi-class Classification Metrics
- **Overall Accuracy**: Correct predictions across all classes
- **Macro-averaged metrics**: Average performance across all classes
- **Per-class metrics**: Individual class performance using One-vs-Rest
- **Confusion Matrix**: Detailed class-by-class prediction accuracy
- **Class-specific ROC curves**: OvR ROC analysis for each class

## Class-Specific Analysis (Multi-class)

For multi-class models, you can access detailed class-specific results:

1. **Class Selection**: Choose specific classes for detailed One-vs-Rest analysis
2. **Individual Class Performance**: View ROC curves, precision-recall curves, and reliability plots for each class
3. **Comparative Analysis**: Compare performance across different classes

## Custom Class Labels

If you provided custom class labels during data upload, they will be displayed throughout the results:
- Table columns show meaningful class names instead of numbers
- Graphs and charts use your custom labels
- Export files include both numerical indices and custom labels

## Table options

The table itself is very flexible and offers many options. First, on the left-hand side, you will notice a star which allows you to mark models as favorites. You may also use the header bar star to toggle showing only favorites.

Next is the table filter which allows you to drill down on results in a variety of ways. Using the `All` drop down, you can filter on all aspects of a run whereas selecting specific options from the drop down ensure the filter only applies to that aspect of the model.

**New Multi-class Filters:**
- **Class Type**: Filter by binary vs multi-class models
- **Class Index**: For multi-class models, filter by specific class results
- **Performance Threshold**: Filter models by minimum performance criteria

You may also tap on any header element to sort the table using that field. The header button can be tapped again to change the sort from descending to ascending.

## Export results

Next to the graphs, you will see a save icon (highlighted with the red box in the image below) which allows you to export PNGs of all three graphs for the model that has been selected within the table (highlighted in grey within the table).

![Export Graphs](./images/image21.png)

Additionally, there is an export button (highlighted with the red box in the image below) at the top header which allows a CSV export of the entire table of results for viewing in any spreadsheet editor (e.g., Excel). 

**Enhanced Export Options:**
- **Full Results**: Complete results including all classes
- **Class-Specific Results**: Export results for individual classes
- **Comparison Reports**: Side-by-side class performance analysis

![Export Data](./images/image22.png)

:::tip
For details on the exported report, please refer to [Glossary for Report Column Definitions](./glossary-report-export.md)
:::

## Run details

In order to see details about the MILO-ML run (e.g., how many models were built and evaluated, how many pipelines and what combination was employed, etc.), you may select one of the two blue buttons located in the middle right side (between the table and the graphs). The first button, `Parameters` allows you to see which pipeline elements were selected during step 3 (i.e., Train step). The second button, `Details` (highlighted with the red box in the image below) gives you some basic information about the number of models built within each algorithm along with what was ingested within the MILO-ML run, as shown below.

**Enhanced Details for Multi-class:**
- **Class Distribution**: Shows the number of samples per class
- **OvR Model Count**: Number of One-vs-Rest models generated
- **Memory Management**: Resource usage during training

![Run Details Button](./images/image23.png)
![Run Details](./images/run-details.png)

## Test model

Each row of a model will have a green play button indicating the ability to run the model for ad hoc testing. For multi-class models, testing supports:

- **Single Predictions**: Test individual cases and see class probabilities
- **Batch Predictions**: Upload a dataset for batch classification
- **Class Probability Display**: View probability scores for all classes

Please see the [Test model](./test-model.md) documentation for more detail.

![Run Model](./images/image24.png)

## Threshold Tuning (Binary and OvR)

For binary classification and One-vs-Rest analysis of multi-class models, you can adjust decision thresholds:
- **Binary Models**: Adjust the standard 0.5 threshold
- **Multi-class OvR**: Tune thresholds for individual class-vs-rest decisions
- **Performance Impact**: See real-time updates to metrics as you adjust thresholds

## Publish model

Each row of a model will also have a blue upload button (highlighted by the red box in the image below) indicating the ability to fix and publish the model for current or future use without the need for new or additional training.

**Multi-class Publishing Features:**
- **Complete Model**: Publish the full multi-class model
- **Class-Specific Models**: Publish individual One-vs-Rest models
- **Custom Naming**: Use meaningful names reflecting your class structure

![Publish Model Button](./images/image32.png)

Once the model of interest is selected (grey highlighted row in the table as shown above), the blue cloud button can be clicked (as highlighted by the red box in the image above) to name and ultimately publish this model on your MILO-ML homepage. You will first be presented with a new window to name your model (no spaces allowed in naming the new model) and if desired, change the default decision threshold (leave empty to use the default value of .5) as shown below.

**Note for Multi-class Models**: Publishing a multi-class model makes the complete classifier available, with the ability to predict any of the trained classes.

![Publish Model Modal](./images/image32.png)
![Publish Model Modal](./images/publish-modal.png)

Please see the [Publish model](./publish-model.md) documentation for more detail.

## Advanced Multi-class Features

### Interactive Results Toggle System

MILO-ML provides dynamic viewing options for multi-class results:

#### View Toggle Options
- **Macro-Average View**: Shows overall multi-class model performance using macro-averaged metrics
- **Individual OvR View**: Displays One-vs-Rest analysis for specific classes
- **Comparison View**: Side-by-side comparison of macro-average vs OvR performance
- **Class-Specific Deep Dive**: Detailed analysis for individual classes

#### Results Navigation
```
Interactive Controls:
├── Analysis Type: [Macro-Average] [OvR Analysis] [Comparison]
├── Class Selector: [All Classes] [Healthy] [Mild] [Moderate] [Severe]
├── Model Source: [Main Model] [Re-optimized OvR] [Both]
└── View Options: [Table] [Graphs] [Performance Summary]
```

**Dynamic Updates**: All graphs, tables, and metrics update in real-time based on your selection, allowing seamless exploration of different analysis perspectives.

### One-vs-Rest (OvR) Analysis Modes

MILO-ML provides two approaches for OvR analysis in multi-class problems:

#### Efficient Mode (Default)
- **Quick Analysis**: Uses the main multi-class model for OvR metrics
- **Resource Efficient**: Minimal additional computational cost
- **Consistent Methodology**: All classes analyzed using the same model framework
- **Use Case**: Initial exploration, resource-constrained environments, balanced datasets

#### Re-optimization Mode (`reoptimize_ovr=true`)
- **Dedicated Models**: Creates specialized binary classifiers for each class vs. all others
- **Enhanced Performance**: Each class gets individually optimized hyperparameters
- **Higher Accuracy**: Potentially better per-class performance through specialization
- **Use Case**: Production deployment, critical applications, imbalanced datasets

**Configuration**: Enable re-optimization mode in the training parameters for maximum per-class accuracy at the cost of increased training time and resources.

### Custom Class Label Display

When custom class labels are provided during upload, they appear throughout the results:

#### Label Integration
- **Results Tables**: Custom labels replace numerical codes (0,1,2 → "Healthy","Mild","Severe")
- **Graph Legends**: Charts and plots use meaningful class names
- **Export Files**: Both numerical codes and custom labels included
- **Toggle Interface**: Class selectors show custom labels for intuitive navigation

#### Benefits
- **Immediate Interpretation**: Results are instantly understandable
- **Professional Presentation**: Export-ready visualizations and reports
- **Reduced Confusion**: No need to remember numerical mappings
- **Stakeholder Communication**: Non-technical audiences can understand results

### Comprehensive Export System

#### Export Options
- **Complete Results**: Full dataset including macro-averaged and all OvR results
- **Class-Specific Exports**: Filter results by individual classes
- **Analysis Type Exports**: Separate macro-average and OvR result files
- **Comparison Reports**: Side-by-side performance analysis

#### Export Formats
```
Available Export Types:
├── report.csv: Complete results (macro + OvR)
├── [ClassName]_report.csv: Class-specific results
├── macro_average_report.csv: Overall multi-class metrics
├── ovr_comparison_report.csv: OvR vs macro comparison
└── performance_summary.csv: Resource usage and training stats
```

### OvR Analysis Features
Access detailed One-vs-Rest analysis for any class in your multi-class model:
- Individual ROC curves for each class vs. all others
- Class-specific precision-recall curves  
- Reliability calibration plots per class
- **Model-Specific Results**: View results from either efficient mode or re-optimized models
- **Performance Comparison**: Compare efficient vs. re-optimized performance when both are available

### Class Comparison Tools
- Side-by-side performance metrics with toggle between views
- Confusion matrix visualization (overall and class-specific)
- Class imbalance analysis and recommendations
- **Interactive Filtering**: Toggle between different classes and analysis types
- **Resource Usage Reports**: Training time and memory usage for different modes

### Model Interpretation
- Feature importance per class with toggle between global and class-specific views
- Class-specific feature effects
- Decision boundary visualization (where applicable)
- **Model Selection Guidance**: Recommendations on when to use efficient vs. re-optimization mode

### Export Options for OvR Models
- **Individual Class Models**: Export specific OvR binary classifiers
- **Mode Selection**: Choose exports from efficient or re-optimized models
- **Performance Reports**: Detailed comparison between approaches
- **Deployment Packages**: Complete packages for production deployment
- **Custom Label Integration**: All exports include meaningful class names

### Memory Management Integration
Results page includes memory management information:
- **Training Resource Usage**: Memory and time consumption during model creation
- **Model Storage Statistics**: Information about compression and storage efficiency
- **Access Performance**: Loading times and memory usage for different result views

For detailed information about choosing between efficient and re-optimization modes, see the [One-vs-Rest Optimization Guide](./ovr-optimization-guide.md).

For comprehensive coverage of all multi-class features, see the [Complete Multi-class Features Guide](./comprehensive-multiclass-guide.md).