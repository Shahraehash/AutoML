# Model building

Before a new run can begin, some options must be configured which will be explained in this section. Keep in mind the default configuration is to enable all options and is the recommended approach. Removing any option will reduce the chance the best model will be found however will speed up the run.

![Pipeline Elements](./images/pipeline-elements.png)

For additional information on each configuration option please reference the following review article which highlights how each algorithm works: [Artificial Intelligence and Machine Learning in Pathology: The Present Landscape of Supervised Methods](https://journals.sagepub.com/doi/pdf/10.1177/2374289519873088).

## Shuffle

Each time the data is split internally, we can choose to shuffle the data to ensure the order of the data is not influencing the model. The default is to have this option checked however it is configurable and can be unchecked.

![Cross Validation Options](./images/cross-validation-options.png)
