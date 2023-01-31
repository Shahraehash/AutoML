# Pre- and Post-Processing Tools

MILO provides a set of pre and post processing tools which allows users to more easily prepare and optimize
their data for use within the MILO Auto-ML tool. These can be accessed from the MILO landing page (under Preprocessing Tools).

![Preprocessing Tools](./images/Preprocessing-Home-Tool1.png)

The tools include:

- **Train and Test Builder**: Converting a single data file to the necessary two datasets (training / initial
validation test and generalization test datasets when needed) that is required within the MILO Auto-ML tool.

- **Multicollinearity Assessment & Removal Tool**: Allows you to observe and assess the correlations between
the variables and to remove high correlates when deemed appropriate.

- **Feature Selector**: As the name implies, this tool will allow you to assess and select the statistical
contributions of the independent variables to the target/outcome variable through two different methods
(an ANOVA F value approach and the Random Forest Importances method). This will allow you to visualize
and select for the most significant features within your dataset when necessary.

- **Column Reducer Tool**: Removes specific user-defined columns/features when needed.

- **Imputation & Encoder Tool** : Allows you to iteratively impute missing values and encode non-numerical data into the numerical data. <span class="badge-style">MILO Pro</span>

- **Automated Preprocessor Tool** : Combines all the tools into one smooth flow to prepare your data for MILO including segementation and imputation. <span class="badge-style">MILO Pro</span>



<style>
.badge-style {
    background: #2a97f3;
    color: white;
    border-radius: 10px;
    padding: 2px 10px;
    font-size: 14px;
    display: inline-block;
    height: 18px;
    line-height: 18px;
}
</style>