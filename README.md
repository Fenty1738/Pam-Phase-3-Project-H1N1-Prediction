## **H1N1 Vaccination Survey Results Project** ##
![Example Image](https://static.toiimg.com/thumb/msid-93451651,imgsize-24858,width-838,resizemode-4/93451651.jpg)

The dataset was sourced from the [DrivenData Flu Shot Learning competition](https://www.drivendata.org/competitions/66/flu-shot-learning/) and is composed of three primary files: `train.csv`, `test.csv`, and `labels.csv`.

**PROJECT OVERVIEW**

In this project, we want to use data from the National Flu Survey (NHFS 2009) to predict whether or not respondents got the H1N1 vaccine. Understanding past vaccination patterns provides an important background for understanding those of more recent pandemics such as COVID-19. The most powerful predictors of Vaccination Status are Doctor Recommendation of H1N1 vaccine, Health Insurance, opinion of H1N1 Vaccine effectiveness, opinion of H1N1 risk. For making the prediction, we used six different machine learning models: Decision Tree Classifier, Logistic Regression, Random Forest, K-Nearest Neighborhood Classifier

**Business Data and Understanding**

The target variable for this evaluation is whether a person will voluntarily receive an influenza or H1N1 vaccine based on past data randomly collected from households in the United States. This was collected by random-digit-dialing of telephones in these US households. There will be two target variables, one for the H1N1 vaccine and one for the Seasonal Flu vaccine. The target variables are clearly defined and binary already, which will aid in ease of preprocessing. This will be a supervised learning experience, as the data set at hand is complete. 

**Data Preparation**

 - The data that is available to solve this problem is concise and appropriate to the business problem. Both testing and training data is available in feature vector format. This will allow for modeling to be applied in a way that will be beneficial without an excessive amount of formatting changes. Not much needs to be done to this dataset before it is processed, though preprocessing steps will be applied. 
 
In the data preparation stage, I evaluated the available data to look for any null values that will prevent the model from being evaluated properly. I replaced null values with mode values where appropriate. Additionally, I identified outlier data that could skew results and eliminated them from the dataset for analysis.
Since the population that was given the survey was completely randomized and scattered across the United States, the data that we are able to collect and the outcomes that we are able to predict should be accurately representative of the larger population in the United States, in theory. 

**Modeling**

The goal for creating a model based on this data is to determine the likelihood of a person meeting certain demographic criterion to receive an H1N1 or seasonal influenza vaccine. 
This will be a supervised learning experience, as the data set at hand is complete. The survey questions that resulted in the dataset have responses that are either binary (yes or no) or they are numeric qualitative categorical variables, though there are demographic responses that are nonnumeric. 

The values for each of the fields are largely categorical data. This means that, though the entries are numerical, the information conveyed by those values are coded qualitative variables, not quantitative. The numerical data for these fields are representative of a corresponding response on the survey, not a mathematical value.

I attempted multiple modeling techniques to achieve the best results possible. I applied logistic regression and decision tree, optimized decision tree, random forest, optimized random forest, k-nearest neighbors, and ensemble modeling. This analysis was done using Python in Google Colaboratory. It was my assumption that the best model will be created by ensemble modeling, in which it will be attempted to use multiple models to create a more accurate, more exhaustive model. The existing models that will be chosen to become part of the ensemble model will be determined based on their individual success rates as they relate to the AUC.

Once the models have been created using the training data set, I will apply the model to the test data set. This will allow me to evaluate the effectiveness of each model as they compare to one another.

Evaluation and Deployment

The most successful model in this process will be evaluated by comparing the area under the curve (AUC/ROC) of each model. The Area Under the Curve (AUC) refers to the area under the Receiver Operating Characteristic (ROC) curve. The Receiver Operating Characteristic (ROC) curve is a popular tool used with binary classifiers. It is very similar to the precision/recall curve. Still, instead of plotting precision versus recall, the ROC curve plots the true positive rate (another name for recall) against the false positive rate (FPR). This evaluation will be performed in Python as well as through DrivenData’s competitive platform, as the test data I will be working with does not have the actual values given.

Typically, accuracy is used as a method of measurement for how well a model is fit. However, what is actually achieved when an AUC is performed over accuracy is something that will strongly discourage people going for models that are representative, but not discriminative, as this will only actually select for models that achieve false positive and true positive rates that are significantly above random chance, which is not guaranteed for accuracy. Essentially, what this means is that the AUC will provide a more usable measure for model prediction efficiency.

So that the model performance can be evaluated accurately, we split our dataset into training and test datasets. The models will be trained with the training data and then evaluated by comparing the predictions for the test set to the actual results. For this scenario, I split the dataset into 80% training and 20% test data. Two different training and test data sets were split. I created one training and test set that identified the target variable as ‘h1n1_vaccine’ and I created one training and test set that identified the target variable as ‘seasonal_vaccine’. Feature scaling was applied to these training and test datasets. Feature scaling normalizes the data so that modeling is easily calculable and in order to standardize the range of functionality of the input dataset.


The problem that this modeling is attempting to solve is a matter of cost. It can be expensive to mass produce vaccines and having an accurate predictor for the number of vaccines that are needed can be beneficial to the National Center for Immunization and Respiratory Diseases (NCIRD), National Center for Health Statistics (NCHS), and Centers for Disease Control and Prevention (CDC) to curtail these expenses. 

Additionally, another application of the findings based on this model could allow these national centers to reach out to people who may be misinformed about the risks and dangers associated with the H1N1 and seasonal flu vaccines. This model could provide a targeted audience to which reference material and educational literature can be distributed.

While gathering this information and creating models can help to identify people who are likely to receive the H1N1 and seasonal flu vaccines, knowing this information will not necessarily solve the problem of overproduction or underproduction of these vaccines. While the information can lead to an educated conclusion, other factors that are not included in the survey may potentially impact someone’s inclination to receive a vaccine. For example, this data was collected in 2009. From 2020 to present day, the world has undergone the COVID-19 pandemic. This will undoubtedly have a great impact on a person’s proclivity to receive an H1N1 or seasonal flu vaccine, especially since the matter of a COVID-19 vaccine is such a hotly-debated topic in our time. 

## Project Structure

The project is organized into several key directories and files for ease of navigation and understanding:

- **`data/`**: This directory contains the dataset files.
  - `training_set_features.csv`: The main dataset file with features data.
  - `test_set_features.csv`: The test dataset for testing the model performance on unseen data.
  - `feature_description.md`: Descriptions of the dataset columns.
- **`notebooks/`**: Jupyter Notebooks for analysis and modeling.
  - `index.ipynb`: The primary notebook containing the data analysis, visualization, and modeling steps.
- **`src/`**: Source code for custom functions and utilities used within the notebooks.
  - `data_preprocessing`: Functions for data cleaning and preparation.
  - `feature_engineering`: Functions for creating new features from the existing data.
  - `model_evaluation`: Utilities for evaluating model performance.
- **`requirements.txt`**: A list of Python packages required to run the project.
- **`LICENSE`**: The MIT License file.
- **`README.md`**: The project overview, setup instructions, and additional information.


## Methodology
The project follows a structured data science process:
- **Data Collection and Inspection:** Gather and inspect the provided dataset.
- **Data Cleaning and Preparation:** Handle missing values, outliers, and incorrect data types.
- **Exploratory Data Analysis (EDA):** Analyze the data to find patterns, relationships, and insights.
- **Modeling:** Build predictive models to predict vaccine uptake based on selected features.
- **Model Evaluation:** Assess the models' performance using appropriate metrics.
- **Interpretation:** Draw conclusions from the model results and provide recommendations.

## Detailed Modeling Section

The modeling phase of this project involves several key steps, from feature selection to model evaluation, aimed at building a robust classification model to predict house prices. Here's a detailed breakdown:

### 1. Feature Selection

- Initial features were selected based on their expected impact on vaccination, as identified during the exploratory data analysis (EDA) phase.
- Correlation analysis was performed to identify highly correlated features that might cause multicollinearity.
- A combination of domain knowledge and statistical tests (e.g., ANOVA for categorical variables) helped refine the feature set.

### 2. Data Preprocessing

- Categorical variables were encoded using one-hot encoding to convert them into a format that could be provided to the model.
- Continuous variables were scaled to have a mean of 0 and a standard deviation of 1, ensuring that no variable would dominate the model due to its scale.

### 3. Model Building

- A linear regression model was chosen as the starting point due to its simplicity, interpretability, and the linear relationship observed between many predictors and the target variable during EDA.
- The model was implemented using the `Logistic Regression` and `Decision Trees` class from `scikit-learn`.

### 4. Model Training

- The dataset was split into training and testing sets, with 80% of the data used for training and 20% for testing, to evaluate the model's performance on unseen data.
- The model was trained using the training set, fitting it to predict the house prices based on the selected features.

### 5. Model Evaluation

- The model's performance was evaluated using several metrics, including accuracy,AUC score, precison, recall and FIscore to assess its accuracy and explanatory power.
- A comparison was made between the training and testing set performances to check for overfitting.

### 6. Model Interpretation

- The coefficients of the model were analyzed to understand the impact of each feature on the vaccine uptake, providing insights into the public health authorities.

### 7. Next Steps

- Based on the initial model's performance, further steps could include exploring more complex models, such as random forest or ensemble methods, and conducting feature engineering to uncover more nuanced relationships within the data.

This detailed approach to modeling ensures a thorough understanding of the factors influencing vaccine uptake, providing a solid foundation for making informed public health vaccination strategies.

## Tools and Libraries Used
The project utilizes a range of tools and libraries for data analysis and machine learning. Here are the key technologies used:

- **Data Analysis and Manipulation:** 
  - ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
  - ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
- **Visualization:** 
  - ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23323330.svg?style=for-the-badge&logo=matplotlib&logoColor=white)
- **Machine Learning:** 
  - ![Scikit-learn](https://img.shields.io/badge/scikit_learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

## How to Run the Project

Before you begin, ensure you have Python 3.8 or later installed on your machine. Follow these steps to set up and run the project:

1. **Clone the Repository:**
   - Open your terminal and run:
     ```
     git https://github.com/Fenty1738/Pam-Phase-3-Project-H1N1-Prediction.git 
     cd your-repository-directory
     ```

2. **Install Dependencies:**
   - Ensure `pip` is up to date:
     ```
     python -m pip install --upgrade pip
     ```
   - Install project dependencies:
     ```
     pip install -r requirements.txt
     ```

3. **Launch Jupyter Notebook:**
   - Run the following command to open the project in Jupyter Notebook:
     ```
     jupyter notebook index.ipynb
     ```
   - Run the cells sequentially to reproduce the analysis and results.

## Data Understanding

The initial phase of the project involves importing necessary libraries such as pandas, numpy, matplotlib, seaborn, and various sklearn modules for preprocessing, model selection, and evaluation. The dataset is then loaded for inspection, revealing it contains 26,707 rows and 36 columns, encompassing a wide range of different individual opinions.

## Conclusion and Recommendations


This project aims to provide a thorough analysis of the H1N1 vaccine uptake to support public health officials and policymakers in making informed decisions regarding vaccination strategies and public outreach. Through detailed exploratory data analysis and model development, we have identified the key factors that significantly influence individuals' decisions to receive the H1N1 vaccine.

### Recommendations

**Targeted Interventions:** Focus educational campaigns on older adults, especially those aged 65 and above, to increase vaccine uptake in this demographic.

**Enhance Doctor Involvement:** Encourage healthcare providers to play a more active role in recommending the vaccine, as their influence is crucial for increasing vaccination rates.

**Address Misconceptions:** Develop public health campaigns to dispel myths and boost confidence in the vaccine's effectiveness, emphasizing clear and evidence-based information.

**Conduct Further Analysis:** Explore additional health-related factors or interactions that might influence vaccination decisions, particularly for those with chronic medical conditions.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

References

DrivenData. (n.d.). Flu Shot Learning: Predict H1N1 and Seasonal Flu Vaccines. DrivenData. Retrieved January 17, 2021, from https://www.drivendata.org/competitions/66/flu-shot-learning/data/

Jin Huang, & Ling, C. X. (2005). Using AUC and accuracy in evaluating learning algorithms. IEEE Transactions on Knowledge and Data Engineering, 17(3), 299–310. https://doi.org/10.1109/tkde.2005.50

ROC Curve in Machine Learning. (2020, July 26). Data Science | Machine Learning | Python | C++ | Coding | Programming | JavaScript. https://thecleverprogrammer.com/2020/07/26/roc-curve-in-machine-learning/#:~:text=ROC%20Curve%20in%20Machine%20Learning%20The%20Receiver%20Operating

