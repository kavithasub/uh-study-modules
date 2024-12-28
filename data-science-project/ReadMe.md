# MSc data science project

## Project Title
Predict stage of lung cancer caused by smoking using machine learning models

## Overview
Lungcancer is the leading cause of cancer death worldwide, with smoking being the primary risk factor. Early detection and diagnosis can greatly increase the chances of survival; however, lung cancer is often diagnosed at an advanced stage due to the lack of early symptoms. Early lung cancer stage prediction is crucial for treatment choices and prognosis, as accurate staging plays a major role. Lung cancer stages are categorized based on the size of the tumor and spreading behavior.

Stage I

Stage I lung cancer is a small tumor that has NOT spread to any lymph nodes.
Stage IA - tumor size is less than or equal to 3 cm. Stage IA subcategorized as IA1 & IA2
Stage IB - tumor size is greater than 3 cm && less than or equal 4 cm. 

Stage II

Stage IIA - tumor size is greater than 4 cm && less than or equal 5 cm (NOT spread to the nearby lymph nodes).
Stage IIB - tumor size is less than or equal to 5 cm (spread to the lymph nodes within the lung) OR tumor size is greater than 5 cm (not spread to the lymph nodes).
Stage II tumors can be removed with surgery, but often additional treatments are recommended.

Stage III & Stage IV

Stage III cancer has not spread to other distant parts of the body where in Stage IV it has spread to more than 1 area in the other lung, the fluid surrounding the lung or the heart, or distant parts of the body through the bloodstream. In Stage III & IV the surgery is not an option and difficult to remove.


Reference - https://www.cancer.org/cancer/types/lung-cancer/detection-diagnosis-staging/staging-nsclc.html
An accurate lung cancer stage prediction might help reduce lung cancer mortality by motivating current smokers to quit and by identifying current smokers at high risk

## Objectives
● Preprocess and analyze demographic and clinical data to identify key features related to lung cancer.

● Develop and compare machine learning models to predict the lung cancer stage.

● Evaluate model performance using accuracy, recall, F1-score and precision.

● Identify the most important features contributing to lung cancer by stage.

● Provide insights and recommendations for smoker lung cancer stage detection based on model findings.

## Implementation
This project was constructed in Python using a variety of model libraries in Google Collaboratory. 

Basic mathematical, visualization, and dataframe operations were performed using the libraries numpy, pandas, matplotlib, and sns. 
Model building was done using the scikit-learn library. 

The first step was to fetch the data file named 'lung-cancer' in csv format from the Kaggle website. A dataframe comprising the information from this data file was created into the Google Colab environment. The features were taken out of this dataframe and put into another dataframe and perfom data preparation which comprised preprocessing and scaling data characteristics.

Next step was splitting data into training and test datasets to make them more compatible with the models and then start building models. The models selected were Logistic Regression, Random Forest and K-Nearest Neighbour classifications. These models were then trained using the data, and their parameters were adjusted to attain the required accuracy levels. Afterward, the models were evaluated using the test dataset.

