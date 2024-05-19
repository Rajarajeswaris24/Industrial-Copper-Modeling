DEMO VIDEO: https://www.linkedin.com/posts/rajarajeswari-s-49671428b_hello-everyone-my-new-project-industrial-activity-7197838115724537856--rwc?utm_source=share&utm_medium=member_desktop

Problem Statement:

The copper industry deals with less complex data related to sales and pricing. However, this data may suffer from issues such as skewness and noisy data, which can affect the accuracy of manual predictions. Dealing with these challenges manually can be time-consuming and may not result in optimal pricing decisions. A machine learning regression model can address these issues by utilizing advanced techniques such as data normalization, feature scaling, and outlier detection, and leveraging algorithms that are robust to skewed and noisy data.

Another area where the copper industry faces challenges is in capturing the leads. A lead classification model is a system for evaluating and classifying leads based on how likely they are to become a customer . You can use the STATUS variable with WON being considered as Success and LOST being considered as Failure and remove data points other than WON, LOST STATUS values.

The solution must include the following steps:

1.Exploring skewness and outliers in the dataset.

2.Transform the data into a suitable format and perform any necessary cleaning and pre-processing steps.

3.ML Regression model which predicts continuous variable ‘Selling_Price’.

4.ML Classification model which predicts Status: WON or LOST.

5.Creating a streamlit page where you can insert each column value and you will get the Selling_Price predicted value or Status(Won/Lost)

Approach: 

Data Understanding: 

Identify the types of variables (continuous, categorical) and their distributions. Some rubbish values are present in ‘Material_Reference’ which starts with ‘00000’ value which should be converted into null. Treat reference columns as categorical variables. INDEX may not be useful. And some negative values in 'quantity tons' and 'selling price' convert into null.

Data Preprocessing: 

1.Handle missing values with mean/median/mode or dropna.

2.Treat Outliers using IQR  from sklearn library.

3.Identify Skewness in the dataset and treat skewness if needed with appropriate data transformations, such as log transformation(which is best suited to transform target variable-train, predict and then reverse transform it back to original scale eg:dollars), boxcox transformation, or other techniques, to handle high skewness in continuous variables.

4.Encode categorical variables using suitable techniques, such as one-hot encoding, label encoding, or ordinal encoding, based on their nature and relationship with the target variable.

5.EDA: Try visualizing outliers and skewness(before and after treating skewness) using Seaborn’s boxplot, distplot, violinplot.

6.Feature Engineering: Engineer new features if applicable, such as aggregating or transforming existing features to create more informative representations of the data. And drop highly correlated columns using SNS HEATMAP.

Model Building and Evaluation:

1.Split the dataset into training and testing/validation sets. 

2.Balance the dataset using Cluster-Centroid Sampling(under_sampling) or SMOTE(over_sampling) or SMOTEENN(over_sampling + under_sampling) for classification.

3.Train and evaluate different classification models, I have used DecisionTree , RandomForest and XGBboost using appropriate evaluation metrics such as r squared score  for regression and  F1 score for classification.

4.Optimize model hyperparameters using techniques such as cross-validation  to find the best-performing model.

5.Interpret the model results and assess its performance based on the defined problem statement.

6..Same steps for Regression modelling.(note: dataset contains more noise and linearity between independent variables so itll perform well only with tree based models).

Model GUI: 

Using streamlit module, create interactive page with

   (1) task input( Regression or Classification) and 
   
   (2) create an input field where you can enter each column value except ‘Selling_Price’ for regression model and  except ‘Status’ for classification model. 
   
   (3) perform the same feature engineering, scaling factors, log/any transformation steps which you used for training ml model and predict this new data from streamlit and display the output.
   
Tips: 

Use pickle module to dump and load models such as encoder(onehot/ label/ str.cat.codes /etc), scaling models(standard scaler), ML models. First fit and then transform in separate line and use transform only for unseen data (Note:Scaling is not mandatory for these algorithms if you want can use it.)

Accuracy and best model:

Regression :

  DcisionTreeRegressor :  0.8444829572463302

  RandomForestRegressor : 0.91146444557533

  XGBRegressor : 0.8527852823149844
  
  So the best model for regression is RandomForestRegressor.
  
Classification:
  
   DecisionTreeClassifier :  0.953829286838429

   RandomForestClassifier : 0.9608614033223533

   XGBClassifier : 0.9502617336527371
  
  So the best model for Classifier is RandomForestClassifier.

  


