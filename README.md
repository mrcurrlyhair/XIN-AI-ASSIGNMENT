High risk accident models .

High risk is classed as >=3 vehicles involved
in the code and data sets, vehicals are called cars_involved or nun_units 

====RESULTS====
--- PRE-REMOVAL OF LEAST IMPORTANT macro-avg (Precision, Recall, F1-Score, Accuracy) ---

Random Forrest - 0.57, 0.72, 0.56, 0.74

XGBoost -  0.65, 0.52, 0.52, 0.92

Gradient Boosting -  0.61, 0.60, 0.61, 0.90


--- POST-REMOVAL OF LEAST IMPORTANT macro-avg (Precision, Recall, F1-Score, Accuracy)--- 


Random Forrest - 0.57, 0.73, 0.63, 0.74

XGBoost -  0.66, 0.52, 0.58, 0.92

Gradient Boosting -  0.61, 0.60, 0.61, 0.90




please make sure the files/folders are available before starting - 

FOLDERS -
least_important
saved_models

FILES - 
clean_data.py
models.py
least_important.py
pre_graphs.py
traffic_accidents.csv

WITH MODELS (optional but reconmended to save time computing) - 
/saved_models/ gb_model.pkl (post-removal models)
/saved_models/ rf_model.pkl (post-removal models)
/saved_models/ xgb_model.pkl (post-removal models)
/least_important/ least_important_gb
/least_important/ least_important_rf
/least_important/ least_important_xgb


Please note , pre and post removal of least important features are located in models_archive.
To switch , please copy and paste the files into the saved_models folder.
The orignal files in saved_models is post removal of least important features. 


Please run in this order :
clean_data.py - cleans and prepared the data
              - this will produce a new CSV file called 'cleaned_traffic_accidents.csv'

pre_graphs.py - produces graphs from the 'cleaned_traffic_accidents.csv'
              - Graphs include a heatmap and bar charts visaulising and comparing data 

models.py - this trains and tests the models 
          - finds the best peramters for each model using gridsearchcv
          - produces f1 and accuracy scores of models
          - saves models using pickle in a folder called saved_models
          - PLEASE NOTE : TRAINING MODELS DOES TAKE 30-45 MINS DEPEDING ON HARDWARE. 

least_important.py - this find the least important features used afetr training the model
                   - this outputs the least important features as csv files 
                   - PLEASE NOTE : to retrain the models without the least important features , please rerun models.py 

models_results.py - this creates graphs of the performance of each model and its features
                  - This creates a new CSV file for the prediction graph called cleaned_traffic_accidents_predictions.csv
                  - Graphs:
                  - Models Performance
                  - Confusion Matrixes
                  - Precision Recall Curves
                  - Feature Importance
                  - Models Prediction

predictor.py WIP - this will be a tool where the user can input options and use the models to see if there will be 3 or more cars involved which will be classed as high risk 







