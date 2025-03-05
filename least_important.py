import numpy as np
import pandas as pd
import pickle


# load test dataset
clean_traffic_data = pd.read_csv('cleaned_traffic_accidents.csv')
x_test = clean_traffic_data.drop(columns=['cars_involved'])

# load random forest model
with open('saved_models/rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# get feature importance from rf
rf_importance = rf_model.feature_importances_
rf_features = x_test.columns

# find 10 least important features
rf_index = np.argsort(rf_importance)
rf_least_important = rf_features[rf_index[:10]]

# save least important features 
rf_df = pd.DataFrame(rf_least_important, columns=['rf features'])
rf_df.to_csv('least_important/least_important_rf.csv', index=False)
print("rf least important features saved")

# load xgboost model
with open('saved_models/xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

# get feature importance from rf
xgb_importance = xgb_model.feature_importances_
xgb_features = x_test.columns

# find 10 least important features
xgb_index = np.argsort(xgb_importance)
xgb_least_important = xgb_features[xgb_index[:10]]

# save least important features 
xgb_df = pd.DataFrame(xgb_least_important, columns=['xgb features'])
xgb_df.to_csv('least_important/least_important_xgb.csv', index=False)
print("xgb least important features saved")