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

# finding all features with no importantce 
rf_least_important = rf_features[rf_importance == 0]

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

# finding all features with no importantce
xgb_least_important = xgb_features[xgb_importance == 0]

# save least important features 
xgb_df = pd.DataFrame(xgb_least_important, columns=['xgb features'])
xgb_df.to_csv('least_important/least_important_xgb.csv', index=False)
print("xgb least important features saved")

# load gradient boost model
with open('saved_models/gb_model.pkl', 'rb') as f:
    gb_model = pickle.load(f)

# get feature importance from gb
gb_importance = gb_model.feature_importances_
gb_features = x_test.columns

# finding all features with no importantce
gb_least_important = gb_features[gb_importance == 0]

# save least important features 
gb_df = pd.DataFrame(gb_least_important, columns=['gb features'])
gb_df.to_csv('least_important/least_important_gb.csv', index=False)
print("gb least important features saved")

