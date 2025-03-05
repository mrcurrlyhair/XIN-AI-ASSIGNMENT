import numpy as np
import pandas as pd
import pickle

# load random forest model
with open('saved_models/rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# load test dataset
clean_traffic_data = pd.read_csv('cleaned_traffic_accidents.csv')
x_test = clean_traffic_data.drop(columns=['cars_involved'])

# get feature importance from rf
rf_importance = rf_model.feature_importances_
features = x_test.columns

# find 10 least important features
rf_index = np.argsort(rf_importance)
rf_least_important = features[rf_index[:10]]

# save least important features 
rf_df = pd.DataFrame(rf_least_important, columns=['features'])
rf_df.to_csv('least_important/least_important_rf.csv', index=False)
print("rf least important features saved")