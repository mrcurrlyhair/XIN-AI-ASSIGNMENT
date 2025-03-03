import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# load cleaned data
clean_traffic_data = pd.read_csv('cleaned_traffic_accidents.csv')

# define features and target
x = clean_traffic_data.drop(columns=['cars_involved'])  
y = (clean_traffic_data['cars_involved'] >= 3).astype(int)  

# split into training and testing 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=28, stratify=y)

# scale int features
scaler = StandardScaler()
int_features = ['crash_hour', 'crash_day_of_week']
x_train[int_features] = scaler.fit_transform(x_train[int_features])
x_test[int_features] = scaler.transform(x_test[int_features])

# fix imbalanced taining data
smote = SMOTE(random_state=28)
x_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)

# train random forest model 
rf_model = RandomForestClassifier(
    n_estimators=100,   
    random_state=28, 
    max_depth=6,       
    min_samples_split=20,  
    min_samples_leaf=15,  
    class_weight={0: 1, 1: 1.5}  
)
rf_model.fit(x_train_balanced, y_train_balanced)

# make predictions
threshold = 0.6
rf_prob = rf_model.predict_proba(x_test)[:, 1]
rf_prediction = (rf_prob >= threshold).astype(int)

# test random forest model
print('Random Forest Model')
print(classification_report(y_test, rf_prediction), accuracy_score(y_test, rf_prediction))

# train xgboost model
xgb_model = XGBClassifier(
    scale_pos_weight=100,
    n_estimators=200,
    max_depth=10,
    learning_rate=0.05,
    random_state=28
)
xgb_model.fit(x_train_balanced, y_train_balanced)

# make predictions
xgb_prob = xgb_model.predict_proba(x_test)[:, 1]
xgb_prediction = (xgb_prob >= threshold).astype(int)

# test xgboost model
print('XGBoost Model')
print(classification_report(y_test, xgb_prediction), accuracy_score(y_test, xgb_prediction))




