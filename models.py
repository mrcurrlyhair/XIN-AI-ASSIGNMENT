import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, binarize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# load cleaned dataset
clean_traffic_data = pd.read_csv('cleaned_traffic_accidents.csv')

# define features and target
x = clean_traffic_data.drop(columns=['cars_involved'])  
y = (clean_traffic_data['cars_involved'] >= 3).astype(int)

# split into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=28, stratify=y)

# scale integer features
scaler = StandardScaler()
int_features = ['crash_hour', 'crash_day_of_week']
x_train[int_features] = scaler.fit_transform(x_train[int_features])
x_test[int_features] = scaler.transform(x_test[int_features])

# balance training data using smote
smote = SMOTE(random_state=28)
x_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)

# define parameter grid for random forest
rf_para = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'min_samples_split': [10, 20, 30],
    'min_samples_leaf': [5, 10, 15],
    'class_weight': [{0:1, 1:1.5}, {0:1, 1:2}],
    'max_features': ['sqrt', 'log2']
}

# run gridsearch for random forest
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=28), 
    rf_para, 
    cv=2, 
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
rf_grid.fit(x_train_balanced, y_train_balanced)

# get best parameters for random forest
best_rf_para = rf_grid.best_params_
print("Best Random Forest Parameters:", best_rf_para)

# train random forest with best parameters
rf_model = RandomForestClassifier(**best_rf_para, random_state=28)
rf_model.fit(x_train_balanced, y_train_balanced)

# make predictions (probability of highrisk accident 1)
rf_threshold = 0.6
rf_prob = rf_model.predict_proba(x_test)[:, 1]
rf_prediction = (rf_prob >= rf_threshold).astype(int)

# test random forest model
print('Random Forest Model')
print(classification_report(y_test, rf_prediction), accuracy_score(y_test, rf_prediction))

# save random forest model
with open('saved_models/rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print('saved rf model')

# define parameter grid for xgboost
xgb_para = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.02, 0.05],
    'max_depth': [3, 4, 5],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

# run gridsearch for xgboost
xgb_grid = GridSearchCV(
    XGBClassifier(random_state=28),
    xgb_para,
    cv=2,
    scoring='f1',
    n_jobs=-1, 
    verbose=1
)
xgb_grid.fit(x_train_balanced, y_train_balanced)

# get best parameters for xgboost
best_xgb_para = xgb_grid.best_params_
print("Best XGBoost Parameters:", best_xgb_para)

# train xgboost with best parameters
xgb_model = XGBClassifier(**best_xgb_para, random_state=28)
xgb_model.fit(x_train_balanced, y_train_balanced)

# make predictions (probability of highrisk accident 1)
xgb_threshold = 0.6
xgb_prob = xgb_model.predict_proba(x_test)[:, 1]
xgb_prediction = (xgb_prob >= xgb_threshold).astype(int)

# test xgboost model
print('XGBoost Model')
print(classification_report(y_test, xgb_prediction), accuracy_score(y_test, xgb_prediction))

# save xgboost model
with open('saved_models/xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
print('saved xgb model')

# define parameter grid for gradient boosting
gb_para = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.02, 0.05],
    'max_depth': [3, 4, 5],
    'subsample': [0.7, 0.8, 0.9],
    'min_samples_split': [5, 10, 15],
    'max_features': ['sqrt', 'log2']
}

# run gridsearch for gradient boosting
gb_grid = GridSearchCV(
    GradientBoostingClassifier(random_state=28), 
    gb_para, 
    cv=2, 
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
gb_grid.fit(x_train_balanced, y_train_balanced)

# get best parameters for gradient boosting
best_gb_para = gb_grid.best_params_
print("Best Gradient Boosting Parameters", best_gb_para)

# train gradient boosting with best parameters
gb_model = GradientBoostingClassifier(**best_gb_para, random_state=28)
gb_model.fit(x_train_balanced, y_train_balanced)

# make predictions (probability of highrisk accident 1)
gb_threshold = 0.6
gb_prob = gb_model.predict_proba(x_test)[:, 1]
gb_prediction = (gb_prob >= gb_threshold).astype(int)

# test gradient boosting model
print('Gradient Boosting Model')
print(classification_report(y_test, gb_prediction), accuracy_score(y_test, gb_prediction))

# save gradient boosting model
with open('saved_models/gb_model.pkl', 'wb') as f:
    pickle.dump(gb_model, f)    
print('saved gb model')
