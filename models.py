import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, binarize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE



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
rf_threshold = 0.6
rf_prob = rf_model.predict_proba(x_test)[:, 1]
rf_prediction = (rf_prob >= rf_threshold).astype(int)

# test random forest model
print('Random Forest Model')
print(classification_report(y_test, rf_prediction), accuracy_score(y_test, rf_prediction))


# train naive bayes model
nb_model = GaussianNB()
nb_model.fit(x_train_balanced, y_train_balanced)

# make predictions
nb_threshold = 0.5
nb_prob = nb_model.predict_proba(x_test)[:, 1]
nb_prediction = (nb_prob >= nb_threshold).astype(int)

# test xgboost model
print('Naive Bayes Model')
print(classification_report(y_test, nb_prediction), accuracy_score(y_test, nb_prediction))
      

# train optimized gradient boosting model
gb_model = GradientBoostingClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6, 
    subsample=0.8, 
    min_samples_split=10, 
    warm_start=True, 
    random_state=28
)

gb_model.fit(x_train_balanced, y_train_balanced)

# make predictions
gb_prob = gb_model.predict_proba(x_test)[:, 1]
gb_prediction = (gb_prob >= rf_threshold).astype(int)

# test optimized gradient boosting model
print('Gradient Boosting Model')
print(classification_report(y_test, gb_prediction), accuracy_score(y_test, gb_prediction))



