import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# load cleaned data
file_path = 'cleaned_traffic_accidents.csv'
traffic_data = pd.read_csv(file_path)

# define features and target
x = traffic_data.drop(columns=['cars_involved'])  
y = (traffic_data['cars_involved'] >= 3).astype(int)  

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


