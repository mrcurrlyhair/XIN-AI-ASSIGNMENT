import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import pandas as pd 
from sklearn.metrics import confusion_matrix, precision_recall_curve, accuracy_score, classification_report

# load cleaned dataset
clean_traffic_data = pd.read_csv('cleaned_traffic_accidents.csv')

# load the models
with open('saved_models/rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('saved_models/gb_model.pkl', 'rb') as f:
    gb_model = pickle.load(f)

with open('saved_models/xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

# drop cars_involved 
x_full = clean_traffic_data.drop(columns=['cars_involved'])

# model performance benchmarks 
def model_bench(model, x_full, y_true):
    y_pred = model.predict(x_full)
    report = classification_report(y_true, y_pred, output_dict=True)
    return report['accuracy'], report['1']['recall'], report['1']['precision'], report['1']['f1-score']

# benchmarks for each model
y_true = (clean_traffic_data['cars_involved'] >= 3).astype(int)
rf_accuracy, rf_recall, rf_precision, rf_f1 = model_bench(rf_model, x_full, y_true)
gb_accuracy, gb_recall, gb_precision, gb_f1 = model_bench(gb_model, x_full, y_true)
xgb_accuracy, xgb_recall, xgb_precision, xgb_f1 = model_bench(xgb_model, x_full, y_true)

m_name = ['Random Forest', 'Gradient Boosting', 'XGBoost']
accuracy = [rf_accuracy, gb_accuracy, xgb_accuracy]
recall = [rf_recall, gb_recall, xgb_recall]
precision = [rf_precision, gb_precision, xgb_precision]
f1_score = [rf_f1, gb_f1, xgb_f1]

# bar chart for model performance
x = np.arange(len(m_name))
width = 0.2

plt.figure(figsize=(10, 6))
plt.bar(x - width, accuracy, width, label='Accuracy')
plt.bar(x, recall, width, label='Recall')
plt.bar(x + width, precision, width, label='Precision')
plt.bar(x + 2 * width, f1_score, width, label='F1-Score')

plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(x, m_name)
plt.legend()
plt.show()
plt.close()

# function for confusion matrixes
def matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not High-Risk', 'High-Risk'], yticklabels=['Not High-Risk', 'High-Risk'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()
    plt.close()

# create confusion matrixes for each model
matrix(y_true, rf_model.predict(x_full), 'Random Forest')
matrix(y_true, gb_model.predict(x_full), 'Gradient Boosting')
matrix(y_true, xgb_model.predict(x_full), 'XGBoost')

# function for precision-recall curves
def pr_curve(model, x_full, y_true, model_name):
    y_probs = model.predict_proba(x_full)[:, 1]
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    plt.plot(recall, precision, label=model_name)

# create precision-recall curves for all models
plt.figure(figsize=(8,6))
pr_curve(rf_model, x_full, y_true, 'Random Forest')
pr_curve(gb_model, x_full, y_true, 'Gradient Boosting')
pr_curve(xgb_model, x_full, y_true, 'XGBoost')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Comparison')
plt.legend()
plt.grid()
plt.show()
plt.close()

# function to plot feature importances for decision tree models
def f_importance(model, model_name, feature_names):
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)
    plt.figure(figsize=(8, 6))
    plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title(f'Top Features - {model_name}')
    plt.show()
    plt.close()

# show feature importances for all tree-based models
f_importance(rf_model, 'Random Forest', x_full.columns)
f_importance(gb_model, 'Gradient Boosting', x_full.columns)
f_importance(xgb_model, 'XGBoost', x_full.columns)

# make predictions on the full dataset
clean_traffic_data['rf_pred'] = rf_model.predict(x_full)
clean_traffic_data['gb_pred'] = gb_model.predict(x_full)
clean_traffic_data['xgb_pred'] = xgb_model.predict(x_full)

# save dataset with predictions
clean_traffic_data.to_csv('cleaned_traffic_accidents_predictions.csv', index=False)
print("predictions added")

# model comparison of predictions
actual_high_risk = y_true.sum()
rf_high_risk = clean_traffic_data['rf_pred'].sum()
gb_high_risk = clean_traffic_data['gb_pred'].sum()
xgb_high_risk = clean_traffic_data['xgb_pred'].sum()

# create a bar chart comparing actual accidents v model predictions
plt.figure(figsize=(8, 5))
plt.bar(['Actual', 'Random Forest', 'Gradient Boosting', 'XGBoost'], 
        [actual_high_risk, rf_high_risk, gb_high_risk, xgb_high_risk], color=['black', 'r', 'g', 'b'])
plt.xlabel('Model')
plt.ylabel('Number of High-Risk Accidents')
plt.title('Actual vs Predicted High-Risk Accident Comparison')
plt.show()
plt.close()
