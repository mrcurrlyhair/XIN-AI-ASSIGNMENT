import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import pandas as pd 
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

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
expected_features_rf = rf_model.feature_names_in_
expected_features_gb = gb_model.feature_names_in_
expected_features_xgb = xgb_model.feature_names_in_
x_full = clean_traffic_data.drop(columns=['cars_involved'])
x_full_rf = x_full[expected_features_rf]
x_full_gb = x_full[expected_features_gb]
x_full_xgb = x_full[expected_features_xgb]

# make predictions and save them to the dataset
clean_traffic_data['rf_pred'] = rf_model.predict(x_full_rf)
clean_traffic_data['gb_pred'] = gb_model.predict(x_full_gb)
clean_traffic_data['xgb_pred'] = xgb_model.predict(x_full_xgb)

# save dataset with predictions
clean_traffic_data.to_csv('cleaned_traffic_accidents_predictions.csv', index=False)
print('Predictions saved') 


# model performance benchmarks 
def model_bench(model, x_full, y_true):
    y_pred = model.predict(x_full)
    report = classification_report(y_true, y_pred, output_dict=True)
    return report['accuracy'], report['1']['recall'], report['1']['precision'], report['1']['f1-score']

# benchmarks for each model
y_true = (clean_traffic_data['cars_involved'] >= 3).astype(int)
rf_accuracy, rf_recall, rf_precision, rf_f1 = model_bench(rf_model, x_full_rf, y_true)
gb_accuracy, gb_recall, gb_precision, gb_f1 = model_bench(gb_model, x_full_gb, y_true)
xgb_accuracy, xgb_recall, xgb_precision, xgb_f1 = model_bench(xgb_model, x_full_xgb, y_true)

m_name = ['Random Forest', 'Gradient Boosting', 'XGBoost']
accuracy = [rf_accuracy, gb_accuracy, xgb_accuracy]
recall = [rf_recall, gb_recall, xgb_recall]
precision = [rf_precision, gb_precision, xgb_precision]
f1_score = [rf_f1, gb_f1, xgb_f1]

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

# function for ROC curve
def roc(model, x_full, y_true, model_name):
    y_probs = model.predict_proba(x_full)[:, 1]
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.2f})')

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

# menu for graph selection
while True:
    print('\nSelect a graph to display')
    print('1 Models Performance')
    print('2 Confusion Matrixes')
    print('3 ROC Curves')
    print('4 Feature Importance')
    print('5 Models Prediction')
    print('6 Quit')
    
    option = input('Enter your choice')
    
    if option == '1':
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
    
    elif option == '2':
        # confusion matrixes
        matrix(y_true, rf_model.predict(x_full_rf), 'Random Forest')
        matrix(y_true, gb_model.predict(x_full_gb), 'Gradient Boosting')
        matrix(y_true, xgb_model.predict(x_full_xgb), 'XGBoost')
    
    elif option == '3':
        # ROC curves
        plt.figure(figsize=(8,6))
        roc(rf_model, x_full_rf, y_true, 'Random Forest')
        roc(gb_model, x_full_gb, y_true, 'Gradient Boosting')
        roc(xgb_model, x_full_xgb, y_true, 'XGBoost')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend()
        plt.grid()
        plt.show()
        plt.close()
    
    elif option == '4':
        # feature importance
        f_importance(rf_model, 'Random Forest', x_full_rf.columns)
        f_importance(gb_model, 'Gradient Boosting', x_full_gb.columns)
        f_importance(xgb_model, 'XGBoost', x_full_xgb.columns)
    
    elif option == '5':
        # model prediction comparison
        actual_high_risk = y_true.sum()
        rf_high_risk = clean_traffic_data['rf_pred'].sum()
        gb_high_risk = clean_traffic_data['gb_pred'].sum()
        xgb_high_risk = clean_traffic_data['xgb_pred'].sum()
        plt.figure(figsize=(8, 5))
        plt.bar(['Actual', 'Random Forest', 'Gradient Boosting', 'XGBoost'], 
                [actual_high_risk, rf_high_risk, gb_high_risk, xgb_high_risk], color=['black', 'r', 'g', 'b'])
        plt.xlabel('Model')
        plt.ylabel('Number of High-Risk Accidents')
        plt.title('Actual vs Predicted High-Risk Accident Comparison')
        plt.show()
        plt.close()
    
    elif option == '6':
        print('Quiting')
        break
    else:
        print('Invalid option, quiting')
