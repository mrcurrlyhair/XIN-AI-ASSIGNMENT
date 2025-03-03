import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from models import y_test, rf_prediction, xgb_prediction, gb_prediction

#  function to plot matrix
def plot_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not High-Risk', 'High-Risk'], yticklabels=['Not High-Risk', 'High-Risk'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

# Random Forest confusion matrix
plot_matrix(y_test, rf_prediction, 'Random Forest')

# XGBoost confusion matrix
plot_matrix(y_test, xgb_prediction, 'XGBoost')

# Gradient Boosting confusion matrix
plot_matrix(y_test, gb_prediction, 'Gradient Boosting')
