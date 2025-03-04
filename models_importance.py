import matplotlib.pyplot as plt
import numpy as np
from models import x_train, rf_model, gb_model, nb_model

# function to plot feature importances
def plot_feature_importance(model, model_name, feature_names):
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[-10:]  # Top 10 features

    plt.figure(figsize=(8, 6))
    plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
    plt.xlabel("Feature Importance")
    plt.title(f"Top 10 Features - {model_name}")
    plt.show()

# show feature importances for Random Forest
plot_feature_importance(rf_model, 'Random Forest', x_train.columns)

# show feature importances for Gradient Boosting
plot_feature_importance(gb_model, 'Gradient Boosting', x_train.columns) 

# function to plot naive bayes feature importance
def plot_naive_bayes_importance(model, feature_names):
    log_prob = np.abs(model.theta_)  
    importance = log_prob.mean(axis=0)  

    sorted_idx = np.argsort(importance)[-10:]  
    plt.figure(figsize=(8, 6))
    plt.barh(np.array(feature_names)[sorted_idx], importance[sorted_idx])
    plt.xlabel("Importance (Log Probabilities)")
    plt.title("Feature Importance - Naive Bayes")
    plt.show()

# show feature importance for naive bayes
plot_naive_bayes_importance(nb_model, x_train.columns)
