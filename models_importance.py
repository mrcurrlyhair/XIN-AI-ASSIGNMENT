import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import plot_tree
from models import x_test, x_train, rf_model, gb_model 

# Function to plot feature importances
def plot_feature_importance(model, model_name, feature_names):
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[-10:]  # Top 10 features

    plt.figure(figsize=(8, 6))
    plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
    plt.xlabel("Feature Importance")
    plt.title(f"Top 10 Features - {model_name}")
    plt.show()

# Show feature importances for Random Forest
plot_feature_importance(rf_model, "Random Forest", x_train.columns)

# Show feature importances for Gradient Boosting
plot_feature_importance(gb_model, "Gradient Boosting", x_train.columns)

