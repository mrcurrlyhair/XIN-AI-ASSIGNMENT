import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, accuracy_score, classification_report
from models import x_train, y_test, x_test, rf_model, gb_model, nb_model

# model performance benchmarks 
def model_bench(model, x_test, y_test):
    y_pred = model.predict(x_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return report['accuracy'], report['1']['recall'], report['1']['precision'], report['1']['f1-score']

# benchmarks for each model
rf_accuracy, rf_recall, rf_precision, rf_f1 = model_bench(rf_model, x_test, y_test)
gb_accuracy, gb_recall, gb_precision, gb_f1 = model_bench(gb_model, x_test, y_test)
nb_accuracy, nb_recall, nb_precision, nb_f1 = model_bench(nb_model, x_test, y_test)

m_name = ['Random Forest', 'Gradient Boosting', 'Naive Bayes']
accuracy = [rf_accuracy, gb_accuracy, nb_accuracy]
recall = [rf_recall, gb_recall, nb_recall]
precision = [rf_precision, gb_precision, nb_precision]
f1_score = [rf_f1, gb_f1, nb_f1]

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

# function for matrixes
def matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not High-Risk', 'High-Risk'], yticklabels=['Not High-Risk', 'High-Risk'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

# create graph for each model
matrix(y_test, rf_model.predict(x_test), 'Random Forest')
matrix(y_test, gb_model.predict(x_test), 'Gradient Boosting')
matrix(y_test, nb_model.predict(x_test), 'Naive Bayes')

# function for precision-recall curves
def pr_curve(model, x_test, y_test, model_name):
    y_probs = model.predict_proba(x_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    plt.plot(recall, precision, label=model_name)

# create precision-recall curves for all models
plt.figure(figsize=(8,6))
pr_curve(rf_model, x_test, y_test, 'Random Forest')
pr_curve(gb_model, x_test, y_test, 'Gradient Boosting')
pr_curve(nb_model, x_test, y_test, 'Naive Bayes')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Comparison')
plt.legend()
plt.grid()
plt.show()

# function to plot feature importances for decision tree modles 
def f_importance(model, model_name, feature_names):
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[-10:]  # Top 10 features

    plt.figure(figsize=(8, 6))
    plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title(f'Top 10 Features - {model_name}')
    plt.show()

# show feature importances for random forest
f_importance(rf_model, 'Random Forest', x_train.columns)

# show feature importances for gradient boosting
f_importance(gb_model, 'Gradient Boosting', x_train.columns)

# function for naive bayes feature importance
def nb_importance(model, feature_names):
    log_prob = np.abs(model.theta_)
    importance = log_prob.mean(axis=0)

    sorted_idx = np.argsort(importance)[-10:]
    plt.figure(figsize=(8, 6))
    plt.barh(np.array(feature_names)[sorted_idx], importance[sorted_idx])
    plt.xlabel('Importance (Log Probabilities)')
    plt.title('Feature Importance - Naive Bayes')
    plt.show()

# show feature importance for naive bayes
nb_importance(nb_model, x_train.columns)
