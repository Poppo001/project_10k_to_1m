# python src/auto_feature_selection.py

import shap
import matplotlib.pyplot as plt

def plot_shap_summary(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X)
    plt.show()
