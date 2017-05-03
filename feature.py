import matplotlib.pyplot as pl
import numpy as np

def feature_plot(importances, X_train, n_features=5):
    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:n_features]]
    values = importances[indices][:n_features]

    # Creat the plot
    fig = pl.figure(figsize=(9, 5))
    pl.title("Normalized Weights for First Five Most Predictive Features", fontsize=16)
    pl.bar(np.arange(n_features), values, width=0.6, align="center", color='#00A000', \
           label="Feature Weight")
    pl.bar(np.arange(n_features) - 0.3, np.cumsum(values), width=0.2, align="center", color='#00A0A0', \
           label="Cumulative Feature Weight")
    pl.xticks(np.arange(n_features), columns)
    pl.ylabel("Weight", fontsize=12)
    pl.xlabel("Feature", fontsize=12)

    pl.legend(loc='upper center')
    pl.tight_layout()
    pl.show()
