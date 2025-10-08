import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_posterior_comparison(bayes_posterior_val, empirical_dict, save_path=None):
    labels = ['Bayes P(Rain|Cloudy)', 'Empirical P(Rain|Cloudy)']
    bayes_val = bayes_posterior_val
    empirical_val = empirical_dict.get(1, {}).get('empirical_prob', np.nan)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, [bayes_val, empirical_val], edgecolor='k')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability')
    ax.set_title('Bayesian Posterior vs Empirical (Cloudy)')
    for i, v in enumerate([bayes_val, empirical_val]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def calibration_plot(df, save_path=None):
    grouped = df.groupby('forecast_bin').agg(
        mean_forecast=('forecast_prob', 'mean'),
        observed_freq=('actual_rain', 'mean')
    ).reset_index().dropna()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(grouped['mean_forecast'], grouped['observed_freq'], marker='o', linestyle='-')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlabel('Mean forecast probability')
    ax.set_ylabel('Observed frequency of rain')
    ax.set_title('Calibration Plot')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
