import numpy as np
import pandas as pd

def compute_prior(df, event_col='actual_rain'):
    total = len(df)
    if total == 0:
        return 0.0
    return df[event_col].sum() / total

def compute_conditional(df, event_col='actual_rain', evidence_col='is_cloudy'):
    event_df = df[df[event_col] == 1]
    not_event_df = df[df[event_col] == 0]
    p_e_given_event = event_df[evidence_col].mean() if len(event_df) > 0 else 0.0
    p_e_given_not_event = not_event_df[evidence_col].mean() if len(not_event_df) > 0 else 0.0
    return {'p_e_given_event': float(p_e_given_event), 'p_e_given_not_event': float(p_e_given_not_event)}

def bayes_posterior(prior, p_e_given_event, p_e_given_not_event, evidence_present=True):
    p_event = prior
    p_e = p_e_given_event * p_event + p_e_given_not_event * (1 - p_event)
    if p_e == 0:
        return p_event
    if evidence_present:
        return (p_e_given_event * p_event) / p_e
    else:
        p_not_e_given_event = 1 - p_e_given_event
        p_not_e_given_not_event = 1 - p_e_given_not_event
        p_not_e = p_not_e_given_event * p_event + p_not_e_given_not_event * (1 - p_event)
        if p_not_e == 0:
            return p_event
        return (p_not_e_given_event * p_event) / p_not_e

def empirical_posterior_from_data(df, evidence_col='is_cloudy', event_col='actual_rain'):
    grouped = df.groupby(evidence_col)[event_col].agg(['mean', 'count']).rename(columns={'mean': 'empirical_prob'})
    out = {}
    for idx, row in grouped.iterrows():
        out[int(idx)] = {'empirical_prob': float(row['empirical_prob']), 'count': int(row['count'])}
    return out
