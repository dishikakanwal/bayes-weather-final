from bayes_model.data_loader import load_or_generate
from bayes_model.preprocessing import basic_cleaning, add_binary_features
from bayes_model.bayes import compute_prior, compute_conditional, bayes_posterior, empirical_posterior_from_data
from bayes_model.visualize import plot_posterior_comparison, calibration_plot
import os

def main():
    df = load_or_generate()
    df = basic_cleaning(df)
    df = add_binary_features(df, cloud_threshold=0.5)

    prior = compute_prior(df, event_col='actual_rain')
    cond = compute_conditional(df, event_col='actual_rain', evidence_col='is_cloudy')
    posterior_cloudy = bayes_posterior(prior, cond['p_e_given_event'], cond['p_e_given_not_event'], evidence_present=True)
    empirical = empirical_posterior_from_data(df, evidence_col='is_cloudy', event_col='actual_rain')

    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    plot_posterior_comparison(posterior_cloudy, empirical, save_path=os.path.join(out_dir, "posterior_compare.png"))
    calibration_plot(df, save_path=os.path.join(out_dir, "calibration.png"))
    df.to_csv(os.path.join(out_dir, "preprocessed_sample.csv"), index=False)
    print("[run] Pipeline complete (50% project code).")

if __name__ == "__main__":
    main()
