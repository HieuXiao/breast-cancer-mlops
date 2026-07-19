import numpy as np
from scipy.stats import ks_2samp
from psi import calculate_psi

def run_test(
    name,
    train,
    production
):
    print("="*50)
    print(name)

    # PSI
    bins = np.linspace(
        min(train.min(), production.min()),
        max(train.max(), production.max()),
        5
    )

    train_hist, _ = np.histogram(
        train,
        bins=bins
    )
    prod_hist, _ = np.histogram(
        production,
        bins=bins
    )

    train_pct = (
        train_hist /
        train_hist.sum()
    )
    prod_pct = (
        prod_hist /
        prod_hist.sum()
    )

    psi = calculate_psi(
        train_pct,
        prod_pct
    )

    # KS test
    ks_stat, p_value = ks_2samp(
        train,
        production
    )

    print(
        f"PSI: {psi:.4f}"
    )
    print(
        f"KS statistic: {ks_stat:.4f}"
    )
    print(
        f"KS p-value: {p_value:.4f}"
    )

    if psi >=0.2:
        print(
            "ALERT: Drift detected"
        )
    else:
        print(
            "No significant drift"
        )

if __name__ == "__main__":
    np.random.seed(42)

    training = np.random.normal(
        0,
        1,
        1000
    )
    no_drift = np.random.normal(
        0,
        1,
        1000
    )
    drift = np.random.normal(
        2,
        1,
        1000
    )

    run_test(
        "Scenario 1: No Drift",
        training,
        no_drift
    )
    run_test(
        "Scenario 2: Drift",
        training,
        drift
    )