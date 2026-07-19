import numpy as np

def calculate_psi(
    expected,
    actual
):
    """
    Calculate Population Stability Index

    expected:
        training distribution
    actual:
        production distribution
    """

    expected = np.array(expected)
    actual = np.array(actual)

    psi = np.sum(
        (actual - expected)
        *
        np.log(actual / expected)
    )
    return psi

if __name__ == "__main__":
    expected = [
        0.2,
        0.3,
        0.3,
        0.2
    ]
    actual = [
        0.1,
        0.2,
        0.4,
        0.3
    ]

    result = calculate_psi(
        expected,
        actual
    )

    print(
        f"PSI = {result:.4f}"
    )