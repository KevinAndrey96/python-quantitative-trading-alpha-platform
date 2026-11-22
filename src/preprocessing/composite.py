from typing import Dict
import pandas as pd


def compute_composite_alpha(
    factor_df: pd.DataFrame,
    factor_columns: Dict[str, float]
) -> pd.Series:
    """
    Computes a composite alpha score as a linear combination of normalized factors:
    Composite = sum(weight_i * factor_i) / sum(weight_i)
    """
    total_weight = sum(abs(w) for w in factor_columns.values())
    if total_weight == 0:
        raise ValueError("Total factor weight cannot be zero")

    composite = pd.Series(0.0, index=factor_df.index)
    for col, weight in factor_columns.items():
        if col not in factor_df.columns:
            raise KeyError(f"Factor column '{col}' not found in DataFrame")
        composite += factor_df[col].fillna(0.0) * weight

    return composite / total_weight
