"""
Cross-sectional median label computation for statistical arbitrage.
Reproduces the binary response variable defined in Section 4.2 of
Krauss, Do & Huck (2016).

Y^s_{t+1} = 1 if R^s_{t+1} > median({R^s_{t+1} | s in valid universe})
           = 0 otherwise

This module is imported by both feature_engineering.py and sequences.py
to ensure a single, consistent label definition across all models.
"""
import numpy as np
import pandas as pd

def build_label_for_date(t_idx: int,
                         returns: pd.DataFrame,
                         valid_permnos: pd.Index) -> pd.Series | None:
    """
    Binary label Y^s_{t+1} = 1 if next-day return of stock s exceeds
    the cross-sectional median return across all valid stocks on day t+1.

    Parameters
    ----------
    t_idx         : integer position of date t in returns.index
    returns       : full cleaned returns matrix (date × permno)
    valid_permnos : permnos to include in the cross-sectional median computation

    Returns
    -------
    pd.Series with index=permno, values in {0, 1}, name="label"
    or None if t+1 is out of bounds or no valid returns exist.

    Notes
    -----
    - No lookahead: uses returns at t+1 only, never beyond.
    - Stocks with NaN return at t+1 are dropped before computing the median.
    - By construction the label should be balanced at ~0.5 since it is
      defined relative to the cross-sectional median.
    """
    if t_idx + 1 >= len(returns):
        return None

    r_next = returns.iloc[t_idx + 1][valid_permnos].dropna()
    if len(r_next) == 0:
        return None

    median = r_next.median()
    label  = (r_next > median).astype(np.int8)
    label.name = "label"
    return label