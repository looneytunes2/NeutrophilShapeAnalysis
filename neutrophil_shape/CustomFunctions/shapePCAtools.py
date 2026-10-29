# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 10:36:41 2022

@author: Aaron
"""

import pandas as pd
import numpy as np
from pathlib import Path
from . import shtools_mod


def filter_extremes_based_on_percentile(
    df: pd.DataFrame,
    features: list,
    pct: float
):

    """
    Exclude extreme data points that fall in the percentile range
    [0,pct] or [100-pct,100] of at least one of the features
    provided.

    Parameters
    --------------------
    df: pandas df
        Input dataframe that contains the features.
    features: List
        List of column names to be used to filter the data
        points.
    pct: float
        Specifies the percentile range; data points that
        fall in the percentile range [0,pct] or [100-pct,100]
        of at least one of the features are removed.

    Returns
    -------
    df: pandas dataframe
        Filtered dataframe.
    """
    
    # Temporary column to store whether a data point is an
    # extreme point or not.
    df["extreme"] = False

    for f in features:
        # Calculated the extreme interval fot the current feature
        finf, fsup = np.percentile(df[f].dropna().values, [pct, 100 - pct])

        # Points in either high or low extreme as flagged
        df.loc[(df[f] < finf), "extreme"] = True
        df.loc[(df[f] > fsup), "extreme"] = True

    # Drop extreme points and temporary column
    df = df.loc[df.extreme == False]
    df = df.drop(columns=["extreme"])

    return df


def digitize_shape_mode(
    df: pd.DataFrame,
    feature: list,
    nbins: int,
    filter_based_on: list,
    filter_extremes_pct: float = 1,
    save: str = None,
    return_freqs_per_structs: bool = False,
    stdlim: float = 2.0,
):

    """
    Discretize a given feature into nbins number of equally
    spaced bins. The feature is first z-scored and the interval
    from -2std to 2std is divided into nbins bins.

    Parameters
    --------------------
    df: pandas df
        Input dataframe that contains the feature to be
        discretized.
    features: str
        Column name of the feature to be discretized.
    nbins: int
        Number of bins to divide the feature into.
    filter_extremes_pct: float
        See parameter pct in function filter_extremes_based_on_percentile
    filter_based_on: list
        List of all column names that should be used for
        filtering extreme data points.
    save: Path
        Path to a file where we save the number of data points
        that fall in each bin
    return_freqs_per_structs: bool
        Wheter or not to return a dataframe with the number of
        data points in each bin stratifyied by structure_name.
    Returns
    -------
        df: pandas dataframe
            Input dataframe with data points filtered according
            to filter_extremes_pct plus a column named "bin"
            that denotes the bin in which a given data point
            fall in.
        bin_indexes: list of tuples
            [(a,b)] where a is the bin number and b is a list
            with the index of all data points that fall into
            that bin.
        bin_centers: list
            List with values of feature at the center of each
            bin
        pc_std: float
            Standard deviation used to z-score the feature.
        df_freq: pd.DataFrame
            dataframe with the number of data points in each
            bin stratifyied by structure_name (returned only
            when return_freqs_per_structs is set to True).

    """
    
    # Check if feature is available
    if feature not in df.columns:
        raise ValueError(f"Column {feature} not found.")

    # Exclude extremeties
    df = filter_extremes_based_on_percentile(
        df = df.copy(deep=True),
        features = filter_based_on,
        pct = filter_extremes_pct
    )
    
    # Get feature values
    values = df[feature].values.astype(np.float32)

    # Should be centered already, but enforce it here
    values -= values.mean()
    # Z-score
    
    pc_std = values.std()
    values /= pc_std

    # Calculate bin half width based on std interval and nbins
    LINF = -stdlim # inferior limit = -2 std
    LSUP = stdlim # superior limit = 2 std
    binw = (LSUP-LINF)/(2*(nbins-1))
    
    # Force samples below/above -/+ 2std to fall into first/last bin
    bin_centers = np.linspace(LINF, LSUP, nbins)
    bin_edges = np.unique([(round(b-binw,4), round(b+binw,4)) for b in bin_centers])
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    
    # Aplly digitization
    df["bin"] = np.digitize(values, bin_edges)

    # Report number of data points in each bin
    df_freq = pd.DataFrame(df["bin"].value_counts(sort=False))
    df_freq.index = df_freq.index.rename(f"{feature}_bin")
    df_freq = df_freq.rename(columns={"bin": "samples"})
    if save is not None:
        with open(f"{save}.txt", "w") as flog:
            print(df_freq, file=flog)

    # Store the index of all data points in each bin
    bin_indexes = []
    df_agg = df.groupby(["bin"]).mean()
    for b, df_bin in df.groupby(["bin"]):
        bin_indexes.append((b, df_bin.index))

    # Optionally return a dataframe with the number of data
    # points in each bin stratifyied by structure_name.
    if return_freqs_per_structs:

        df_freq = (
            df[["structure_name", "bin"]].groupby(["structure_name", "bin"]).size()
        )
        df_freq = pd.DataFrame(df_freq)
        df_freq = df_freq.rename(columns={0: "samples"})
        df_freq = df_freq.unstack(level=1)
        return df, bin_indexes, (bin_centers, pc_std), df_freq

    return df, bin_indexes, (bin_centers, pc_std)



def get_mesh_from_series(row, alias, lmax):
    coeffs = np.zeros((2, lmax, lmax), dtype=np.float32)
    for l in range(lmax):
        for m in range(l + 1):
            try:
                # Cosine SHE coefficients
                coeffs[0, l, m] = row[
                    [f for f in row.keys() if f"{alias}_shcoeffs_L{l}M{m}C" in f]
                ]
                # Sine SHE coefficients
                coeffs[1, l, m] = row[
                    [f for f in row.keys() if f"{alias}_shcoeffs_L{l}M{m}S" in f]
                ]
            # If a given (l,m) pair is not found, it is assumed to be zero
            except: pass
    mesh, _ = shtools_mod.get_reconstruction_from_coeffs(coeffs)
    return mesh



def get_shcoeff_columns(l_order: int):
    """
    Names of the SH coefficient columns up to order l_order, in the order that
    reshapes to the (2, l_order+1, l_order+1) array of cosine and sine
    coefficients the reconstruction functions take (the order that
    shparam_mod.get_shcoeffs_mesh returns them in).
    """
    return [f'shcoeffs_L{l}M{m}{c}' for c in 'CS' for l in range(l_order + 1) for m in range(l_order + 1)]


def get_nonzero_shcoeff_columns(l_order: int):
    """
    The subset of get_shcoeff_columns (same order) that can be non-zero: the
    cosine terms with m <= l and the sine terms with 1 <= m <= l. The rest
    (m > l, and the m = 0 sine terms) are always zero, so they are not saved;
    load_shcoeffs puts them back as zeros.
    """
    return [f'shcoeffs_L{l}M{m}{c}' for c in 'CS' for l in range(l_order + 1) for m in range(l_order + 1)
            if m <= l and not (c == 'S' and m == 0)]


def pca_inverse_to_shcoeffs(
        pca, # fitted PCA whose features are the SH coefficients
        pcs, # PC values to invert, shape (npcs,) or (n, npcs)
        l_order: int, # order of the coefficients to return, i.e. the config's common.l_order
        ):
    """
    Inverse PCA transform of pcs back to SH coefficients in the full layout of
    get_shcoeff_columns (reshapes to (2, l_order+1, l_order+1)).
    The PCA is fit on only the coefficients that can be non-zero
    (get_nonzero_shcoeff_columns), so its inverse transform returns just those;
    they are put back into their places and the always-zero ones are zeros.
    A PCA fit on every coefficient is also handled (returned as is).
    """
    columns = get_shcoeff_columns(l_order)
    nonzero = get_nonzero_shcoeff_columns(l_order)
    pcs = np.asarray(pcs)
    single = pcs.ndim == 1  # sklearn needs 2D input
    inverted = pca.inverse_transform(np.atleast_2d(pcs))
    if single:
        inverted = inverted[0]
    if inverted.shape[-1] == len(columns):
        return inverted
    if inverted.shape[-1] != len(nonzero):
        raise ValueError(f'PCA has {inverted.shape[-1]} features, expected {len(nonzero)} '
                         f'(nonzero) or {len(columns)} (all) for l_order {l_order}')
    position = {c: i for i, c in enumerate(columns)}
    full = np.zeros(inverted.shape[:-1] + (len(columns),), dtype=inverted.dtype)
    full[..., [position[c] for c in nonzero]] = inverted
    return full


def load_shcoeffs(
        datadir, # directory with the SH coefficient files saved by image_processing.extract_shape_metrics
        l_order: int, # order of the coefficients to return, i.e. the config's common.l_order
        pattern: str = 'SH_Coefficients_*.csv', # which of the files in datadir to read
        ):
    """
    SH coefficients of every cell in the files of datadir matching pattern, as a
    dataframe with the 'cell' column that identifies each row followed by the
    coefficients up to order l_order (see get_shcoeff_columns). The files hold
    more orders than any analysis uses (shparam_mod.SHAPE_INFO_L_ORDER), so this
    is how the coefficients are recalled at the config's order. Combine them with
    other per-cell data by merging on 'cell'.
    The files only hold the coefficients that can be non-zero (see
    get_nonzero_shcoeff_columns); the always-zero ones are filled in as zeros here,
    so the result has the full layout whether the file was saved compactly or not.
    """
    columns = get_shcoeff_columns(l_order)
    nonzero = get_nonzero_shcoeff_columns(l_order)
    position = {c: i for i, c in enumerate(columns)}
    files = sorted(Path(datadir).glob(pattern))
    if not files:
        raise FileNotFoundError(f'No files matching {pattern} in {datadir}')
    frames = []
    for f in files:
        #older files have every column, so only read the ones that can be non-zero
        saved = pd.read_csv(f, usecols=['cell'] + nonzero)
        full = np.zeros((len(saved), len(columns)), dtype=np.float64)
        full[:, [position[c] for c in nonzero]] = saved[nonzero].values
        frames.append(pd.concat([saved[['cell']], pd.DataFrame(full, columns=columns)], axis=1))
    return pd.concat(frames, ignore_index=True)
