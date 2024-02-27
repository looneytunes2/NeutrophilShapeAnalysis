# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 10:36:41 2022

@author: Aaron
"""

import pandas as pd
import numpy as np
from aicsshparam import shtools


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
        finf, fsup = np.percentile(df[f].values, [pct, 100 - pct])

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
    mesh, _ = shtools.get_reconstruction_from_coeffs(coeffs)
    return mesh

