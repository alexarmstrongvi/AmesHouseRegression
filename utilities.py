#!/usr/bin/env python3
"""
Useful functions for exploratory data analysis

Author:
    Alex Armstrong <alex.armstrong.vi@gmail.com> 
    March 2020

Example:
    import utilities as utils
"""
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import pingouin as pg
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
import numpy as np

from collections import defaultdict

from typing import Mapping, List, Callable, Any


def categorize_data_type(
        df             : pd.DataFrame, 
        continuous_thr : int = 50, 
        override       : Mapping = None
        ) -> pd.DataFrame:
    """
    Attempt to categorize the data types of DataFrame columns into nominal, ordinal, or
    continuous. No attempt to determine discrete data types. 

    Arguments:
        df : 
            DataFrame to be categorized
        continuous_thr :
            Minimum threshold on the number of unique values to be considered continuous
        override : dict[feature name] = data type
            Override automatically determined data type

    Returns:
       Data type of each feature in a Series 
    """

    if not override:
        override = {}

    data_types = []
    for feature in df:
        if feature in override:
            data_types.append(override[feature])
            continue

        data = df[feature].dropna()
        n_unique = len(data.unique())
        if data.dtype == 'object':
            data_type = 'Nominal'
        elif data.dtype == 'float64' and not np.allclose(data%1, 0):
            data_type = 'Continuous'
        elif data.dtype in ['int64', 'float64'] and n_unique >= continuous_thr:
            data_type = 'Continuous'
        elif data.dtype == 'int64':
            data_type = 'Ordinal'
        else:
            data_type = 'Unknown'

        data_types.append(data_type)

    rv = pd.Series(data_types, index=df.columns, name='Data type')
    return rv

def corr_to_target(
        df : pd.DataFrame,
        target : Any,
        cat_features : Any,
    ) -> pd.DataFrame:
    """
    Determine correlation of target feature to all other features with an effect size
    that is comparable between categorical and numerical features. 

    Arguments:
        df :
            Data
        target :
            Target feature key in DataFrame with respect to which correlations are determined
        cat_features :
            Keys for categorical features in DataFrame
    Returns:
        DataFrame with correlations measures

    """
    result = defaultdict(dict)
    cat_target = target in cat_features
    num_target = not cat_target
    
    for f in df.columns.drop(target):
        data = df[[f, target]].dropna()
        cat_feature = f in cat_features
        num_feature = not cat_feature
        result['Categorical'][f] = int(cat_feature)

        if cat_target and cat_feature:
            pass
        elif cat_target and num_feature:
            pass
        elif num_target and cat_feature:
            if data[f].value_counts().min() > 1:
                n2, pval = pg.welch_anova(data=data, dv=target, between=f).loc[0,['np2','p-unc']].values
                result['R2'][f] = n2
                result['pval'][f] = pval
            mi = mutual_info_classif(data[[target]], data[f])[0]
            result['MI'][f] = mi
        elif num_target and num_feature:
            r, pval = stats.pearsonr(data[f], data[target])
            result['R2'][f] = r**2
            result['pval'][f] = pval
            mi = mutual_info_regression(data[[f]], data[target])[0]
            result['MI'][f] = mi

    return pd.DataFrame(result)

def local_sigma_bands(
    x            : np.ndarray,
    y            : np.ndarray,
    n_regions    : int   = 100,
    window_scale : float = 0.3,
    percentiles  : bool  = False,
    zero         : bool  = False
    ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Estimate local mean and standard deviation
    Arguments:
        x :
            Predictor variable
        y :
            Response variable
        n_regions :
            Number of localized regions in x to process
        window_scale :
            Relative window size defining each local x region. Scale ranges from 0 to 1.
            Default behavior calculates window width relative to the range of x values; value in [0,1]
            For `percentiles=True`, window scale is in units of percentiles; value in [0,100]
        percentiles :
            Window includes a percentile range of data points instead of a fixed width range
        zero :
            Center the sigma bands around zero
    Returns:
        x_local :
            Central x values for regions used in calculating local mean and std
        y_mid :
            Mean for each local region 
        y_lo, y_hi :
            One standard deviation below and above the mean at each local x
    """
    # Sort data points by x values
    idx_sort = np.argsort(x)
    x, y = x[idx_sort], y[idx_sort]

    # Configure parameters
    if percentiles:
        x_scan = np.linspace(0,100, n_regions)
        window_width = window_scale
    else:
        x_scan = np.linspace(min(x), max(x), n_regions)
        window_width = window_scale * (max(x) - min(x))

    # Calculate bands
    x_local, y_lo, y_mid, y_hi = [],[],[],[]
    for i, xi in enumerate(x_scan):
        if percentiles:
            p_lo = max(0,   xi - window_width/2)
            p_hi = min(100, xi + window_width/2)
            x_mid, x_lo, x_hi = np.percentile(x, [xi, p_lo, p_hi])
            x_local.append(x_mid)
        else:
            x_local.append(xi)
            x_lo, x_hi = xi - window_width/2, xi + window_width/2

        vals = y[(x_lo <= x) & (x <= x_hi)]
        if len(vals) > 2:
            y_mid.append(vals.mean())
            y_lo.append(vals.mean() - vals.std(ddof=1))
            y_hi.append(vals.mean() + vals.std(ddof=1))
        else: # Not enough values to calculate std. Skip region
            x_local.pop()

    # Format output
    x_local = np.array(x_local)
    y_lo  = np.array(y_lo)
    y_mid = np.array(y_mid)
    y_hi  = np.array(y_hi)
    if zero:
        offset = (y_hi + y_lo)/2
        y_lo -= offset
        y_hi -= offset

    return (x_local, y_mid, y_lo, y_hi)

def truncated_hist(
    x   : pd.Series,
    val : Any = 'mode',
    ax  : plt.Axes = None
    ) -> plt.Axes:
    """
    Truncated histogram to visualize more values when one dominates

    Arguments:
        x :
            Data Series
        val :
            Value to truncate in histogram. 'mode' will truncate the data mode.
        ax :
            matplotlib Axes object to draw plot onto
    Returns:
        ax :
            Returns the Axes object with the plot drawn onto it
    """
    # Setup Axes
    if not ax:
        fig, ax = plt.subplots()
    ax.set_xlabel(x.name)
    ax.set_ylabel('Counts')

    if val is None:
        ax.hist(x, bins='auto')
        return

    if val == 'mode':
        val = x.mode().iloc[0]

    # Plot without selected value
    sel_vals = x[x != val]
    bin_vals, bin_edges, _ = ax.hist(sel_vals, bins='auto')
    ax_min, ax_max = ax.get_ylim()
    ax.set_ylim(ax_min, ax_max*1.1)

    # Expand x-axis to include removed value and then annotate with the value's count
    ax_min, ax_max = ax.get_xlim()
    if val < min(sel_vals): # Lower xmin
        buff = abs(ax_max - max(sel_vals))
        ax.set_xlim(val - buff, ax_max)
        horiztonal_alignment = 'left'
        arrow_relpos = (0, 0)
    elif val > max(sel_vals): # Increase xmax
        buff = abs(ax_min - min(sel_vals))
        ax.set_xlim(ax_min, val + buff)
        horiztonal_alignment = 'right'
        arrow_relpos = (1, 1)
    else: # No change to x range
        horiztonal_alignment = 'center'
        arrow_relpos = (0.5, 0.5)
    val_count = (x == val).sum()
    val_perc = val_count/len(x)
    ax.annotate(f'{val} (count={val_count}; {val_perc:.0%} of total)',
                xy=(val, 0), xytext=(val, max(bin_vals)*1.1), xycoords='data',
                ha=horiztonal_alignment,
                arrowprops=dict(arrowstyle = '<-', color = 'black', lw = '4', relpos=arrow_relpos)
               )

    return ax

def truncated_countplot(
    x   : pd.Series,
    val : Any = 'mode',
    ax  : plt.Axes = None
    ) -> plt.Axes:
    """
    Truncated count plot to visualize more values when one dominates

    Arguments:
        x :
            Data Series
        val :
            Value to truncate in count plot. 'mode' will truncate the data mode.
        ax :
            matplotlib Axes object to draw plot onto
    Returns:
        ax :
            Returns the Axes object with the plot drawn onto it
    """
    # Setup Axes
    if not ax:
        fig, ax = plt.subplots()
    ax.set_xlabel(x.name)
    ax.set_ylabel('Counts')

    if val is None:
        sns.countplot(x=x, ax=ax)
        return

    if val == 'mode':
        val = x.mode().iloc[0]

    # Plot and truncate
    splot = sns.countplot(x=x, ax=ax)
    ymax = x[x != val].value_counts().iloc[0]*1.4
    ax.set_ylim(0, ymax)

    # Annotate truncated bin
    xticklabels = [x.get_text() for x in ax.get_xticklabels()]
    val_ibin = xticklabels.index(str(val))
    val_bin = splot.patches[val_ibin]
    xloc = val_bin.get_x() + 0.5*val_bin.get_width()
    yloc = ymax
    ax.annotate('', xy=(xloc, 0), xytext=(xloc, yloc), xycoords='data',
                arrowprops=dict(arrowstyle = '<-', color = 'black', lw = '4')
               )
    val_count = (x == val).sum()
    val_perc = val_count / len(x)
    ax.annotate(f'{val} (count={val_count}; {val_perc:.0%} of total)',
                xy=(0.5, 0), xytext=(0.5, 0.9), xycoords='axes fraction',
                ha='center'
               )

    return ax

def plot_grid(
    df        : pd.DataFrame,
    features  : List,
    plottype  : str = None,
    plot_func : Callable[[pd.DataFrame, Any, plt.Axes], Any] = None,
    ncols     : int   = 4,
    vscale    : float = 5,
    hscale    : float = 5
    ) -> plt.Axes:
    """
        Draw several univariate plots in a grid
    Arguments:
        df :
            Dataframe with data to plot
        features :
            Dataframe features to plot
        plottype : Type of plot
            * 'histplot'
            * 'histplot_trunc'
            * 'histplot_discrete'
            * 'countplot_trunc'
            * 'countplot_few'
            * 'countplot_many'
            * None -> must provide `plot_func`
        plot_func :
            Arbitrary plot function (`plot_func(DataFrame, feature, Axes)`)
        ncols :
            Number of columns in grid
        vscale, hscale :
            Vertical and horizonital scale of each plot in the grid (i.e. `figsize`)

    Returns:
        fig : Figure
        ax  : array of Axes objects.
    """

    # Initialize figure
    nrows = len(features)//ncols + (len(features)%ncols > 0)
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*hscale,nrows*vscale))

    # Draw plots from left -> right and top -> bottom
    for i, c in enumerate(features):
        ax = axs[i//ncols, i%ncols] if nrows > 1 else axs[i]
        if plottype == 'histplot':
            sns.histplot(df, x=c, ax=ax)
        elif plottype == 'histplot_crop':
            truncated_hist(df[c], val='mode', ax=ax)
        elif plottype == 'histplot_discrete':
            sns.histplot(df, x=c, discrete=True, shrink=0.9, ax=ax)
        elif plottype == 'countplot_crop':
            truncated_countplot(df[c], val='mode', ax=ax)
        elif plottype == 'countplot_few':
            sns.countplot(data=df, x=c, ax=ax)
        elif plottype == 'countplot_many':
            sns.countplot(data=df, y=c, ax=ax)
        elif plot_func:
            plot_func(df, c, ax)
        else:
            print('Plot type no recognized:', plottype)
            break

    # Remove any axes not filled
    if len(features)%ncols:
        for i in range(len(features)%ncols, ncols):
            ax = axs[nrows-1, i%ncols] if nrows > 1 else axs[i]
            ax.axis('off')

    return fig, axs
