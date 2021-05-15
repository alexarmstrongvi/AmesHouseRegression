#!/usr/bin/env python3
"""
Useful functions for exploratory data analysis

Author:
    Alex Armstrong <alex.armstrong.vi@gmail.com> 
    March 2020

Example:
    import utilities as utils
"""
import phik
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import pingouin as pg
import statsmodels.api as sm
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats, optimize
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

def nearest_neighbor_value(vals, val):
    '''
    Return array entry with value closest to target entry
    
    Arguments :
        vals : array of comparable values (e.g. float, string)
        val  : target value
    Returns:
        nearest neighbor value
    
    >>> nearest_neighbor_value([2,3,5],2)
    3
    >>> nearest_neighbor_value([2,5,3],2)
    3
    '''
    vals = np.sort(vals)
    index = np.where(vals == val)[0][0]
    if index == len(vals)-1: # target entry in back
        nneighbor = vals[index-1]
    elif index == 0: # target entry in front
        nneighbor = vals[1]
    else: # Target entry in middle
        diff_lo = abs(val - vals[index-1])
        diff_hi = abs(val - vals[index+1])
        nneighbor = vals[index-1] if diff_lo < diff_hi else vals[index+1]
    return nneighbor

def outlier_check(df, index, num_features, discrete_thr=10, print_result=False):
    """
    Calculate outlier statistics for each feature of a given entry
    
    Arguments:
        df           : Input DataFrame
        index        : Index of entry
        num_features : Numerical features in df
        discrete_thr : Threshold on number of unique values for calculating numerical outlier statistics
        print_result : print results instead of returning them
    
    Returns:
        (df_numerical, df_categorical) : Outlier statatistis for numerical and categorical variables
    """
    num_scores = defaultdict(dict)
    cat_scores = defaultdict(dict)
    for f in df.columns:
        vals = df[f]
        val = vals[index]
        vc = vals.value_counts()
        mode = vc.index[0]
        if f in num_features and len(vc) > discrete_thr:
            num_scores['Value'][f] = val
            if pd.isnull(val): continue
            num_scores['Mean'][f] = vals.mean()
            num_scores['Median'][f] = vals.median()
            num_scores['NN value'][f] = nearest_neighbor_value(vals, val)
            num_scores['|Z-score|'][f] = abs(stats.zscore(vals)[index])
            #num_scores['percentile-score'][f] = stats.percentileofscore(vals, val)
            median_resids = np.abs(vals - vals.median())
            median_resid = np.abs(val - vals.median())
            num_scores['percentile-score'][f] = stats.percentileofscore(median_resids, median_resid)
            # MAD score 
            # must account for cases where >50% of values are the same
            mad = stats.median_abs_deviation(vals)
            if mad > 0: 
                mad_score = median_resid/mad 
            elif val == mode:
                mad_score = 0
            else: # val != mode
                mad_score = float('inf')
            num_scores['|MAD-score|'][f] = mad_score
        else : # Categorical
            cat_scores['Value'][f] = val
            if pd.isnull(val): continue
            ival = vc.index.get_loc(val)
            inn = ival-1 if ival > 0 else 0
            cat_scores['Mode'][f] = mode
            cat_scores['freq'][f] = vc[val]
            cat_scores['freq rank'][f] = ival + 1
            cat_scores['# unique'][f] = len(vc)
            cat_scores['rel freq'][f] = vc[val]/len(df)
            cat_scores['cum freq'][f] = vc.iloc[ival:].sum()/len(df)
            cat_scores['max freq ratio'][f] = vc[val]/vc[mode]
            cat_scores['NN freq ratio'][f] = vc.iloc[ival]/vc.iloc[ival-1] if ival>0 else 1


    df_num_scores = pd.DataFrame(num_scores)
    df_cat_scores = pd.DataFrame(cat_scores)
    if print_result:
        print('Outlier test for entry', index)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            # Show two tables: one for continuous and another for categorical
            print("Numerical Outliers")
            display(df_num_scores.sort_values(['|Z-score|'], ascending=False))
            print("Categorical Outliers")
            display(df_cat_scores.sort_values(['cum freq']))
    else:
        return df_num_scores, df_cat_scores

def r2_adjusted(r2, n, k):
    return 1 - ((1-r2)*(n-1))/(n-k-1)

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
            vc = data[f].value_counts()
            if vc.min() > 1:
                n2, pval = pg.welch_anova(data=data, dv=target, between=f).loc[0,['np2','p-unc']].values
                result['R2'][f] = n2
                result['R2_adj'][f] = r2_adjusted(n2,len(data),len(vc))
                result['pval'][f] = pval
            #mi = mutual_info_classif(data[[target]], data[f])[0]
            #result['MI'][f] = mi
        elif num_target and num_feature:
            r, pval = stats.pearsonr(data[f], data[target])
            result['R2'][f] = r**2
            result['R2_adj'][f] = r2_adjusted(r**2, len(data), 1)
            result['pval'][f] = pval
            #mi = mutual_info_regression(data[[f]], data[target])[0]
            #result['MI'][f] = mi

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

def power_fit(
        x : np.ndarray, 
        y : np.ndarray, 
        neg_p   : bool = False,
        maxfev  : int  = None,
        use_jac : bool = False
        ) -> (float, float, float, float):
    """
    Fit quasi-polynomial a*(x-s)^p + b to inputs. Useful for checking linearity

    Arguments:
        x, y    : Data for fit
        neg_p   : Initialize fit with p = -1. Improves convergence if true p < 0
        maxfev  : Manually set max function evaluations 
        use_jac : Use Jacobian in curve fit
    
    Returns:
        (a, s, p, b) : fit parameters

    """
    def f(x, a, s, p, b):
        return a*((x-s)**p) + b


    def jac(x, a, s, p, b):
        return np.array([
            (x-s)**p,                 # df/da
            -p*((x-s)**(p-1)),        # df/ds
            np.log(x-s) * ((x-s)**p), # df/dp
            np.ones(len(x))           # df/db
        ]).T
    if not use_jac:
        jac = None

    # Constraints : x-s > 0 or else discontinuous behavior since df/fp undefined
    s_max = x.min()-1e-3*np.ptp(x)

    # Initialize fit parameters to give OLS fit line
    X = sm.add_constant(x.reshape(-1,1))
    result = sm.OLS(y, X).fit()
    a0 = result.params[1] # OLS Slope
    s0 = s_max
    p0 = -1 if neg_p else 1
    b0 = result.predict([[1, s_max]])[0]
    param0 = (a0, s0, p0, b0)

    # Define bounds
    #p_min, p_max = (-3, -1/3) if neg_p else (1/3, 3)
    p_min, p_max = (-2, -1/2) if neg_p else (1/2, 2)
    bounds = [
        # a, s, p, b
        [-np.inf, -np.inf, p_min, -np.inf], # lower
        [np.inf, s_max, p_max,  np.inf] # upper
    ]

    # Fit
    try:
        popt, _ = optimize.curve_fit(f, x, y, p0=param0, bounds=bounds, method='trf', jac=jac, maxfev=maxfev)
    except RuntimeError as e:
        print('Power fit failed:',e)
        return param0

    return popt

def inverse_crosstab(table):
    n = table.sum().sum()
    x = []
    y = []
    for i, row in enumerate(table.index):
        for j, col in enumerate(table.columns):
            count = table.at[row,col]
            x += [j]*count
            y += [i]*count
    return np.array(x), np.array(y)


# Cochran’s rule of thumb is that at least 80% of the expected cell frequencies is 5 counts or more,
# and that no expected cell frequency is less than 1 count. For a 2x2 contingency table, Cochran
# recommends that the test should be used only if the expected frequency in each cell is at
# least 5 counts.

def phi_coef(cont_table):
    """
    - a measure of association for two binary variables
    - the Pearson correlation coefficient reduces to the phi coefficient in the 2×2 case
    """
    chi2 = stats.chi2_contingency(cont_table, correction=False)[0]
    N = cont_table.sum().sum()
    return np.sqrt(chi2/N)


def tschuprows_t(cont_table):
    """
    - Tschuprow, 1925, 1939
    - Symmetric, order independent measure of correlation between discrete variables with >=2 levels
    - varies from 0 (no association) to 1 (complete association)
    - Reduces to the phi coefficient in the 2×2 case 
    """
    phi = phi_coef(cont_table)
    c, r = cont_table.shape
    correction = ((r-1)*(c-1))**(-1/4)
    return correction * phi

def contingency_coef(cont_table, adjust=True):
    """
    - An adjustment to phi coefficient, intended to adapt it to tables 
    larger than 2-by-2.
    
    - The larger the table your chi-squared coefficient is calculated from,
    the closer to 1 a perfect association will approach. That’s why some 
    statisticians suggest using the contingency coefficient only if you’re 
    working with a 5 by 5 table or larger.
    (https://www.statisticshowto.com/contingency-coefficient/)
    """
    phi = phi_coef(cont_table)
    correction = 1 / np.sqrt(1 + phi**2)
    if adjust: # normalize across different contingency table dimensions
        r, c = cont_table.shape
        theoretical_max = ((r-1)/r * (c-1)/c)**(1/4)
    else:
        theoretical_max = 1
    return (correction * phi) / theoretical_max
    # return 1/np.sqrt(1 + N/chi2) / theoretical_max

def cramers_v(cont_table):
    """
    - Cramer, 1946
    - Symmetric, order independent measure of correlation between discrete variables with >=2 levels
    - Varies from 0 (no association) to 1 (complete association)
    - Reduces to the phi coefficient in the 2×2 case
    """
    phi = phi_coef(cont_table)
    min_dim = min(cont_table.shape)
    correction = 1 / np.sqrt(min_dim-1)
    return correction * phi

def conditional_entropy(p_xy):
    if not np.allclose(p_xy.sum(), 1): # not normalized
        p_xy = p_xy/p_xy.sum()
    p_y = p_xy.sum(axis=1, keepdims=True)
    return -np.sum(p_xy * np.log(p_xy/p_y))

def uncertainty_coef(cont_table, sym=False):
    """
    - Asymmetric (Y|X)
    - Varies from 0 (no association) to 1 (complete association)
    - Theil (1970) derived a large part of the uncertainty coefficient, 
    so it’s occasionally referred to as “Theil’s U”. This is a little misleading, 
    because the term Theil’s U usually refers to a completely different U used in finance.
    """
    if sym:
        sx = stats.entropy(cont_table.sum(axis=0)) 
        sy = stats.entropy(cont_table.sum(axis=1))
        ux = uncertainty_coef(cont_table)
        uy = uncertainty_coef(cont_table.T)
        return (sx*ux + sy*uy)/(sx+sy) # Entropy weighted average
    
    sx = stats.entropy(cont_table.sum(axis=0))
    x, y = inverse_crosstab(cont_table)
    mi = mutual_info_classif(x.reshape(-1,1), y, discrete_features=True)[0]
    #mi = hx - conditional_entropy(cont_table.to_numpy())
    return mi/sx

def goodman_kruskals_lambda(cont_table, sym=False):
    """
    - Goodman, L.A., Kruskal, W.H. (1954) "Measures of association for cross classifications"
    - default asymmetric (Y|X) but can symmetric version available
    """
    
    n = cont_table.sum().sum() # Total entries
    s = cont_table.max(axis=0).sum() # Sum of col max
    r = cont_table.sum(axis=1).max() # Max of row sum

    if sym:
        sx = cont_table.max(axis=1).sum()
        rx = cont_table.sum(axis=0).max()
        s = (sx+s)/2
        r = (rx+r)/2
        
    e1 = 1 - r/n # P(error) from always guessing Y from max P(Y)
    e2 = 1 - s/n # P(error) from always guessing Y from max P(X=x,Y)
    return (e1 - e2) / e1 # = (s-r)/(n-r)

def phik_coef(cont_table):
    pk = phik.phik.phik_from_hist2d(cont_table.to_numpy())
    pk_pval, pk_z = phik.significance.significance_from_hist2d(cont_table.to_numpy())
    return (pk, pk_pval, pk_z)

