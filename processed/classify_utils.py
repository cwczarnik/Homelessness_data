from __future__ import division
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
# import matplotlib as mpl
# import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
from sklearn.metrics import f1_score

import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import patsy
from IPython.display import display

def classify(X_train, X_test, y_train, y_test, classifier,
             class_weight=None,
             penalty='l1', n_estimators=200,
             min_samples_split=2, min_samples_leaf=1,
             n_jobs=10, verbose=0, print_n_features=50,
             print_it=True, plot_it=True):
    # true class is second column (y_pred_proba[:,1])
    if (classifier == 'lr'):
        myLR = LogisticRegression(penalty=penalty, class_weight=class_weight, n_jobs=n_jobs, verbose=verbose)
        myLR.fit(X_train, y_train)
        y_pred = myLR.predict(X_test)
        y_pred_proba = myLR.predict_proba(X_test)
    elif (classifier == 'rf'):
        myRF = RandomForestClassifier(n_estimators=n_estimators, class_weight=class_weight, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, n_jobs=n_jobs, verbose=verbose)
        myRF.fit(X_train, y_train)
        y_pred = myRF.predict(X_test)
        y_pred_proba = myRF.predict_proba(X_test)

    if len(np.unique(y_train)) == 2 & len(np.unique(y_test)) == 2:
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba[:,1])
        myAuc = auc(fpr,tpr)
    else:
        myAuc = None

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = 2*((precision * recall) / (precision + recall))

    if print_it:
        if len(np.unique(y_train)) == 2 & len(np.unique(y_test)) == 2:
            print('\nAUC: %.5f' % myAuc)
        print('Precision: %.4f' % precision)
        print('Recall: %.4f' % recall)
        print('F1 score: %.4f' % f1)

        print('\nConfusion Matrix')
        #print(confusion_matrix(y_test, y_pred))
        print(pd.crosstab(y_test, y_pred, rownames=['Truth'], colnames=['Prediction'], margins=True))
        print('\nPercent of outcomes classified')
        print(pd.crosstab(y_test, y_pred, rownames=['Truth'], colnames=['Prediction']).apply(lambda r: 100.0 * r/r.sum()))

    if (classifier == 'lr'):
        myModel = myLR
        results = pd.DataFrame(np.array([myModel.coef_[0], np.exp(myModel.coef_[0])]).T, index=X_train.columns, columns=['coef', 'odds'])
        results = results.reindex(results['coef'].abs().sort_values(ascending=False).index)
        results.loc[results['odds'] < 1, 'odds'] = results.loc[results['odds'] < 1, 'odds'].apply(lambda x: 1/x)
        if print_it:
            print('\nAs feature increases, more likely to be in positive class:')
            print(results[results['coef'] >= 0].head(print_n_features))
            print('\nAs feature increases, less likely be in positive class:')
            print(results[results['coef'] < 0].head(print_n_features))
    elif (classifier == 'rf'):
        myModel = myRF
        results = pd.DataFrame(myModel.feature_importances_, index=X_train.columns, columns=['importance'])
        results['std'] = np.std([tree.feature_importances_ for tree in myModel.estimators_], axis=0)
        results = results.sort_values(['importance'], ascending=False)
        if print_it:
            print('\nFeatures, ranked by importance')
            print(results.head(print_n_features))

    if plot_it & (len(np.unique(y_train)) == 2 & len(np.unique(y_test)) == 2):
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_pred_proba[:,1])
        f1_curve = 2*((precision_curve * recall_curve) / (precision_curve + recall_curve))

        ncols = 3
        nrows = 1
        fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols*5,nrows*5))

        ax[0].plot(fpr, tpr)
        ax[0].set_aspect('equal')
        ax[0].set_xlabel('False Positive Rate')
        ax[0].set_ylabel('True Positive Rate')
        ax[0].set_title('ROC [AUC=%0.3f]' % (myAuc))

        ax[1].plot(roc_thresholds, tpr, 'b')
        ax[1].plot(roc_thresholds, fpr, 'r')
        ax[1].set_xlim((0, 1))
        ax[1].set_aspect('equal')
        ax[1].legend(['True Pos Rate', 'False Pos Rate'])
        ax[1].set_xlabel('Threshold')

        ax[2].plot(pr_thresholds, precision_curve[:-1], 'b')
        ax[2].plot(pr_thresholds, recall_curve[:-1], 'r')
        ax[2].plot(pr_thresholds, f1_curve[:-1], 'g')
        ax[2].set_xlim((0, 1))
        ax[2].set_aspect('equal')
        ax[2].legend(['Precision', 'Recall', 'F1 Score'])
        ax[2].set_xlabel('Threshold')
    return (results, myAuc, precision, recall, f1, myModel)

def plot_hist(df_pos, df_neg, pos_str='Positive class', neg_str='Negative class',
              results=None, n=None,
              max_col=4, normed=False,
              sharex=None, sharey=None,
              figsize_multuplier=5,
              log_trans=None,
              ):
    dummy_str = 'null999'
    if results is None:
        these_cols = df_pos.columns.tolist()
    elif type(results) is list:
        these_cols = results
    elif type(results) is str:
        these_cols = [results]
    elif hasattr(results, 'index'):
        if n is None:
            n=40
        these_cols = results.index.tolist()[:n]

    if log_trans is not None:
        if type(log_trans) is str:
            log_trans = [log_trans]
        for log_col in log_trans:
            df_pos[log_col] = df_pos[log_col].apply(lambda x: np.log(x))
            df_neg[log_col] = df_neg[log_col].apply(lambda x: np.log(x))

    if sharex is None:
        sharex = False
    if sharey is None:
        if normed is True:
            sharey = True
        else:
            sharey = False
    else:
        sharey = False

    if max_col < 2:
        max_col = 2
    if len(these_cols) == 1:
        these_cols.append(dummy_str)
    if len(these_cols) < max_col:
        ncols = len(these_cols)
    else:
        ncols = max_col
    nrows = int(np.ceil(len(these_cols)/ncols))

    fig, axes = plt.subplots(ncols=ncols, nrows=nrows,
                             figsize=(ncols*figsize_multuplier,nrows*figsize_multuplier),
                             sharex=sharex, sharey=sharey)

    count = -1
    for ax, cat in zip(axes.ravel(),these_cols):
        if cat == dummy_str:
            continue
        count += 1
        min_val = min(int(df_pos[cat].min()), df_neg[cat].min())
        max_val = max(int(df_pos[cat].max()), int(df_neg[cat].max()))
        if max_val <= 1:
            bin_spacing = .1
            if min_val == 0:
                bin_spacing = .25
                if max_val == 1:
                    max_val += .1
                elif max_val == 0:
                    max_val = 1.1
        elif max_val < 4:
            bin_spacing = .2
        elif max_val < 50:
            bin_spacing = .5
        elif max_val < 100:
            bin_spacing = 1
        elif max_val < 500:
            bin_spacing = 5
        elif max_val < 1000:
            bin_spacing = 10
        elif max_val < 5000:
            bin_spacing = 50
        elif max_val < 10000:
            bin_spacing = 100
        elif max_val < 50000:
            bin_spacing = 500
        else:
            bin_spacing = 1000

        bins = np.arange(min_val, max_val, bin_spacing)
        try:
            df_pos[cat].plot(ax=ax, kind='hist', alpha=.5, color='red', normed=normed, bins=bins);
        except:
            print('pos {}'.format(cat))
            print('min_val {}'.format(min_val))
            print('max_val {}'.format(max_val))
            print('bin_spacing {}'.format(bin_spacing))
            print(bins)
        try:
            df_neg[cat].plot(ax=ax, kind='hist', alpha=.5, color='blue', normed=normed, bins=bins);
        except:
            print('neg {}'.format(cat))
            print('min_val {}'.format(min_val))
            print('max_val {}'.format(max_val))
            print('bin_spacing {}'.format(bin_spacing))
            print(bins)
        if hasattr(results, 'columns'):
            if 'coef' in results.columns:
                title_str = cat
                if results.ix[count, 'coef'] < 0:
                    title_str = '%s: %.1f' % (title_str, -results.ix[count, 'odds'])
                elif results.ix[count, 'coef'] > 0:
                    title_str = '%s: %.1f' % (title_str, results.ix[count, 'odds'])
            elif 'importance' in results.columns:
                #title_str = '%s: %.2f' % (cat, results.ix[count, 'importance'])
                title_str = cat
            else:
                title_str = cat
        else:
            title_str = cat
        ax.set_title(title_str)
        if normed is False:
            ax.set_ylabel('Count')
        ax.legend([pos_str, neg_str], loc='upper right');
    fig.tight_layout()

def chunkify(lst, n=1, to_remove=None):
    """Chunk a list into n approximately equal groups. Also remove hardcoded items."""
    if to_remove is not None:
        lst = [x for x in lst if x not in to_remove]
    return [ lst[i::n] for i in xrange(n) ]

def binary_jitter(x, jitter_amount = .05):
    '''
    Add jitter to a 0/1 vector of data for plotting.
    '''
    jitters = np.random.rand(*x.shape) * jitter_amount
    x_jittered = x + np.where(x == 1, -1, 1) * jitters
    return x_jittered

def sm_logit(df, f=None, features=None,
             outcome='outcome_field',
             add_constant=True,
             categorical=None,
             maxiter=35,
             reg_method=None,
             reg_alpha=10,
             missing='raise',
             #log_trans=None,
             #sort_by='z',
             #outcome_behavior=None,
             subset=None,
             method='newton',
             ):
    """reg_method: regularization method. None (default), 'l1' or 'l1_cvxopt_cp'.

    reg_alpha: weight to apply regularization penalty. Default: 1.0.
    higher alpha = more coeff equal to zero

    missing: what to do with rows with missing values. 'raise' (default) or 'drop'.
    """
    if features is not None:
        df = df[features]
    else:
        features = df.columns.tolist()

    #if add_constant:
    #    df = sm.tools.add_constant(df, prepend=False, has_constant='raise')

    if f is None:
        these_features = [x for x in features if x != outcome]
        if categorical is not None:
            f = '{} ~ '.format(outcome) + ' + '.join(['C({})'.format(x) if x in categorical else x for x in these_features])
        else:
            f = '{} ~ '.format(outcome) + ' + '.join([x for x in these_features])
    # debug
    print(f)

    if reg_method is not None:
        # if subset is not None:
        #     df = df.loc[subset, :]
        # y, X = patsy.dmatrices(f, df, return_type='dataframe')

        # reg_alpha = reg_alpha * np.ones(X.shape[1])
        # reg_alpha[X.columns.tolist().index('Intercept')] = 0

        # results_log = sm.Logit(y, X, missing=missing).fit_regularized(method=reg_method, alpha=reg_alpha)

        results_log = smf.logit(f, df, subset=subset, missing=missing).fit_regularized(method=reg_method, alpha=reg_alpha)
    else:
        #results_log = sm.Logit.from_formula(f, df, missing='raise').fit(maxiter=maxiter)
        results_log = smf.logit(f, df, subset=subset, missing=missing).fit(maxiter=maxiter, method=method)

    #print_sm_logit_results(results_log, sort_by=sort_by, log_trans=log_trans, outcome_behavior=outcome_behavior)

    return results_log

def sm_ols(df, f=None, features=None,
             outcome='outcome_field',
             add_constant=True,
             categorical=None,
             maxiter=35,
             reg_method=None,
             reg_alpha=0,
             missing='raise',
             #log_trans=None,
             #sort_by='z',
             #outcome_behavior=None,
             ):
    """NOT TESTED

    reg_method: regularization method. None (default), 'l1' or 'l1_cvxopt_cp'.

    reg_alpha=1.0 means pure LASSO. 0 means OLS. In between represents elastic net regression.
    Higher alpha means more coefficients equal to zero.

    missing: what to do with rows with missing values. 'raise' (default) or 'drop'.
    """
    if features is not None:
        df = df[features]
    else:
        features = df.columns.tolist()

    #if add_constant:
    #    df = sm.tools.add_constant(df, prepend=False, has_constant='raise')

    if f is None:
        these_features = [x for x in features if x != outcome]
        if categorical is not None:
            f = '{} ~ '.format(outcome) + ' + '.join(['C({})'.format(x) if x in categorical else x for x in these_features])
        else:
            f = '{} ~ '.format(outcome) + ' + '.join([x for x in these_features])
    # debug
    print(f)

    if reg_method is not None:
        y, X = patsy.dmatrices(f, df, return_type='dataframe')

        #if reg_method == 'l1':
        #    reg_alpha = 1.0 # pure lasso
        #elif reg_method == 'l2':
        #    reg_alpha = # ridge

        # higher alpha = more coeff equal to zero
        reg_alpha = reg_alpha * np.ones(X.shape[1])
        reg_alpha[X.columns.tolist().index('Intercept')] = 0

        results_ols = sm.OLS(y, X, missing=missing).fit_regularized(method=reg_method, alpha=reg_alpha)
    else:
        #results_ols = sm.OLS.from_formula(f, df, missing='raise').fit(maxiter=maxiter)
        results_ols = smf.ols(f, df, missing=missing).fit(maxiter=maxiter)

    return results_ols

def print_sm_logit_results(results, sort_by='z', log_trans=None, print_n=None, print_p_limit=.05, outcome_behavior=None):
    #summary = results.summary()
    summary = results.summary2()
    display(summary.tables[0])

    if outcome_behavior is None:
        outcome_behavior = 'to be in positive class'

    #if log_trans is not None:
    #    if isinstance(log_trans, str):
    #        log_trans = [log_trans]
    #    for feat in log_trans:
    #        if feat in results.params.index.tolist():
    #            results.params.ix[feat] = np.exp(results.params.ix[feat])

    coef_str = 'Coef.'
    odds_str = 'Odds Ratio'
    p_str = 'P>|z|'
    results_print = pd.DataFrame(np.array([results.params, np.exp(results.params)]).T, index=results.params.index, columns=[coef_str, odds_str])
    results_print.loc[results_print[odds_str] < 1, odds_str] = results_print.loc[results_print[odds_str] < 1, odds_str].apply(lambda x: 1/x)

    ci = results.conf_int()
    ci.columns = ['[0.025', '0.975]']

    results_print = results_print.join(
        pd.DataFrame(results.bse, columns=['Std.Err.'])).join(
        pd.DataFrame(results.tvalues, columns=['z'])).join(
        pd.DataFrame(results.pvalues, columns=[p_str])).join(
        ci)
    results_more = results_print[(results_print[coef_str] >= 0) & (results_print[p_str] <= print_p_limit)].sort_values(by=sort_by, ascending=False)
    results_less = results_print[(results_print[coef_str] < 0) & (results_print[p_str] <= print_p_limit)].sort_values(by=sort_by, ascending=True)

    if print_n is None:
        print_n = results_more.shape[0]

    print('As feature increases, more likely {}:'.format(outcome_behavior))
    if print_n > results_more.shape[0]:
        print_n = results_more.shape[0]
    display(results_more.iloc[:print_n, :])

    print('\nAs feature increases, less likely {}:'.format(outcome_behavior))
    if print_n > results_less.shape[0]:
        print_n = results_less.shape[0]
    display(results_less.iloc[:print_n, :])

    if len(summary.tables) > 2:
        for i in range(2, len(summary.tables)):
            display(summary.tables[i])
    return (results_more, results_less)

def df_find_existing_col(df, col, result_col=None):
    '''Find and return the columns in list col that exist in dataframe df,
    returning that intersecting subset (and corresponding entries of result_col).
    '''
    if result_col == None:
        col = list(set(df.columns.tolist()).intersection(col))
    else:
        if len(col) != len(result_col):
            logging.warning('col and result_col are not the same length.')
            return
        # make a dictionary out of the columns we want in the numerator
        ind_dict = dict((k,i) for i,k in enumerate(col))
        # find the overlap between requested columns and what actually exists in the df
        inter = set(ind_dict).intersection(df.columns.tolist())
        # get their indices and overwrite with overlapping set
        indices = [ind_dict[x] for x in inter]
        col = [col[i] for i in indices]
        result_col = [result_col[i] for i in indices]
    if len(col) == 0:
        logging.warning('None of the columns are in the dataframe.')
    return (col, result_col)

def quantile_nonzero_values(df, regex_col='^namestart', quant_dim='quantity', q=4, quant_col_prefix='', col_prefix_to_remove=None):
    """Calculate quantile of quant_dim for regex_col regex match with >0 values
    """
    cols = df.filter(regex=regex_col).columns.tolist()
    if len(cols) > 0:
        for col in cols:
            if col_prefix_to_remove is not None:
                col_rm = col.replace('prefix_', '')
            else:
                col_rm = col
            quant_col = '{qcp}{cr}_quant'.format(qcp=quant_col_prefix,cr=col_rm)
            # fill with nan so we can operate on non-zero values later
            df.loc[:, quant_col] = np.nan
            if len(df[(df[col] > 0)][quant_dim]) > 0 and df[(df[col] > 0)][quant_dim].std() > 0:
                labels = range(1, q+1)
                # # option 1: use qcut
                # col_q = pd.qcut(df[(df[col] > 0)][quant_dim], q, labels=range(1, q+1))

                # # option 2: manual bins for when there might not be unique bins, only based on unique values
                # bins = pd.core.algorithms.quantile(np.unique(df[(df[col] > 0)][quant_dim]),
                #                                    np.linspace(0, 1, q+1))

                # option 3: manual bins for when there might not be unique bins, only based on unique values
                bins = pd.core.algorithms.quantile(df[(df[col] > 0)][quant_dim],
                                                   np.linspace(0, 1, q+1))
                if len(np.unique(bins)) < len(labels):
                    bins = np.unique(bins)
                    # # option 1: change q to length of unique bins
                    # labels = range(1, len(bins))

                    # option 2: interpolate bin numbers, keeping min and max, skipping where needed
                    labels = np.round(np.linspace(1, len(labels), len(bins)-1))

                col_q = pd.tools.tile._bins_to_cuts(df[(df[col] > 0)][quant_dim],
                                                    bins, labels=labels, include_lowest=True)
                df.loc[(df[col] > 0), quant_col] = col_q
            #df[quant_col] = df[quant_col].fillna(np.nan)
    return df

def calc_similarity(df, columns=None, field_prefix=''):
    '''Calculates similarity, using two different measures: Schmidt number and entropy.

    The Schmidt number (K) measures the factorability of a matrix in terms of the singular values from SVD.
    Here, 1-K it represents how broadly the rows (independent events) sample the possible space.
    If every row is the same (every event is the same every time), K=1. K increases with variability across rows.
    The maximum value of K is either N columns (components that make up an event) or M rows (events), whichever is smaller.
    This maximum value is achieved when rows contain the same frequency of the different columns.

    NB: This function returns 1 minus a normalized K, normalized by its maximum possible value
        to keep it within the interval [0, 1]. It is subtracted from 1 to give a similarity value,
        where 0 means completely dissimilar and 1 means completely similar.

    For the entropy measure, it returns e^(-entropy), to give a measure of similarity.
    This will return 1 when events are identical and moves toward 0 as they differ.
    '''
    if columns is None:
        columns = df.columns.tolist()
    freq = df.as_matrix(columns=columns).T
    U, s, Vh = svd(freq)
    s_norm = s / s.sum()

    # schmidt decomposition-based similarity
    K = 1 / (s_norm**2).sum()
    # normalize the schmidt number by the maximum possible value of K: (K - 1) / [ min(M,N) - 1]
    K_norm = (K - 1) / (min(freq.shape) - 1)

    # entropy-based similarity
    entropy = np.nansum(-np.log(s_norm) * s_norm)

    return pd.Series({'{}similarity_schmidt'.format(field_prefix): 1 - K_norm,
                      '{}similarity_entropy'.format(field_prefix): np.exp(-entropy)})

def bin_residuals(resid, var, bins=40):
    '''
    Compute average residuals within bins of a variable.

    Returns a dataframe indexed by the bins, with the bin midpoint,
    the residual average within the bin, and the confidence interval
    bounds.

    Original source: https://nbviewer.jupyter.org/github/carljv/Will_it_Python/blob/master/ARM/ch5/arsenic_wells_switching.ipynb
    '''
    resid_df = pd.DataFrame({'var': var, 'resid': resid})
    resid_df['bins'] = pd.qcut(var, bins)
    bin_group = resid_df.groupby('bins')
    bin_df = bin_group['var', 'resid'].mean()
    bin_df['count'] = bin_group['resid'].count()
    bin_df['lower_ci'] = -2 * (bin_group['resid'].std() /
                               np.sqrt(bin_group['resid'].count()))
    bin_df['upper_ci'] =  2 * (bin_group['resid'].std() /
                               np.sqrt(bin_df['count']))
    bin_df = bin_df.sort('var')
    return(bin_df)

def plot_binned_residuals(bin_df):
    '''
    Plotted binned residual averages and confidence intervals.

    Original source: https://nbviewer.jupyter.org/github/carljv/Will_it_Python/blob/master/ARM/ch5/arsenic_wells_switching.ipynb
    '''
    plt.plot(bin_df['var'], bin_df['resid'], '.')
    plt.plot(bin_df['var'], bin_df['lower_ci'], '-r')
    plt.plot(bin_df['var'], bin_df['upper_ci'], '-r')
    plt.axhline(0, color = 'gray', lw = .5)