from typing import Tuple, List, Dict, Optional
import pandas as pd
import numpy as np
import matplotlib
import scipy
from pygam import GAM, LinearGAM, s, te
from pygam.terms import TermList
from sklearn.preprocessing import StandardScaler

color_mapping = {
    "rnn_spillover" : {"name": "RNN Surprisal w/spillover", "pred_type" : "surprisal"},
    "bigram_spillover" : {"name": "Bigram Surprisal w/spillover", "pred_type" : "surprisal"},
    "brown_bigram_surprisal" : {"name": "Bigram Surprisal", "pred_type" : "surprisal"},
    "rnn_surprisal" : {"name": "RNN Surprisal", "pred_type" : "surprisal"},
    'free_energy': {"name": "VFE", "pred_type" : "both"},
    'arclength' : {"name" : "Arclen", "pred_type" : "both"},
    'li_top_down': {"name" : "Integral Top", "pred_type" : "state"},
    'li_bottom_up' : {"name" : "Integral Bottom", "pred_type" : "observation"},
    'wnorm_xy' : {"name" : "WY Norm", "pred_type": "observation"},
    'wnorm_xx' :{"name" : "WX Norm", "pred_type" : "state"},
    'bnorm_x' : {"name" : "BX Norm", "pred_type" : "state"},
    'bnorm_y' :{"name": "BY Norm", "pred_type": "observation"},
    'cosdist_y' : {"name": "Obs CosDist", "pred_type" : "observation"},
    'cosdist_posterior_v_posterior' : {"name": "Posterior CosDist", "pred_type" : "both"},
    'cosdist_prior_v_posterior' : {"name" : "Prior CosDist" , "pred_type": "state"},
    'bayesian_surprise' :{"name" : "KLDiv", "pred_type" : "state"},
    'iters' : {"name" : "InfIters", "pred_type": "both"},
    'cosdist_likelihood_v_posterior' : {"name" : "Likelihood CosDist", "pred_type": "both"}
}


def run_crossval(indices : Tuple[List, List], df : pd.DataFrame, predictor_list : List[str], surprisal_spillover : int, is_linear : bool) -> Dict[str, List[float]]:
    delta_loglik = {}
    for predictor in predictor_list:
        print(f"Running 10-fold CV for {predictor}")
        delta_loglik[predictor] = []
        for index_set in indices:
            train_indices, test_indices = index_set[0], index_set[1]
            training_data, test_data = df.iloc[train_indices], df.iloc[test_indices]
            num_spillover = 0
            if "surprisal" in predictor:
                num_spillover = surprisal_spillover
            model, predictor_names = fit_gam(training_data, predictor, num_spillover = num_spillover, return_predictors = True, linear = is_linear, baseline = False)
            training_data = training_data[predictor_names + ['RT']].dropna()
            test_data = test_data[predictor_names + ['RT']].dropna()
            train_x, train_y, test_x, test_y = np.array(training_data[predictor_names]), np.array(training_data['RT']), np.array(test_data[predictor_names]), np.array(test_data['RT'])
            baseline = fit_gam(training_data, predictor, num_spillover = num_spillover, return_predictors = False, linear = is_linear, baseline = True)
            delta_loglik[predictor].append((calc_loglik(model, test_x, train_x, train_y, test_y) \
                                            - calc_loglik(baseline, test_x, train_x, train_y, test_y)))
        print(f"Average Delta LogLik: {np.mean(delta_loglik[predictor])}")
    return delta_loglik

def build_TermList(baseline : bool, num_spillover : int, is_linear : bool) -> TermList:
    # this assumes that the predictors are defined at each spillover level: ie frequency, length, prev freq , prev len
    term_list = TermList()
    if not baseline:
        for i in range(num_spillover + 1):
            if is_linear:
                term_list += s(i)
            else:
                term_list += s(i, n_splines = 6, spline_order = 3) # from wilcox et al 23 and hoover et al
    first_control_index = num_spillover + 1
    total_control_terms = (num_spillover + 1) * 2
    for i in range(first_control_index, total_control_terms, 2):
        if is_linear:
            term_list += te(i, i + 1)
        else:
            term_list += te(i, i + 1, spline_order = 3)
    return term_list

def fit_gam(df : pd.DataFrame, predictor_name : str, num_spillover : int,
                     return_predictors : bool, linear : bool, baseline : bool):
    predictors = []
    predictors += [predictor_name] + [f'prev_{predictor_name}_{i}' for i in range(1, num_spillover + 1)]
    predictors += ['word_length', 'log_freq'] + \
        np.array([[f'prev_len_{i}', f'prev_freq_{i}'] for i in range(1, num_spillover + 1)]).flatten().tolist()
    # the features are based on the indices of the predictors
    rt_column = 'RT'
    model_data = df[predictors + [rt_column]].dropna()
    standardizer = StandardScaler()
    X = standardizer.fit_transform(np.array(model_data[predictors]))
    y = np.array(model_data[rt_column])
    terms = build_TermList(baseline, num_spillover, linear)
    if linear:
        gam = LinearGAM(terms)
    else:
        gam = GAM(terms)
    gam.fit(X, y)
    if return_predictors:
        return gam, predictors
    return gam

def plot_gam(model : GAM, original_values : pd.Series, predictor_name : str, 
             ax : matplotlib.axes._axes.Axes, y_bounds : Tuple, corpus_name : str):
    XX = model.generate_X_grid(term=0)
    pdep, confi = model.partial_dependence(term=0, X=XX, width=0.95)
    unstandardized = (XX[:,0] * np.std(original_values) + np.mean(original_values))
    ax.plot(unstandardized, pdep)
    ax.plot(unstandardized, confi, c='r', ls='--')
    print(np.mean(confi[:,1] - confi[:,0]))
    ax.set(xlabel = predictor_name, ylabel = f"Slowdown in RT due to {predictor_name}", ylim = y_bounds, title = corpus_name)

def calc_loglik(model, test_x, train_x, train_y, test_y):
    standardizer = StandardScaler()
    standardizer.fit(train_x)
    train_x, test_x = standardizer.transform(train_x), standardizer.transform(test_x)
    predictions = model.predict(test_x)
    residuals = train_y - model.predict(train_x)
    stdev = np.std(residuals)
    return np.mean(scipy.stats.norm.logpdf(test_y, loc=predictions, scale=stdev))
    # converted from Wilcox et al 2020, calculating per token LogLik

def format_logliks(tpc_delta_loglik : pd.DataFrame):
    # this adjusts the format of the dataframe containing delta log likelihood so that it can be plotted.
    # metrics are colorcoded by what type of predictor they are
    melted_loglik = tpc_delta_loglik.melt()
    melted_loglik['metric_type'] = melted_loglik['variable'].apply(lambda metric_name : color_mapping[metric_name]['pred_type'])
    melted_loglik['predictor_name'] = melted_loglik['variable'].apply(lambda metric_name : color_mapping[metric_name]['name'])
    return melted_loglik

def transform_logliks(df, is_linear : bool):
    df = df.melt()
    df['linear'] = is_linear
    return df

def compare_with_surprisal(tpc_delta_loglik, tpc_predictors, alternative_hyp = "two-sided"):
    vs_bigram = []
    vs_rnn = []
    for predictor in tpc_predictors:
        rnn_gam = scipy.stats.wilcoxon(tpc_delta_loglik[predictor], tpc_delta_loglik["rnn_surprisal"], alternative = alternative_hyp)
        bigram_gam = scipy.stats.wilcoxon(tpc_delta_loglik[predictor], tpc_delta_loglik["brown_bigram_surprisal"], alternative = alternative_hyp)
        vs_bigram.append({
            "predictor" : predictor,
            "gam_effect" : bigram_gam.statistic,
            "gam_p" : bigram_gam.pvalue,
        })
        vs_rnn.append({
            "predictor" : predictor,
            "gam_effect" : rnn_gam.statistic,
            "gam_p" : rnn_gam.pvalue,
        })
    return vs_bigram, vs_rnn

def significance_surprisal(effects):
    surprisal_comparison = pd.DataFrame(effects)
    surprisal_comparison['gam_significance'] = surprisal_comparison['gam_p'].apply(significance_level)
    return surprisal_comparison

def calc_p_value(loglik_data):
    wilcoxon_test = lambda predictor : scipy.stats.wilcoxon(loglik_data[predictor], np.zeros(10), alternative="greater")
    return [
        {"predictor_name" : predictor,
        "stat" : wilcoxon_test(predictor).statistic,
        "p_value" : wilcoxon_test(predictor).pvalue}
        for predictor in loglik_data.columns.values
    ]

def significance_level(p_value):
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"

def significance_table(delta_logliks):
    tpc_p_values = pd.DataFrame(calc_p_value(delta_logliks))
    significance = tpc_p_values['p_value'].apply(significance_level)
    tpc_p_values['significance'] = significance
    return tpc_p_values
