import string
from typing import Optional, List

import numpy as np
import pandas as pd
from wordfreq import word_frequency

def clean_str(token):
    return token.replace('[^\w\s]','').replace(",", "").replace(".", "").lower()

# functions for predictors
def word_length(rt_data: pd.DataFrame, token_col: str):
    return rt_data.apply(lambda row: len(row[token_col]), axis = 1)

def compute_log_frequency(word, language):
    freq = word_frequency(word, language)
    return -np.log2(freq)

def join_log_freq(rt_data: pd.DataFrame, from_file : bool, lang : Optional[str], filepath : Optional[str]):
    if from_file:
        freq_table = pd.read_table(filepath, delim_whitespace=True, names=('prob', 'word', 'backoff_weight'))
        rt_data = rt_data.merge(freq_table[['prob', 'word']], how = 'left', left_on = 'token', right_on = 'word')
    else:
        rt_data['token'].apply(compute_log_frequency, language = lang)
    rt_data.rename(columns={'prob': 'log_freq'}, inplace = True)
    return rt_data

def prev_token_predictors(rt_data: pd.DataFrame, num_tokens: int, predictors : List[str]):
    for i in range(1, num_tokens + 1):
        rt_data[f'prev_freq_{str(i)}'] = rt_data['log_freq'].shift(i)
        rt_data[f'prev_len_{str(i)}'] = rt_data['word_length'].shift(i)
        for predictor in predictors:
            rt_data[f'prev_{predictor}_{str(i)}'] = rt_data[predictor].shift(i)
    return rt_data

"""
IMPORTANT: this function assumes that each row of the surprisal df has predictors
 associated with the same word as the RT data, and the RT data has values to exclude (marked under an `exclude` column).
 See the next function to generate the exclude value for an RT dataset (following Goodkind & Bicknell)
"""
def merge_with_rt_data(surprisal_df : pd.DataFrame, rt_data : pd.DataFrame, predictor_name: str, spillover_count : int,
                       frequency_file : Optional[str], language : Optional[str]):
    rt_data['surprisal'] = surprisal_df[predictor_name]
    rt_data['word_length'] = word_length(rt_data, 'token')
    if frequency_file:
        rt_data = join_log_freq(rt_data, True, filepath=frequency_file)
    else:
        rt_data = join_log_freq(rt_data, False, lang = language)
    rt_data = prev_token_predictors(rt_data, spillover_count)
    rt_data = rt_data[rt_data['exclude'] == 0].dropna()
    return rt_data

def exclude_token(df: pd.DataFrame, index: int, token_column: str):
    """
    exclude the token (specified by token_column) if it precedes or follows punctuation, or if it is not alphabetic
    example usage on RT data: 
    rt_data['exclude'] = [exclude_token(rt_data, i, 'token') for i in range(len(rt_data.index))]
    """
    token = str(df.iloc[index][token_column])
    if token[0] in string.punctuation or token[-1] in string.punctuation or not token.isalpha():
        return 1
    if index != 0:
        prev_token = str(df.iloc[index - 1][token_column])
        # exclude this token if the previous token ended w/a punctuation mark or if it's non alphabetic
        if prev_token[-1] in string.punctuation or not prev_token.isalpha():
            return 1
    if index != len(df.index - 1):
        next_token = str(df.iloc[index + 1][token_column])
        # exclude this token if the next token starts w/a punctuation mark or isn't alphabetic
        if next_token[0] in string.punctuation or not next_token.isalpha():
            return 1
    return 0