import math
import re
import nltk
from nltk.corpus import brown
import pandas as pd
from tqdm import tqdm
import gensim.downloader
import numpy as np

UNK_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"


def load_embeddings(filepath):
    # can check a list of default embeddings vs loading in trained local embeddings
    embeddings = gensim.downloader.load(filepath)
    embeddings[UNK_TOKEN] = np.ones(300)*1e-6  
    embeddings[PAD_TOKEN] = np.ones(300)*1e-6  
    return embeddings 



def encode_batch(batch, key2index, tokenizer):
    # ids for each token in each sentence
    batch_ids = [tokenizer.encode(sentence).ids for sentence in batch["sentence"]]
    # max sentence length for the batch
    max_batch_len = max(len(ids) for ids in batch_ids)
    # tuple (ids, mask) per sentence for ids with padding to longest sentence in batch
    padded_batch_ids_and_masks = [
        (ids+[key2index[PAD_TOKEN]]*(max_batch_len - len(ids)),
        [1]*len(ids) + [0]*(max_batch_len - len(ids))) for ids in batch_ids
    ]
    padded_batch_ids, masks = list(zip(*padded_batch_ids_and_masks))
    batch["padded_batch_ids"] = padded_batch_ids
    batch["masks"] = masks
    return batch

def clean_token(token, vocab):
    token = token.lower()
    if token in vocab:
        final_token = token
    else:
        token_cand1 = re.sub("[^a-z0-9-]", "", token)
        if token_cand1 in vocab:
            final_token = token_cand1
        else:
            token_cand2 = token.split("-")[-1]
            if token_cand2 in vocab:
                final_token =  token_cand2
            else:
                final_token = token
    return final_token


def get_brown_old():
    id2sent_raw = dict()
    id2sent_clean = dict()
    id_count = 0
    model = spacy.load("en_core_web_sm")
    nltk.download('brown') # get data if not already available

    for sentence in tqdm(brown.sents()):
        doc = model(" ".join(sentence))

        cleaned_tokens = []
        for token in doc:
            cleaned_token = clean_token(token.text)
            if cleaned_token != "":
                cleaned_tokens.append(cleaned_token)

        if len(cleaned_tokens) < 2:
            continue

        raw_sentence = " ".join([token.text for token in doc])
        clean_sentence = " ".join(cleaned_tokens)

        id2sent_raw[f"brown.{id_count}"] = raw_sentence
        id2sent_clean[f"brown.{id_count}"] = clean_sentence
        id_count+=1
    
    id2sent_raw_df = pd.DataFrame.from_dict({
        "id": id2sent_raw.keys(),
        "sentence": id2sent_raw.values()
    })
    id2sent_raw_df.to_csv("./corpora/brown/raw_brown.csv", index=False)

    id2sent_clean_df = pd.DataFrame.from_dict({
        "id": id2sent_clean.keys(),
        "sentence": id2sent_clean.values()
    })
    id2sent_clean_df.to_csv("./corpora/brown/clean_brown.csv", index=False)


def get_brown():
    vocab = load_embeddings("glove-wiki-gigaword-300").index_to_key
    id2sent_raw = dict()
    id2sent_clean = dict()
    id_count = 0
    nltk.download('brown') # get data if not already available
    for sentence in tqdm(brown.sents()):
        cleaned_tokens = []
        for token in sentence:
            cleaned_token = clean_token(token, vocab)
            if cleaned_token != "":
                cleaned_tokens.append(cleaned_token)
        raw_sentence = " ".join(sentence)
        clean_sentence = " ".join(cleaned_tokens)
        id2sent_raw[f"brown.{id_count}"] = raw_sentence
        id2sent_clean[f"brown.{id_count}"] = clean_sentence
        id_count += 1
    id2sent_raw_df = pd.DataFrame.from_dict({
        "id": id2sent_raw.keys(),
        "sentence": id2sent_raw.values()
    })
    id2sent_raw_df.to_csv("./corpora/raw_brown_glove.csv", index=False)
    id2sent_clean_df = pd.DataFrame.from_dict({
        "id": id2sent_clean.keys(),
        "sentence": id2sent_clean.values()
    })
    id2sent_clean_df.to_csv("./corpora/clean_brown_glove.csv", index=False)


def split_brown(split):
    train, val, test = split
    full_data = pd.read_csv("./corpora/clean_brown_glove.csv")
    shuffled_data = full_data.sample(frac=1)
    train_end_index = math.floor(len(shuffled_data)*train)
    val_end_index = math.floor(len(shuffled_data)*(train+val))
    train_data = shuffled_data[:train_end_index]
    val_data = shuffled_data[train_end_index:val_end_index]
    test_data = shuffled_data[val_end_index:]
    train_data.to_csv("./corpora/train_brown.csv", index=False)
    val_data.to_csv("./corpora/val_brown.csv", index=False)
    test_data.to_csv("./corpora/test_brown.csv", index=False)


def get_dundee_sentences():
    vocab = load_embeddings("glove-wiki-gigaword-300").index_to_key
    id2sent = dict()
    id_count = 0
    tokens = []
    with open("./text_files/dundee.txt", "r") as f:
        for line in f:
            for token in line.split():
                tokens.append(clean_token(token, vocab))
            id2sent[id_count] = " ".join(tokens)
            id_count += 1
            tokens = []
    id2sent_df = pd.DataFrame.from_dict({
        "id": id2sent.keys(),
        "sentence": id2sent.values()
    })
    id2sent_df.to_csv("./corpora/dundee.csv", index=False)

def get_natural_stories_sentences():
    vocab = load_embeddings("glove-wiki-gigaword-300").index_to_key
    id2sent = dict()
    id_count = 0
    tokens = []
    with open("./text_files/natural_stories.txt", "r") as f:
        for line in f:
            for token in line.split():
                tokens.append(clean_token(token, vocab))
            id2sent[id_count] = " ".join(tokens)
            id_count += 1
            tokens = []
    id2sent_df = pd.DataFrame.from_dict({
        "id": id2sent.keys(),
        "sentence": id2sent.values()
    })
    id2sent_df.to_csv("./corpora/natural_stories.csv", index=False)

def get_alice_sentences():
    vocab = load_embeddings("glove-wiki-gigaword-300").index_to_key
    df = pd.read_csv('./csvs/AliceChapterOne-EEG.csv')
    id2sent = dict()
    id_count = 0
    tokens = []
    prev_sent_num = 1
    for index, row in df.iterrows():
        if row["Sentence"] != prev_sent_num:
            id2sent[id_count] = " ".join(tokens)
            id_count += 1
            tokens = []
            prev_sent_num = row["Sentence"]
        tokens.append(clean_token(row["Word"], vocab))
    id2sent[id_count] = " ".join(tokens)
    id2sent_df = pd.DataFrame.from_dict({
        "id": id2sent.keys(),
        "sentence": id2sent.values()
    })
    id2sent_df.to_csv("./corpora/alice.csv", index=False)