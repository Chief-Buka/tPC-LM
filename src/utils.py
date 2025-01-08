import math
import re
import nltk
from nltk.corpus import brown
import spacy
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
    #TODO: zero mask the [UNK] tokens
    # separate into 2d lists of ids and masks
    padded_batch_ids, masks = list(zip(*padded_batch_ids_and_masks))
    batch["padded_batch_ids"] = padded_batch_ids
    batch["masks"] = masks
    return batch

def clean_token(token):
    token = re.sub("n't", "not", token)
    token = re.sub("'m", "am", token)
    token = re.sub("'re", "are", token)
    token = re.sub("[^a-zA-Z0-9]", "", token).lower()
    return token


def get_brown():
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
            
def split_brown(split):
    train, val, test = split

    full_data = pd.read_csv("./corpora/brown/clean_brown.csv")
    shuffled_data = full_data.sample(frac=1)

    train_end_index = math.floor(len(shuffled_data)*train)
    val_end_index = math.floor(len(shuffled_data)*(train+val))

    train_data = shuffled_data[:train_end_index]
    val_data = shuffled_data[train_end_index:val_end_index]
    test_data = shuffled_data[val_end_index:]

    train_data.to_csv("./corpora/brown/train_brown.csv", index=False)
    val_data.to_csv("./corpora/brown/val_brown.csv", index=False)
    test_data.to_csv("./corpora/brown/test_brown.csv", index=False)


def get_dundee():
    df = pd.read_csv("metrics_full.csv")
    id2sent = dict()
    id_count = 0
    tokens = []
    for index, row in df.iterrows():
        if row["sentence_num"] != id_count:
            id2sent[id_count] = " ".join(tokens)
            id_count += 1
            tokens = []
        tokens.append(clean_token(row["original_tokens"]))

    id2sent[id_count] = " ".join(tokens)
        
    id2sent_df = pd.DataFrame.from_dict({
        "id": id2sent.keys(),
        "sentence": id2sent.values()
    })
    id2sent_df.to_csv("./corpora/dundee/dundee.csv", index=False)


def get_appleseed():
    all_tokens = []
    segment_nums = []
    sent_nums = []
    token_nums = []
    sent_num = 0
    token_num = 0
    id2sent = dict()
    segments = [f"{i+1}" for i in range(11)] + ['11b']
    for seg in segments:
        with open(f"./appleseed_text/appleseed{seg}.txt", "r") as f:
            for line in f.readlines():
                sent = []
                tokens = line.split()
                if len(tokens) > 0:
                    for token in tokens:
                        all_tokens.append(clean_token(token))
                        sent.append(all_tokens[-1])
                        segment_nums.append(seg)
                        sent_nums.append(sent_num)
                        token_nums.append(token_num)
                        token_num += 1
                    sent_num += 1
                    id2sent[sent_num] = " ".join(sent)

    id2sent_df = pd.DataFrame.from_dict({
        "id": id2sent.keys(),
        "sentence": id2sent.values()
    })

    tokens_df = pd.DataFrame({
        "tokens": all_tokens,
        "segment_num": segment_nums,
        "sentence_num": sent_nums,
        "token_num": token_nums
    })

    id2sent_df.to_csv("./corpora/appleseed.csv", index=False)
    tokens_df.to_csv("./appleseed_base.csv", index=False)


def get_natural_stories():
    df = pd.read_csv("natural_stories_rts.csv")
    id2sent = dict()
    id_count = 0
    tokens = []
    for index, row in df.iterrows():
        tokens.append(clean_token(row["token"]))
        if "." in row["token"]:
            id2sent[id_count] = " ".join(tokens)
            id_count += 1
            tokens = []

    id2sent_df = pd.DataFrame.from_dict({
        "id": id2sent.keys(),
        "sentence": id2sent.values()
    })
    id2sent_df.to_csv("./corpora/natural_stories.csv", index=False)
