import math
import re
import nltk
from nltk.corpus import brown
import spacy
import pandas as pd
from tqdm import tqdm
import gensim.downloader



def get_brown():
    id2sent_raw = dict()
    id2sent_clean = dict()
    id_count = 0
    model = spacy.load("en_core_web_sm")
    nltk.download('brown') # get data if not already available

    def clean_token(token):
        token = re.sub("n't", "not", token)
        token = re.sub("'m", "am", token)
        token = re.sub("[^a-zA-Z0-9]", "", token).lower()
        return token

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