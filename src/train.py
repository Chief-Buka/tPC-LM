from models import AutoTPC
import argparse
import gensim.downloader
import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers
from datasets import load_dataset
import torch

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

    # mac sentence length for the batch
    max_batch_len = max(len(ids) for ids in batch_ids)

    # tuple (ids, mask) per sentence for ids with padding to longest sentence in batch
    padded_batch_ids_and_masks = [
        (ids+[key2index[PAD_TOKEN]]*(max_batch_len - len(ids)),
        [1]*len(ids) + [0]*(max_batch_len - len(ids))) for ids in batch_ids
    ]

    # separate into 2d lists of ids and masks
    padded_batch_ids, masks = list(zip(*padded_batch_ids_and_masks))

    batch["padded_batch_ids"] = np.array(padded_batch_ids).T 
    batch["masks"] = torch.BoolTensor(masks).T

    return batch



def run_helper(model, data, embeddings, tokenizer, mode):
    # Shuffle, batch and create features for the dataset
    data = data.shuffle().batch(batch_size=model.batch_size)
    data = data.map(
        encode_batch, 
        batched=False, 
        fn_kwargs={"tokenizer": tokenizer, "key2index": embeddings.key_to_index}
    )

    # iterate over batches
    for batch_num, batch in tqdm(enumerate(train_data)):
        model.reset(reset_state=True, reset_error=True)
        model.set_random_prev()

        # iterate over token ids and masks
        for k, (ids, mask) in enumerate(zip(batch["padded_batch_ids"], batch["masks"])):
            model.y = torch.Tensor(embeddings[ids]).to(model.device) #shape=(batch_size,embd_dim)
            model.mask = torch.BoolTensor(mask).to(model.device) #shape=batch_size

            # Infer the hidden statee
            for _ in range(model.inf_iters):
                model.step()

            # Compute end of inference energy
            energy += model.compute_energy().cpu().numpy()

            if mode == "train":
                # Update model parameters
                model.update_weights()

            # Update previous state and observation
            model.update_prev()

    if mode == "train":
        output = {"model": model, "energy": energy}
    elif mode == "val":
        output = {"energy": energy}


def run(model, embeddings, dataset, tokenizer, epochs, savedir):
    train_data = dataset["train"]
    val_data = dataset["val"]

    best_validation_energy = float('inf')

    for epoch in range(epochs):

        # train
        train_output = run_helper(model, data, embeddings, tokenizer, mode="train")
        print(f"Train Energy: {train_output['energy']:.3f}")
        model = train_output["model"]

        # validation
        val_output = run_helper(model, data, embeddings, tokenizer, mode="val")
        print(f"Validation Energy: {val_output['energy']:.3f}")

        # save model parameters if current parameters are better than current
        if val_output["energy"] < best_validation_energy:
            model.save_parameters(epoch, energy, savedir)





def main(args):
    print("Initializing model...")
    model = AutoTPC(
        autoregressive=args.autoregressive,
        y_size=args.y_size,
        x_size=args.x_size,
        batch_size=args.batch_size,
        delta_t_x=args.delta_t_x,
        delta_t_w=args.delta_t_w,
        inf_iters=args.inf_iters,
        error_units=args.error_units,
        device=args.device
    )

    print("Loading embeddings...")
    embeddings = load_embeddings(args.embedding_path)

    print("Loading dataset")
    tokenizer = Tokenizer(models.WordLevel(embeddings.key_to_index, UNK_TOKEN))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    dataset = load_dataset("csv", data_files={"train":args.train_data_path, "val":args.val_data_path})

    print("Starting training...")
    run(
        model=model, 
        embeddings=embeddings, 
        dataset=dataset,
        tokenizer=tokenizer,
        epochs=args.epochs,
        savedir=args.savedir
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--autoregressive", type=bool, required=True)
    parser.add_argument("--y_size", type=int, required=True)
    parser.add_argument("--x_size", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--inf_iters", type=int, required=True)
    parser.add_argument("--delta_t_x", type=float, required=True)
    parser.add_argument("--delta_t_w", type=float, required=True)
    parser.add_argument("--error_units", type=float, default=False)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--embedding_path", type=str, default="word2vec-google-news-300")
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--savedir", type=str, default="./")
    args = parser.parse_args()
    main(args)