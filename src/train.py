from models import TPC
import argparse
import gensim.downloader
import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers
from datasets import load_dataset
import torch
from tqdm import tqdm
from utils import load_embeddings, encode_batch, UNK_TOKEN
import pdb




def run_helper(model, data, embeddings, tokenizer, mode):
    # Shuffle, batch and create features for the dataset
    data = data.shuffle().batch(batch_size=model.batch_size)
    data = data.map(
        encode_batch, 
        batched=False, 
        fn_kwargs={"tokenizer": tokenizer, "key2index": embeddings.key_to_index},

    )

    energy = 0.0

    # iterate over batches
    for batch_num, batch in tqdm(enumerate(data)):

        batch_ids = np.array(batch["padded_batch_ids"]).T #(max_sent_len, batch_size)
        batch_masks = torch.BoolTensor(batch["masks"]).T #(max_sent_len, batch_size)

        model.batch_size = batch_ids.shape[1]
        model.reset(reset_state=True, reset_error=True)
        model.set_random_prev()

        batch_energy = 0.0

        # iterate over token ids and masks
        for k, (ids, masks) in enumerate(zip(batch_ids, batch_masks)):
            model.y = torch.Tensor(embeddings[ids]).to(model.device).T #shape=(embd_dim, batch_size)
            model.mask = masks.to(model.device) #shape=batch_size

            # Infer the hidden statee
            pdb.set_trace()
            for t in range(model.inf_iters):
                model.step(t)

            # Compute end of inference energy
            batch_energy += model.compute_energy().cpu().numpy()

            if mode == "train":
                # Update model parameters
                model.update_weights()

            # Update previous state and observation
            model.update_prev()
        energy += batch_energy
        #print(batch_energy)

    if mode == "train":
        output = {"model": model, "energy": energy}
    elif mode == "val":
        output = {"energy": energy}

    return output


def run(model, embeddings, dataset, tokenizer, args):
    train_data = dataset["train"]
    val_data = dataset["val"]

    best_validation_energy = float('inf')

    for epoch in range(args.epochs):

        # train
        train_output = run_helper(model, train_data, embeddings, tokenizer, mode="train")
        print(f"Train Energy: {train_output['energy']:.3f}")
        model = train_output["model"]
        model.batch_size = args.batch_size

        # validation
        model.batch_size = args.batch_size
        val_output = run_helper(model, val_data, embeddings, tokenizer, mode="val")
        print(f"Validation Energy: {val_output['energy']:.3f}")
        model.batch_size = args.batch_size

        # save model parameters if current parameters are better than current
        if val_output["energy"] < best_validation_energy:
            model.save_parameters(epoch, val_output["energy"], args.savedir)
            best_validation_energy = val_output["energy"]





def main(args):
    print("Initializing model...", end="")
    model = TPC(
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
    print(model)

    print("done\nLoading embeddings...", end="")
    if args.embedding_path == "word2vec":
        embeddings_path = "word2vec-google-news-300"
    elif args.embedding_path == "glove":
        embeddings_path = "glove-wiki-gigaword-300"
    embeddings = load_embeddings(embeddings_path)

    print("done\nLoading dataset...", end="")
    tokenizer = Tokenizer(models.WordLevel(embeddings.key_to_index, UNK_TOKEN))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    dataset = load_dataset("csv", data_files={"train":args.train_data_path, "val":args.val_data_path})
    print("done\nStarting training...")

    run(
        model=model, 
        embeddings=embeddings, 
        dataset=dataset,
        tokenizer=tokenizer,
        args=args
    )
    print("Finished training!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--autoregressive", action="store_true")
    parser.add_argument("--y_size", type=int, required=True)
    parser.add_argument("--x_size", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--inf_iters", type=int, required=True)
    parser.add_argument("--delta_t_x", type=float, required=True)
    parser.add_argument("--delta_t_w", type=float, required=True)
    parser.add_argument("--error_units", action='store_true')
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--embedding_path", type=str, default="glove")
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--savedir", type=str, default="../results")
    parser.add_argument("--normalize", action='store_true')
    arguments = parser.parse_args()
    main(arguments)