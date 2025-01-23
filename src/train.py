from models import AutoTPC
import argparse
import gensim.downloader
import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers
from datasets import load_dataset, disable_caching, disable_progress_bar
import torch
from tqdm import tqdm
from utils import load_embeddings, encode_batch, UNK_TOKEN
import json
import pdb
from pathlib import Path 




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

        batch_energy = []

        # iterate over token ids and masks
        #pdb.set_trace()
        for k, (ids, masks) in enumerate(zip(batch_ids, batch_masks)):
            model.y = torch.Tensor(embeddings[ids]).to(model.device).T #shape=(embd_dim, batch_size)
            model.y = model.y/torch.linalg.norm(model.y+1e-12, dim=0)
            model.mask = masks.to(model.device) #shape=batch_size

            # if model is linear can solve directly, faster than iteration but is unstable for large matrices
            if model.g_type == "linear" and model.x_size <= 1400:
                model.solve()
            else:
            # Infer the hidden state
                for t in range(model.inf_iters):
                    model.step(t)

            # Compute end of inference energy
            batch_energy.append(model.compute_energy().cpu().numpy())

            if mode == "train":
                # Update model parameters
                model.update_weights()

            # Update previous state and observation
            model.update_prev()

        # compute avg energy per token or this batch
        batch_energy = np.array(batch_energy) #seqlen x batchsize
        energy += np.mean(batch_energy.sum(axis=0)/np.sum(batch_energy!=0,axis=0))

    energy = energy / len(data) #avg loss per token over the epoch

    if mode == "train":
        output = {"model": model, "energy": energy}
    elif mode == "val":
        output = {"energy": energy}

    return output


def run(model, embeddings, dataset, tokenizer, args):
    train_data = dataset["train"]
    val_data = dataset["val"]

    train_losses = []
    val_losses = []
    best_validation_energy = float('inf')

    for epoch in range(model.epoch+1, model.epoch+args.epochs):

        print(f"EPOCH {epoch}")

        # train
        train_output = run_helper(model, train_data, embeddings, tokenizer, mode="train")
        train_metrics = [
            f"Train Energy {epoch}: {train_output['energy']:.4f}",
        ]
        print(" ".join(train_metrics))
        train_losses.append(train_output['energy'])
        model = train_output["model"]
        model.batch_size = args.batch_size

        # validation
        model.batch_size = args.batch_size
        val_output = run_helper(model, val_data, embeddings, tokenizer, mode="val")
        val_metrics = [
            f"Validation Energy {epoch}: {val_output['energy']:.4f}",
        ]
        print(" ".join(val_metrics))
        val_losses.append(val_output['energy'])
        model.batch_size = args.batch_size

        # save model parameters if current parameters are better than current
        if val_output["energy"] < best_validation_energy:
            model.save_parameters(epoch, val_output["energy"], args.savedir)
            best_validation_energy = val_output["energy"]

    output = {
        "train_losses": np.array(train_losses), 
        "val_losses": np.array(val_losses),
    }

    return output


def main(args):
    print("---------- ARGS ----------")
    print("\n".join([f"{k}: {v}" for k,v in vars(args).items()]))
    print("Initializing model...")

    if args.model_path:
        model = AutoTPC()
        model.load_parameters(args.model_path)
    else:
        model = AutoTPC(
            K=args.K,
            y_size=args.y_size, 
            x_size=args.x_size, 
            batch_size=args.batch_size, 
            delta_t_x=args.delta_t_x, 
            delta_t_w=args.delta_t_w, 
            inf_iters=args.inf_iters, 
            error_units=args.error_units, 
            device=args.device,
            f_type=args.f_type,
            g_type=args.g_type,
            h_type=args.h_type
        )
    print("done")
    print("---------- MODEL ----------")
    print(model)

    print("\nLoading embeddings and dataset...", end="")
    if args.embedding_path == "word2vec":
        embeddings_path = "word2vec-google-news-300"
    elif args.embedding_path == "glove":
        embeddings_path = "glove-wiki-gigaword-300"
    embeddings = load_embeddings(embeddings_path)

    tokenizer = Tokenizer(models.WordLevel(embeddings.key_to_index, UNK_TOKEN))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    dataset = load_dataset("csv", data_files={"train":args.train_data_path, "val":args.val_data_path})
    print("done\nStarting training...")

    # Save the args for this run
    Path(args.savedir).mkdir(parents=True, exist_ok=True)
    with open(f"{args.savedir}/args.json", "w") as f:
        json.dump(vars(args), f)

    output = run(
        model=model, 
        embeddings=embeddings, 
        dataset=dataset,
        tokenizer=tokenizer,
        args=args
    )
    np.save(f"{args.savedir}/train_losses1.npy", output["train_losses"])
    np.save(f"{args.savedir}/val_losses1.npy", output["val_losses"])

    print("Finished training!")

if __name__ == "__main__":
    disable_caching()
    disable_progress_bar()
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument('--f_type', type=str, default='linear')
    parser.add_argument('--g_type', type=str, default='linear')
    parser.add_argument('--h_type', type=str, default='linear')
    parser.add_argument("--model_path", type=str, default=None)
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
    arguments = parser.parse_args()
    main(arguments)