import pdb

import numpy as np
import torch
import pandas as pd
from torch.nn.functional import cosine_similarity
from gensim.models import word2vec
from models import TPC
from collections import defaultdict
from tqdm import tqdm
from argparse import ArgumentParser
import os
from pathlib import Path
from datasets import load_dataset
from utils import load_embeddings, encode_batch, UNK_TOKEN
from tokenizers import Tokenizer, models, pre_tokenizers


METRICS = [
    "free_energy",
    "arclength",
    "li_bottom_up",
    "li_top_down",
    "wnorm_xy",
    "wnorm_xx",
    "cosdist_x",
    "cosdist_y",
    "kl_div",
    "iters",
    "amplitude"
]
METRICS_AR = METRICS + ["wnorm_yx"]
THRESHOLD = 1e-2
MAX_INF_ITERS = 600


# TODO: this keeps returning inf
def kl_divergence(model):
    prior_precision = torch.unsqueeze(1.0/model.cx, 1)*torch.eye(model.x_size).to(model.device)
    prior_variance = torch.unsqueeze(model.cx, 1)*torch.eye(model.x_size).to(model.device)
    likelihood_precision = torch.unsqueeze(1.0/model.cy, 1)*torch.eye(model.y_size).to(model.device)
    jacobian_de_dx = -torch.einsum('oi,ib->oi', model.Wxy, model.g_deriv(model.x))
    posterior_precision = prior_precision + jacobian_de_dx.T@likelihood_precision@jacobian_de_dx
    posterior_variance = torch.linalg.inv(posterior_precision)
    kl_div = 0.5*(
        model.error_x.T@prior_precision@model.error_x + 
        torch.trace(prior_precision@posterior_variance) +
        -torch.log(torch.det(posterior_variance)/torch.det(prior_variance)) + 
        -model.x.shape[0]
    )
    return kl_div
    


def compute_metrics(dataset, model, embeddings, tokenizer, metrics):
    metrics_records = []
    normed_gradients = []
    dataset = dataset["train"].shuffle().batch(batch_size=model.batch_size)
    dataset = dataset.map(
        encode_batch, 
        batched=False, 
        fn_kwargs={"tokenizer": tokenizer, "key2index": embeddings.key_to_index},
    )

    # iterate over batches
    for batch_num, batch in tqdm(enumerate(dataset)):

        batch_ids = np.array(batch["padded_batch_ids"]).T #(max_sent_len, batch_size)
        batch_masks = torch.BoolTensor(batch["masks"]).T #(max_sent_len, batch_size)
        model.batch_size = batch_ids.shape[1]
        model.reset(reset_state=True, reset_error=True)
        model.set_random_prev()


        # iterate over token ids and masks
        for k, (ids, masks) in enumerate(zip(batch_ids, batch_masks)):
            model.y = torch.Tensor(embeddings[ids]).to(model.device).T #shape=(embd_dim, batch_size)
            model.mask = masks.to(model.device) #shape=batch_size

            # If want to start inference at the predicted next state
            # if start_at_prediction:
            #     model.predict()
            #     model.x = model.pred_x

            gradient = []
            bottom_up = []
            top_down = []
            # Infer the hidden statee
            for t in range(model.inf_iters):
                model.step(t)
                gradient.append(model.delta_x.cpu().numpy().squeeze())
                bottom_up.append(model.bottom_up.cpu().numpy().squeeze())
                top_down.append(model.top_down.cpu().numpy().squeeze())
                if torch.linalg.norm(model.delta_x.squeeze()) <= THRESHOLD:
                    break

            # Compute end of inference energy
            energy = model.compute_energy().cpu().numpy()

            # Update model parameters
            model.update_weights()

            #pdb.set_trace()
            gradient = np.array(gradient)
            top_down = np.array(top_down)
            bottom_up = np.array(bottom_up)
            normed_gradient = np.linalg.norm(gradient, axis=1)
            amplitude = np.max(normed_gradient)
            arclength = np.sum(normed_gradient * model.delta_t_x)
            li_bottom_up = np.sum(np.sum(bottom_up*gradient, axis=1) * model.delta_t_x)
            li_top_down = np.sum(np.sum(top_down*gradient, axis=1) * model.delta_t_x)
            wnorm_xx = torch.linalg.norm(model.delta_Wxx).cpu().numpy()
            wnorm_xy = torch.linalg.norm(model.delta_Wxy).cpu().numpy()
            if "wnorm_yx" in metrics:
                wnorm_yx = torch.linalg.norm(model.delta_Wyx).cpu().numpy()
            else:
                wnorm_yx = None
            cosdist_x = 1.0 - cosine_similarity(model.pred_x.T, model.x.T).cpu().numpy()
            cosdist_y = 1.0 - cosine_similarity((model.Wxy@model.g(model.pred_x) + model.by).T, model.y.T).cpu().numpy()
            # need to fix this keeps returning inf
            kl_div = kl_divergence(model)
            token = tokenizer.decode(ids)

            #TODO: save and calculate the commented metrics
            metrics_records.append({
                "energy": energy.item(),
                "arclength": arclength,
                "amplitude": amplitude,
                "li_top_down": li_top_down,
                "li_bottom_up": li_bottom_up,
                "wnorm_xx": wnorm_xx.item(),
                "wnorm_xy": wnorm_xy.item(),
                "wnorm_yx": wnorm_yx.item(),
                "cosdist_x": cosdist_x.item(),
                "cosdist_y": cosdist_y.item(),
                "kl_div": kl_div.item(), # kldiv1
                #kldiv2 = kldiv between where state started (instead of prediction) and where it end
                "iters": t+1,
                "token": token
            })
            normed_gradients.append({
                "norm_grad": normed_gradient,
                "token": token
            })

            # Update previous state and observation
            model.update_prev()

    return {
        "metrics": pd.DataFrame(metrics_records).set_index("token"), 
        "norm_grad": pd.DataFrame(normed_gradients)
    }


def main(args):
    print("Loading model and computing metrics...")
    model = TPC()
    model.load_parameters(args.model_path)
    model.batch_size = 1
    model.delta_t_w = 0.0
    model.inf_iters = MAX_INF_ITERS
    model.error_units = True
    print(model)
    embeddings = load_embeddings("word2vec-google-news-300")
    tokenizer = Tokenizer(models.WordLevel(embeddings.key_to_index, UNK_TOKEN))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    dataset = load_dataset("csv", data_files=args.corpus_path)
    metrics = METRICS_AR if model.autoregressive else METRICS

    df = compute_metrics(
        dataset=dataset, model=model, 
        embeddings=embeddings, tokenizer=tokenizer,
        metrics=metrics
    )
    print("done\nSaving csv...", end="")

    Path(args.savedir).mkdir(exist_ok=True)
    if model.autoregressive:
        df['metrics'].to_csv(os.path.join(args.savedir, "metrics_autotpc.csv"))
        df['norm_grad'].to_parquet(os.path.join(args.savedir, 'norm_grads_autotpc.parquet'))
    else:
        df['metrics'].to_csv(os.path.join(args.savedir, "metrics_tpc.csv"))
        df['norm_grad'].to_parquet(os.path.join(args.savedir, 'norm_grads_autotpc.parquet'))
    print("done")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--corpus_path")
    parser.add_argument("--savedir")
    parser.add_argument("--embedding_path", type=str, default="word2vec-google-news-300")
    arguments = parser.parse_args()
    main(arguments)
