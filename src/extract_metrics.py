import pdb

import numpy as np
import torch
import pandas as pd
from torch.nn.functional import cosine_similarity
from gensim.models import word2vec
from models import AutoTPC
from collections import defaultdict
from tqdm import tqdm
from argparse import ArgumentParser
import os
from pathlib import Path
from datasets import load_dataset, disable_caching
from utils import load_embeddings, encode_batch, UNK_TOKEN, PAD_TOKEN
from tokenizers import Tokenizer, models, pre_tokenizers
import json




def kl_divergence(model, b, mode="bayesian-surprise"):
    def log_prob(x):
        error_y = model.y[:,b] - (model.Wxy@model.g(x) + model.by[:,0])
        error_x = x - model.pred_x[:,b]
        energy = 0.5 * (
            torch.dot((error_x * model.px.squeeze()), error_x) + 
            torch.dot((error_y * model.py.squeeze()), error_y) + 
            torch.logdet(torch.diag(model.cx)) + 
            torch.logdet(torch.diag(model.cy)) +
            model.x_size*torch.log(torch.Tensor([2*torch.pi]).to(model.device)) + 
            model.y_size*torch.log(torch.Tensor([2*torch.pi]).to(model.device)) 
        )
        return energy

    posterior_precision = torch.func.hessian(log_prob)(torch.clone(model.x[:,b]).requires_grad_(True)).detach()
    posterior = torch.distributions.multivariate_normal.MultivariateNormal(model.x[:,b], precision_matrix=posterior_precision)

    prior_precision = torch.diag(model.px.squeeze()) 
    prior = torch.distributions.multivariate_normal.MultivariateNormal(model.pred_x[:,b], precision_matrix=prior_precision)
    
    kl_div = torch.distributions.kl.kl_divergence(posterior, prior)

    output = {
        "kl_div": kl_div.item(),
        "posterior_cov": torch.linalg.inv(posterior_precision)
    }
    return output

def update_batch_metrics(batch_metrics, model, tokenizer, ids, gradient, top_down, bottom_up, energy, times):
    normed_gradient = np.linalg.norm(gradient, axis=1)
    for b in range(model.batch_size):
        token = tokenizer.decode([ids[b]])
        if token != PAD_TOKEN:
            amplitude = np.max(normed_gradient[:times[b],b])
            arclength = np.sum(normed_gradient[:times[b],b] * model.delta_t_x)
            li_bottom_up = np.sum(np.sum(bottom_up[:times[b],:,b]*gradient[:times[b],:,b], axis=1) * model.delta_t_x)
            li_top_down = np.sum(np.sum(top_down[:times[b],:,b]*gradient[:times[b],:,b], axis=1) * model.delta_t_x)
            wnorm_xy = torch.linalg.norm(model.batched_delta_Wxy[b]).item()
            wnorm_xx = torch.linalg.norm(model.batched_delta_Wxx[b]).item()
            bnorm_x = torch.linalg.norm(model.batched_bx[b]).item()
            bnorm_y = torch.linalg.norm(model.batched_by[b]).item()
            cosdist_y = 1.0 - cosine_similarity((model.Wxy@model.g(model.pred_x[:,b]) + model.by[:,0]), model.y[:,b], dim=0).item()

            cosdist_prior_v_posterior = 1.0 - cosine_similarity(model.pred_x[:,b], model.x[:,b], dim=0).item() # prior x_k vs posterior x_k
            cosdist_posterior_v_posterior = 1.0 - cosine_similarity(model.prev_x[:,b], model.x[:,b], dim=0).item() # posterior x_{k-1} vs posterior x_k
            
            z = torch.linalg.lstsq(model.Wxy, (model.y[:,b] - model.by[:,0])).solution
            if model.g_type == "linear":
                x_likelihood = z
            elif model.g_type == "leaky_relu":
                x_likelihood = torch.where(z >= 0, z, z/model.negative_slope)
            cosdist_likelihood_v_posterior = 1.0 - cosine_similarity(x_likelihood, model.x[:,b], dim=0).item() # likelihood x_k vs posterior x_k
                
            output = kl_divergence(model, b, mode="bayesian-surprise")
            bayesian_surprise = output["kl_div"]
            free_energy = (
                energy[b] -
                0.5*(model.x_size*torch.log(torch.Tensor([2*torch.pi]).to(model.device)) + torch.logdet(output["posterior_cov"]))
            ).item()

            iters = times[b].item()

            batch_metrics[b].append({
                "free_energy": free_energy,
                "arclength": arclength,
                "amplitude": amplitude,
                "li_top_down": li_top_down,
                "li_bottom_up": li_bottom_up,
                "wnorm_xy": wnorm_xy,
                "wnorm_xx": wnorm_xx,
                "bnorm_x": bnorm_x,
                "bnorm_y": bnorm_y,
                "cosdist_y": cosdist_y,
                "cosdist_prior_v_posterior": cosdist_prior_v_posterior,
                "cosdist_posterior_v_posterior": cosdist_posterior_v_posterior,
                "cosdist_likelihood_v_posterior": cosdist_likelihood_v_posterior,
                "bayesian_surprise": bayesian_surprise,
                "iters": times[b].item()+1,
                "normed_gradient": normed_gradient[:iters, b],
                "model_token": token
            })
            for k in range(model.K):
                batch_metrics[b][-1][f"wnorm_yx{k}"] = torch.linalg.norm(model.batched_delta_Wyx[k][b]).item()

            
    return batch_metrics


def update_full_metrics(metrics, batch_metrics):
    for i in sorted(list(batch_metrics.keys())):
        metrics.extend(batch_metrics[i])
    return metrics


def compute_metrics(dataset, model, embeddings, tokenizer):
    metrics = []
    dataset = dataset["train"].batch(batch_size=model.batch_size)
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

        batch_metrics = defaultdict(list)

        # iterate over token ids and masks
        for k, (ids, masks) in enumerate(zip(batch_ids, batch_masks)):
            model.y = torch.Tensor(embeddings[ids]).to(model.device).T #shape=(embd_dim, batch_size)
            model.y = model.y/torch.linalg.norm(model.y+1e-12, dim=0)
            model.mask = masks.to(model.device) #shape=batch_size

            # If want to start inference at the predicted next state
            if model.start_at_prediction:
                model.predict()
                model.x = torch.clone(model.pred_x)

            times = torch.ones(model.batch_size).to(model.device) * float('inf')
            gradient = []
            bottom_up = []
            top_down = []
            # Infer the hidden state
            #pdb.set_trace()
            for t in range(model.inf_iters):
                model.step(t)
                gradient.append(model.delta_x.cpu().numpy())
                bottom_up.append(model.bottom_up.cpu().numpy())
                top_down.append(model.top_down.cpu().numpy())

                # for updating when convergence was reached for each item in batch
                new_times = (torch.linalg.norm(model.delta_x, dim=0) < model.threshold)*t 
                new_times = torch.where(new_times == 0, float('inf'), new_times)
                times = torch.minimum(times, new_times)
                if torch.sum(times == float('inf')) == 0:
                    break
            times = times.to(torch.int64)

            # Compute end of inference energy
            energy = model.compute_energy(comp_metrics=True).cpu().numpy()

            # Update model parameters
            model.update_weights()

            #pdb.set_trace()
            gradient = np.array(gradient)
            top_down = np.array(top_down)
            bottom_up = np.array(bottom_up)
            
            batch_metrics = update_batch_metrics(batch_metrics, model, tokenizer, ids, gradient, top_down, bottom_up, energy, times)

            # Update previous state and observation
            model.update_prev()
            model.reset(reset_state=False, reset_error=True)
        #pdb.set_trace()
        metrics = update_full_metrics(metrics, batch_metrics)
        
    metrics_df = pd.DataFrame(metrics)

    return metrics_df
        

def main(args):
    print(vars(args))
    print("Initializing model...")
    model = AutoTPC()
    model.load_parameters(args.model_path)
    model.eval_mode(args)
    print(model)

    print("done\nLoading embeddings and dataset...", end="")
    if args.embedding_path == "glove":
        embeddings_path = "glove-wiki-gigaword-300"
    embeddings = load_embeddings(embeddings_path)
    
    tokenizer = Tokenizer(models.WordLevel(embeddings.key_to_index, UNK_TOKEN))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    dataset = load_dataset("csv", data_files=args.corpus_path)
    print("done\nStarting training...")

    # Save the args for this run
    Path(args.savedir).mkdir(parents=True, exist_ok=True)
    with open(f"{args.savedir}/args.json", "w") as f:
        json.dump(vars(args), f)

    df = compute_metrics(
        dataset=dataset, 
        model=model, 
        embeddings=embeddings, 
        tokenizer=tokenizer,
    )
    print("done\nSaving parquet...", end="")

    Path(args.savedir).mkdir(exist_ok=True)
    df.to_parquet(os.path.join(args.savedir, "metrics_autotpc.parquet"))
    print("done")


if __name__ == "__main__":
    disable_caching()
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--max_inf_iters", type=int, default=2000) #300
    parser.add_argument("--K", type=int, default=0)
    parser.add_argument("--delta_t_x", type=float, default=1e-2) #3e-2
    parser.add_argument("--threshold", type=float, default=1e-3)
    parser.add_argument("--start_at_prediction", action="store_true")
    parser.add_argument("--error_units", action="store_true")
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--corpus_path")
    parser.add_argument("--savedir")
    parser.add_argument("--embedding_path", type=str, default="glove")
    arguments = parser.parse_args()
    main(arguments)
