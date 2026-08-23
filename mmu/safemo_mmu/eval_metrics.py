from __future__ import annotations

from collections import OrderedDict
from typing import Dict

import numpy as np
import torch

from closd.diffusion_planner.data_loaders.humanml.utils.metrics import (
    calculate_activation_statistics,
    calculate_diversity,
    calculate_frechet_distance,
    calculate_top_k,
    euclidean_distance_matrix,
)


def normalize_batch(batch):
    if not isinstance(batch, (list, tuple)):
        raise TypeError("Expected tuple/list batch")
    if len(batch) == 7:
        return batch
    if len(batch) >= 10:
        return batch[0], batch[1], batch[2], batch[3], batch[8], batch[5], batch[6]
    raise ValueError("Unexpected batch length {}".format(len(batch)))


def matching(eval_wrapper, loaders):
    scores, retrieval, activations = OrderedDict(), OrderedDict(), OrderedDict()
    for name, loader in loaders.items():
        embeddings_all, total, score_sum, top_k_count = [], 0, 0.0, 0
        with torch.no_grad():
            for raw in loader:
                words, pos, _, sent_lens, motions, motion_lens, _ = normalize_batch(raw)
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                    word_embs=words,
                    pos_ohot=pos,
                    cap_lens=sent_lens,
                    motions=motions,
                    m_lens=motion_lens,
                )
                distances = euclidean_distance_matrix(
                    text_embeddings.cpu().numpy(), motion_embeddings.cpu().numpy()
                )
                score_sum += float(distances.trace())
                top_k_count += calculate_top_k(
                    np.argsort(distances, axis=1), top_k=3
                ).sum(axis=0)
                total += int(text_embeddings.shape[0])
                embeddings_all.append(motion_embeddings.cpu().numpy())
        if total == 0:
            raise RuntimeError("Empty evaluator loader: {}".format(name))
        scores[name] = score_sum / total
        retrieval[name] = top_k_count / total
        activations[name] = np.concatenate(embeddings_all, axis=0)
    return scores, retrieval, activations


def fids(eval_wrapper, ground_truth_loader, activations):
    embeddings_all = []
    with torch.no_grad():
        for raw in ground_truth_loader:
            _, _, _, _, motions, motion_lens, _ = normalize_batch(raw)
            embeddings_all.append(
                eval_wrapper.get_motion_embeddings(motions=motions, m_lens=motion_lens)
                .cpu()
                .numpy()
            )
    reference = np.concatenate(embeddings_all, axis=0)
    reference_mu, reference_cov = calculate_activation_statistics(reference)
    result = OrderedDict()
    for name, values in activations.items():
        mu, cov = calculate_activation_statistics(values)
        result[name] = float(
            calculate_frechet_distance(reference_mu, reference_cov, mu, cov)
        )
    return result


def diversities(activations, pairs: int = 300):
    result = OrderedDict()
    for name, values in activations.items():
        times = max(1, min(pairs, len(values) - 1))
        result[name] = float(calculate_diversity(values, times))
    return result


def evaluate_once(eval_wrapper, ground_truth_loader, generated_loader, generated_name: str) -> Dict[str, object]:
    loaders = OrderedDict(
        [("ground_truth", ground_truth_loader), (generated_name, generated_loader)]
    )
    scores, retrieval, activations = matching(eval_wrapper, loaders)
    fid = fids(eval_wrapper, ground_truth_loader, activations)
    diversity = diversities(activations)
    r = np.asarray(retrieval[generated_name], dtype=float)
    return {
        "FID": fid[generated_name],
        "Diversity": diversity[generated_name],
        "R@1": float(r[0]),
        "R@2": float(r[1]),
        "R@3": float(r[2]),
        "Matching Score": float(scores[generated_name]),
        "ground_truth": {
            "FID": fid["ground_truth"],
            "Diversity": diversity["ground_truth"],
            "R@1": float(retrieval["ground_truth"][0]),
            "R@2": float(retrieval["ground_truth"][1]),
            "R@3": float(retrieval["ground_truth"][2]),
            "Matching Score": float(scores["ground_truth"]),
        },
    }
