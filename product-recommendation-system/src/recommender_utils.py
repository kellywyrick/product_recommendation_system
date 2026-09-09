"""Utility functions for the recommendation-system portfolio project."""
from collections import defaultdict
import numpy as np

def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """Compute macro-averaged precision@k, recall@k, and F1@k from Surprise predictions."""
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
    precisions, recalls = [], []
    for ratings in user_est_true.values():
        ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum(true_r >= threshold for _, true_r in ratings)
        top_k = ratings[:k]
        n_rec = sum(est >= threshold for est, _ in top_k)
        n_rel_rec = sum((true_r >= threshold) and (est >= threshold) for est, true_r in top_k)
        precisions.append(n_rel_rec / n_rec if n_rec else 0.0)
        recalls.append(n_rel_rec / n_rel if n_rel else 0.0)
    precision = float(np.mean(precisions))
    recall = float(np.mean(recalls))
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    return precision, recall, f1

def top_n_products(ratings, n=5, min_interactions=50):
    """Popularity baseline: highest mean rating among sufficiently reviewed products."""
    summary = ratings.groupby('prod_id')['rating'].agg(['mean','count'])
    eligible = summary[summary['count'] >= min_interactions]
    return eligible.sort_values(['mean','count'], ascending=False).head(n)
