"""Evaluation metrics for recommendation quality and popularity bias."""

import math
from typing import List, Dict, Set
import numpy as np

from data.base import BaseDataset


class Evaluator:
    """Computes Precision, nDCG, MAP, Novelty, Coverage, and Avg Popularity."""
    
    def __init__(self, dataset: BaseDataset):
        self.dataset = dataset
        self._popularity = None
        self._num_users = None
        self._catalog_size = None
    
    @property
    def popularity(self) -> Dict[int, int]:
        if self._popularity is None:
            self._popularity = self.dataset.get_popularity()
        return self._popularity
    
    @property
    def num_users(self) -> int:
        if self._num_users is None:
            self._num_users = len(self.dataset.get_all_users())
        return self._num_users
    
    @property
    def catalog_size(self) -> int:
        if self._catalog_size is None:
            self._catalog_size = len(self.dataset.get_all_items())
        return self._catalog_size
    
    def precision_at_k(self, recommended: List[int], relevant: Set[int], k: int = 5) -> float:
        if k <= 0:
            return 0.0
        hits = sum(1 for item in recommended[:k] if item in relevant)
        return hits / k
    
    def recall_at_k(self, recommended: List[int], relevant: Set[int], k: int = 5) -> float:
        if not relevant:
            return 0.0
        hits = sum(1 for item in recommended[:k] if item in relevant)
        return hits / len(relevant)
    
    def ndcg_at_k(self, recommended: List[int], relevant: Set[int], k: int = 5) -> float:
        if not relevant or k <= 0:
            return 0.0
        dcg = sum(1.0 / math.log2(i + 2) for i, item in enumerate(recommended[:k]) if item in relevant)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
        return dcg / idcg if idcg > 0 else 0.0
    
    def average_precision_at_k(self, recommended: List[int], relevant: Set[int], k: int = 5) -> float:
        if not relevant or k <= 0:
            return 0.0
        hits, sum_prec = 0, 0.0
        for i, item in enumerate(recommended[:k]):
            if item in relevant:
                hits += 1
                sum_prec += hits / (i + 1)
        return sum_prec / min(len(relevant), k)
    
    def novelty(self, recommended: List[int], k: int = 5) -> float:
        top_k = recommended[:k]
        if not top_k:
            return 0.0
        scores = []
        for item in top_k:
            pop = self.popularity.get(item, 1)
            prob = pop / self.num_users
            if prob > 0:
                scores.append(-math.log2(prob))
        return sum(scores) / len(scores) if scores else 0.0
    
    def average_popularity(self, recommended: List[int], k: int = 5) -> float:
        top_k = recommended[:k]
        if not top_k:
            return 0.0
        return sum(self.popularity.get(item, 0) for item in top_k) / len(top_k)
    
    def coverage(self, all_recommendations: List[List[int]], k: int = 5) -> float:
        unique = set()
        for recs in all_recommendations:
            unique.update(recs[:k])
        return (len(unique) / self.catalog_size) * 100
    
    def evaluate_user(self, recommended: List[int], relevant: Set[int], k: int = 5) -> Dict[str, float]:
        return {
            'precision': self.precision_at_k(recommended, relevant, k),
            'recall': self.recall_at_k(recommended, relevant, k),
            'ndcg': self.ndcg_at_k(recommended, relevant, k),
            'map': self.average_precision_at_k(recommended, relevant, k),
            'novelty': self.novelty(recommended, k),
            'avg_popularity': self.average_popularity(recommended, k),
        }
    
    def evaluate_all(self, recommendations: Dict[int, List[int]], 
                     ground_truth: Dict[int, Set[int]], k: int = 5) -> Dict[str, float]:
        metrics = {m: [] for m in ['precision', 'recall', 'ndcg', 'map', 'novelty', 'avg_popularity']}
        all_recs = []
        
        for uid, recs in recommendations.items():
            relevant = ground_truth.get(uid, set())
            user_m = self.evaluate_user(recs, relevant, k)
            for m, v in user_m.items():
                metrics[m].append(v)
            all_recs.append(recs)
        
        results = {}
        for m, vals in metrics.items():
            results[m] = np.mean(vals) if vals else 0.0
            results[f'{m}_std'] = np.std(vals) if vals else 0.0
        results['coverage'] = self.coverage(all_recs, k)
        return results
    
    def format_results(self, results: Dict[str, float], k: int = 5) -> str:
        return f"""Evaluation Results (K={k})
========================================
Precision@{k}: {results['precision']:.4f}
Recall@{k}:    {results.get('recall', 0):.4f}
nDCG@{k}:      {results['ndcg']:.4f}
MAP@{k}:       {results['map']:.4f}
Novelty:       {results['novelty']:.2f}
Avg Popularity: {results['avg_popularity']:.0f}
Coverage:      {results['coverage']:.1f}%"""


if __name__ == "__main__":
    from data.movielens import MovieLensDataset
    ds = MovieLensDataset()
    ev = Evaluator(ds)
    items = ds.get_all_items()
    pop = ds.get_popularity()
    top5 = sorted(items, key=lambda x: pop.get(x, 0), reverse=True)[:5]
    print(f"Top 5 popular - Novelty: {ev.novelty(top5, 5):.2f}, AvgPop: {ev.average_popularity(top5, 5):.0f}")
