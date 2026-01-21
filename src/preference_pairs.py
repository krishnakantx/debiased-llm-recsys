"""Preference Pair Construction for SFT and DPO training."""

from typing import List, Tuple, Dict
import random

from data.base import BaseDataset
from src.preference_extraction import PreferenceExtractor


class PreferencePairConstructor:
    """
    Constructs preference pairs programmatically from dataset statistics.
    - Preferred: low-popularity, genre-similar items
    - Rejected: high-popularity items (no genre filter)
    """
    
    def __init__(self, dataset: BaseDataset, num_recommendations: int = 5,
                 min_genre_similarity: float = 0.1):
        self.dataset = dataset
        self.num_recommendations = num_recommendations
        self.min_genre_similarity = min_genre_similarity
        self._low_pop = None
        self._high_pop = None
        self._popularity = None
    
    @property
    def low_popularity_items(self) -> set:
        if self._low_pop is None:
            self._low_pop = self.dataset.get_low_popularity_items()
        return self._low_pop
    
    @property
    def high_popularity_items(self) -> set:
        if self._high_pop is None:
            self._high_pop = self.dataset.get_high_popularity_items()
        return self._high_pop
    
    @property
    def popularity(self) -> Dict[int, int]:
        if self._popularity is None:
            self._popularity = self.dataset.get_popularity()
        return self._popularity
    
    def _genre_sim_to_user(self, item_id: int, liked_items: List[int]) -> float:
        if not liked_items:
            return 0.0
        sims = [self.dataset.get_genre_similarity(item_id, lid) for lid in liked_items]
        return sum(sims) / len(sims)
    
    def construct_preferred_response(self, user_id: int, extractor: PreferenceExtractor,
                                      excluded: set = None) -> List[str]:
        if excluded is None:
            excluded = extractor.get_user_rated_items(user_id)
        
        history = self.dataset.get_user_history(user_id)
        liked_items = history[history['rating'] >= 4]['item_id'].tolist()
        candidates = self.low_popularity_items - excluded
        
        scored = []
        for iid in candidates:
            sim = self._genre_sim_to_user(iid, liked_items)
            if sim >= self.min_genre_similarity:
                scored.append((iid, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        selected = [iid for iid, _ in scored[:self.num_recommendations]]
        if len(selected) < self.num_recommendations:
            remaining = list(candidates - set(selected))
            random.shuffle(remaining)
            selected.extend(remaining[:self.num_recommendations - len(selected)])
        
        return [self.dataset.get_item_title(iid) for iid in selected]
    
    def construct_rejected_response(self, user_id: int, extractor: PreferenceExtractor,
                                     excluded: set = None) -> List[str]:
        if excluded is None:
            excluded = extractor.get_user_rated_items(user_id)
        
        candidates = list(self.high_popularity_items - excluded)
        candidates.sort(key=lambda x: self.popularity.get(x, 0), reverse=True)
        selected = candidates[:self.num_recommendations]
        
        return [self.dataset.get_item_title(iid) for iid in selected]
    
    def construct_pair(self, user_id: int, extractor: PreferenceExtractor) -> Tuple[str, str, str]:
        prompt = extractor.format_prompt(user_id)
        excluded = extractor.get_user_rated_items(user_id)
        preferred = self.construct_preferred_response(user_id, extractor, excluded)
        rejected = self.construct_rejected_response(user_id, extractor, excluded)
        return prompt, "\n".join(preferred), "\n".join(rejected)
    
    def construct_sft_target(self, user_id: int, extractor: PreferenceExtractor) -> Tuple[str, str]:
        prompt = extractor.format_prompt(user_id)
        excluded = extractor.get_user_rated_items(user_id)
        target = self.construct_preferred_response(user_id, extractor, excluded)
        return prompt, "\n".join(target)
    
    def generate_sft_dataset(self, extractor: PreferenceExtractor, 
                             user_ids: List[int] = None, max_samples: int = None) -> List[Dict]:
        if user_ids is None:
            user_ids = self.dataset.get_all_users()
        if max_samples:
            user_ids = user_ids[:max_samples]
        
        data = []
        for uid in user_ids:
            try:
                prompt, target = self.construct_sft_target(uid, extractor)
                if target:
                    data.append({'prompt': prompt, 'completion': target, 'user_id': uid})
            except:
                continue
        return data
    
    def generate_dpo_dataset(self, extractor: PreferenceExtractor,
                             user_ids: List[int] = None, max_samples: int = None) -> List[Dict]:
        if user_ids is None:
            user_ids = self.dataset.get_all_users()
        if max_samples:
            user_ids = user_ids[:max_samples]
        
        data = []
        for uid in user_ids:
            try:
                prompt, chosen, rejected = self.construct_pair(uid, extractor)
                if chosen and rejected:
                    data.append({'prompt': prompt, 'chosen': chosen, 'rejected': rejected, 'user_id': uid})
            except:
                continue
        return data


if __name__ == "__main__":
    from data.movielens import MovieLensDataset
    ds = MovieLensDataset()
    ext = PreferenceExtractor(ds)
    con = PreferencePairConstructor(ds)
    prompt, pref, rej = con.construct_pair(ds.get_all_users()[0], ext)
    print(f"Prompt: {prompt[:80]}...")
    print(f"Preferred: {pref.split(chr(10))[0]}")
    print(f"Rejected: {rej.split(chr(10))[0]}")
