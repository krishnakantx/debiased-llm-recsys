"""Stage 1: Preference Extraction - transforms user history into prompts."""

from typing import List, Tuple
import pandas as pd

from data.base import BaseDataset


class PreferenceExtractor:
    """Extracts top-k liked and bottom-m disliked items as prompts."""
    
    def __init__(self, dataset: BaseDataset, k_positive: int = 10, 
                 m_negative: int = 5, num_recommendations: int = 5):
        self.dataset = dataset
        self.k_positive = k_positive
        self.m_negative = m_negative
        self.num_recommendations = num_recommendations
    
    def extract_preferences(self, user_id: int) -> Tuple[List[str], List[str]]:
        history = self.dataset.get_user_history(user_id)
        if len(history) == 0:
            return [], []
        
        history = history.sort_values('rating', ascending=False)
        liked = history.head(self.k_positive)['title'].tolist()
        
        low_rated = history[history['rating'] <= 2]
        if len(low_rated) > 0:
            disliked = low_rated.tail(self.m_negative)['title'].tolist()
        else:
            disliked = history.tail(self.m_negative)['title'].tolist()
        
        return liked, disliked
    
    def format_prompt(self, user_id: int) -> str:
        liked, disliked = self.extract_preferences(user_id)
        
        parts = []
        if liked:
            parts.append(f"User likes: {', '.join(liked)}.")
        if disliked:
            parts.append(f"User dislikes: {', '.join(disliked)}.")
        parts.append(f"Recommend {self.num_recommendations} movies.")
        
        return " ".join(parts)
    
    def format_prompt_with_system(self, user_id: int) -> dict:
        return {
            'system': "You are a movie recommendation assistant. Recommend movies based on user preferences. List only titles, one per line.",
            'user': self.format_prompt(user_id)
        }
    
    def get_user_rated_items(self, user_id: int) -> set:
        return set(self.dataset.get_user_history(user_id)['item_id'].tolist())
    
    def get_user_liked_genres(self, user_id: int) -> set:
        history = self.dataset.get_user_history(user_id)
        top_rated = history[history['rating'] >= 4]
        genres = set()
        for _, row in top_rated.iterrows():
            genres.update(self.dataset.get_item_genres(row['item_id']))
        return genres


if __name__ == "__main__":
    from data.movielens import MovieLensDataset
    ds = MovieLensDataset()
    ext = PreferenceExtractor(ds)
    for uid in ds.get_all_users()[:3]:
        print(f"User {uid}: {ext.format_prompt(uid)[:100]}...")
