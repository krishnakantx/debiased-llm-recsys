"""Abstract base class for recommendation datasets."""

from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple
import pandas as pd


class BaseDataset(ABC):
    """Interface for recommendation datasets (MovieLens, Netflix, etc.)."""
    
    @abstractmethod
    def load_ratings(self) -> pd.DataFrame:
        """Returns DataFrame with [user_id, item_id, rating, timestamp]."""
        pass
    
    @abstractmethod
    def load_items(self) -> pd.DataFrame:
        """Returns DataFrame with [item_id, title, genres]."""
        pass
    
    @abstractmethod
    def get_user_history(self, user_id: int) -> pd.DataFrame:
        """Get all ratings for a user."""
        pass
    
    @abstractmethod
    def get_popularity(self) -> Dict[int, int]:
        """Returns dict: item_id -> interaction_count."""
        pass
    
    @abstractmethod
    def get_item_genres(self, item_id: int) -> Set[str]:
        """Get genres for an item."""
        pass
    
    def get_genre_similarity(self, item1: int, item2: int) -> float:
        """Jaccard similarity between item genres."""
        g1, g2 = self.get_item_genres(item1), self.get_item_genres(item2)
        if not g1 or not g2:
            return 0.0
        return len(g1 & g2) / len(g1 | g2)
    
    def get_median_popularity(self) -> float:
        values = sorted(self.get_popularity().values())
        n = len(values)
        if n % 2 == 0:
            return (values[n//2 - 1] + values[n//2]) / 2
        return values[n//2]
    
    def get_low_popularity_items(self) -> Set[int]:
        pop = self.get_popularity()
        median = self.get_median_popularity()
        return {i for i, c in pop.items() if c < median}
    
    def get_high_popularity_items(self) -> Set[int]:
        pop = self.get_popularity()
        median = self.get_median_popularity()
        return {i for i, c in pop.items() if c >= median}
    
    @abstractmethod
    def get_all_users(self) -> List[int]:
        pass
    
    @abstractmethod
    def get_all_items(self) -> List[int]:
        pass
    
    @abstractmethod
    def get_item_title(self, item_id: int) -> str:
        pass
    
    def get_train_test_split(self, test_ratio: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split ratings into train/test per user."""
        ratings = self.load_ratings().sample(frac=1, random_state=random_state)
        train, test = [], []
        
        for _, user_df in ratings.groupby('user_id'):
            n_test = max(1, int(len(user_df) * test_ratio))
            test.append(user_df.head(n_test))
            train.append(user_df.tail(len(user_df) - n_test))
        
        return pd.concat(train, ignore_index=True), pd.concat(test, ignore_index=True)
