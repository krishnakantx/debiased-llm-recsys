"""Flickscore Dataset Loader (stub)."""

from typing import Dict, List, Set
import pandas as pd
from .base import BaseDataset


class FlickscoreDataset(BaseDataset):
    def __init__(self, data_dir: str = None):
        raise NotImplementedError("Flickscore dataset not implemented.")
    
    def load_ratings(self) -> pd.DataFrame: pass
    def load_items(self) -> pd.DataFrame: pass
    def get_user_history(self, user_id: int) -> pd.DataFrame: pass
    def get_popularity(self) -> Dict[int, int]: pass
    def get_item_genres(self, item_id: int) -> Set[str]: pass
    def get_all_users(self) -> List[int]: pass
    def get_all_items(self) -> List[int]: pass
    def get_item_title(self, item_id: int) -> str: pass
