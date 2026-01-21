"""MovieLens 1M Dataset Loader."""

import os
import zipfile
import requests
from pathlib import Path
from typing import Dict, List, Set
import pandas as pd

from .base import BaseDataset


class MovieLensDataset(BaseDataset):
    """MovieLens 1M dataset (~1M ratings, 6K users, 4K movies)."""
    
    DOWNLOAD_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "datasets" / "movielens"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self._ratings = None
        self._items = None
        self._popularity = None
        self._item_genres = None
        
        self._ensure_downloaded()
    
    def _ensure_downloaded(self):
        if not (self.data_dir / "ratings.dat").exists():
            print("Downloading MovieLens 1M dataset...")
            self._download_and_extract()
            print("Download complete!")
    
    def _download_and_extract(self):
        zip_path = self.data_dir / "ml-1m.zip"
        
        response = requests.get(self.DOWNLOAD_URL, stream=True)
        response.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(self.data_dir.parent)
        
        ml1m_dir = self.data_dir.parent / "ml-1m"
        if ml1m_dir.exists():
            for file in ml1m_dir.iterdir():
                target = self.data_dir / file.name
                if not target.exists():
                    file.rename(target)
        
        if zip_path.exists():
            zip_path.unlink()
    
    def load_ratings(self) -> pd.DataFrame:
        if self._ratings is None:
            self._ratings = pd.read_csv(
                self.data_dir / "ratings.dat",
                sep='::', engine='python',
                names=['user_id', 'item_id', 'rating', 'timestamp'],
                encoding='latin-1'
            )
        return self._ratings
    
    def load_items(self) -> pd.DataFrame:
        if self._items is None:
            self._items = pd.read_csv(
                self.data_dir / "movies.dat",
                sep='::', engine='python',
                names=['item_id', 'title', 'genres'],
                encoding='latin-1'
            )
        return self._items
    
    def get_user_history(self, user_id: int) -> pd.DataFrame:
        ratings = self.load_ratings()
        items = self.load_items()
        user_df = ratings[ratings['user_id'] == user_id].merge(items, on='item_id')
        return user_df.sort_values('rating', ascending=False)
    
    def get_popularity(self) -> Dict[int, int]:
        if self._popularity is None:
            self._popularity = self.load_ratings().groupby('item_id').size().to_dict()
        return self._popularity
    
    def get_item_genres(self, item_id: int) -> Set[str]:
        if self._item_genres is None:
            self._item_genres = {}
            for _, row in self.load_items().iterrows():
                self._item_genres[row['item_id']] = set(row['genres'].split('|'))
        return self._item_genres.get(item_id, set())
    
    def get_all_users(self) -> List[int]:
        return self.load_ratings()['user_id'].unique().tolist()
    
    def get_all_items(self) -> List[int]:
        return self.load_items()['item_id'].unique().tolist()
    
    def get_item_title(self, item_id: int) -> str:
        items = self.load_items()
        row = items[items['item_id'] == item_id]
        return row.iloc[0]['title'] if len(row) > 0 else f"Unknown ({item_id})"
    
    def get_stats(self) -> dict:
        ratings = self.load_ratings()
        items = self.load_items()
        n_users, n_items = ratings['user_id'].nunique(), items['item_id'].nunique()
        return {
            'num_users': n_users,
            'num_items': n_items,
            'num_ratings': len(ratings),
            'sparsity': 1 - len(ratings) / (n_users * n_items),
        }


if __name__ == "__main__":
    ds = MovieLensDataset()
    print(ds.get_stats())
