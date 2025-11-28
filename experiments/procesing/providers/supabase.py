import os
import pandas as pd
import requests
from typing import List
from supabase import create_client, Client
from .base import DataProvider

class SupabaseProvider(DataProvider):
    """Concrete Supabase + backend API implementation"""

    def __init__(self,
                 supabase_url: str = None,
                 supabase_key: str = None,):
        self.supabase_url = supabase_url or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)

    def fetch_products(self, store_mode: str) -> pd.DataFrame:
        resp = self.supabase.table(f'{store_mode}_products').select(
            "id, room_type, date_index, metadata, availability"
        ).execute()
        return pd.DataFrame(resp.data) if resp.data else pd.DataFrame()

    def fetch_experiments(self, experiment_ids: List[str]) -> pd.DataFrame:
        if not experiment_ids:
            return pd.DataFrame()

        resp = self.supabase.table('experiments').select(
            'id, subject_name, xp_human_only, xp_market_mode, xp_task_id, '
            'task:tasks(task_name, task_description, task_def_of_done)'
        ).in_('id', experiment_ids).execute()

        return pd.DataFrame(resp.data) if resp.data else pd.DataFrame()
