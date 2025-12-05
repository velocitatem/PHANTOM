import pandas as pd
from procesing.steps.base import BaseContextStep

class JoinExperimentsStep(BaseContextStep):
    """Join experiment metadata to interactions"""

    def transform(self, data: tuple):
        """
        Args:
            data: (interactions_df, experiments_df)
        Returns:
            merged interactions dataframe
        """
        interactions_df, experiments_df = data

        if experiments_df.empty:
            return interactions_df

        # Flatten nested task field if present
        if 'task' in experiments_df.columns and experiments_df['task'].notnull().any():
            task_norm = pd.json_normalize(experiments_df['task'].dropna())
            task_norm.index = experiments_df[experiments_df['task'].notnull()].index
            experiments_df = experiments_df.drop('task', axis=1).join(task_norm, rsuffix='_task')

        # Rename for clarity
        experiments_df = experiments_df.rename(columns={
            'id': 'experimentId',
            'subject_name': 'exp_subject',
            'xp_human_only': 'exp_human_only',
            'xp_market_mode': 'exp_market_mode',
            'xp_task_id': 'exp_task_id'
        })

        return interactions_df.merge(experiments_df, on='experimentId', how='left')

class JoinProductFeaturesStep(BaseContextStep):
    """Join product features to interactions"""

    def transform(self, data: tuple):
        """
        Args:
            data: (interactions_df, products_df)
        Returns:
            merged interactions dataframe
        """
        demand_df, price_df = data

        if price_df.empty:
            return demand_df
        return demand_df.merge(price_df, on='productId', how='left')
