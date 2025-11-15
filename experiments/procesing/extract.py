from kafka import KafkaConsumer
import pandas as pd
import json
import numpy as np
import os
from dotenv import load_dotenv
from sklearn.base import BaseEstimator, TransformerMixin
# import matplotlib.pyplot as plt
# from IPython.display import display, SVG, Image
load_dotenv()


KAFKA_HOST=os.getenv("KAFKA_HOST", "localhost")
KAFKA_PORT=os.getenv("KAFKA_PORT", 9092)
TOPIC = os.getenv("KAFKA_TOPIC", "user-interactions")
N_PRICE_BUCKETS = 5

def get_data_from_kafka() -> pd.DataFrame:
    consumer = KafkaConsumer(
        TOPIC,
        enable_auto_commit=True,
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        auto_offset_reset='earliest',
        bootstrap_servers=[f"{KAFKA_HOST}:{KAFKA_PORT}"]
    )
    messages=consumer.poll(timeout_ms=1000,max_records=10000)
    df = []
    for m in messages.values():
        for i in m:
            df.append(i.value)
    df = pd.DataFrame(df)
    """
 0   sessionId             73 non-null     object
 1   eventName             73 non-null     object
 2   page                  73 non-null     object
 3   productId             67 non-null     object
 4   storeMode             73 non-null     object
 5   userAgent             73 non-null     object
 6   ts                    73 non-null     object
 7   metadata_referrer     6 non-null      object
 8   metadata_roomType     45 non-null     object
 9   metadata_price        45 non-null     float64
 10  metadata_nights       45 non-null     float64
 11  metadata_elementText  22 non-null     object
 12  metadata_dwellTime    22 non-null     float64
    """
    # explode metadata col json
    df = df.join(pd.json_normalize(df.pop("metadata"), sep=".").add_prefix("metadata_"))
    df = df.dropna(subset=['eventName'])
    return df


def join_with_experiments(df: pd.DataFrame) -> pd.DataFrame:
    # TODO: Get experiments db from supabase and join on session_id
    return df


def augment_event_titles(df: pd.DataFrame) -> pd.DataFrame:
    # from taking standard view_item_page in eventName to view_item_page_{metadata_schema}
    # we want metadata schema to create product specific event names
    price_buckets = pd.qcut(
        df["metadata_price"],
        q=N_PRICE_BUCKETS,
        labels=[f"PB_{i+1}" for i in range(N_PRICE_BUCKETS)]
    )
    # metadata_schema: _product_id@price_bucket_{i} only if we have product metadata otherswise keep original event name
    # TODO: make this adaptive, if we have hover_over_title we append the title, if its view_page we say which page
    df["metadata_schema"] = np.where(
        df["productId"].notnull() & df["metadata_price"].notnull(),
        "_" + df["productId"].astype(str) + "@" + price_buckets.astype(str),
        ""
    )
    df["eventName"] = df["eventName"] + df["metadata_schema"].astype(str)
    return df


def extract() -> pd.DataFrame:
    df = get_data_from_kafka()
    df = join_with_experiments(df)
    df = augment_event_titles(df)
    return df


class DataExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None):
        return extract()


if __name__ == "__main__":
    df = extract()
    print(df.head())
    print(df.tail())
    print(df.info())
