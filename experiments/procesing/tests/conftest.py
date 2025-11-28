import pytest
import pandas as pd
from typing import List
from procesing.providers.base import DataProvider
from procesing.context import PipelineContext


class MockProvider(DataProvider):
    """Mock provider for testing, holds in-memory fixtures"""

    def __init__(self, products_df=None, experiments_df=None, kafka_data=None):
        self._products = products_df if products_df is not None else pd.DataFrame()
        self._experiments = experiments_df if experiments_df is not None else pd.DataFrame()
        self._kafka_data = kafka_data if kafka_data is not None else {}

    def fetch_products(self, store_mode: str) -> pd.DataFrame:
        return self._products.copy()

    def fetch_experiments(self, experiment_ids: List[str]) -> pd.DataFrame:
        if self._experiments.empty:
            return pd.DataFrame()
        return self._experiments[
            self._experiments['id'].isin(experiment_ids)
        ].copy()

    def fetch_kafka_topic(self, topic: str) -> pd.DataFrame:
        return self._kafka_data.get(topic, pd.DataFrame()).copy()


@pytest.fixture
def mock_products():
    """Standard product catalog fixture with realistic IDs from test data"""
    return pd.DataFrame({
        'id': [
            'd018efc1-25e9-4284-b276-80386e048b25',
            '51266ddb-5b07-47b7-89ee-5b5cae94bb11',
            '2cd7f756-fc65-4ba0-ab01-74521c1fff43'
        ],
        'name': ['Junior Suite', 'Superior Room', 'Deluxe Room'],
        'base_price': [200.0, 150.0, 180.0]
    })


@pytest.fixture
def mock_interactions_raw_kafka():
    """Raw Kafka message structure for interactions, matches production format"""
    return [
        {
            'partitionID': 0, 'offset': 203, 'timestamp': 1764102082676,
            'value': {
                'payload': {
                    'sessionId': 'd423ce8a-77aa-4c9a-94d4-d1adddcc3472',
                    'experimentId': '53aefd07-f66a-4d7f-ba8b-7ea1fc562d35',
                    'eventName': 'learn_more_about_item',
                    'page': '/hotel/products/d018efc1-25e9-4284-b276-80386e048b25',
                    'productId': 'd018efc1-25e9-4284-b276-80386e048b25',
                    'metadata': {'type': 'hotel', 'dateIndex': 1, 'roomType': 'Junior Suite'},
                    'storeMode': 'hotel',
                    'ts': '2025-11-25T20:21:22.674Z'
                }
            }
        },
        {
            'partitionID': 0, 'offset': 204, 'timestamp': 1764102086982,
            'value': {
                'payload': {
                    'sessionId': 'd423ce8a-77aa-4c9a-94d4-d1adddcc3472',
                    'experimentId': '53aefd07-f66a-4d7f-ba8b-7ea1fc562d35',
                    'eventName': 'page_view',
                    'page': '/hotel/products',
                    'productId': None,
                    'metadata': {'referrer': ''},
                    'storeMode': 'hotel',
                    'ts': '2025-11-25T20:21:26.947Z'
                }
            }
        },
        {
            'partitionID': 0, 'offset': 205, 'timestamp': 1764102091825,
            'value': {
                'payload': {
                    'sessionId': 'd423ce8a-77aa-4c9a-94d4-d1adddcc3472',
                    'experimentId': '53aefd07-f66a-4d7f-ba8b-7ea1fc562d35',
                    'eventName': 'hover_over_title',
                    'page': '/hotel/products',
                    'productId': '51266ddb-5b07-47b7-89ee-5b5cae94bb11',
                    'metadata': {'elementText': 'Superior Room', 'dateIndex': 1, 'dwellTime': 1200},
                    'storeMode': 'hotel',
                    'ts': '2025-11-25T20:21:31.823Z'
                }
            }
        },
        {
            'partitionID': 0, 'offset': 206, 'timestamp': 1764102094193,
            'value': {
                'payload': {
                    'sessionId': 'd423ce8a-77aa-4c9a-94d4-d1adddcc3472',
                    'experimentId': 'bbbbcccc-dddd-eeee-ffff-000011112222',
                    'eventName': 'hover_over_paragraph',
                    'page': '/hotel/products',
                    'productId': '51266ddb-5b07-47b7-89ee-5b5cae94bb11',
                    'metadata': {'elementText': 'price', 'dateIndex': 1, 'dwellTime': 1307},
                    'storeMode': 'hotel',
                    'ts': '2025-11-25T20:21:34.191Z'
                }
            }
        },
        {
            'partitionID': 0, 'offset': 207, 'timestamp': 1764102101970,
            'value': {
                'payload': {
                    'sessionId': 'd423ce8a-77aa-4c9a-94d4-d1adddcc3472',
                    'experimentId': 'bbbbcccc-dddd-eeee-ffff-000011112222',
                    'eventName': 'hover_over_paragraph',
                    'page': '/hotel/products',
                    'productId': 'd018efc1-25e9-4284-b276-80386e048b25',
                    'metadata': {'elementText': 'price', 'dateIndex': 1, 'dwellTime': 1201},
                    'storeMode': 'hotel',
                    'ts': '2025-11-25T20:21:41.967Z'
                }
            }
        }
    ]


@pytest.fixture
def mock_interactions(mock_interactions_raw_kafka):
    """Processed interaction DataFrame (what provider.fetch_kafka_topic returns)"""
    records = [msg['value']['payload'] for msg in mock_interactions_raw_kafka]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['ts'])
    return df


@pytest.fixture
def mock_price_logs_raw_kafka():
    """Raw Kafka message structure for price logs, matches production format"""
    return [
        {
            'partitionID': 0, 'offset': 32, 'timestamp': 1764104757969,
            'value': {
                'payload': {
                    'productId': '2cd7f756-fc65-4ba0-ab01-74521c1fff43',
                    'price': 162.47,
                    'sessionId': 'd423ce8a-77aa-4c9a-94d4-d1adddcc3472',
                    'experimentId': '53aefd07-f66a-4d7f-ba8b-7ea1fc562d35',
                    'storeMode': 'shop',
                    'ts': '2025-11-25T21:05:57.967Z'
                }
            }
        },
        {
            'partitionID': 0, 'offset': 33, 'timestamp': 1764104757995,
            'value': {
                'payload': {
                    'productId': '2ddabbfc-4127-48fc-86dc-ebc4c677efa2',
                    'price': 743.49,
                    'sessionId': 'd423ce8a-77aa-4c9a-94d4-d1adddcc3472',
                    'experimentId': '53aefd07-f66a-4d7f-ba8b-7ea1fc562d35',
                    'storeMode': 'shop',
                    'ts': '2025-11-25T21:05:57.993Z'
                }
            }
        },
        {
            'partitionID': 0, 'offset': 34, 'timestamp': 1764104758011,
            'value': {
                'payload': {
                    'productId': '2cd7f756-fc65-4ba0-ab01-74521c1fff43',
                    'price': 163.87,
                    'sessionId': 'd423ce8a-77aa-4c9a-94d4-d1adddcc3472',
                    'experimentId': '53aefd07-f66a-4d7f-ba8b-7ea1fc562d35',
                    'storeMode': 'shop',
                    'ts': '2025-11-25T21:05:58.009Z'
                }
            }
        },
        {
            'partitionID': 0, 'offset': 35, 'timestamp': 1764104758050,
            'value': {
                'payload': {
                    'productId': '2ddabbfc-4127-48fc-86dc-ebc4c677efa2',
                    'price': 397.46,
                    'sessionId': 'd423ce8a-77aa-4c9a-94d4-d1adddcc3472',
                    'experimentId': '53aefd07-f66a-4d7f-ba8b-7ea1fc562d35',
                    'storeMode': 'shop',
                    'ts': '2025-11-25T21:05:58.049Z'
                }
            }
        },
        {
            'partitionID': 0, 'offset': 36, 'timestamp': 1764104768865,
            'value': {
                'payload': {
                    'productId': '2cd7f756-fc65-4ba0-ab01-74521c1fff43',
                    'price': 401.66,
                    'sessionId': 'd423ce8a-77aa-4c9a-94d4-d1adddcc3472',
                    'experimentId': '53aefd07-f66a-4d7f-ba8b-7ea1fc562d35',
                    'storeMode': 'shop',
                    'ts': '2025-11-25T21:06:08.864Z'
                }
            }
        }
    ]


@pytest.fixture
def mock_price_logs(mock_price_logs_raw_kafka):
    """Processed price logs DataFrame (what provider.fetch_kafka_topic returns)"""
    # extract payloads and flatten
    records = [msg['value']['payload'] for msg in mock_price_logs_raw_kafka]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['ts'])
    return df


@pytest.fixture
def mock_experiments():
    """Standard experiment metadata fixture matching Supabase schema"""
    return pd.DataFrame({
        'id': ['53aefd07-f66a-4d7f-ba8b-7ea1fc562d35', 'bbbbcccc-dddd-eeee-ffff-000011112222'],
        'created_at': pd.to_datetime(['2025-11-25T20:00:00Z', '2025-11-26T10:00:00Z']),
        'subject_name': ['Session A', 'Session B'],
        'xp_human_only': [True, False],
        'xp_market_mode': ['hotel', 'shop'],
        'xp_task_id': [None, None]
    })


@pytest.fixture
def mock_provider(mock_products, mock_experiments, mock_interactions, mock_price_logs):
    """Fully configured mock provider"""
    return MockProvider(
        products_df=mock_products,
        experiments_df=mock_experiments,
        kafka_data={
            'user-interactions': mock_interactions,
            'price-logs': mock_price_logs
        }
    )


@pytest.fixture
def pipeline_context(mock_provider):
    """Standard pipeline context for testing"""
    return PipelineContext(
        provider=mock_provider,
        store_mode='hotel',
        window_size='30s',
        n_price_buckets=3
    )


@pytest.fixture
def empty_provider():
    """Provider with no data, for edge case testing"""
    return MockProvider(
        products_df=pd.DataFrame(columns=['id', 'name', 'base_price']),
        experiments_df=pd.DataFrame(columns=['id', 'created_at', 'subject_name', 'xp_human_only', 'xp_market_mode', 'xp_task_id']),
        kafka_data={'user-interactions': pd.DataFrame(), 'price-logs': pd.DataFrame()}
    )


@pytest.fixture
def empty_context(empty_provider):
    """Context with empty provider"""
    return PipelineContext(
        provider=empty_provider,
        store_mode='hotel',
        window_size='30s'
    )
