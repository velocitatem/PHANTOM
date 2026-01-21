"""PHANTOM shared library
Exports unified utilities for features, state, config, kafka, and model registry
"""
from .config import (
    PROJECT_ROOT, DATA_DIR, EXPERIMENTS_DIR,
    AGENT_DATA_DIR, HUMAN_DATA_DIR, SIM_RUNS_DIR, MODEL_REGISTRY_DIR,
    COLLECTED_DATA_DIR, NOTEBOOK_OUTPUT_DIR,
    ensure_dir, get_data_path, get_experiments_path, get_sim_path,
    KAFKA_HOST, KAFKA_PORT, KAFKA_BROKER,
    REDIS_HOST, REDIS_PORT,
    SUPABASE_URL, SUPABASE_ANON_KEY,
    BACKEND_PORT, PROVIDER_PORT
)
from .state import (
    make_state_repr, event_to_state, parse_state,
    get_event_name, get_timestamp,
    create_state_fn, create_event_name_fn, create_timestamp_fn
)
from .features import (
    transition_histogram, temporal_signature, state_coverage, transition_entropy,
    event_type_distribution, featurize_trajectory, parse_timestamp
)

__all__ = [
    # config
    'PROJECT_ROOT', 'DATA_DIR', 'EXPERIMENTS_DIR',
    'AGENT_DATA_DIR', 'HUMAN_DATA_DIR', 'SIM_RUNS_DIR', 'MODEL_REGISTRY_DIR',
    'COLLECTED_DATA_DIR', 'NOTEBOOK_OUTPUT_DIR',
    'ensure_dir', 'get_data_path', 'get_experiments_path', 'get_sim_path',
    'KAFKA_HOST', 'KAFKA_PORT', 'KAFKA_BROKER',
    'REDIS_HOST', 'REDIS_PORT',
    'SUPABASE_URL', 'SUPABASE_ANON_KEY',
    'BACKEND_PORT', 'PROVIDER_PORT',
    # state
    'make_state_repr', 'event_to_state', 'parse_state',
    'get_event_name', 'get_timestamp',
    'create_state_fn', 'create_event_name_fn', 'create_timestamp_fn',
    # features
    'transition_histogram', 'temporal_signature', 'state_coverage', 'transition_entropy',
    'event_type_distribution', 'featurize_trajectory', 'parse_timestamp',
]
