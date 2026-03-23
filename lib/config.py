"""Unified path configuration for PHANTOM project
All hardcoded paths should reference this module
Paths can be overridden via environment variables
"""

import os
from pathlib import Path

# project root (directory containing lib/, experiments/, sim/, web/, backend/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# data directories
DATA_DIR = Path(os.getenv("PHANTOM_DATA_DIR", PROJECT_ROOT / "data"))
EXPERIMENTS_DIR = Path(
    os.getenv("PHANTOM_EXPERIMENTS_DIR", PROJECT_ROOT / "experiments")
)

# agent/human interaction data
AGENT_DATA_DIR = Path(os.getenv("PHANTOM_AGENT_DATA_DIR", DATA_DIR / "agents"))
HUMAN_DATA_DIR = Path(os.getenv("PHANTOM_HUMAN_DATA_DIR", DATA_DIR / "humans"))

# RL simulation runs
SIM_RUNS_DIR = Path(
    os.getenv("PHANTOM_SIM_RUNS_DIR", PROJECT_ROOT / "sim" / "rl" / "runs")
)

# model artifacts
MODEL_REGISTRY_DIR = Path(os.getenv("PHANTOM_MODEL_REGISTRY_DIR", DATA_DIR / "models"))

# collected experiment data
COLLECTED_DATA_DIR = Path(
    os.getenv(
        "PHANTOM_COLLECTED_DATA_DIR", EXPERIMENTS_DIR / "agents" / "collected_data"
    )
)

# notebook outputs
NOTEBOOK_OUTPUT_DIR = Path(
    os.getenv("PHANTOM_NOTEBOOK_OUTPUT_DIR", EXPERIMENTS_DIR / "notebooks" / "outputs")
)


def ensure_dir(path: Path) -> Path:
    """ensure directory exists, create if needed"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_path(*parts: str) -> Path:
    """construct path relative to DATA_DIR"""
    return DATA_DIR.joinpath(*parts)


def get_experiments_path(*parts: str) -> Path:
    """construct path relative to EXPERIMENTS_DIR"""
    return EXPERIMENTS_DIR.joinpath(*parts)


def get_sim_path(*parts: str) -> Path:
    """construct path relative to SIM_RUNS_DIR"""
    return SIM_RUNS_DIR.joinpath(*parts)


# service configuration (from .env)
KAFKA_HOST = os.getenv("KAFKA_HOST", "localhost")
KAFKA_PORT = os.getenv("KAFKA_PORT", "9092")
KAFKA_BROKER = f"{KAFKA_HOST}:{KAFKA_PORT}"

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY", "")

BACKEND_PORT = int(os.getenv("BACKEND_PORT", "5000"))
PROVIDER_PORT = int(os.getenv("PROVIDER_PORT", "5001"))

# huggingface dataset repo for collected behavioral data
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "velocitatem/phantom-collected-data")
