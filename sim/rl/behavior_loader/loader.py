import os
from pydantic import BaseModel as Base
import json

class PayloadModel(Base):
    sessionId: str
    experimentId: str | None
    eventName: str
    page: str | None
    productId: str | None
    metadata: dict
    storeMode: str
    userAgent: str
    ts: str

class ValueModel(Base):
    payload: PayloadModel
    encoding: str
    isPayloadNull: bool
    schemaId: int
    size: int

class InteractionModel(Base):
    partitionID: int
    offset: int
    timestamp: int
    compression: str
    isTransactional: bool
    headers: list
    key: dict
    value: ValueModel

class Loader:
    def __init__(self, src_dir: str):
        self.src_dir = src_dir
        self.entries = os.listdir(src_dir)
        if not self.entries: raise ValueError("empty directory")
        self.data = self._load_sessions()

    def _is_admin_page(self, interaction: InteractionModel) -> bool:
        page = interaction.value.payload.page
        return page and page.startswith("/admin/")

    def _load_sessions(self) -> dict:
        sessions = {}
        for entry in self.entries:
            int_path = f"{self.src_dir}/{entry}/int.json"
            raw = json.load(open(int_path))
            ints = [InteractionModel(**i) for i in raw]
            sessions[entry] = [i for i in ints if not self._is_admin_page(i)]
        return sessions

    def get_data(self) -> dict:
        return self.data

    def get_entries(self) -> tuple[list[str], int]:
        return self.entries, len(self.entries)

class AgentLoader(Loader):
    """Loader for agent interaction data with simplified schema (direct PayloadModel format)"""

    def _is_admin_page_simple(self, interaction: PayloadModel) -> bool:
        return interaction.page and interaction.page.startswith("/admin/")

    def _load_sessions(self) -> dict:
        sessions = {}
        for entry in self.entries:
            int_path = f"{self.src_dir}/{entry}/int.json"
            raw = json.load(open(int_path))
            ints = [PayloadModel(**i) for i in raw]
            sessions[entry] = [i for i in ints if not self._is_admin_page_simple(i)]
        return sessions

if __name__ == "__main__":
    DIR = "/home/velocitatem/Documents/Projects/PHANTOM/experiments/agents/collected_data/"
    loader = AgentLoader(DIR)
    _, n = loader.get_entries()
    print(f"Loaded {n} sessions from {DIR}")

    DIR = "/home/velocitatem/Documents/Projects/PHANTOM/experiments/collected_data/"
    loader = Loader(DIR)
    _, n = loader.get_entries()
    print(f"Loaded {n} sessions from {DIR}")
