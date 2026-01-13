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

class JointLoader:
    """Loader for combined human (Kafka) and agent (direct) data without discrimination"""

    def __init__(self, human_dir: str, agent_dir: str):
        self.human_dir = human_dir
        self.agent_dir = agent_dir
        self.human_loader = Loader(human_dir)
        self.agent_loader = AgentLoader(agent_dir)
        self.data = self._load_joint_sessions()
        self.entries = list(self.data.keys())

    def _load_joint_sessions(self) -> dict:
        sessions = {}
        # load human sessions (unwrap from Kafka format to PayloadModel)
        for sid, evts in self.human_loader.get_data().items():
            sessions[f"human_{sid}"] = [evt.value.payload for evt in evts]
        # load agent sessions (already PayloadModel)
        for sid, evts in self.agent_loader.get_data().items():
            sessions[f"agent_{sid}"] = evts
        return sessions

    def get_data(self) -> dict:
        return self.data

    def get_entries(self) -> tuple[list[str], int]:
        return self.entries, len(self.entries)

if __name__ == "__main__":
    AGENT_DIR = "/home/velocitatem/Documents/Projects/PHANTOM/experiments/agents/collected_data/"
    loader = AgentLoader(AGENT_DIR)
    _, n = loader.get_entries()
    print(f"Loaded {n} agent sessions from {AGENT_DIR}")

    HUMAN_DIR = "/home/velocitatem/Documents/Projects/PHANTOM/experiments/collected_data/"
    loader = Loader(HUMAN_DIR)
    _, n = loader.get_entries()
    print(f"Loaded {n} human sessions from {HUMAN_DIR}")

    joint_loader = JointLoader(HUMAN_DIR, AGENT_DIR)
    _, n = joint_loader.get_entries()
    print(f"Loaded {n} total sessions (combined) from joint loader")
