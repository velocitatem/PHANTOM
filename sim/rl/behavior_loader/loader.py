import os
import json
from pydantic import BaseModel as Base

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

def _is_admin(page: str | None) -> bool:
    return page is not None and page.startswith("/admin/")

class Loader:
    def __init__(self, src_dir: str):
        self.src_dir = src_dir
        self.entries = os.listdir(src_dir)
        if not self.entries: raise ValueError("empty directory")
        self.data = self._load_sessions()

    def _load_sessions(self) -> dict:
        sessions = {}
        for entry in self.entries:
            with open(f"{self.src_dir}/{entry}/int.json") as f:
                raw = json.load(f)
            ints = [InteractionModel(**i) for i in raw]
            sessions[entry] = [i for i in ints if not _is_admin(i.value.payload.page)]
        return sessions

    def get_data(self) -> dict:
        return self.data

    def get_entries(self) -> tuple[list[str], int]:
        return self.entries, len(self.entries)

class AgentLoader(Loader):
    def _load_sessions(self) -> dict:
        sessions = {}
        for entry in self.entries:
            with open(f"{self.src_dir}/{entry}/int.json") as f:
                raw = json.load(f)
            ints = [PayloadModel(**i) for i in raw]
            sessions[entry] = [i for i in ints if not _is_admin(i.page)]
        return sessions

class JointLoader:
    def __init__(self, human_dir: str, agent_dir: str):
        self.human_loader = Loader(human_dir)
        self.agent_loader = AgentLoader(agent_dir)
        self.data = self._merge()
        self.entries = list(self.data.keys())

    def _merge(self) -> dict:
        return {
            **{f"human_{sid}": [e.value.payload for e in evts]
               for sid, evts in self.human_loader.get_data().items()},
            **{f"agent_{sid}": evts
               for sid, evts in self.agent_loader.get_data().items()}
        }

    def get_data(self) -> dict:
        return self.data

    def get_entries(self) -> tuple[list[str], int]:
        return self.entries, len(self.entries)

if __name__ == "__main__":
    agent_dir = "/home/velocitatem/Documents/Projects/PHANTOM/experiments/agents/collected_data/"
    human_dir = "/home/velocitatem/Documents/Projects/PHANTOM/experiments/collected_data/"

    for name, cls, path in [("agent", AgentLoader, agent_dir),
                             ("human", Loader, human_dir),
                             ("joint", lambda d: JointLoader(human_dir, d), agent_dir)]:
        ldr = cls(path) if name != "joint" else cls(agent_dir)
        print(f"Loaded {len(ldr.get_entries()[0])} {name} sessions")
