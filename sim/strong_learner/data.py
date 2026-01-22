import os
import requests
try:
    import py7zr  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    py7zr = None
import pandas as pd
from typing import Generator
try:
    from sim.rl.behavior_loader.loader import PayloadModel, ValueModel, InteractionModel, Loader
except ImportError:
    from loader import PayloadModel, ValueModel, InteractionModel, Loader

class YooChooseLoader(Loader):
    URL = "https://s3-eu-west-1.amazonaws.com/yc-rdata/yoochoose-data.7z"
    CLICK_COLS = ['session_id', 'ts', 'item_id', 'category']
    BUY_COLS = ['session_id', 'ts', 'item_id', 'price', 'quantity']

    def __init__(self, root_dir: str = "data/yoochoose", chunk_size: int = 500_000, max_sessions: int = 1000):
        self.root = root_dir
        self.chunk_size = chunk_size
        self.max_sessions = max_sessions
        self.click_path = f"{root_dir}/yoochoose-clicks.dat"
        self.buy_path = f"{root_dir}/yoochoose-buys.dat"
        if not os.path.exists(self.click_path): self._setup()
        self.data = self._load_sessions(max_sessions)
        self.entries = list(self.data.keys())

    def _setup(self):
        if py7zr is None:
            raise RuntimeError("py7zr is required to unpack YooChoose dataset. Install py7zr first.")
        os.makedirs(self.root, exist_ok=True)
        zip_path = f"{self.root}/temp.7z"
        with requests.get(self.URL, stream=True) as r:
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
        with py7zr.SevenZipFile(zip_path, 'r') as z:
            z.extractall(self.root)
        os.remove(zip_path)

    def _make_interaction(self, sid: str, ts: str, item_id: str, event: str, page: str, meta: dict) -> InteractionModel:
        payload = PayloadModel(
            sessionId=sid, experimentId=None, eventName=event,
            page=page, productId=item_id, metadata=meta,
            storeMode="yoochoose", userAgent="dataset", ts=ts
        )
        return InteractionModel(
            partitionID=0, offset=0, timestamp=0, compression="",
            isTransactional=False, headers=[], key={},
            value=ValueModel(payload=payload, encoding="json", isPayloadNull=False, schemaId=1, size=0)
        )

    def _parse_category(self, cat) -> str:
        if pd.isna(cat) or cat == "0": return "unknown"
        if cat == "S": return "special_offer"
        try:
            n = int(cat)
            return f"category_{n}" if 1 <= n <= 12 else f"brand_{n}"
        except: return str(cat)

    def stream_clicks(self) -> Generator[InteractionModel, None, None]:
        with pd.read_csv(self.click_path, names=self.CLICK_COLS, chunksize=self.chunk_size, header=None) as reader:
            for chunk in reader:
                for r in chunk.itertuples(index=False):
                    yield self._make_interaction(
                        str(r.session_id), r.ts, str(r.item_id),
                        "view_item_page", self._parse_category(r.category), {}
                    )

    def stream_buys(self) -> Generator[InteractionModel, None, None]:
        with pd.read_csv(self.buy_path, names=self.BUY_COLS, chunksize=self.chunk_size, header=None) as reader:
            for chunk in reader:
                for r in chunk.itertuples(index=False):
                    yield self._make_interaction(
                        str(r.session_id), r.ts, str(r.item_id),
                        "purchase_complete", "/checkout", {"price": r.price, "quantity": r.quantity}
                    )

    def stream(self) -> Generator[InteractionModel, None, None]:
        yield from self.stream_clicks()
        yield from self.stream_buys()

    def _load_sessions(self, max_sessions: int | None = None) -> dict:
        sessions = {}
        for interaction in self.stream():
            sid = interaction.value.payload.sessionId
            if sid not in sessions:
                if max_sessions and len(sessions) >= max_sessions: continue
                sessions[sid] = []
            sessions[sid].append(interaction)
        for sid in sessions: sessions[sid].sort(key=lambda x: x.value.payload.ts)
        return sessions

    def get_data(self) -> dict:
        return self.data

    def get_entries(self) -> tuple[list[str], int]:
        return self.entries, len(self.entries)

if __name__ == "__main__":
    loader = YooChooseLoader(max_sessions=100)
    views, purchases = 0, 0
    for sid, evts in loader.get_data().items():
        for e in evts:
            if e.value.payload.eventName == "view_item_page": views += 1
            elif e.value.payload.eventName == "purchase_complete": purchases += 1
    print(f"Loaded {len(loader.entries)} sessions: {views} view_item_page, {purchases} purchase_complete")
