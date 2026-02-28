"""Contrastive encoder via trajectory windowing. Classification by prototype distance."""
import sys
sys.path.insert(0, "/home/velocitatem/Documents/Projects/PHANTOM/sim/rl/behavior_loader")
sys.path.insert(0, "/home/velocitatem/Documents/Projects/PHANTOM/experiments/ml")

from sim.rl.behavior_loader.loader import JointLoader, PayloadModel
from arch import TrajectoryEncoder, featurize_trajectory, nt_xent_loss
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np, torch, torch.nn.functional as F, random, optuna
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

RUNS = "/home/velocitatem/Documents/Projects/PHANTOM/experiments/ml/runs"
AGENT_DIR = "/home/velocitatem/Documents/Projects/PHANTOM/experiments/agents/collected_data/"
HUMAN_DIR = "/home/velocitatem/Documents/Projects/PHANTOM/experiments/collected_data/"


@dataclass
class Window:
    events: List[PayloadModel]
    traj_id: str
    label: int  # 0=human, 1=agent


def extract_windows(events: List[PayloadModel], traj_id: str, label: int,
                    sizes: List[int] = [5, 10, 15], stride: int = 2) -> List[Window]:
    """Multi-scale overlapping windows from trajectory"""
    n = len(events)
    wins = [Window(events[i:i+s], traj_id, label) for s in sizes if n >= s for i in range(0, n-s+1, stride)]
    if n >= 3: wins.append(Window(events, traj_id, label))  # full traj
    return wins


def build_windows(data: Dict[str, List], sizes=[5,10,15], stride=2) -> List[Window]:
    return [w for tid, evts in data.items()
            for w in extract_windows(evts, tid, 0 if tid.startswith('human_') else 1, sizes, stride)]


class WindowDataset(Dataset):
    """Yields (anchor, positive) pairs from same class"""
    def __init__(self, windows: List[Window], dim: int = 64):
        self.wins, self.dim = windows, dim
        self.by_label = {0: [i for i,w in enumerate(windows) if w.label==0],
                         1: [i for i,w in enumerate(windows) if w.label==1]}
        self.by_traj = {}
        for i, w in enumerate(windows): self.by_traj.setdefault(w.traj_id, []).append(i)

    def __len__(self): return len(self.wins)

    def _feat(self, evts): return featurize_trajectory(evts, None, self.dim)

    def _aug(self, evts):  # subsample 70-100%
        if len(evts) < 4: return evts
        k = max(3, int(len(evts) * random.uniform(0.7, 1.0)))
        start = random.randint(0, len(evts) - k)
        return evts[start:start+k]

    def __getitem__(self, idx):
        w = self.wins[idx]
        pool = [i for i in self.by_label[w.label] if self.wins[i].traj_id != w.traj_id]
        pos_idx = random.choice(pool) if pool else idx
        a = torch.tensor(self._feat(self._aug(w.events)), dtype=torch.float32)
        p = torch.tensor(self._feat(self._aug(self.wins[pos_idx].events)), dtype=torch.float32)
        return a, p, w.label


class PrototypeClassifier:
    """Classify by distance to class centroids"""
    def __init__(self, encoder: TrajectoryEncoder, device = 'cuda', dim=64):
        self.enc, self.dev, self.dim = encoder, device, dim
        self.centroids = {0: None, 1: None}

    def fit(self, windows: List[Window]):
        self.enc.eval()
        embs = {0: [], 1: []}
        with torch.no_grad():
            for w in windows:
                x = torch.tensor(featurize_trajectory(w.events, None, self.dim), dtype=torch.float32)
                z = self.enc(x.unsqueeze(0).unsqueeze(1).to(self.dev))
                embs[w.label].append(z)
        self.centroids = {k: torch.cat(v).mean(0, keepdim=True) if v else None for k, v in embs.items()}
        return self

    def predict(self, events: List[PayloadModel]) -> Tuple[int, float, Dict]:
        """Returns (pred, confidence, debug). Confidence via softmax over -distances."""
        self.enc.eval()
        with torch.no_grad():
            x = torch.tensor(featurize_trajectory(events, None, self.dim), dtype=torch.float32)
            z = self.enc(x.unsqueeze(0).unsqueeze(1).to(self.dev))
            dists = {k: torch.norm(z - c, dim=1).item() for k, c in self.centroids.items() if c is not None}
            if not dists: return 0, 0.0, {'d': {}, 'p': [0.5, 0.5]}
            pred = min(dists, key=dists.get)
            d0, d1 = dists.get(0, 1e6), dists.get(1, 1e6)  # softmax(-d) gives higher prob to closer centroid
            probs = F.softmax(torch.tensor([[-d0, -d1]]), dim=1).squeeze()
        return pred, probs[pred].item(), {'d': dists, 'p': probs.tolist()}


def train(epochs=200, lr=5e-4, batch=16, dim=64, emb=32, temp=0.5,
          sizes=[5,10,15], stride=2, name=None, verbose=True):
    data = JointLoader(HUMAN_DIR, AGENT_DIR).get_data()
    wins = build_windows(data, sizes, stride)
    if verbose: print(f"Windows: {len(wins)} ({sum(w.label==0 for w in wins)}h/{sum(w.label==1 for w in wins)}a)")

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc = TrajectoryEncoder(dim, emb).to(dev)
    opt = Adam(enc.parameters(), lr=lr)
    loader = DataLoader(WindowDataset(wins, dim), batch_size=batch, shuffle=True, drop_last=True)

    name = name or f"enc_{dim}_{emb}_{datetime.now():%Y%m%d_%H%M%S}"
    writer = SummaryWriter(f"{RUNS}/encoder/{name}")

    for ep in range(epochs):
        enc.train()
        total, n = 0.0, 0
        for a, p, _ in loader:
            loss = nt_xent_loss(enc(a.unsqueeze(1).to(dev)), enc(p.unsqueeze(1).to(dev)), temp)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); n += 1
        avg = total / max(n, 1)
        writer.add_scalar('loss-ntxent', avg, ep)
        if verbose and (ep+1) % 20 == 0: print(f"Epoch {ep+1}: {avg:.4f}")

    writer.close()
    return enc, wins, dev


def loocv(epochs=100, lr=5e-4, dim=64, emb=32, temp=0.5, sizes=[5,10,15], stride=2, verbose=True):
    """Leave-one-trajectory-out CV"""
    data = JointLoader(HUMAN_DIR, AGENT_DIR).get_data()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []

    for test_id in data:
        train_data = {k: v for k, v in data.items() if k != test_id}
        if not any(k.startswith('human_') for k in train_data) or not any(k.startswith('agent_') for k in train_data):
            continue

        wins = build_windows(train_data, sizes, stride)
        enc = TrajectoryEncoder(dim, emb).to(dev)
        opt = Adam(enc.parameters(), lr=lr)
        loader = DataLoader(WindowDataset(wins, dim), batch_size=min(16, len(wins)//2 or 1),
                            shuffle=True, drop_last=len(wins)>2)

        for _ in range(epochs):
            enc.train()
            for a, p, _ in loader:
                loss = nt_xent_loss(enc(a.unsqueeze(1).to(dev)), enc(p.unsqueeze(1).to(dev)), temp)
                opt.zero_grad(); loss.backward(); opt.step()

        clf = PrototypeClassifier(enc, dev, dim).fit(wins)
        pred, conf, dbg = clf.predict(data[test_id])
        actual = 0 if test_id.startswith('human_') else 1
        results.append((pred, actual, conf))
        if verbose: print(f"{test_id[:18]}: pred={pred} conf={conf:.2f} actual={actual} {'OK' if pred==actual else 'MISS'}")

    if results:
        acc = sum(p==a for p,a,_ in results) / len(results)
        if verbose: print(f"\nAccuracy: {acc:.1%} ({sum(p==a for p,a,_ in results)}/{len(results)})")
        return acc, results
    return 0.0, []


def hparam_tune(n_trials=50, epochs=60, n_jobs=2, verbose=True):
    """Optuna hyperparameter search maximizing LOOCV accuracy"""
    def objective(trial):
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        dim = trial.suggest_categorical('dim', [32, 64, 128, 256])
        emb = trial.suggest_categorical('emb', [16, 32, 64, 128])
        temp = trial.suggest_float('temp', 0.05, 1.0)
        stride = trial.suggest_int('stride', 1, 4)
        sizes = [trial.suggest_int(f's{i}', 3, 20) for i in range(3)]
        sizes = sorted(set(sizes))  # unique sorted
        acc, _ = loocv(epochs, lr, dim, emb, temp, sizes, stride, verbose=False)
        return acc

    study = optuna.create_study(direction='maximize', study_name='encoder_hparam',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=verbose)

    best = study.best_params
    if verbose:
        print(f"\nBest accuracy: {study.best_value:.1%}")
        print(f"Best params: {best}")
    return best, study


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['train', 'eval', 'hparam'], default='train')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--dim', type=int, default=128)
    p.add_argument('--emb', type=int, default=64)
    p.add_argument('--temp', type=float, default=0.1)
    p.add_argument('--sizes', type=str, default='5,10,15')
    p.add_argument('--stride', type=int, default=2)
    p.add_argument('--n_trials', type=int, default=50)
    args = p.parse_args()
    sizes = [int(x) for x in args.sizes.split(',')]

    if args.mode == 'train':
        enc, wins, dev = train(args.epochs, args.lr, 16, args.dim, args.emb, args.temp, sizes, args.stride)
    elif args.mode == 'hparam':
        best, study = hparam_tune(args.n_trials, min(args.epochs, 60))
    else:
        loocv(args.epochs, args.lr, args.dim, args.emb, args.temp, sizes, args.stride)
