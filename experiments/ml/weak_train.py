import sys
sys.path.insert(0, "/home/velocitatem/Documents/Projects/PHANTOM/sim/rl/behavior_loader")
sys.path.insert(0, "/home/velocitatem/Documents/Projects/PHANTOM/experiments/ml")

from sim.rl.behavior_loader.loader import AgentLoader, Loader, JointLoader, PayloadModel
from sim.rl.behavior_loader.models import JointBehaviorModel
from arch import ContrastiveWeakClassifier, contrastive_loss, featurize_trajectory
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from copy import deepcopy
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

RUNS_DIR = "/home/velocitatem/Documents/Projects/PHANTOM/experiments/ml/runs"
agent_dir = "/home/velocitatem/Documents/Projects/PHANTOM/experiments/agents/collected_data/"
human_dir = "/home/velocitatem/Documents/Projects/PHANTOM/experiments/collected_data/"


def _perturb_ts(evt: PayloadModel, jitter_ms: int = 500) -> PayloadModel:
    """Add random jitter to event timestamp"""
    new_evt = deepcopy(evt)
    try:
        ts = datetime.fromisoformat(evt.ts.replace('Z', '+00:00'))
        delta = timedelta(milliseconds=random.randint(-jitter_ms, jitter_ms))
        new_evt.ts = (ts + delta).isoformat()
    except:
        pass
    return new_evt


def augment_trajectory(trajectory: List[PayloadModel], rate: float = 0.1) -> List[PayloadModel]:
    """Apply random augmentation to trajectory for contrastive learning"""
    if len(trajectory) < 2:
        return trajectory

    aug_type = random.choice(['window', 'shuffle', 'noise', 'drop'])

    if aug_type == 'window':  # random contiguous sub-sequence (70-100% length)
        min_len = max(2, int(len(trajectory) * 0.7))
        sub_len = random.randint(min_len, len(trajectory))
        start = random.randint(0, len(trajectory) - sub_len)
        return trajectory[start:start + sub_len]

    elif aug_type == 'shuffle':  # swap adjacent pairs with probability rate
        result = list(trajectory)
        for i in range(len(result) - 1):
            if random.random() < rate:
                result[i], result[i + 1] = result[i + 1], result[i]
        return result

    elif aug_type == 'drop':  # drop events with probability rate
        result = [e for e in trajectory if random.random() > rate]
        return result if len(result) >= 2 else trajectory[:2]

    elif aug_type == 'noise':  # perturb timestamps
        return [_perturb_ts(e, jitter_ms=500) for e in trajectory]

    return trajectory


class TripletDataset(Dataset):
    """Generate (anchor, positive, negative) triplets on-the-fly with augmentation"""
    def __init__(self, data: Dict[str, List[PayloadModel]], mdp: Optional[Dict], augment_fn, input_dim: int = 64, multiplier: int = 10):
        self.sessions = list(data.items())
        self.human_ids = [i for i, (sid, _) in enumerate(self.sessions) if sid.startswith('human_')]
        self.agent_ids = [i for i, (sid, _) in enumerate(self.sessions) if sid.startswith('agent_')]
        self.mdp = mdp
        self.augment = augment_fn
        self.input_dim = input_dim
        self.multiplier = multiplier

        if not self.human_ids or not self.agent_ids:
            raise ValueError(f"Need both human ({len(self.human_ids)}) and agent ({len(self.agent_ids)}) sessions")

    def __len__(self) -> int:
        return len(self.sessions) * self.multiplier

    def __getitem__(self, idx: int):
        anchor_idx = idx % len(self.sessions)
        sid, events = self.sessions[anchor_idx]
        is_human = sid.startswith('human_')

        anchor = featurize_trajectory(events, self.mdp, self.input_dim)
        positive = featurize_trajectory(self.augment(events), self.mdp, self.input_dim)

        neg_pool = self.agent_ids if is_human else self.human_ids
        neg_idx = random.choice(neg_pool)
        negative = featurize_trajectory(self.sessions[neg_idx][1], self.mdp, self.input_dim)

        label = 0 if is_human else 1  # 0=human, 1=agent
        return (torch.tensor(anchor, dtype=torch.float32),
                torch.tensor(positive, dtype=torch.float32),
                torch.tensor(negative, dtype=torch.float32),
                torch.tensor(label, dtype=torch.long))


def train(epochs: int = 100, lr: float = 1e-3, batch_size: int = 4, input_dim: int = 64,
          embed_dim: int = 32, margin: float = 0.3, verbose: bool = True, run_name: str = None):
    """Train contrastive weak classifier on human/agent trajectories"""
    joint = JointLoader(human_dir, agent_dir)
    data = joint.get_data()
    if verbose:
        print(f"Loaded {len(data)} sessions")

    joint_model = JointBehaviorModel(human_dir, agent_dir)
    ref_mdp = joint_model.build_MDP()

    dataset = TripletDataset(data, ref_mdp, augment_trajectory, input_dim=input_dim)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = ContrastiveWeakClassifier(input_dim=input_dim, embed_dim=embed_dim, margin=margin)
    model.to_device()

    run_name = run_name or f"d{input_dim}_e{embed_dim}_lr{lr}_m{margin}_{datetime.now():%Y%m%d_%H%M%S}"
    writer = SummaryWriter(f"{RUNS_DIR}/train/{run_name}")

    optimizer = Adam(list(model.encoder.parameters()) + list(model.classifier.parameters()), lr=lr)
    ce_loss_fn = torch.nn.CrossEntropyLoss()

    best_loss = float('inf')
    for epoch in range(epochs):
        model.encoder.train()
        model.classifier.train()
        total_loss, n_batches = 0.0, 0

        for anchor, positive, negative, labels in loader:
            anchor, positive, negative, labels = [t.to(model.device) for t in [anchor, positive, negative, labels]]
            z_a, z_p, z_n = [model.encoder(t.unsqueeze(1)) for t in [anchor, positive, negative]]

            trip_loss = contrastive_loss(z_a, z_p, z_n, margin=model.margin)
            ce = ce_loss_fn(model.classifier(z_a), labels)
            loss = trip_loss + 0.5 * ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        writer.add_scalar('loss', avg_loss, epoch)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss

    writer.close()
    if verbose:
        print(f"Done. Best={best_loss:.4f} TB:{RUNS_DIR}/train/{run_name}")

    return model, ref_mdp


def evaluate_loocv(input_dim: int = 64, embed_dim: int = 32, epochs_per_fold: int = 50,
                   lr: float = 1e-3, margin: float = 0.3, run_name: str = None):
    """Leave-one-out cross-validation given limited samples"""
    joint = JointLoader(human_dir, agent_dir)
    data = joint.get_data()
    session_ids = list(data.keys())

    joint_model = JointBehaviorModel(human_dir, agent_dir)
    ref_mdp = joint_model.build_MDP()

    run_name = run_name or f"loocv_d{input_dim}_e{embed_dim}_m{margin}_{datetime.now():%Y%m%d_%H%M%S}"
    writer = SummaryWriter(f"{RUNS_DIR}/eval/{run_name}")

    predictions, actuals = [], []

    for fold_idx, test_sid in enumerate(session_ids):
        train_data = {k: v for k, v in data.items() if k != test_sid}
        test_events = data[test_sid]
        test_label = 0 if test_sid.startswith('human_') else 1

        n_human = sum(1 for k in train_data if k.startswith('human_'))
        n_agent = sum(1 for k in train_data if k.startswith('agent_'))
        if n_human == 0 or n_agent == 0:
            continue

        try:
            dataset = TripletDataset(train_data, ref_mdp, augment_trajectory, input_dim=input_dim, multiplier=5)
            loader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)

            model = ContrastiveWeakClassifier(input_dim=input_dim, embed_dim=embed_dim, margin=margin)
            model.to_device()
            optimizer = Adam(list(model.encoder.parameters()) + list(model.classifier.parameters()), lr=lr)

            model.encoder.train()
            model.classifier.train()
            for _ in range(epochs_per_fold):
                for anchor, positive, negative, labels in loader:
                    z_a, z_p, z_n = [model.encoder(t.unsqueeze(1).to(model.device)) for t in [anchor, positive, negative]]
                    loss = contrastive_loss(z_a, z_p, z_n, margin=margin)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            test_feat = featurize_trajectory(test_events, ref_mdp, input_dim)
            pred = model.predict(test_feat.reshape(1, -1))[0]
            predictions.append(pred)
            actuals.append(test_label)
            print(f"  {test_sid[:12]}...: pred={pred}, actual={test_label}, {'OK' if pred == test_label else 'MISS'}")

        except Exception as e:
            print(f"Error: {e}")

    if predictions:
        acc = sum(p == a for p, a in zip(predictions, actuals)) / len(predictions)
        tp = sum(1 for p, a in zip(predictions, actuals) if p == 1 and a == 1)
        fp = sum(1 for p, a in zip(predictions, actuals) if p == 1 and a == 0)
        fn = sum(1 for p, a in zip(predictions, actuals) if p == 0 and a == 1)
        prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-10)
        writer.add_scalar('accuracy', acc, 0)
        writer.add_scalar('f1', f1, 0)
        writer.add_scalar('precision', prec, 0)
        writer.add_scalar('recall', rec, 0)
        writer.close()
        print(f"\nAccuracy: {acc:.2%} F1: {f1:.3f} TB:{RUNS_DIR}/eval/{run_name}")
        return acc, predictions, actuals
    writer.close()
    return 0.0, [], []


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--margin', type=float, default=0.3)
    parser.add_argument('--input-dim', type=int, default=64)
    parser.add_argument('--embed-dim', type=int, default=32)
    parser.add_argument('--run-name', type=str, default=None)
    args = parser.parse_args()

    if args.mode == 'train':
        model, mdp = train(epochs=args.epochs, lr=args.lr, input_dim=args.input_dim,
                           embed_dim=args.embed_dim, margin=args.margin, run_name=args.run_name)
    else:
        evaluate_loocv(input_dim=args.input_dim, embed_dim=args.embed_dim, epochs_per_fold=args.epochs,
                       lr=args.lr, margin=args.margin, run_name=args.run_name)
