import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from models import BehaviorModel, AgentBehaviorModel, aggregate_event_transitions, kl_divergence

def event_frequency_distribution(mdp):
    evt_cnt, total = defaultdict(int), 0
    for s, trans in mdp['transitions'].items():
        evt = s.split('|')[2]
        for cnt in mdp['trans_counts'][s].values():
            evt_cnt[evt] += cnt
            total += cnt
    return {evt: cnt/total for evt, cnt in evt_cnt.items()} if total > 0 else {}

def transition_distribution(mdp):
    trans_cnt, total = defaultdict(int), 0
    for s, trans in mdp['trans_counts'].items():
        src = s.split('|')[2]
        for s_next, cnt in trans.items():
            dst = s_next.split('|')[2]
            trans_cnt[f"{src}->{dst}"] += cnt
            total += cnt
    return {t: cnt/total for t, cnt in trans_cnt.items()} if total > 0 else {}

def kl_color(kl):
    return '#d62828' if kl > 2.0 else '#f77f00' if kl > 0.5 else '#2a9d8f'

def plot_comparison(ax, human_vals, agent_vals, labels, title, ylabel, kl_val=None):
    x, w = np.arange(len(labels)), 0.35
    ax.bar(x - w/2, human_vals, w, label='Human', alpha=0.8, color='#2E86AB')
    ax.bar(x + w/2, agent_vals, w, label='Agent', alpha=0.8, color='#A23B72')
    ax.set_ylabel(ylabel, fontsize=9 if len(labels) > 10 else 11, fontweight='bold')
    ax.set_title(title if not kl_val else f"{title}\nKL={kl_val:.4f}",
                 fontsize=10 if len(labels) > 10 else 12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    return ax

if __name__ == "__main__":
    base_dir = "/home/velocitatem/Documents/Projects/PHANTOM/experiments"
    human_dir, agent_dir = f"{base_dir}/collected_data/", f"{base_dir}/agents/collected_data/"

    human_model, agent_model = BehaviorModel(human_dir), AgentBehaviorModel(agent_dir)
    human_mdp, agent_mdp = human_model.build_MDP(), agent_model.build_MDP()

    human_evt, agent_evt = aggregate_event_transitions(human_mdp), aggregate_event_transitions(agent_mdp)
    common = set(human_evt.keys()) & set(agent_evt.keys())
    kl_results = sorted([(e, kl_divergence(human_evt[e], agent_evt[e])) for e in common],
                        key=lambda x: x[1], reverse=True)

    fig = plt.figure(figsize=(16, 10))
    n_rows, n_cols = (len(kl_results) + 1) // 2, 2

    for idx, (evt, kl) in enumerate(kl_results):
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        h_dist, a_dist = human_evt.get(evt, {}), agent_evt.get(evt, {})
        dests = sorted(set(h_dist.keys()) | set(a_dist.keys()))
        if not dests: continue

        h_probs, a_probs = [h_dist.get(d, 0) for d in dests], [a_dist.get(d, 0) for d in dests]
        plot_comparison(ax, h_probs, a_probs, dests, f'From: {evt}', 'Probability')
        ax.set_ylim([0, max(max(h_probs + a_probs, default=0) * 1.1, 0.1)])
        ax.text(0.95, 0.95, f'KL={kl:.2f}', transform=ax.transAxes, fontsize=11,
                fontweight='bold', va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor=kl_color(kl), alpha=0.3))

    plt.tight_layout()
    plt.savefig('kl_divergence_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved visualization to kl_divergence_comparison.png")

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    evts, kls = zip(*kl_results) if kl_results else ([], [])
    colors = [kl_color(kl) for kl in kls]
    bars = ax2.barh(evts, kls, color=colors, alpha=0.8)
    ax2.set_xlabel('KL Divergence D(Human || Agent)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Event Type', fontsize=12, fontweight='bold')
    ax2.set_title('Behavioral Divergence Between Human and Agent Traffic', fontsize=14, fontweight='bold')
    if kls:
        ax2.axvline(x=np.mean(kls), color='black', linestyle='--', linewidth=2,
                    alpha=0.5, label=f'Mean={np.mean(kls):.2f}')
        for bar, kl in zip(bars, kls):
            ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                     f'{kl:.2f}', ha='left', va='center', fontsize=10, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig('kl_summary.png', dpi=300, bbox_inches='tight')
    print("Saved KL summary to kl_summary.png")

    h_freq, a_freq = event_frequency_distribution(human_mdp), event_frequency_distribution(agent_mdp)
    h_trans, a_trans = transition_distribution(human_mdp), transition_distribution(agent_mdp)
    freq_kl, trans_kl = kl_divergence(h_freq, a_freq), kl_divergence(h_trans, a_trans)

    print(f"\n=== Global Distribution KL Divergence ===")
    print(f"Event frequency KL: {freq_kl:.4f}")
    print(f"Transition pair KL: {trans_kl:.4f}")

    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    all_evts = sorted(set(h_freq.keys()) | set(a_freq.keys()))
    h_freqs, a_freqs = [h_freq.get(e, 0) for e in all_evts], [a_freq.get(e, 0) for e in all_evts]
    plot_comparison(ax1, h_freqs, a_freqs, all_evts, 'Event Frequency Distribution',
                    'Frequency', freq_kl)

    all_trans = sorted(set(h_trans.keys()) | set(a_trans.keys()))
    top_trans = [t for t, _ in sorted([(t, h_trans.get(t, 0) + a_trans.get(t, 0))
                                        for t in all_trans], key=lambda x: x[1], reverse=True)[:15]]
    h_tprobs, a_tprobs = [h_trans.get(t, 0) for t in top_trans], [a_trans.get(t, 0) for t in top_trans]
    plot_comparison(ax2, h_tprobs, a_tprobs, top_trans, 'Top Transition Pairs Distribution',
                    'Probability', trans_kl)

    plt.tight_layout()
    plt.savefig('global_distributions.png', dpi=300, bbox_inches='tight')
    print("Saved global distributions to global_distributions.png")
