# Glossary

Short definitions point to the thesis **Terminology** appendix in the [PDF](https://pub-d5b94a3c29fd40c6b3881946e463fdb7.r2.dev/thesis-latest.pdf) for full precision.

| Term | Meaning (operational) |
| --- | --- |
| **COI (Cost of Information)** | Expected price premium above a floor under the platform’s policy; thesis KPI for pricing power. |
| **Trajectory \(\tau_s\)** | Ordered session events used as the behavioral record. |
| **Demand proxy \(\hat{q}\)** | Weighted aggregation of actions—what the platform observes instead of true demand. |
| **Contamination \(\alpha\)** | Agent share in the mixture demand model (thesis); not automatically “% of bots” in raw logs. |
| **Transition kernel \(\hat{\mathcal{T}}\)** | MLE Markov model over behavioral states / events for class \(H\) or \(A\). |
| **\(\Delta_H,\Delta_A\)** | Divergence scores vs human/agent prototypes (thesis notation). |
| **\(f(\tau)\)** | Weak agent probability from trajectory (implementation: `engine/lib/coi.py`). |
| **\(\mathcal{G}(\alpha)\)** | Contamination generator: synthetic agent trajectories to reach mixture level \(\alpha\). |
| **DR-RL** | Distributionally robust reinforcement learning training narrative in the thesis. |
| **Ambiguity set / Wasserstein** | Robust optimization neighborhood around an empirical demand law. |
| **Kappa–Lambda architecture** | Thesis term for streaming (online) vs batch/offline learning loops. |
