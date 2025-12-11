# Multi-Task Learning Architecture - Quick Reference

## Current System (Baseline)

```
┌─────────────────────────────────────────────────────────────────┐
│                        CURRENT STATE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Browser Events → Next.js → FastAPI → Kafka (user-interactions)  │
│                                           ↓                       │
│                                      Airflow (every 15min)       │
│                                           ↓                       │
│                              [Messy SessionState Pipeline]       │
│                                           ↓                       │
│                        Simple Rule-Based Pricing:                │
│                        - Surge (if demand > 10)                  │
│                        - Elasticity formula                      │
│                        - Velocity threshold for agents           │
│                                           ↓                       │
│                                    Redis (prices)                │
│                                           ↓                       │
│                                  Pricing Provider API            │
│                                                                   │
│  ISSUES:                                                          │
│  ✗ O(n²) feature extraction                                      │
│  ✗ No supervised ML for agent detection                          │
│  ✗ Simple heuristics (velocity > 5 → agent)                      │
│  ✗ No learning from data                                         │
│  ✗ Margin leakage not effectively addressed                      │
└─────────────────────────────────────────────────────────────────┘
```

## Proposed System (Multi-Task Learning)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           PHASE 1: DATA PIPELINE                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Kafka (user-interactions)                                                │
│         ↓                                                                 │
│  ┌─────────────────────────────────────┐                                 │
│  │  VECTORIZED FEATURE PIPELINE        │                                 │
│  ├─────────────────────────────────────┤                                 │
│  │  1. TemporalFeatureExtractor        │  → 8 features (velocity, etc.)  │
│  │  2. BehavioralFeatureExtractor      │  → 10 features (carts, hovers)  │
│  │  3. ProductFeatureExtractor         │  → 8 features (prices, depth)   │
│  │  4. UserAgentParser                 │  → 3 features (browser type)    │
│  │  5. SessionAggregator               │  → Session-level matrix         │
│  │  6. ExperimentLabelJoiner           │  → Join with xp_human_only      │
│  └─────────────────────────────────────┘                                 │
│         ↓                                                                 │
│  Feature Matrix: [sessionId, 29 features, 3 labels]                      │
│                                                                            │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: SUPERVISED AGENT CLASSIFIER                   │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Feature Matrix (29 features)                                             │
│         ↓                                                                 │
│  ┌────────────────────┐                                                   │
│  │   XGBoost Model    │                                                   │
│  ├────────────────────┤                                                   │
│  │  Input: 29 dims    │                                                   │
│  │  Output: P(agent)  │                                                   │
│  │  Loss: BCE         │                                                   │
│  └────────────────────┘                                                   │
│         ↓                                                                 │
│  Target: ROC-AUC > 0.90                                                   │
│                                                                            │
│  DEPLOYMENT:                                                              │
│  - Real-time inference in Pricing Provider                                │
│  - Dynamic markup: P(agent) > 0.7 → 1.3x price                           │
│  - Retrain daily via Airflow                                              │
│                                                                            │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                  PHASE 3: MULTI-TASK LEARNING MODEL                       │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Input: Session Features (29) + Product Features (10) + Current Price    │
│         ↓                                                                 │
│  ┌───────────────────────────────────────────────────────────┐           │
│  │              MULTI-TASK NEURAL NETWORK                    │           │
│  ├───────────────────────────────────────────────────────────┤           │
│  │                                                             │           │
│  │   ┌──────────────────────┐                                │           │
│  │   │  Session Encoder     │  (Shared)                      │           │
│  │   │  [29] → [128] → [64] │                                │           │
│  │   └──────────┬───────────┘                                │           │
│  │              │                                             │           │
│  │              ├────────────┬───────────────┐                │           │
│  │              ↓            ↓               ↓                │           │
│  │       ┌─────────┐   ┌─────────┐   ┌─────────────┐        │           │
│  │       │ Task A  │   │ Product │   │   Task B    │        │           │
│  │       │ Agent   │   │ Encoder │   │  Purchase   │        │           │
│  │       │ Head    │   │ [10]→16 │   │  Prob Head  │        │           │
│  │       └────┬────┘   └────┬────┘   └──────┬──────┘        │           │
│  │            ↓             └────┬────────────┘               │           │
│  │       P(agent)               ↓                            │           │
│  │                          P(purchase|price)                 │           │
│  │                                                             │           │
│  │   Loss = α·BCE(agent) + β·BCE(purchase)                   │           │
│  │   α=1.0, β=2.0 (tune these weights)                       │           │
│  └───────────────────────────────────────────────────────────┘           │
│         ↓                                                                 │
│  OUTPUTS:                                                                 │
│  1. Agent probability (like Phase 2)                                     │
│  2. Purchase probability given price                                      │
│  3. Session embedding (for knowledge distillation)                        │
│                                                                            │
│  USE CASE:                                                                │
│  Optimal Price = argmax_p [ p · P(purchase|p) · (1 + λ·P(agent)) ]      │
│                                                                            │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                     KNOWLEDGE DISTILLATION BRANCH                         │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Multi-Task Model (teacher)                                               │
│         ↓                                                                 │
│  Generate predictions on validation set                                   │
│         ↓                                                                 │
│  ┌──────────────────────────────────────┐                                │
│  │  Distill to Decision Tree (student)  │                                │
│  ├──────────────────────────────────────┤                                │
│  │  Input: 29 session features          │                                │
│  │  Output: Optimal markup multiplier   │                                │
│  │  Max depth: 5 (interpretable)        │                                │
│  └──────────────────────────────────────┘                                │
│         ↓                                                                 │
│  Extract Human-Readable Rules:                                            │
│                                                                            │
│  IF interaction_velocity > 10 AND cart_to_view_ratio < 0.1:              │
│      markup = 1.3  (likely agent reconnaissance)                          │
│  ELIF unique_products_viewed < 3 AND session_duration > 300:             │
│      markup = 0.9  (engaged human, offer discount)                       │
│  ELSE:                                                                    │
│      markup = 1.0  (baseline)                                             │
│                                                                            │
│  Also: SHAP values for feature importance analysis                        │
│                                                                            │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│              PHASE 4: SYNTHETIC DYNAMIC PRICING SIMULATOR                 │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  PURPOSE: Fast experimentation without real users                         │
│                                                                            │
│  ┌────────────────────────────────────────────────────┐                  │
│  │         DynamicPricingEnv (Gymnasium)              │                  │
│  ├────────────────────────────────────────────────────┤                  │
│  │                                                      │                  │
│  │  State:   [demand, inventory, hour, agent_frac,    │                  │
│  │            avg_velocity]                            │                  │
│  │                                                      │                  │
│  │  Action:  price_multiplier ∈ [0.7, 1.5]           │                  │
│  │                                                      │                  │
│  │  Dynamics:                                          │                  │
│  │  - Simulate user arrivals (Poisson)                │                  │
│  │  - Split into humans (30%) vs agents (70%)         │                  │
│  │  - Purchase probability:                            │                  │
│  │      P_human(buy) = logistic(price, sensitivity=2) │                  │
│  │      P_agent(buy) = logistic(price, sensitivity=5) │                  │
│  │                                                      │                  │
│  │  Reward:  revenue - 0.5 * margin_leakage           │                  │
│  │           where margin_leakage = (oracle_price -   │                  │
│  │                                   actual_price) ×   │                  │
│  │                                   agent_purchases   │                  │
│  └────────────────────────────────────────────────────┘                  │
│         ↓                                                                 │
│  ┌────────────────────────────────────────┐                              │
│  │  Train RL Agent (PPO)                  │                              │
│  ├────────────────────────────────────────┤                              │
│  │  Learn policy: State → Optimal Price   │                              │
│  │  100k timesteps training                │                              │
│  └────────────────────────────────────────┘                              │
│         ↓                                                                 │
│  BENCHMARK vs Baselines:                                                  │
│  - Fixed pricing: 1.0x always                                             │
│  - Simple surge: 1.2x if demand > 10, else 0.9x                         │
│  - Elasticity-based: formula                                              │
│  - RL policy: learned                                                     │
│  - Multi-task + RL: Use MT model predictions as state features           │
│                                                                            │
│  VALIDATION:                                                              │
│  - Calibrate simulator from historical data                               │
│  - Run counterfactuals ("what if agent_frac=0.8?")                       │
│  - A/B test winner on real traffic                                        │
│                                                                            │
└──────────────────────────────────────────────────────────────────────────┘
```

## Data Flow (Production)

```
┌─────────────┐
│   Browser   │
│ (User/Agent)│
└──────┬──────┘
       │ POST /api/ingest (events + experimentId)
       ↓
┌──────────────┐
│  Next.js API │
└──────┬───────┘
       │ Forward events
       ↓
┌──────────────┐
│ FastAPI      │
│ /api/kafka   │
│ /ingest      │
└──────┬───────┘
       │ Publish
       ↓
┌─────────────────────────┐
│ Kafka                   │
│ Topic: user-interactions│
└──────┬──────────────────┘
       │
       ├──────────────────┬──────────────────┐
       ↓                  ↓                  ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│ Airflow      │  │ Real-Time    │  │ Kafka Streams    │
│ (Batch)      │  │ Inference    │  │ (Feature Cache)  │
│              │  │              │  │                  │
│ Daily:       │  │ On Price     │  │ Rolling window   │
│ - Retrain    │  │ Request:     │  │ compute session  │
│   classifier │  │ - Get session│  │ features, push   │
│ - Retrain MT │  │   features   │  │ to Redis         │
│   model      │  │ - Predict    │  │                  │
│ - Publish to │  │   P(agent)   │  │ TTL: 1 hour      │
│   registry   │  │ - Predict    │  │                  │
│              │  │   P(purchase)│  │                  │
│              │  │ - Compute    │  │                  │
│              │  │   optimal_p  │  │                  │
└──────┬───────┘  └──────┬───────┘  └────────┬─────────┘
       │                 │                   │
       ↓                 ↓                   ↓
┌──────────────────────────────────────────────┐
│          Redis (Model Registry)              │
├──────────────────────────────────────────────┤
│ Keys:                                         │
│ - classifier:agent_detector:latest (pickle)  │
│ - multitask_model:latest (state_dict)        │
│ - session_features:{sessionId} (json, TTL)   │
│ - prices:latest (DataFrame)                  │
│ - elasticity:latest (DataFrame)              │
└──────────────────┬───────────────────────────┘
                   │
                   ↓
         ┌─────────────────────┐
         │ Pricing Provider    │
         │ /api/{mode}/price/  │
         │ {productId}         │
         │                     │
         │ GET sessionId       │
         │ → Load features     │
         │ → Load models       │
         │ → Predict           │
         │ → Return price      │
         └─────────┬───────────┘
                   │
                   ↓
         ┌─────────────────────┐
         │   Frontend          │
         │   (Display price)   │
         └─────────────────────┘
```

## Key Metrics

### Model Performance
| Metric | Target | Current | Phase |
|--------|--------|---------|-------|
| Agent Classifier ROC-AUC | >0.90 | N/A (rule-based) | Phase 2 |
| Purchase Predictor ROC-AUC | >0.75 | N/A | Phase 3 |
| Pricing Latency (p99) | <100ms | ~50ms | All |
| Retraining Frequency | Daily | Every 15min (rules) | Phase 2+ |

### Business Impact
| Metric | Target | Current | Phase |
|--------|--------|---------|-------|
| Margin Leakage Reduction | -30% | Baseline | Phase 2-4 |
| Human Conversion Rate | No change | Baseline | All |
| Agent Detection Rate | >85% precision | ~60% (velocity) | Phase 2 |
| Revenue Uplift | +10% | Baseline | Phase 3-4 |

## File Structure (New)

```
experiments/
  ml/
    __init__.py

    # Phase 1: Features
    features/
      __init__.py
      temporal.py           # TemporalFeatureExtractor
      behavioral.py         # BehavioralFeatureExtractor
      product.py            # ProductFeatureExtractor
      useragent.py          # UserAgentParser
      aggregator.py         # SessionAggregator

    pipeline.py             # build_feature_pipeline()
    datasets.py             # load_events_from_kafka(), etc.

    # Phase 2: Classifier
    train_classifier.py     # XGBoost training script

    # Phase 3: Multi-Task
    models/
      __init__.py
      multitask.py          # MultiTaskPricingModel (PyTorch)

    train_multitask.py      # Multi-task training script
    distill.py              # Knowledge distillation

    # Phase 4: Simulator
    simulator/
      __init__.py
      env.py                # DynamicPricingEnv (Gymnasium)
      agents.py             # HumanUser, AgentUser
      train_rl.py           # PPO training

    # Inference
    inference/
      __init__.py
      pricing_service.py    # gRPC service (optional)
      feature_cache.py      # Redis feature store client

    # Notebooks
    notebooks/
      01_eda.ipynb
      02_feature_analysis.ipynb
      03_model_evaluation.ipynb
      04_simulator_calibration.ipynb
```

## Critical Code Changes

### 1. Replace Messy SessionState
**Before:** `experiments/procesing/steps/session.py` (O(n²) loops)
**After:** `experiments/ml/pipeline.py` (vectorized pipeline)

### 2. Upgrade Pricing Provider
**Before:** Simple velocity threshold
**After:** ML model inference with agent probability

### 3. Add Real-Time Feature Store
**Before:** No feature caching
**After:** Kafka Streams → Redis (session features)

### 4. Airflow DAG Upgrades
**Before:** `surge_pricing_pipeline` (rule-based)
**After:** Add `agent_classifier_training_pipeline` (daily retrain)

## Next Actions (Start Here)

1. ✅ **Read gameplan**: See `/home/user/PHANTOM/docs/GAMEPLAN_MULTITASK_PRICING.md`

2. **Create directory structure**:
   ```bash
   mkdir -p experiments/ml/{features,models,simulator,inference,notebooks}
   ```

3. **Pull sample data**:
   ```python
   # experiments/ml/notebooks/01_eda.ipynb
   from kafka import KafkaConsumer
   # Pull 1 week of events, join with experiments table
   # Analyze label distribution, feature correlations
   ```

4. **Prototype first feature extractor**:
   ```python
   # experiments/ml/features/temporal.py
   # Start with TemporalFeatureExtractor
   # Test on 10k events, validate output schema
   ```

5. **Review with team**: Discuss tradeoffs, priorities, timeline

## Questions to Resolve

1. **Label Quality**: How confident are we in `xp_human_only` labels? Should we add manual verification?

2. **Compute Budget**: Do we have GPU access for PyTorch training? (Phase 3)

3. **Latency Requirements**: Is 100ms p99 acceptable for pricing API?

4. **A/B Testing**: Do we have infrastructure for traffic splitting? (Deployment)

5. **Monitoring**: Who owns the Grafana dashboards? What alerting thresholds?

---

**For detailed implementation, see:** `/home/user/PHANTOM/docs/GAMEPLAN_MULTITASK_PRICING.md`
