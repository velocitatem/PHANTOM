# Multi-Task Learning for Dynamic Pricing & Agent Detection
## Comprehensive Gameplan & Architecture

**Date:** 2025-12-11
**Objective:** Transform raw interaction data into a multi-task learning system that (1) discriminates human vs agent behavior and (2) learns pricing heuristics to prevent margin leakage in dynamic pricing algorithms.

---

## Executive Summary

### Current State
- ✅ Rich event collection system capturing 15+ event types (views, carts, hovers, filters)
- ✅ experimentId-based labeling via `xp_human_only` flag
- ✅ Kafka streaming to `user-interactions` topic
- ✅ Airflow pipeline running every 15min with rule-based pricing (surge, elasticity)
- ❌ SessionState pipeline is O(n²) inefficient, messy feature extraction
- ❌ No supervised ML for agent detection (only velocity threshold heuristics)
- ❌ No multi-task learning architecture
- ❌ Pricing models are analytical formulas, not learned from data

### Proposed Solution
1. **Refactor SessionState Pipeline**: Vectorized feature engineering with proper augmentation
2. **Supervised Agent Classifier**: Binary classifier (human/agent) using session features
3. **Multi-Task Learning**: Joint model learning both tasks:
   - Task A: Human/Agent Classification (cross-entropy loss)
   - Task B: Price Sensitivity Prediction (regression loss)
4. **Synthetic Pricing Simulator**: Fast RL-style environment for testing pricing strategies
5. **Knowledge Distillation**: Distill insights from classifier into pricing heuristics

---

## I. DATA ANALYSIS

### What We Actually Collect

**Event Schema** (`web/src/lib/events.ts`):
```typescript
{
  sessionId: string,           // Session UUID (httpOnly cookie)
  experimentId?: string,        // Experiment UUID → LABEL SOURCE
  storeMode: 'hotel' | 'airline',
  ts: string,                  // ISO8601 timestamp
  page: string,                // URL path
  eventName: EventName,        // 15 event types
  productId?: string,          // Product interaction
  metadata?: Record<string, unknown>,  // Product details (price, amenities)
  userAgent?: string
}
```

**Event Types Breakdown:**
- **Navigation:** `page_view`, `view_item_page`, `learn_more_about_item`
- **Cart:** `add_item_to_cart`, `remove_item`, `checkout_start`, `purchase_complete`
- **Filters:** `search`, `filter_for_date`, `filter_for_amenities`, `filter_for_price`, `sort_change`
- **Dwell Signals:** `hover_over_title`, `hover_over_paragraph`, `hover_over_link`, `hover_over_button`
- **Session:** `session_start`

### Ground Truth Labels

**Labeling Strategy:**
```sql
-- Experiment table schema (Supabase)
CREATE TABLE experiments (
  id UUID PRIMARY KEY,              -- This becomes experimentId in events
  subject_name TEXT,
  xp_human_only BOOLEAN,            -- KEY FIELD for labeling
  xp_market_mode TEXT,
  xp_task_id UUID
);
```

**Label Assignment:**
1. Admin creates experiment with `xp_human_only=true` (human) or `false` (agent allowed)
2. Generates start link: `https://phantom-hotel.vercel.app/start-task?uuid={experimentId}`
3. All events from that session carry `experimentId`
4. Join events with experiments table on `experimentId` → get `xp_human_only` label

**Training Set Construction:**
```python
# Pseudocode
events_df = fetch_from_kafka('user-interactions')
experiments_df = fetch_from_supabase('experiments')

labeled_df = events_df.merge(
    experiments_df[['id', 'xp_human_only']],
    left_on='experimentId',
    right_on='id',
    how='inner'  # Only keep sessions with known experimentId
)

# Label mapping
labeled_df['is_agent'] = ~labeled_df['xp_human_only']  # Flip boolean
```

### What We're Missing

**For Pricing Task:**
- ❌ Price paid vs price shown (conversion price tracking)
- ❌ Product availability at interaction time
- ❌ Competitor prices (external data)
- ✅ Demand proxy (view counts, cart adds) - **WE HAVE THIS**
- ✅ Base prices from product metadata - **WE HAVE THIS**

**For Agent Detection:**
- ✅ Interaction velocity, dwell times, product view depth - **WE HAVE THIS**
- ✅ userAgent strings - **WE HAVE THIS**
- ❌ Mouse movement patterns (only have hover events, not trajectories)
- ❌ Typing speed in search fields

---

## II. PHASE 1: Fix SessionState Pipeline

### Current Issues (`experiments/procesing/steps/session.py`)

**Problem 1: O(n²) Complexity**
```python
# Line 77-93: _apply_to_slice()
for idx, row in df.iterrows():  # For EVERY row
    session_df = df[df['sessionId'] == row['sessionId']]  # Scan ALL events
    current_time_events = session_df[session_df['ts'] <= row['ts']]
    features = _extract_features_for_session(current_time_events)
    # This scans the full dataset O(n) for each of n rows = O(n²)
```

**Problem 2: No Proper Feature Augmentation**
- Features computed only at aggregate session level
- No rolling window statistics
- No temporal decay of signals
- Missing product-level features (price seen, amenity preferences)

### Proposed Solution: Vectorized Feature Engineering

**New Pipeline Architecture:**
```
Raw Events (Kafka)
    ↓
SessionGrouper (group by sessionId + experimentId)
    ↓
TemporalFeatureExtractor (vectorized time diffs, velocity)
    ↓
BehavioralFeatureExtractor (interaction counts, ratios)
    ↓
ProductFeatureExtractor (price sensitivity, preference patterns)
    ↓
SessionAggregator (final session-level features)
    ↓
LabelJoiner (merge with experiments table)
    ↓
Feature Matrix + Labels (ready for ML)
```

**Feature Categories:**

**A. Temporal Features (Vectorized)**
```python
class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    def transform(self, events_df):
        # Sort by session and time
        df = events_df.sort_values(['sessionId', 'ts'])

        # Vectorized time differences
        df['ts_dt'] = pd.to_datetime(df['ts'])
        df['time_diff'] = df.groupby('sessionId')['ts_dt'].diff().dt.total_seconds()

        # Rolling statistics (vectorized with groupby)
        df['velocity_5min'] = df.groupby('sessionId').rolling(
            window='5min', on='ts_dt'
        )['eventName'].count().reset_index(drop=True)

        df['avg_time_between_events'] = df.groupby('sessionId')['time_diff'].transform('mean')
        df['std_time_between_events'] = df.groupby('sessionId')['time_diff'].transform('std')

        return df
```

**B. Behavioral Features**
```python
class BehavioralFeatureExtractor:
    def transform(self, events_df):
        session_agg = events_df.groupby('sessionId').agg({
            # Interaction counts
            'eventName': 'count',
            'productId': lambda x: x.nunique(),

            # Event type breakdowns
            'page_view': lambda x: (x == 'page_view').sum(),
            'add_item_to_cart': lambda x: (x == 'add_item_to_cart').sum(),
            'hover_over_*': lambda x: x.str.startswith('hover').sum(),

            # Session duration
            'ts': lambda x: (x.max() - x.min()).total_seconds()
        }).rename(columns={...})

        # Derived ratios
        session_agg['cart_to_view_ratio'] = (
            session_agg['cart_adds'] / (session_agg['item_views'] + 1)
        )
        session_agg['hover_intensity'] = (
            session_agg['total_hovers'] / session_agg['total_interactions']
        )

        return session_agg
```

**C. Product Interaction Features**
```python
class ProductFeatureExtractor:
    def transform(self, events_df):
        # Extract price from metadata when available
        events_df['price_seen'] = events_df['metadata'].apply(
            lambda x: x.get('base_price') if isinstance(x, dict) else None
        )

        product_agg = events_df[events_df['productId'].notna()].groupby('sessionId').agg({
            # Price sensitivity
            'price_seen': ['mean', 'min', 'max', 'std'],

            # Product diversity
            'productId': lambda x: x.nunique(),

            # Max view depth (how many times most-viewed product was viewed)
            'productId': lambda x: x.value_counts().iloc[0] if len(x) > 0 else 0
        })

        # Filter usage (proxy for deliberate search)
        filter_events = events_df[events_df['eventName'].str.contains('filter|search')]
        filter_counts = filter_events.groupby('sessionId').size()

        product_agg['filter_usage'] = filter_counts

        return product_agg
```

**D. Final Feature Matrix**
```python
# Target schema (session-level)
features = pd.DataFrame({
    # Identifiers
    'sessionId': str,
    'experimentId': str,

    # Temporal (8 features)
    'session_duration_sec': float,
    'total_interactions': int,
    'avg_time_between_events': float,
    'std_time_between_events': float,
    'interaction_velocity': float,  # interactions/min
    'max_velocity_5min': float,
    'min_time_between_events': float,
    'session_start_hour': int,  # time of day (0-23)

    # Behavioral (10 features)
    'page_views': int,
    'item_views': int,
    'cart_adds': int,
    'purchases': int,
    'hover_events': int,
    'filter_events': int,
    'cart_to_view_ratio': float,
    'conversion_rate': float,
    'hover_intensity': float,
    'unique_pages_visited': int,

    # Product interaction (8 features)
    'unique_products_viewed': int,
    'product_view_depth': int,  # max times single product viewed
    'avg_price_seen': float,
    'min_price_seen': float,
    'max_price_seen': float,
    'std_price_seen': float,
    'price_range_explored': float,  # max - min
    'filter_usage_count': int,

    # userAgent parsing (3 features)
    'is_headless_browser': bool,  # "HeadlessChrome" in UA
    'is_automation_tool': bool,   # "Selenium", "Playwright", etc.
    'browser_family': str,

    # LABELS
    'is_agent': bool,  # from experiments.xp_human_only
    'purchased': bool,
    'total_revenue': float  # for pricing task
})
```

### Implementation Files

**Create new directory structure:**
```
experiments/
  ml/
    __init__.py
    features/
      __init__.py
      temporal.py           # TemporalFeatureExtractor
      behavioral.py         # BehavioralFeatureExtractor
      product.py            # ProductFeatureExtractor
      useragent.py          # UserAgentParser
    pipeline.py             # Full feature pipeline (sklearn Pipeline)
    datasets.py             # Data loaders (Kafka → Pandas)
```

**Replace messy session.py with:**
```python
# experiments/ml/pipeline.py
from sklearn.pipeline import Pipeline

def build_feature_pipeline():
    return Pipeline([
        ('temporal', TemporalFeatureExtractor()),
        ('behavioral', BehavioralFeatureExtractor()),
        ('product', ProductFeatureExtractor()),
        ('useragent', UserAgentParser()),
        ('aggregator', SessionAggregator()),
        ('label_joiner', ExperimentLabelJoiner())
    ])

# Usage:
pipeline = build_feature_pipeline()
feature_matrix = pipeline.fit_transform(raw_events_df)
# Output: DataFrame with 29 features + 3 labels per session
```

---

## III. PHASE 2: Supervised Agent Classifier

### Model Architecture

**Objective:** Binary classification `is_agent ∈ {0, 1}` from session features

**Model Choice: Gradient Boosting (XGBoost/LightGBM)**
- Handles mixed feature types (continuous, categorical, boolean)
- Interpretable feature importance
- Strong baseline performance
- Fast inference for real-time detection

**Alternative: Neural Network (if needed for multi-task later)**
```python
class AgentClassifier(nn.Module):
    def __init__(self, n_features=29, hidden_dims=[128, 64, 32]):
        super().__init__()

        layers = []
        in_dim = n_features
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))  # Binary logit
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)  # Shape: [batch, 1]
```

### Training Pipeline

**File: `experiments/ml/train_classifier.py`**

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

def train_agent_classifier(feature_matrix_df):
    # Separate features and labels
    X = feature_matrix_df.drop(columns=['sessionId', 'experimentId', 'is_agent', 'purchased', 'total_revenue'])
    y = feature_matrix_df['is_agent'].astype(int)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # XGBoost with class imbalance handling
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc',
        early_stopping_rounds=20
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=10
    )

    # Evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    # Feature importance
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return model, importance_df

# Publish to ModelRegistry
def publish_classifier(model, registry):
    import joblib
    model_bytes = joblib.dumps(model)
    registry.redis_client.set('classifier:agent_detector:latest', model_bytes)
```

### Inference Integration

**Update Pricing Provider** (`backend/provider/app.py`):
```python
from lib.model_registry import ModelRegistry
import joblib

registry = ModelRegistry()

# Load classifier on startup
classifier_bytes = registry.redis_client.get('classifier:agent_detector:latest')
agent_classifier = joblib.loads(classifier_bytes)

@app.get("/api/{mode}/price/{productId}")
def get_price(mode, productId, sessionId, experimentId):
    # Fetch session features from real-time feature store
    session_features = compute_session_features(sessionId)  # New function

    # Predict agent probability
    agent_prob = agent_classifier.predict_proba([session_features])[0, 1]

    # Apply dynamic markup based on agent probability
    base_price = get_base_price(productId)

    if agent_prob > 0.7:  # High confidence agent
        markup = 1.3
    elif agent_prob > 0.4:  # Uncertain
        markup = 1.1
    else:  # Likely human
        markup = 1.0

    final_price = base_price * markup

    return PriceResponse(
        productId=productId,
        price=final_price,
        agent_probability=agent_prob,
        markup=markup
    )
```

**Real-Time Feature Computation:**
```python
# backend/provider/features.py
def compute_session_features(sessionId: str) -> np.ndarray:
    """
    Compute features for active session from recent Kafka events
    Uses sliding window (last 5 minutes)
    """
    consumer = KafkaConsumer('user-interactions')

    # Fetch recent events for this session
    events = []
    for msg in consumer:
        event = msg.value
        if event['sessionId'] == sessionId:
            events.append(event)

        # Stop after collecting enough history or timeout
        if len(events) > 100 or time_elapsed > 300:
            break

    events_df = pd.DataFrame(events)

    # Apply feature pipeline (same as training)
    feature_pipeline = load_feature_pipeline()
    features = feature_pipeline.transform(events_df)

    return features.values[0]  # Return single row as array
```

---

## IV. PHASE 3: Multi-Task Learning Architecture

### Conceptual Framework

**Two Tasks:**
1. **Task A: Agent Classification** (what we just built)
   - Input: Session features
   - Output: P(is_agent)
   - Loss: Binary cross-entropy

2. **Task B: Price Sensitivity Prediction**
   - Input: Session features + Product features + Current price
   - Output: P(purchase | price) or Willingness-to-Pay (WTP)
   - Loss: Binary cross-entropy (purchase) or MSE (WTP)

**Shared Representation Hypothesis:**
- Both tasks benefit from learning session behavioral patterns
- Agent sessions have different price sensitivity than human sessions
- Shared feature encoder can learn robust representations

### Multi-Task Network Architecture

```python
class MultiTaskPricingModel(nn.Module):
    """
    Shared encoder for session features
    Two task-specific heads
    """
    def __init__(self, n_session_features=29, n_product_features=10):
        super().__init__()

        # Shared session encoder
        self.session_encoder = nn.Sequential(
            nn.Linear(n_session_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # Product encoder (processes product features + price)
        self.product_encoder = nn.Sequential(
            nn.Linear(n_product_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        # Task A: Agent classification head
        self.agent_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Logit for is_agent
        )

        # Task B: Purchase probability head (uses both session + product)
        self.purchase_head = nn.Sequential(
            nn.Linear(64 + 16, 32),  # Concatenate session + product encodings
            nn.ReLU(),
            nn.Linear(32, 1)  # Logit for purchase probability
        )

    def forward(self, session_features, product_features):
        # Encode session (shared)
        session_enc = self.session_encoder(session_features)

        # Encode product
        product_enc = self.product_encoder(product_features)

        # Task A prediction
        agent_logit = self.agent_head(session_enc)

        # Task B prediction
        combined = torch.cat([session_enc, product_enc], dim=1)
        purchase_logit = self.purchase_head(combined)

        return {
            'agent_logit': agent_logit,
            'purchase_logit': purchase_logit,
            'session_embedding': session_enc  # For knowledge distillation
        }
```

### Multi-Task Loss Function

```python
def multi_task_loss(outputs, targets, task_weights={'agent': 1.0, 'purchase': 2.0}):
    """
    Weighted combination of task losses

    Args:
        outputs: dict with 'agent_logit', 'purchase_logit'
        targets: dict with 'is_agent', 'purchased'
        task_weights: importance weights for each task
    """
    # Task A: Agent classification
    loss_agent = F.binary_cross_entropy_with_logits(
        outputs['agent_logit'].squeeze(),
        targets['is_agent'].float()
    )

    # Task B: Purchase prediction
    loss_purchase = F.binary_cross_entropy_with_logits(
        outputs['purchase_logit'].squeeze(),
        targets['purchased'].float()
    )

    # Weighted sum
    total_loss = (
        task_weights['agent'] * loss_agent +
        task_weights['purchase'] * loss_purchase
    )

    return total_loss, {
        'loss_agent': loss_agent.item(),
        'loss_purchase': loss_purchase.item(),
        'loss_total': total_loss.item()
    }
```

### Training Loop

**File: `experiments/ml/train_multitask.py`**

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

def train_multitask_model(feature_matrix_df, n_epochs=100):
    # Prepare data
    X_session = feature_matrix_df[SESSION_FEATURE_COLS].values
    X_product = feature_matrix_df[PRODUCT_FEATURE_COLS].values
    y_agent = feature_matrix_df['is_agent'].values
    y_purchase = feature_matrix_df['purchased'].values

    # Convert to tensors
    dataset = TensorDataset(
        torch.FloatTensor(X_session),
        torch.FloatTensor(X_product),
        torch.FloatTensor(y_agent),
        torch.FloatTensor(y_purchase)
    )

    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Initialize model
    model = MultiTaskPricingModel(
        n_session_features=X_session.shape[1],
        n_product_features=X_product.shape[1]
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    # Training loop
    for epoch in range(n_epochs):
        model.train()
        epoch_losses = {'agent': [], 'purchase': [], 'total': []}

        for X_sess_batch, X_prod_batch, y_agent_batch, y_purch_batch in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(X_sess_batch, X_prod_batch)

            # Compute loss
            loss, loss_dict = multi_task_loss(
                outputs,
                {'is_agent': y_agent_batch, 'purchased': y_purch_batch}
            )

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track losses
            for k, v in loss_dict.items():
                epoch_losses[k.replace('loss_', '')].append(v)

        # Log epoch stats
        avg_loss = np.mean(epoch_losses['total'])
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, "
              f"Agent={np.mean(epoch_losses['agent']):.4f}, "
              f"Purchase={np.mean(epoch_losses['purchase']):.4f}")

        scheduler.step(avg_loss)

    return model
```

### Knowledge Distillation for Pricing Heuristics

**Goal:** Extract interpretable pricing rules from the learned multi-task model

**Method 1: Decision Tree Distillation**
```python
from sklearn.tree import DecisionTreeClassifier, export_text

def distill_pricing_rules(multitask_model, feature_matrix_df):
    """
    Train interpretable decision tree to mimic multi-task model's pricing decisions
    """
    # Get embeddings from multi-task model
    X_session = torch.FloatTensor(feature_matrix_df[SESSION_FEATURE_COLS].values)
    X_product = torch.FloatTensor(feature_matrix_df[PRODUCT_FEATURE_COLS].values)

    with torch.no_grad():
        outputs = multitask_model(X_session, X_product)
        session_embeddings = outputs['session_embedding'].numpy()
        agent_probs = torch.sigmoid(outputs['agent_logit']).numpy()
        purchase_probs = torch.sigmoid(outputs['purchase_logit']).numpy()

    # Compute optimal markup based on model predictions
    # High agent prob + low purchase prob → high markup
    optimal_markup = compute_markup_from_probs(agent_probs, purchase_probs)

    # Train decision tree to predict optimal markup
    distilled_tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=50)
    distilled_tree.fit(feature_matrix_df[SESSION_FEATURE_COLS], optimal_markup)

    # Extract human-readable rules
    rules_text = export_text(distilled_tree, feature_names=SESSION_FEATURE_COLS)

    print("Distilled Pricing Rules:")
    print(rules_text)

    return distilled_tree, rules_text

def compute_markup_from_probs(agent_probs, purchase_probs):
    """
    Heuristic:
    - High agent prob → increase markup (exploit reconnaissance)
    - Low purchase prob → decrease markup (improve conversion)
    """
    markup = 1.0 + 0.5 * agent_probs - 0.3 * purchase_probs
    return np.clip(markup, 0.7, 1.5)  # Bounds: [70% to 150%]
```

**Method 2: SHAP Values for Feature Importance**
```python
import shap

def explain_pricing_model(multitask_model, feature_matrix_df):
    """Use SHAP to understand which features drive pricing decisions"""

    # Wrap model for SHAP
    def model_predict(X_session):
        X_session_t = torch.FloatTensor(X_session)
        X_product_t = torch.zeros(len(X_session), n_product_features)  # Placeholder

        with torch.no_grad():
            outputs = multitask_model(X_session_t, X_product_t)
            agent_probs = torch.sigmoid(outputs['agent_logit']).numpy()

        return agent_probs

    # Compute SHAP values
    explainer = shap.KernelExplainer(model_predict, feature_matrix_df[SESSION_FEATURE_COLS].sample(100))
    shap_values = explainer.shap_values(feature_matrix_df[SESSION_FEATURE_COLS].sample(200))

    # Visualize
    shap.summary_plot(shap_values, feature_matrix_df[SESSION_FEATURE_COLS].sample(200))

    return shap_values
```

---

## V. PHASE 4: Synthetic Dynamic Pricing Simulator

### Motivation

**Problem:** Real-world pricing experiments are:
- Slow (need to wait for user sessions)
- Expensive (margin leakage during exploration)
- Risky (can break revenue if pricing goes wrong)

**Solution:** Fast synthetic environment to:
1. Simulate user/agent behavior
2. Test pricing strategies in closed loop
3. Train RL agents for dynamic pricing
4. Validate multi-task model predictions

### Simulator Architecture

**File: `experiments/ml/simulator/env.py`**

```python
import gymnasium as gym
import numpy as np
from typing import Dict, Tuple

class DynamicPricingEnv(gym.Env):
    """
    Synthetic environment for dynamic pricing with human/agent users

    State: [current_demand, inventory_level, time_of_day, agent_fraction, avg_session_velocity]
    Action: price_multiplier ∈ [0.7, 1.5] (continuous)
    Reward: revenue - margin_leakage_penalty
    """

    def __init__(self,
                 n_products=20,
                 n_timesteps=1000,
                 human_price_sensitivity=2.0,  # Elasticity
                 agent_price_sensitivity=5.0,  # Agents more price-aware
                 agent_fraction=0.3):
        super().__init__()

        self.n_products = n_products
        self.n_timesteps = n_timesteps
        self.human_sensitivity = human_price_sensitivity
        self.agent_sensitivity = agent_price_sensitivity
        self.agent_fraction = agent_fraction

        # State space: [demand, inventory, hour, agent_frac, velocity]
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),
            high=np.array([100, 100, 24, 1, 20]),
            dtype=np.float32
        )

        # Action space: price multiplier
        self.action_space = gym.spaces.Box(
            low=0.7, high=1.5, shape=(1,), dtype=np.float32
        )

        # Internal state
        self.base_prices = np.random.uniform(100, 500, n_products)
        self.inventory = np.full(n_products, 50)
        self.timestep = 0
        self.demand_history = []

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.inventory = np.full(self.n_products, 50)
        self.timestep = 0
        self.demand_history = []
        return self._get_state(), {}

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one pricing decision

        Args:
            action: price_multiplier

        Returns:
            state, reward, terminated, truncated, info
        """
        price_multiplier = action[0]

        # Simulate user arrivals (Poisson process)
        hour = self.timestep % 24
        arrival_rate = self._get_arrival_rate(hour)
        n_arrivals = np.random.poisson(arrival_rate)

        # Separate humans and agents
        n_agents = np.random.binomial(n_arrivals, self.agent_fraction)
        n_humans = n_arrivals - n_agents

        # Simulate purchase decisions
        current_price = self.base_prices[0] * price_multiplier  # Simplified: 1 product

        # Human purchase probability (logistic demand curve)
        human_purchase_prob = self._purchase_probability(
            current_price,
            self.base_prices[0],
            self.human_sensitivity
        )
        human_purchases = np.random.binomial(n_humans, human_purchase_prob)

        # Agent purchase probability (more sensitive)
        agent_purchase_prob = self._purchase_probability(
            current_price,
            self.base_prices[0],
            self.agent_sensitivity
        )
        agent_purchases = np.random.binomial(n_agents, agent_purchase_prob)

        # Compute revenue
        revenue = current_price * (human_purchases + agent_purchases)

        # Compute margin leakage (agents getting oracle prices)
        oracle_price = self.base_prices[0] * 1.2  # Optimal price for agents
        margin_leakage = (oracle_price - current_price) * agent_purchases
        margin_leakage = max(0, margin_leakage)  # Only count if we underpriced

        # Reward = revenue - margin leakage penalty
        reward = revenue - 0.5 * margin_leakage

        # Update inventory
        total_purchases = human_purchases + agent_purchases
        self.inventory[0] -= total_purchases

        # Track demand
        self.demand_history.append(n_arrivals)

        # Next state
        self.timestep += 1
        terminated = self.timestep >= self.n_timesteps
        truncated = self.inventory[0] <= 0

        state = self._get_state()
        info = {
            'revenue': revenue,
            'margin_leakage': margin_leakage,
            'human_purchases': human_purchases,
            'agent_purchases': agent_purchases,
            'agent_fraction': n_agents / n_arrivals if n_arrivals > 0 else 0
        }

        return state, reward, terminated, truncated, info

    def _get_state(self) -> np.ndarray:
        """Current state observation"""
        demand = np.mean(self.demand_history[-10:]) if self.demand_history else 0
        inventory = self.inventory[0]
        hour = self.timestep % 24
        agent_frac = self.agent_fraction  # Simplified
        velocity = np.random.uniform(1, 10)  # Placeholder

        return np.array([demand, inventory, hour, agent_frac, velocity], dtype=np.float32)

    def _get_arrival_rate(self, hour: int) -> float:
        """Simulate time-of-day arrival patterns"""
        # Higher demand during business hours (9am-5pm)
        if 9 <= hour <= 17:
            return 10.0
        else:
            return 3.0

    def _purchase_probability(self, current_price, base_price, sensitivity):
        """
        Logistic demand curve
        P(purchase) = 1 / (1 + exp(sensitivity * (log(current_price) - log(base_price))))
        """
        price_ratio = current_price / base_price
        logit = sensitivity * (np.log(price_ratio))
        prob = 1 / (1 + np.exp(logit))
        return np.clip(prob, 0.01, 0.99)
```

### Behavioral Models

**File: `experiments/ml/simulator/agents.py`**

```python
class SyntheticUser:
    """Base class for simulated users"""
    def __init__(self, price_sensitivity, session_velocity):
        self.price_sensitivity = price_sensitivity
        self.session_velocity = session_velocity  # interactions/min

    def generate_session(self) -> List[Event]:
        """Generate event sequence"""
        raise NotImplementedError

class HumanUser(SyntheticUser):
    """Simulates human browsing behavior"""
    def __init__(self):
        super().__init__(
            price_sensitivity=2.0,  # Moderate
            session_velocity=np.random.uniform(1, 3)  # Slow browsing
        )

    def generate_session(self) -> List[Event]:
        events = []

        # Human pattern: slow, exploratory
        n_page_views = np.random.poisson(5)
        n_hovers = np.random.poisson(8)
        n_products_viewed = np.random.randint(2, 6)

        # Time between events (seconds)
        inter_event_times = np.random.exponential(20, size=n_page_views)

        for i in range(n_page_views):
            events.append({
                'eventName': 'page_view',
                'timestamp': sum(inter_event_times[:i]),
                'velocity': self.session_velocity
            })

        # Purchase decision (price-dependent)
        purchase_prob = self._compute_purchase_prob(current_price)
        if np.random.rand() < purchase_prob:
            events.append({'eventName': 'purchase_complete', ...})

        return events

class AgentUser(SyntheticUser):
    """Simulates agent/bot behavior"""
    def __init__(self):
        super().__init__(
            price_sensitivity=5.0,  # Highly price-aware
            session_velocity=np.random.uniform(10, 20)  # Fast reconnaissance
        )

    def generate_session(self) -> List[Event]:
        events = []

        # Agent pattern: fast, systematic
        n_products_viewed = np.random.randint(15, 30)  # Views many products
        inter_event_times = np.random.exponential(2, size=n_products_viewed)  # Fast

        for i in range(n_products_viewed):
            events.append({
                'eventName': 'view_item_page',
                'timestamp': sum(inter_event_times[:i]),
                'velocity': self.session_velocity
            })

        # Agents rarely purchase (reconnaissance)
        purchase_prob = 0.05
        if np.random.rand() < purchase_prob:
            events.append({'eventName': 'purchase_complete', ...})

        return events
```

### RL Training for Pricing Policy

**File: `experiments/ml/simulator/train_rl.py`**

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def train_pricing_policy():
    """Train RL agent to learn optimal pricing strategy"""

    # Create environment
    env = DynamicPricingEnv(agent_fraction=0.3)
    env = DummyVecEnv([lambda: env])

    # Train PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        verbose=1,
        tensorboard_log="./logs/"
    )

    model.learn(total_timesteps=100_000)

    # Evaluate
    obs = env.reset()
    total_reward = 0
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            print(f"Episode reward: {total_reward}")
            break

    return model

# Compare with baseline strategies
def benchmark_strategies():
    strategies = {
        'fixed': lambda state: np.array([1.0]),  # No markup
        'surge': lambda state: np.array([1.2]) if state[0] > 10 else np.array([0.9]),
        'rl': lambda state: rl_model.predict(state)[0]
    }

    for name, policy in strategies.items():
        env = DynamicPricingEnv()
        total_reward = evaluate_policy(env, policy, n_episodes=100)
        print(f"{name}: avg reward = {total_reward:.2f}")
```

### Integration with Multi-Task Model

**Use multi-task model predictions as state features:**

```python
class EnhancedPricingEnv(DynamicPricingEnv):
    """Environment augmented with multi-task model predictions"""

    def __init__(self, multitask_model):
        super().__init__()
        self.multitask_model = multitask_model

    def _get_state(self):
        # Get base state
        base_state = super()._get_state()

        # Generate synthetic session features
        session_features = self._generate_session_features()

        # Get multi-task model predictions
        with torch.no_grad():
            outputs = self.multitask_model(
                torch.FloatTensor(session_features).unsqueeze(0),
                torch.zeros(1, n_product_features)
            )
            agent_prob = torch.sigmoid(outputs['agent_logit']).item()
            purchase_prob = torch.sigmoid(outputs['purchase_logit']).item()

        # Augment state with model predictions
        enhanced_state = np.concatenate([
            base_state,
            [agent_prob, purchase_prob]
        ])

        return enhanced_state
```

---

## VI. IMPLEMENTATION ROADMAP

### Phase 1: Foundation (Week 1-2)

**1.1 Refactor Feature Pipeline** ✓
- [ ] Create `experiments/ml/features/` directory
- [ ] Implement `TemporalFeatureExtractor` (vectorized)
- [ ] Implement `BehavioralFeatureExtractor`
- [ ] Implement `ProductFeatureExtractor`
- [ ] Implement `UserAgentParser`
- [ ] Create end-to-end `build_feature_pipeline()`
- [ ] Write unit tests for each extractor
- [ ] Benchmark: ensure <5min processing for 100k events

**1.2 Data Loading** ✓
- [ ] Implement `experiments/ml/datasets.py`
- [ ] Add function `load_events_from_kafka(topic, time_range)`
- [ ] Add function `load_experiments_from_supabase()`
- [ ] Add function `create_labeled_dataset()` (joins events + experiments)
- [ ] Add data validation (check for missing experimentIds, etc.)

**1.3 Model Registry Updates** ✓
- [ ] Extend `lib/model_registry.py` to store:
  - Pickled sklearn/XGBoost models
  - PyTorch model state_dicts
  - Feature column names (schema versioning)
- [ ] Add `publish_classifier(model, metadata)`
- [ ] Add `publish_multitask_model(model, metadata)`

### Phase 2: Supervised Classifier (Week 2-3)

**2.1 Training Pipeline** ✓
- [ ] Create `experiments/ml/train_classifier.py`
- [ ] Load labeled dataset with feature pipeline
- [ ] Train XGBoost classifier
- [ ] Evaluate: ROC-AUC, precision/recall, confusion matrix
- [ ] Generate feature importance report
- [ ] Publish model to registry

**2.2 Inference Integration** ✓
- [ ] Update `backend/provider/app.py`
- [ ] Implement `compute_session_features(sessionId)` for real-time inference
- [ ] Add `/api/detect-agent/{sessionId}` endpoint (debugging)
- [ ] Modify `/api/{mode}/price/{productId}` to:
  - Compute agent probability
  - Apply dynamic markup based on probability
  - Log (sessionId, agent_prob, markup) to Kafka

**2.3 Monitoring** ✓
- [ ] Create Airflow DAG: `agent_classifier_training_pipeline`
  - Runs daily
  - Fetches last 7 days of data
  - Retrains classifier
  - Publishes to registry
- [ ] Add Grafana dashboard:
  - Agent detection rate over time
  - Precision/recall metrics
  - Markup distribution (human vs agent sessions)

### Phase 3: Multi-Task Learning (Week 3-5)

**3.1 Model Development** ✓
- [ ] Create `experiments/ml/models/multitask.py`
- [ ] Implement `MultiTaskPricingModel` (PyTorch)
- [ ] Implement `multi_task_loss()`
- [ ] Create `experiments/ml/train_multitask.py`

**3.2 Training** ✓
- [ ] Prepare product-level dataset (not just session-level)
  - Each row: (session_features, product_features, price, is_agent, purchased)
  - Requires joining events with product catalog
- [ ] Train multi-task model
- [ ] Evaluate both tasks:
  - Agent classification: ROC-AUC
  - Purchase prediction: ROC-AUC, calibration plot
- [ ] Hyperparameter tuning (task weights, learning rate, architecture)

**3.3 Knowledge Distillation** ✓
- [ ] Implement `distill_pricing_rules()` (decision tree)
- [ ] Implement `explain_pricing_model()` (SHAP)
- [ ] Generate interpretable pricing rule report
- [ ] Validate: compare distilled tree performance vs full model

**3.4 Deployment** ✓
- [ ] Serialize PyTorch model to ONNX for fast inference
- [ ] Update pricing provider to use multi-task model:
  - Predict agent probability
  - Predict purchase probability given current price
  - Compute optimal price (maximize expected revenue)
- [ ] A/B test: multi-task pricing vs baseline elasticity pricing

### Phase 4: Synthetic Simulator (Week 5-6)

**4.1 Environment** ✓
- [ ] Create `experiments/ml/simulator/`
- [ ] Implement `DynamicPricingEnv` (Gymnasium)
- [ ] Implement `HumanUser` and `AgentUser` behavioral models
- [ ] Validate environment:
  - Test human_sensitivity vs agent_sensitivity differentiation
  - Visualize demand curves for both user types

**4.2 RL Training** ✓
- [ ] Create `experiments/ml/simulator/train_rl.py`
- [ ] Train PPO agent
- [ ] Benchmark against baseline strategies (fixed, surge, elasticity)
- [ ] Visualize: reward curves, pricing decisions over time

**4.3 Integration with Multi-Task Model** ✓
- [ ] Implement `EnhancedPricingEnv` (uses multi-task model predictions)
- [ ] Train RL agent in enhanced environment
- [ ] Compare: RL with model predictions vs RL without

**4.4 Validation** ✓
- [ ] Run counterfactual simulations:
  - "What if agent fraction was 50%?"
  - "What if agents were less price-sensitive?"
- [ ] Generate risk analysis report for pricing strategies

### Phase 5: Production Integration (Week 7-8)

**5.1 Real-Time Feature Store** ✓
- [ ] Implement session feature caching (Redis)
- [ ] Create Kafka Streams job:
  - Consumes `user-interactions`
  - Computes rolling session features
  - Publishes to Redis with TTL
- [ ] Update `compute_session_features()` to read from cache

**5.2 Pricing Service Refactor** ✓
- [ ] Decouple pricing logic from provider service
- [ ] Create `experiments/ml/inference/pricing_service.py`
- [ ] Expose gRPC API for low-latency pricing
- [ ] Load model once on startup (not per-request)

**5.3 Observability** ✓
- [ ] Add OpenTelemetry tracing to pricing service
- [ ] Create Grafana dashboard:
  - Pricing latency (p50, p99)
  - Revenue metrics (human vs agent)
  - Model prediction distribution
- [ ] Set up alerts:
  - Agent detection rate spike
  - Margin leakage threshold exceeded
  - Model inference failures

**5.4 Documentation** ✓
- [ ] Write model cards for both models:
  - Intended use
  - Training data characteristics
  - Performance metrics
  - Limitations & biases
- [ ] Create runbook for retraining pipelines
- [ ] API documentation for pricing service

---

## VII. TECHNICAL STACK

### Data Engineering
- **Streaming:** Kafka (existing)
- **Storage:** Supabase (PostgreSQL) for metadata, Kafka for events
- **Feature Store:** Redis (for real-time session features)
- **Orchestration:** Airflow (existing)

### Machine Learning
- **Feature Engineering:** Pandas, NumPy, Scikit-learn
- **Classifier:** XGBoost / LightGBM (for speed + interpretability)
- **Multi-Task Model:** PyTorch
- **RL:** Stable-Baselines3 (PPO)
- **Explainability:** SHAP, Scikit-learn decision trees
- **Model Registry:** Redis (lightweight) or MLflow (if scaling up)

### Inference
- **Serving:** ONNX Runtime (for PyTorch models) or direct PyTorch
- **API:** FastAPI (existing backend)
- **Caching:** Redis

### Monitoring
- **Metrics:** Prometheus
- **Dashboards:** Grafana
- **Tracing:** OpenTelemetry

---

## VIII. KEY DECISIONS & TRADEOFFS

### Decision 1: XGBoost vs Neural Network for Classifier
**Choice:** Start with XGBoost, migrate to NN if multi-task learning shows benefit

**Rationale:**
- XGBoost: Faster training, interpretable, strong baseline
- Neural Network: Required for multi-task learning with shared encoder
- **Approach:** Build both, use XGBoost for initial deployment

### Decision 2: Real-Time vs Batch Feature Engineering
**Choice:** Hybrid approach
- **Batch:** Full feature pipeline in Airflow (daily retraining)
- **Real-Time:** Lightweight rolling features in Kafka Streams (for inference)

**Tradeoff:** Complexity vs latency
- Real-time adds complexity but enables dynamic pricing
- Batch is simpler but can't adapt mid-session

### Decision 3: Separate Simulator vs Production Trace Replay
**Choice:** Separate simulator (not trace replay)

**Rationale:**
- **Trace replay:** More realistic but slow, requires large historical dataset
- **Simulator:** Fast iteration, can test edge cases (high agent fraction, etc.)
- **Approach:** Use simulator for development, validate on historical traces before production

### Decision 4: Knowledge Distillation Method
**Choice:** Decision tree distillation + SHAP explanations

**Rationale:**
- Decision trees: Generate actionable rules for non-ML stakeholders
- SHAP: Understand feature importance for model debugging
- Both complement each other (global rules + local explanations)

---

## IX. SUCCESS METRICS

### Model Performance
- **Agent Classifier:**
  - ROC-AUC > 0.90
  - Precision > 0.85 at 80% recall (minimize false agent flags)
- **Purchase Predictor:**
  - ROC-AUC > 0.75
  - Calibration error < 0.1 (predicted probs match observed rates)

### Business Impact
- **Margin Leakage Reduction:** Target 30% reduction in `L_agent`
- **Revenue:** No degradation in human conversion rate
- **Latency:** p99 pricing latency < 100ms

### Operational
- **Retraining Frequency:** Daily (automated)
- **Model Drift Detection:** Alert if agent detection rate changes >20% week-over-week
- **Uptime:** 99.9% availability for pricing service

---

## X. RISKS & MITIGATION

### Risk 1: Label Noise
**Issue:** `xp_human_only` flag may be incorrect (humans acting like agents, vice versa)

**Mitigation:**
- Use soft labels (agent probability) instead of hard classification
- Implement active learning: flag uncertain sessions for manual review
- Combine flag with behavioral heuristics (velocity > threshold)

### Risk 2: Concept Drift
**Issue:** Agent behavior evolves to evade detection

**Mitigation:**
- Monitor model performance metrics daily
- Implement champion/challenger A/B testing for new models
- Retrain weekly with latest data

### Risk 3: Revenue Impact
**Issue:** Aggressive agent markup hurts conversion

**Mitigation:**
- Start with conservative markup (10% max)
- Run A/B test with 10% traffic
- Monitor revenue closely, rollback if <5% degradation

### Risk 4: Simulator Misalignment
**Issue:** Simulator doesn't match real-world dynamics

**Mitigation:**
- Calibrate simulator parameters from historical data
- Validate RL policies on historical traces before production
- Run small-scale live experiments

---

## XI. NEXT STEPS (Immediate Actions)

1. **Review this document** with team for alignment
2. **Set up project structure:**
   ```bash
   mkdir -p experiments/ml/{features,models,simulator,inference}
   mkdir -p experiments/ml/notebooks  # For EDA
   ```
3. **Create sample dataset:**
   - Pull 1 week of events from Kafka
   - Join with experiments table
   - Validate label distribution (human/agent ratio)
4. **Prototype feature pipeline:**
   - Start with `TemporalFeatureExtractor`
   - Benchmark on 10k events
   - Ensure output matches expected schema
5. **Kickoff Phase 1** tasks from roadmap

---

## XII. REFERENCES & RESOURCES

### Papers
- [Multi-Task Learning Using Uncertainty to Weigh Losses](https://arxiv.org/abs/1705.07115) - Kendall et al.
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) - Hinton et al.
- [Deep Reinforcement Learning for Online Pricing](https://arxiv.org/abs/1912.02572)

### Code Examples
- Existing pipeline: `/home/user/PHANTOM/experiments/procesing/pipelines.py`
- Existing pricers: `/home/user/PHANTOM/experiments/procesing/pricers/`
- Agent runner: `/home/user/PHANTOM/experiments/agents/agent.py`

### Internal Documentation
- Event schema: `/home/user/PHANTOM/web/src/lib/events.ts`
- Pricing provider API: `/home/user/PHANTOM/backend/provider/app.py`
- Airflow DAGs: `/home/user/PHANTOM/experiments/airflow/dags/`

---

## APPENDIX A: Feature Schema

```python
SESSION_FEATURE_COLS = [
    # Temporal (8)
    'session_duration_sec',
    'total_interactions',
    'avg_time_between_events',
    'std_time_between_events',
    'interaction_velocity',
    'max_velocity_5min',
    'min_time_between_events',
    'session_start_hour',

    # Behavioral (10)
    'page_views',
    'item_views',
    'cart_adds',
    'purchases',
    'hover_events',
    'filter_events',
    'cart_to_view_ratio',
    'conversion_rate',
    'hover_intensity',
    'unique_pages_visited',

    # Product (8)
    'unique_products_viewed',
    'product_view_depth',
    'avg_price_seen',
    'min_price_seen',
    'max_price_seen',
    'std_price_seen',
    'price_range_explored',
    'filter_usage_count',

    # UserAgent (3)
    'is_headless_browser',
    'is_automation_tool',
    'browser_family'
]

PRODUCT_FEATURE_COLS = [
    'base_price',
    'current_demand',
    'inventory_level',
    'avg_demand_last_hour',
    'price_elasticity',
    'n_amenities',
    'is_refundable',
    'days_until_date',
    'competitor_price_diff',  # If available
    'view_count_last_hour'
]

LABEL_COLS = [
    'is_agent',        # Binary: 0=human, 1=agent
    'purchased',       # Binary: 0=no purchase, 1=purchase
    'total_revenue'    # Float: revenue from this session
]
```

---

## APPENDIX B: Sample Code Snippets

### Data Loading
```python
# experiments/ml/datasets.py
from kafka import KafkaConsumer
import pandas as pd
from supabase import create_client

def load_events_from_kafka(topic='user-interactions', hours=24):
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers='localhost:9092',
        auto_offset_reset='earliest',
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )

    events = []
    cutoff_time = datetime.now() - timedelta(hours=hours)

    for message in consumer:
        event = message.value
        if pd.to_datetime(event['ts']) > cutoff_time:
            events.append(event)

    return pd.DataFrame(events)

def load_experiments_from_supabase():
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    response = client.table('experiments').select('*').execute()
    return pd.DataFrame(response.data)

def create_labeled_dataset():
    events_df = load_events_from_kafka(hours=168)  # 1 week
    experiments_df = load_experiments_from_supabase()

    # Join
    labeled_df = events_df.merge(
        experiments_df[['id', 'xp_human_only']],
        left_on='experimentId',
        right_on='id',
        how='inner'
    )

    # Create labels
    labeled_df['is_agent'] = ~labeled_df['xp_human_only']

    # Apply feature pipeline
    feature_pipeline = build_feature_pipeline()
    feature_matrix = feature_pipeline.fit_transform(labeled_df)

    return feature_matrix
```

### Feature Extraction
```python
# experiments/ml/features/temporal.py
from sklearn.base import BaseEstimator, TransformerMixin

class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, events_df):
        df = events_df.copy()
        df['ts_dt'] = pd.to_datetime(df['ts'])
        df = df.sort_values(['sessionId', 'ts_dt'])

        # Time diffs (vectorized)
        df['time_diff'] = df.groupby('sessionId')['ts_dt'].diff().dt.total_seconds()

        # Session-level aggregates
        session_temporal = df.groupby('sessionId').agg({
            'ts_dt': [
                ('session_duration_sec', lambda x: (x.max() - x.min()).total_seconds()),
                ('session_start_hour', lambda x: x.min().hour)
            ],
            'time_diff': [
                ('avg_time_between_events', 'mean'),
                ('std_time_between_events', 'std'),
                ('min_time_between_events', 'min')
            ],
            'eventName': [
                ('total_interactions', 'count')
            ]
        })

        # Flatten multi-index columns
        session_temporal.columns = [col[1] if col[1] else col[0] for col in session_temporal.columns]

        # Compute velocity
        session_temporal['interaction_velocity'] = (
            session_temporal['total_interactions'] /
            (session_temporal['session_duration_sec'] / 60 + 1e-6)  # per minute
        )

        return session_temporal.reset_index()
```

---

**END OF GAMEPLAN**
