<img width="200" align="left" src="https://github.com/user-attachments/assets/d148b00d-e9f9-4280-89cc-0cc866e17251" />

### PHANTOM

[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-sm.svg)](https://huggingface.co/datasets/velocitatem/whoclickedit)
[![Build PDF](https://github.com/velocitatem/PHANTOM/actions/workflows/latex.yml/badge.svg)](https://github.com/velocitatem/PHANTOM/actions/workflows/latex.yml)
[![Paper](https://img.shields.io/badge/Paper-PDF-red?logo=adobe-acrobat-reader)](https://pub-d5b94a3c29fd40c6b3881946e463fdb7.r2.dev/thesis-latest.pdf)
[![TPU Research Cloud](https://img.shields.io/badge/TPU%20Research%20Cloud-TRC%20supported-4285F4?logo=googlecloud&logoColor=white)](https://sites.research.google/trc/faq/)
[![Vercel Deploy](https://deploy-badge.vercel.app/?url=https://phantom-hotel.vercel.app&name=Hotel)](https://phantom-hotel.vercel.app)
[![Vercel Deploy](https://deploy-badge.vercel.app/?url=https://phantom-airline.vercel.app&name=Airline)](https://phantom-airline.vercel.app)



```mermaid
mindmap
  PHANTOM((PHANTOM Project))
    North Star
      Study how automated actors change markets 
      Build an experimentation platform for real-world-like commerce 
      Two-loop learning system
        Online observation loop 
        Offline "defense gym" loop 
    Core Economic Questions
      Price Discovery
        How prices respond to demand signals
        How signal quality changes with bots/agents
      Demand & Elasticity
        Shifts in willingness-to-pay
        Short-run vs long-run elasticity
      Market Efficiency & Welfare
        Consumer surplus vs producer surplus
        Deadweight loss from frictions/manipulation
      Price Discrimination & Segmentation
        Behavioral feature-based segmentation
        Fairness vs profitability tradeoffs
      Information Asymmetry
        Agents amplify search and arbitrage
        Sellers infer more about buyers; buyers infer more about sellers
      Strategic Interaction
        Consumers vs firms vs agents
        Feedback loops: policy ↔ behavior ↔ price
      Market Power & Competition
        Algorithmic pricing as competitive tool
        Risks: tacit coordination / "algorithmic collusion"
      Externalities
        Congestion and attention costs
        Spillovers: one segment’s behavior affects others’ prices
    System-Level View
      Participants
        Humans
        Agents (automated buyers/actors) 
        Firms (pricing decision-makers)
        Platform (measurement + control layer)
      Markets Simulated
        Repeated transactions
        Limited inventory / capacity constraints (conceptually)
        Time dynamics (learning over time)
      Interventions
        Pricing policies
        Experiment assignment / randomized exposure
        Agent behavioral policies (task-driven)
    Measurement & Causal Inference
      What is observed
        Actions (search, click, purchase intent)
        Context (product attributes, time, exposure)
        Outcomes (conversion, revenue, churn proxies)
      Identification strategy
        A/B tests and randomization
        Counterfactual baselines
        Robustness checks (offline replay)
      Key metrics
        Revenue / profit proxies
        Conversion & bounce
        Price volatility / stability
        Welfare proxies (e.g., dispersion, access)
    Risk, Governance, and Ethics
      Manipulation & Integrity
        Bot-driven demand distortion
        Measurement contamination
      Fairness & Transparency
        Differential pricing concerns
        Explainability and auditability
      Safety Constraints
        Guardrails on price moves
        Monitoring for runaway feedback loops
    Outputs
      Insights
        When do agents raise/lower prices via behavior shifts?
        Which market designs are robust to automation?
      Defenses
        Agent-aware pricing policies (robust control)
        Detection + mitigation strategies (feature-level separability)
      Platform Value
        Reusable testbed for market + AI-agent research
```
