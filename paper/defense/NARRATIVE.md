---
present_time: 15 minutes
qa: 15 minutes
---

> Notes for presentation deck: keep minimal text, highlight only key metrics or keywords and diagrams, if possible do progressive reveal of items on slides, if going through a list, make each appear progressively on new slides like an animation.

# Introduction [2min]
> Hook: Extracting margin in markets with high density of AI agents.
- Say what today's agenda is (show in the blocks at the botton of each slide and with each slide indicate which stage we are at)
- Highlight problem (add financial consequence)
  - What are we trying to answer?

# First Stage (Platform Development)  [4min]
- Talk about designing the platform (nextjs design and apache airflow and kafka)

## About the Platform
- Show an architecture diagram.

## Dataset Brief 
- Screenshot of the HF dataset and highlight some key features of the dataset with big numbers indicated.

## Experimental Design
- Say how we collected data and how we used AI Agents

### AI Agents
- browser use
- models used (say we used the LLM router for different models)

# Second Stage (Distinguishability Construction) [4min]
- Explain kernels of behavior (what is a kernel)
- How we separate kernels and finally how we turn that into a probability.

# DR-RL [4min]
- Explain simple wesserstein balls and ambiguity
- Highlight computational complexity 

## Results [1min]
- Empirical results from experiments

# Conclusions
- Consequences of our work (financial and future implications for pricing systems)
- Did we answer what we wanted? How?

# Appendix

## Derivation of the COI theorem
## Reward Structure Composition
## On our Sample Size


