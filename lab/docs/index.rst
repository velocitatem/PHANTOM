Quote-Control Simulator
=======================

Research-grade platform for dynamic pricing and market making experiments.

The platform abstracts pricing as: **Quote → Arrival → Execution → Position**

Supports multiple mechanisms:

* **PostedPrice**: retail dynamic pricing
* **TwoSided**: market making with bid-ask spreads
* **Auction**: reserve/shading for auction settings

Quick Start
-----------

.. code-block:: python

   from lab.config import make_retail_platform
   from lab.experiments import rollout, fixed_price_policy

   platform = make_retail_platform()
   policy = fixed_price_policy(platform.instruments.refs)
   result = rollout(platform, policy, n_steps=100)
   print(f"Total PnL: {result.total_pnl:.2f}")

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   system_overview
   modules/outlet
   modules/population
   modules/experiments

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
