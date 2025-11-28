
# Products
# Agents
# Pipeline

Our pipeline technically should follow principles in a style like this:
- Each step should be defined as an inheriting child of an scikit pipeline step, the granularity of the steps is dictated by the following: a step should be a transformation, augmentation or computation independently, no single stage should run multiple in-itself. This way we can modularize properly all the components and track properly in airflow. A stage can be defined as an sklearn step but then must be transalted to a function that takes the context in our DAG of airflow. All parametrization must be done via contexts.

