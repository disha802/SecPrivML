"""
privml — PrivBench-Adult
========================

A unified privacy–utility evaluation framework for tabular machine learning.
Every privacy mechanism from the five course modules is plugged into the same
data pipeline and attacked with the same attack suite, so that their
privacy–utility trade-offs can be compared on equal terms.

Modules
-------
data         Download, provenance (SHA-256), cleaning, encoding, splits   (all)
attacks      Membership inference, attribute inference, re-identification (Module 1)
dp           Laplace / Gaussian mechanisms, DP-SGD, RDP accountant         (Module 2)
ldp          GRR, OUE, Piecewise mechanism, LDP data collection            (Module 3)
ppdm         Mondrian k-anonymity, l-diversity, information loss           (Module 4)
compressive  PCA, DCA and the proposed Desensitised DCA (D-DCA)            (Module 5)
models       Non-private baselines (LR, RF, MLP) and the MLP architecture
metrics      Utility metrics
"""

__version__ = "1.0.0"
