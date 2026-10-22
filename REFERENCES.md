# References and Sources

Sources used in the PrivBench-Adult project (IA-1, IA-2 and the manuscript draft).

## Papers, standards and legislation

1. C. Dwork and A. Roth, “The Algorithmic Foundations of Differential Privacy,” *Foundations and Trends in Theoretical Computer Science*, vol. 9, no. 3–4, pp. 211–407, 2014.
2. M. Abadi, A. Chu, I. Goodfellow, H. B. McMahan, I. Mironov, K. Talwar and L. Zhang, “Deep learning with differential privacy,” in *Proc. ACM CCS*, 2016, pp. 308–318.
3. I. Mironov, “Rényi differential privacy,” in *Proc. IEEE CSF*, 2017, pp. 263–275.
4. I. Mironov, K. Talwar and L. Zhang, “Rényi differential privacy of the sampled Gaussian mechanism,” arXiv:1908.10530, 2019.
5. B. Balle, G. Barthe, M. Gaboardi, J. Hsu and T. Sato, “Hypothesis testing interpretations and Rényi differential privacy,” in *Proc. AISTATS*, 2020.
6. S. L. Warner, “Randomized response: A survey technique for eliminating evasive answer bias,” *J. American Statistical Association*, vol. 60, no. 309, pp. 63–69, 1965.
7. P. Kairouz, K. Bonawitz and D. Ramage, “Discrete distribution estimation under local privacy,” in *Proc. ICML*, 2016.
8. T. Wang, J. Blocki, N. Li and S. Jha, “Locally differentially private protocols for frequency estimation,” in *Proc. USENIX Security*, 2017, pp. 729–745.
9. N. Wang, X. Xiao, Y. Yang, J. Zhao, S. C. Hui, H. Shin, J. Shin and G. Yu, “Collecting and analyzing multidimensional data with local differential privacy,” in *Proc. IEEE ICDE*, 2019, pp. 638–649.
10. Ú. Erlingsson, V. Pihur and A. Korolova, “RAPPOR: Randomized aggregatable privacy-preserving ordinal response,” in *Proc. ACM CCS*, 2014.
11. L. Sweeney, “k-anonymity: A model for protecting privacy,” *Int. J. Uncertainty, Fuzziness and Knowledge-Based Systems*, vol. 10, no. 5, pp. 557–570, 2002.
12. K. LeFevre, D. J. DeWitt and R. Ramakrishnan, “Mondrian multidimensional k-anonymity,” in *Proc. IEEE ICDE*, 2006.
13. A. Machanavajjhala, D. Kifer, J. Gehrke and M. Venkitasubramaniam, “l-diversity: Privacy beyond k-anonymity,” *ACM TKDD*, vol. 1, no. 1, 2007.
14. N. Li, T. Li and S. Venkatasubramanian, “t-closeness: Privacy beyond k-anonymity and l-diversity,” in *Proc. IEEE ICDE*, 2007.
15. J. Xu, W. Wang, J. Pei, X. Wang, B. Shi and A. W.-C. Fu, “Utility-based anonymization using local recoding,” in *Proc. ACM KDD*, 2006.
16. R. Agrawal and R. Srikant, “Privacy-preserving data mining,” in *Proc. ACM SIGMOD*, 2000.
17. S. Y. Kung, “Compressive privacy: From information/estimation theory to machine learning,” *IEEE Signal Processing Magazine*, vol. 34, no. 1, pp. 94–112, 2017.
18. S. Y. Kung, “A compressive privacy approach to generalized information bottleneck and privacy funnel problems,” *Journal of the Franklin Institute*, vol. 355, no. 4, pp. 1846–1872, 2018.
19. R. Shokri, M. Stronati, C. Song and V. Shmatikov, “Membership inference attacks against machine learning models,” in *Proc. IEEE S&P*, 2017.
20. S. Yeom, I. Giacomelli, M. Fredrikson and S. Jha, “Privacy risk in machine learning: Analyzing the connection to overfitting,” in *Proc. IEEE CSF*, 2018.
21. N. Carlini, S. Chien, M. Nasr, S. Song, A. Terzis and F. Tramèr, “Membership inference attacks from first principles,” in *Proc. IEEE S&P*, 2022.
22. A. Narayanan and V. Shmatikov, “Robust de-anonymization of large sparse datasets,” in *Proc. IEEE S&P*, 2008.
23. B. Becker and R. Kohavi, “Adult” [Dataset], UCI Machine Learning Repository, 1996. doi:10.24432/C5XW20.
24. R. Kohavi, “Scaling up the accuracy of naive-Bayes classifiers: A decision-tree hybrid,” in *Proc. KDD*, 1996.
25. F. Ding, M. Hardt, J. Miller and L. Schmidt, “Retiring Adult: New datasets for fair machine learning,” in *Proc. NeurIPS*, 2021.
26. A. Yousefpour et al., “Opacus: User-friendly differential privacy library in PyTorch,” arXiv:2109.12298, 2021.
27. N. Papernot and T. Steinke, “Hyperparameter tuning with Rényi differential privacy,” in *Proc. ICLR*, 2022.
28. E. Bagdasaryan, O. Poursaeed and V. Shmatikov, “Differential privacy has disparate impact on model accuracy,” in *Proc. NeurIPS*, 2019.
29. S. Ravfogel, Y. Elazar, H. Gonen, M. Twiton and Y. Goldberg, “Null it out: Guarding protected attributes by iterative nullspace projection,” in *Proc. ACL*, 2020.
30. N. Natarajan, I. S. Dhillon, P. Ravikumar and A. Tewari, “Learning with noisy labels,” in *Proc. NeurIPS*, 2013.
31. Government of India, *The Digital Personal Data Protection Act, 2023* (Act No. 22 of 2023).
32. European Parliament and Council, *Regulation (EU) 2016/679 (General Data Protection Regulation)*, 2016.

## Dataset

- B. Becker and R. Kohavi, "Adult" [Dataset], UCI Machine Learning Repository, 1996. DOI: [10.24432/C5XW20](https://doi.org/10.24432/C5XW20). Licence: CC BY 4.0. Source: <https://archive.ics.uci.edu/dataset/2/adult>. Provenance and SHA-256 hashes: [data/provenance.json](data/provenance.json).

## Software and tools

| Tool | Use |
|---|---|
| [Python 3.12](https://www.python.org/) | Language |
| [NumPy](https://numpy.org/) | MLP, DP-SGD, LDP mechanisms, projections |
| [SciPy](https://scipy.org/) | Log-sum-exp/log-gamma for the RDP accountant; generalised eigensolver |
| [pandas](https://pandas.pydata.org/) | Data handling, k-anonymity tables, result CSVs |
| [scikit-learn](https://scikit-learn.org/) | LR/RF baselines, gradient-boosting attacker, metrics |
| [matplotlib](https://matplotlib.org/) | Figures |
| [python-docx](https://python-docx.readthedocs.io/) | Generating the Word reports |
| [pytest](https://pytest.org/) | Unit tests |
| [Opacus 1.5.2](https://opacus.ai/) | Reference only: used to validate the RDP accountant (not a runtime dependency) |
| Claude (Anthropic) | Generative-AI assistant used for code development, running experiments and drafting reports, as declared in the reports |
