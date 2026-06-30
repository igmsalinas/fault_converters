# Self-Supervised Deep Learning for Unsupervised Anomaly Detection in Power Electronic Converters using Frequency Response Signatures

**Ignacio Martin-Salinas** $^1$, **Jose A. Belloch** $^1$, **Cristina Fernandez** $^1$

$^1$ *Depto. de Tecnología Electrónica, Universidad Carlos III de Madrid, Madrid, Spain*

This work presents a comprehensive, modular deep learning framework for unsupervised and self-supervised anomaly detection in Power Electronic Converters (PECs). Designed for high-reliability applications such as aerospace, electric vehicles, and critical industrial systems, the system leverages frequency-domain transfer function (Bode plot) signatures—comprising log-spaced amplitude and phase measurements—to isolate subtle, early-stage parametric degradation before catastrophic failure occurs.

The framework integrates three key components:
1. **A Physics-Aware Simulation Pipeline**: Generates a high-fidelity dataset of 227,698 unique frequency response transfer functions (spanning 100 Hz to 50 kHz across 101 log-spaced points) using PSIM simulations under continuous parametric variations of critical components, including the output capacitor ($C_{\text{out}}$), inductor ($L$), switch on-resistances ($R_{\text{ds},1}, R_{\text{ds},2}$), and equivalent series resistances (ESR of both inductor and capacitor). Variations within the $[-5\%, +5\%]$ range (swept at a fine $1\%$ step resolution) represent simulated normal behavior, while variations in the ranges of $[-20\%, -5\%)$ and $(+5\%, +20\%]$ (swept at $5\%$ step resolution) represent anomalous states.
2. **A Modular Deep Autoencoder Suite**: Implements six interchangeable neural architectures sharing a unified `BaseAutoencoder` API, including Convolutional 1D (Conv1D-AE), Multi-Layer Perceptron (MLP-AE), recurrent models (LSTM-AE and GRU-AE), probabilistic Variational Autoencoders (VAE), and attention-based sequence models (Transformer-AE).
3. **A Self-Supervised Contrastive Engine (CARLA)**: Enhances representation capabilities by coupling encoder backends with an MLP projection head. During training, it computationally injects synthetic frequency anomalies (resonance shifts, amplitude scaling, phase distortions, spectral noise, and localized point drops) matching converter physics to serve as negative samples optimized via Normalized Temperature-scaled Cross Entropy (NT-Xent) loss.

During evaluation, anomaly scoring is conducted non-parametrically via a $k$-Nearest Neighbors ($k$-NN) density and distance estimator in the projection space, mapping the boundary between healthy and degraded states. All models are tuned using automated Bayesian, Random, and Grid hyperparameter search spaces targeting maximum ROC-AUC and F1 scores. 

Experimental results on the held-out test set show that traditional reconstruction-only autoencoders yield strong baselines on this dataset, with the Conv1D-AE achieving a ROC-AUC of $98.4\%$ (F1: $96.3\%$) and the LSTM-AE achieving $95.5\%$ (F1: $91.6\%$). The trained CARLA-Conv1D model achieves a ROC-AUC of $83.1\%$ (F1: $82.5\%$) on the test set with its default configuration. The proposed contrastive framework effectively partitions complex multi-component degradation profiles in the projection space, demonstrating a high sensitivity to subtle parametric deviations. These findings underscore the potential of self-supervised contrastive learning using frequency signatures as a highly sensitive, non-invasive diagnostic solution for real-time predictive maintenance in power electronic systems.

**Keywords**: Anomaly Detection, Power Electronics, Self-Supervised Learning, Contrastive Learning (CARLA), Autoencoders, Transfer Function

### References:
[1] S. Darban, G. A. A. C. V. S. B. P. A., “CARLA: Self-supervised Contrastive Representation Learning for Time Series Anomaly Detection,” *arXiv preprint arXiv:2308.09296*, 2023.  
[2] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you need,” in *Advances in Neural Information Processing Systems*, pp. 5998–6008, 2017.  
