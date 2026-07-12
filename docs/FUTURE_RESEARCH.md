# Future Research Directions

This document summarizes recent papers and emerging techniques that could enhance the Buck Converter anomaly detection project.

---

## 1. State Space Models (Mamba)

### Overview
Mamba is a new state space model architecture offering **linear complexity** (vs. quadratic for Transformers) while maintaining strong sequence modeling capabilities.

### Relevant Papers

| Paper | arXiv | Key Contribution |
|-------|-------|------------------|
| **Bi-Mamba+: Bidirectional Mamba for Time Series Forecasting** | 2404.15772 | Forget gate + bidirectional processing for better long-range dependencies |
| **TSMamba: A Foundation Model for Time Series** | 2411.02941 | Zero-shot time series forecasting with transfer learning from LLMs |
| **Is Mamba Effective for Time Series Forecasting?** | 2403.11144 | S-Mamba: bidirectional Mamba + FFN, near-linear complexity |
| **SOR-Mamba: Sequential Order-Robust Mamba** | 2410.23356 | Addresses channel ordering bias in multivariate time series |

### Potential Implementation
```python
# Future: MambaAutoencoder for transfer function anomaly detection
class MambaAE(BaseAutoencoder):
    """Linear complexity alternative to Transformer-AE"""
    # Bidirectional Mamba encoder
    # Forget gate for selective memory
    # Channel-independent processing for multivariate data
```

**Benefits for project:**
- 8x faster than Transformers on long sequences
- 62% less GPU memory usage
- Better handling of long-range dependencies in transfer functions

---

## 2. Foundation Models for Time Series

### Overview
Large pre-trained models that can perform zero-shot or few-shot time series analysis.

### Relevant Papers

| Paper | arXiv | Key Contribution |
|-------|-------|------------------|
| **Time Series Foundational Models: Role in Anomaly Detection** | 2412.19286 | Critical evaluation of TSFMs for anomaly detection |
| **Can Multimodal LLMs Perform Time Series Anomaly Detection?** | 2502.17812 | VisualTimeAnomaly benchmark, MLLMs for TSAD |
| **Anomaly Detection of Tabular Data Using LLMs** | 2406.16308 | GPT-4 as zero-shot anomaly detector |
| **LLM-Mixer: Multiscale Mixing in LLMs** | 2410.11674 | Combining multiscale decomposition with frozen LLMs |
| **TabPFN-TS: Tables to Time** | 2501.02945 | Tabular foundation model for time series (11M params) |

### Potential Implementation
```python
# Future: LLM-based anomaly scoring
class LLMAnomalyDetector:
    """Zero-shot anomaly detection using multimodal LLMs"""
    # Convert transfer functions to visual representations
    # Use GPT-4V or open-source MLLMs for anomaly scoring
    # Combine with traditional autoencoders for hybrid approach
```

**Key insights:**
- MLLMs better at detecting **range-wise anomalies** than point-wise
- Robust to 25% missing data
- Open-source MLLMs competitive with proprietary models

---

## 3. Kolmogorov-Arnold Networks (KAN)

### Overview
Novel architecture based on the Kolmogorov-Arnold representation theorem, replacing fixed activation functions with learnable spline-based functions.

### Relevant Papers

| Paper | arXiv | Key Contribution |
|-------|-------|------------------|
| **Kolmogorov-Arnold Networks for Time Series** | 2405.08790 | KANs outperform MLPs with fewer parameters |
| **Wav-KAN: Wavelet Kolmogorov-Arnold Networks** | 2405.12832 | Wavelet-based KANs for multi-resolution analysis |
| **KAN for Time Series Classification** | 2408.07314 | Robustness advantages, lower Lipschitz constants |
| **TSKANMixer: KAN with MLP-Mixer** | 2502.18410 | Hybrid architecture for time series forecasting |
| **P-KAN: Probabilistic Kolmogorov-Arnold Networks** | 2510.16940 | Uncertainty-aware predictions with KANs |
| **KKANs: Kurkova-Kolmogorov-Arnold Networks** | 2412.16738 | Universal approximator with improved learning dynamics |

### Potential Implementation
```python
# Future: KAN-based Autoencoder
class KANAutoencoder(BaseAutoencoder):
    """Interpretable autoencoder using Kolmogorov-Arnold Networks"""
    # Learnable spline-based activation functions
    # Symbolic equation extraction for interpretability
    # 60-230x fewer parameters than traditional MLPs
```

**Benefits for project:**
- **Interpretability**: Extract symbolic equations explaining anomalies
- **Parameter efficiency**: Much smaller models
- **Robustness**: Lower Lipschitz constants reduce sensitivity to perturbations

---

## 4. Contrastive & Self-Supervised Learning

### Overview
Learn representations by contrasting normal vs. anomalous patterns without requiring extensive labels.

### Relevant Papers

| Paper | arXiv | Key Contribution |
|-------|-------|------------------|
| **CARLA: Contrastive Representation Learning for TSAD** | 2308.09296 | Inject anomalies as negative samples during contrastive learning |
| **MAG: Multi-GNN Augmented Graph Contrastive Framework** | 2305.02496 | Graph contrastive learning for anomaly detection |
| **Self-Distilled Representation Learning for Time Series** | 2311.11335 | data2vec-style self-distillation for time series |
| **GADT3: Cross-Domain Graph Anomaly Detection** | 2502.14293 | Test-time training with homophily-guided self-supervision |
| **In/Out-of-Distribution SSL for ECG Arrhythmia** | 2304.06427 | SwAV achieves best performance; strong OOD generalization |

### Potential Implementation
```python
# Future: Contrastive Autoencoder
class ContrastiveAE(BaseAutoencoder):
    """Self-supervised contrastive learning for anomaly detection"""
    # Positive pairs: augmented normal samples
    # Negative pairs: synthetically injected anomalies
    # Learn to distinguish normal from anomalous patterns
```

**Key insights from CARLA:**
- Inject various anomaly types as negative samples
- Learn both normal behavior AND deviation patterns
- Nearest-neighbor classification in latent space

---

## 5. Physics-Informed Neural Networks (PINNs)

### Overview
Embed physical laws and domain knowledge directly into neural network training.

### Relevant Papers

| Paper | arXiv | Key Contribution |
|-------|-------|------------------|
| **BPINN for Inverter-dominated Power Systems** | 2403.13602 | Bayesian PINNs for system identification under uncertainty |
| **PINNSim: Power System Dynamics Simulator** | 2303.10256 | PINNs for accelerated time-domain simulations |
| **PINN Control for Power Electronics** | 2406.15787 | Combines model-driven and data-driven for stability |
| **Physics-Informed Real NVP for Satellite Fault Detection** | 2405.17339 | PI normalizing flows for EPS fault detection |

### Potential Implementation
```python
# Future: Physics-Informed Autoencoder
class PhysicsInformedAE(BaseAutoencoder):
    """Embed Buck converter physics into autoencoder"""
    def physics_loss(self, x, x_reconstructed):
        # Enforce transfer function physical constraints
        # dc_gain_constraint: |H(0)| should match expected value
        # stability_constraint: poles in left half-plane
        # Bode magnitude/phase relationships
        return reconstruction_loss + lambda * physics_violations
```

**Key insight from BPINN paper:**
- Orders of magnitude lower errors than SINDy
- Transfer learning reduces training time by 80%
- Uncertainty quantification built-in

---

## 6. Diffusion Models for Time Series

### Overview
Generative models that have achieved state-of-the-art in image generation, now being applied to time series.

### Relevant Papers

| Paper | arXiv | Key Contribution |
|-------|-------|------------------|
| **The Rise of Diffusion Models in Time-Series Forecasting** | 2401.03006 | Comprehensive survey of 11 diffusion-based TS methods |
| **RobustTAD: Robust Time Series Anomaly Detection** | 2002.09545 | Seasonal-trend decomposition + CNN encoder-decoder |

### Potential Implementation
```python
# Future: Diffusion-based Anomaly Detection
class DiffusionAnomalyDetector:
    """Score-based anomaly detection using diffusion models"""
    # Train diffusion model on normal transfer functions
    # Anomaly score = difficulty of denoising from corrupted input
    # Can generate counterfactual "normal" versions of anomalies
```

---

## 7. Graph Neural Networks for Anomaly Detection

### Overview
Model relationships between components/parameters as graphs.

### Relevant Papers

| Paper | arXiv | Key Contribution |
|-------|-------|------------------|
| **Detecting Contextual Network Anomalies with GNNs** | 2312.06342 | GNNs for traffic anomaly detection |
| **Multi-Flow: Multi-View Normalizing Flows** | 2504.03306 | Cross-view message passing for industrial anomaly detection |

### Potential Implementation
```python
# Future: Graph-based parameter relationship modeling
class GraphAnomalyDetector:
    """Model Buck converter parameter interactions as graph"""
    # Nodes: individual parameters (Cout, Rds_1, Rds_2, Esr_L, Esr_C)
    # Edges: physical relationships/correlations
    # GNN learns normal parameter configurations
```

---

## 8. Quantum Autoencoders

### Overview
Quantum computing approaches to anomaly detection showing promise.

### Relevant Papers

| Paper | arXiv | Key Contribution |
|-------|-------|------------------|
| **Quantum Autoencoders for Time Series Anomaly Detection** | 2410.04154 | 60-230x fewer parameters, 5x fewer training iterations |

### Key Finding
- Quantum autoencoders **outperform classical deep learning autoencoders** on multiple datasets
- Practical for resource-constrained scenarios

---

## 9. Multi-Signal Fusion & Fault Identifiability Analysis

### Overview
The models are currently trained on a **single** measurement — the control-to-output
Bode $G_{vd}(j\omega)$. This makes several component degradations *confounded*: distinct
faults map to nearly the same frequency response, so the detector can flag "a fault"
but cannot reliably attribute it to a component, and healthy multi-component tolerance
excursions can masquerade as faults. This direction proposes (A) **quantifying** those
overlaps and (B) **breaking** them with complementary signals.

### Why the overlaps exist (grounded in the plant model)
Using the canonical buck $G_{vd}(s)$ (`src/data/physics_anomaly.py`):

- **Capacitance vs. inductance** — both shift the resonance $\omega_0 = 1/\sqrt{LC}$, so
  they are *degenerate in resonant frequency*. They are **not** fully degenerate: since
  $Q = R\sqrt{C/L}$, a cap drop *lowers* the peak while an inductance drop *raises* it
  (verified: $C\times0.7 \Rightarrow$ peak $-0.9$ dB; $L\times0.7 \Rightarrow +0.8$ dB, both
  at $f_{res}\!\approx\!+20\%$). Separation therefore lives in the **~1 dB peak-shape
  difference**, demanding fine resolution and low noise (<~0.3 dB) around resonance.
- **Cap-ESR vs. switch/inductor resistance** — both add series resistance that lowers $Q$
  and flattens the peak; ESR additionally drags the ESR zero $\omega_z = 1/(R_cC)$ toward
  the origin, but for this converter $\omega_z\!\approx\!80$ kHz sits **above** $f_{sw}/2 = 50$
  kHz (the averaged-model / sweep ceiling), so that disambiguating signature is
  unobservable — extending the sweep does **not** help (see the note in
  `data/generate_data.py`).
- **Correlated ageing** — a real electrolytic ages with $C\!\downarrow$ **and** $\text{ESR}\!\uparrow$
  *together*; modelling them as independent single-component faults is a simplification.

### Two research thrusts

**(A) Fault identifiability analysis** (purely analytic — no PSIM/hardware needed)
- Build the sensitivity Jacobian $J = \partial(\text{Bode})/\partial\theta$ at nominal
  ($\theta = [C, L, R_c, R_{ds}, \dots]$); the **SVD / condition number** reveals which
  fault combinations are well- vs. poorly-observed (near-parallel columns = confounded).
- Add a noise model to get the **Fisher Information Matrix** and **Cramér–Rao bounds**
  per component; the off-diagonal correlations *quantify* each overlap and give a
  principled, measurement-aware answer to "which faults can this signal set resolve?".

**(B) Multi-signal fusion** — add measurements that are orthogonal to $G_{vd}$:

| Extra signal | Breaks which overlap | Physics |
|---|---|---|
| Output impedance $Z_{out}(s)$ | C / ESR vs. L | $Z_{out}\approx \text{ESR} + 1/(sC)$ |
| Inductor-current ripple | L vs. C / ESR | $\Delta I_L \propto 1/(L\,f_{sw})$ |
| Output-voltage ripple @ $f_{sw}$ | ESR vs. C | amp $\sim \text{ESR}\cdot\Delta I$; slope $\sim \Delta I/C$ |
| Load-step transient | L vs. C / ESR | di/dt vs. hold-up |
| Temperature | resistive (Rds/DCR/ESR) vs. reactive | resistive loss = heat |
| Operating-point diversity $(D, R)$ | L vs. load; RHP-zero faults | $Q(R)$, $\omega_{rhp}(R, L)$ |

### Potential Implementation
```python
# Future: expose complementary transfer functions on the converter model,
# then fuse them as extra channels / run identifiability analysis.
class PowerConverter(ABC):
    def transfer_function(self) -> TransferFunctionModel: ...   # G_vd(s)  (exists)
    def output_impedance(self)  -> TransferFunctionModel: ...   # Z_out(s) (NEW)
    def audiosusceptibility(self) -> TransferFunctionModel: ... # G_vg(s)  (NEW)

# Identifiability study (analytic):
#   J = numerical d(Bode)/d(theta) at nominal
#   U, S, Vt = svd(J);  cond = S[0]/S[-1];  FIM = J.T @ Sigma_inv @ J
#   -> report confounded component pairs, Cramer-Rao bounds, and the
#      improvement when Z_out(s) is stacked alongside G_vd(s).

# Multi-channel CARLA: feed (G_vd, Z_out, ripple stats, T) as feature channels;
# correlated fault injection: sample joint (C-down, ESR-up) ageing trajectories
# instead of axis-aligned single-component faults.
```

**Benefits for project:**
- Turns "anomaly vs. normal" into **component-level attribution** (which part is degrading).
- Quantifies, before any hardware change, whether a proposed signal/sweep actually
  improves separability (answers the "widen the sweep?" question rigorously).
- Correlated fault injection aligns the synthetic negatives with real ageing manifolds,
  reducing false positives from stacked healthy tolerances.

**Relevant concepts:** structural/practical identifiability, Fisher information &
Cramér–Rao bound, multi-transfer-function (EIS-style) system identification. See also
`docs/REFERENCES.md` → "Component Tolerances & Degradation Ranges".

---

## Priority Roadmap for Implementation

### Phase 1: Near-term (1-3 months)
1. ✅ **Contrastive Learning Enhancement** (CARLA approach) - **IMPLEMENTED**
   - Add anomaly injection during training
   - Implement nearest-neighbor scoring in latent space
   - **See:** `src/data/anomaly_injection.py`, `src/models/contrastive_ae.py`, 
     `src/training/carla_trainer.py`

2. **Physics-Informed Loss**
   - Add Buck converter physics constraints
   - Validate against known fault signatures

### Phase 2: Medium-term (3-6 months)
3. **Mamba-based Autoencoder**
   - Implement Bi-Mamba architecture
   - Compare with Transformer-AE on speed/accuracy

4. **KAN Integration**
   - Wav-KAN for frequency domain analysis
   - Extract interpretable symbolic equations

5. **Fault Identifiability & Multi-Signal Fusion** (see §9)
   - Analytic sensitivity/Fisher-information study to quantify confounded faults
     (C↔L, ESR↔Rds) — no hardware needed
   - Add `output_impedance()` / `audiosusceptibility()` to `PowerConverter`;
     stack as extra CARLA channels; correlated (C↓ & ESR↑) fault injection

### Phase 3: Long-term (6-12 months)
6. **Foundation Model Integration**
   - Fine-tune TSFMs on Buck converter data
   - Explore multimodal approaches (visual transfer functions)

7. **Diffusion-based Detection**
   - Score-based anomaly scoring
   - Counterfactual generation for explainability

---

## BibTeX Citations

```bibtex
% Mamba / State Space Models
@article{liang2024bimamba,
  title={Bi-Mamba+: Bidirectional Mamba for Time Series Forecasting},
  author={Liang, Aobo and Jiang, Xingguo and Sun, Yan and Shi, Xiaohou and Li, Ke},
  journal={arXiv preprint arXiv:2404.15772},
  year={2024}
}

@article{wang2024mamba,
  title={Is Mamba Effective for Time Series Forecasting?},
  author={Wang, Zihan and Kong, Fanheng and Feng, Shi and Wang, Ming and Yang, Xiaocui and Zhao, Han and Wang, Daling and Zhang, Yifei},
  journal={arXiv preprint arXiv:2403.11144},
  year={2024}
}

% Foundation Models
@article{shyalika2024tsfm,
  title={Time Series Foundational Models: Their Role in Anomaly Detection and Prediction},
  author={Shyalika, Chathurangi and Bagga, Harleen Kaur and Bhatt, Ahan and Prasad, Renjith and Al Ghazo, Alaa and Sheth, Amit},
  journal={arXiv preprint arXiv:2412.19286},
  year={2024}
}

@article{xu2025mllm,
  title={Can Multimodal LLMs Perform Time Series Anomaly Detection?},
  author={Xu, Xiongxiao and Wang, Haoran and Liang, Yueqing and Yu, Philip S. and Zhao, Yue and Shu, Kai},
  journal={arXiv preprint arXiv:2502.17812},
  year={2025}
}

% Kolmogorov-Arnold Networks
@article{vaca2024kan,
  title={Kolmogorov-Arnold Networks (KANs) for Time Series Analysis},
  author={Vaca-Rubio, Cristian J. and Blanco, Luis and Pereira, Roberto and Caus, M{\`a}rius},
  journal={arXiv preprint arXiv:2405.08790},
  year={2024}
}

@article{bozorgasl2024wavkan,
  title={Wav-KAN: Wavelet Kolmogorov-Arnold Networks},
  author={Bozorgasl, Zavareh and Chen, Hao},
  journal={arXiv preprint arXiv:2405.12832},
  year={2024}
}

% Contrastive Learning
@article{darban2023carla,
  title={CARLA: Self-supervised Contrastive Representation Learning for Time Series Anomaly Detection},
  author={Darban, Zahra Zamanzadeh and Webb, Geoffrey I. and Pan, Shirui and Aggarwal, Charu C. and Salehi, Mahsa},
  journal={arXiv preprint arXiv:2308.09296},
  year={2023}
}

% Physics-Informed Neural Networks
@article{stock2024bpinn,
  title={Bayesian Physics-informed Neural Networks for System Identification of Inverter-dominated Power Systems},
  author={Stock, Simon and Babazadeh, Davood and Becker, Christian and Chatzivasileiadis, Spyros},
  journal={arXiv preprint arXiv:2403.13602},
  year={2024}
}

@article{hui2024pinncontrol,
  title={On Physics-Informed Neural Network Control for Power Electronics},
  author={Hui, Peifeng and Cui, Chenggang and Lin, Pengfeng and Ghias, Amer M. Y. M. and Niu, Xitong and Zhang, Chuanlin},
  journal={arXiv preprint arXiv:2406.15787},
  year={2024}
}

@article{cena2024pifault,
  title={Physics-Informed Real NVP for Satellite Power System Fault Detection},
  author={Cena, Carlo and Albertin, Umberto and Martini, Mauro and Bucci, Silvia and Chiaberge, Marcello},
  journal={arXiv preprint arXiv:2405.17339},
  year={2024}
}

% Diffusion Models
@article{meijer2024diffusion,
  title={The Rise of Diffusion Models in Time-Series Forecasting},
  author={Meijer, Caspar and Chen, Lydia Y.},
  journal={arXiv preprint arXiv:2401.03006},
  year={2024}
}

% Signal Processing & Deep Learning
@article{parhi2023sparse,
  title={Deep Learning Meets Sparse Regularization: A Signal Processing Perspective},
  author={Parhi, Rahul and Nowak, Robert D.},
  journal={arXiv preprint arXiv:2301.09554},
  year={2023}
}

% Quantum Computing
@article{frehner2024quantum,
  title={Applying Quantum Autoencoders for Time Series Anomaly Detection},
  author={Frehner, Robin and Stockinger, Kurt},
  journal={arXiv preprint arXiv:2410.04154},
  year={2024}
}
```

---

*Last updated: January 2026*
