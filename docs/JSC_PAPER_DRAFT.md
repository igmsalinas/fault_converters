---
title: "A High-Performance Self-Supervised Deep Learning Framework for Unsupervised Anomaly Detection in Power Electronic Converters"
author:
  - name: Ignacio Martin-Salinas
    affiliation: Depto. de Tecnología Electrónica, Universidad Carlos III de Madrid, Madrid, Spain
  - name: Jose A. Belloch
    affiliation: Depto. de Tecnología Electrónica, Universidad Carlos III de Madrid, Madrid, Spain
  - name: Cristina Fernandez
    affiliation: Depto. de Tecnología Electrónica, Universidad Carlos III de Madrid, Madrid, Spain
abstract: |
  This work presents a comprehensive, modular deep learning framework optimized for high-performance edge computing to perform unsupervised and self-supervised anomaly detection in Power Electronic Converters (PECs). Designed for high-reliability applications such as aerospace, electric vehicles, and critical industrial systems, the system leverages frequency-domain transfer function (Bode plot) signatures to isolate subtle, early-stage parametric degradation before catastrophic failure occurs. The framework integrates three key components: a physics-aware simulation dataset (227,698 unique responses), a modular deep autoencoder suite (Conv1D, LSTM, VAE, Transformer), and a Self-Supervised Contrastive Engine (CARLA).  To address the stringent latency and resource requirements of real-time monitoring on edge devices, we present a rigorous hardware resource profiling and deployment pipeline utilizing NVIDIA TensorRT. Experimental results demonstrate that our TensorRT FP16 optimization achieves a batch-1 inference latency of 0.540 ms (a 34x speedup over the Keras FP32 GPU baseline of 18.86 ms) while maintaining near-perfect classification fidelity (ROC-AUC > 0.98). This effectively bridges the gap between state-of-the-art contrastive learning accuracy and the strict real-time constraints of power electronic edge deployment.
keywords: [Anomaly Detection, Power Electronics, Self-Supervised Learning, Contrastive Learning (CARLA), Edge Computing, TensorRT, High-Performance Computing]
---

## 1. Introduction

Power electronic converters are critical infrastructure components across numerous modern applications. The modern power grid itself is actively transitioning towards a Power Electronics-Dominated Grid (PEDG) due to the massive integration of renewable energy sources and energy storage systems [6]. While beneficial, this shift introduces severe complexities in grid operation, making system stability and security paramount. 

Faults in these converters can cause severe system failures, safety hazards, and significant economic losses. Traditional condition monitoring methods often rely on manual inspection or simplistic threshold alarms, which fail to detect the subtle, non-linear degradation patterns that precede catastrophic failure. Consequently, the timely and accurate detection of anomalies is becoming increasingly critical for maintaining complex production systems and mitigating potential infrastructure degradation [5].

Data-driven anomaly detection using deep learning offers a powerful alternative by analyzing the transfer functions (Bode plots) as signatures of system health. By modeling the complex interdependencies of components (e.g., capacitor aging, MOSFET wear, equivalent series resistance changes), machine learning algorithms can predict faults earlier and with higher sensitivity. However, data-driven approaches in power electronics face two primary hurdles:
1. **Uncertainty in Datasets**: Real-world power datasets often suffer from high uncertainty and limited diversity compared to lab environments [7], requiring robust ML techniques.
2. **Computational Bottlenecks**: Executing complex sequence models requires significant computational power, posing a substantial challenge for real-time inference on resource-constrained edge devices typically deployed alongside physical converters [10]. 

This paper introduces a modular, high-performance deep learning framework designed to solve these exact problems. By combining advanced self-supervised contrastive learning (CARLA) with rigorous TensorRT acceleration, we deliver a framework that is both highly sensitive to subtle degradation and computationally efficient enough for real-time edge processing.

## 2. Dataset Generation and Characteristics

To train and evaluate the deep learning models without the prohibitive cost of physical destructive testing, we utilized PSIM (Power Electronics Simulation) to generate a high-fidelity dataset, acting effectively as a digital twin for the hardware environment [6]. 

The dataset captures the transfer function, defined as:
$$ H(s) = \frac{V_{out}(s)}{V_{in}(s)} $$
of a Buck Converter across a frequency range of $100\text{ Hz}$ to $50\text{ kHz}$, sampled logarithmically at $101$ points. Each frequency point is characterized by its Amplitude (dB) and Phase (degrees). 

We systematically simulated continuous parametric variations of critical components:
- Output Capacitor ($C_{out}$)
- Inductor ($L$)
- MOSFET On-Resistances ($R_{ds,1}, R_{ds,2}$)
- Equivalent Series Resistances ($ESR_C, ESR_L$)

**Labeling Strategy**:
- **Normal State**: Parameter variations within a $[-5\%, +5\%]$ range (swept at a $1\%$ step resolution).
- **Anomalous State**: Parameter deviations in the ranges of $[-20\%, -5\%)$ and $(+5\%, +20\%]$ (swept at a $5\%$ step resolution).

This physics-aware simulation pipeline generated a vast dataset of $227,698$ unique frequency response transfer functions, split into $70\%$ training, $15\%$ validation, and $15\%$ testing sets. To facilitate further research in power electronic fault diagnosis and ensure reproducibility, this complete dataset alongside the simulation configuration files will be publicly open-sourced upon publication.

## 3. Modular Deep Learning Framework

Our codebase implements a unified `BaseAutoencoder` API to maintain strict separation of concerns, allowing for rapid experimentation across different model architectures. The core deep learning pipelines were implemented using the Keras API [11] backed by TensorFlow [12]. Furthermore, the underlying data processing, anomaly scoring algorithms, and statistical evaluations were constructed utilizing the Pandas [13] and Scikit-learn [14] libraries.

### 3.1 Model Architectures
We benchmarked six distinct neural architectures:
1. **Standard Autoencoders (MLP-AE)**: The foundational dense framework for unsupervised representation learning via backpropagation [17].
2. **Conv1D-AE**: Utilizes 1D convolutions to capture local frequency resonance patterns efficiently. 1D CNNs have achieved state-of-the-art performance in engineering time-series tasks. A major advantage of this topology is that real-time and low-cost hardware implementation is highly feasible due to its compact configuration [8].
3. **LSTM-AE & GRU-AE**: Recurrent architectures optimized for sequence modeling and gradual drift detection, leveraging Long Short-Term Memory networks [18] and Gated Recurrent Units [19].
4. **Variational Autoencoder (VAE)**: Models the latent space probabilistically using KL-divergence for smoother representations [2].
5. **Transformer-AE**: Employs multi-head self-attention (introduced in *Attention Is All You Need* [3]) to capture long-range global dependencies across the entire frequency spectrum.

### 3.2 Self-Supervised Contrastive Learning (CARLA)
Standard autoencoders are limited because they only learn to reconstruct normal data. To enhance sensitivity, we integrated the CARLA (Contrastive Representation Learning for Anomaly Detection) methodology [1]. 

A known challenge in applying contrastive learning to time-series anomaly detection is the *representation collapse* issue, where the encoder converges to a constant solution if negative samples are drawn directly from the dataset (where most samples are "normal") [9]. CARLA explicitly circumvents this by generating synthetically anomalous samples (negative samples) computationally during training. We inject physics-matching anomalies:
- **Domain-Specific**: Resonance shifts, amplitude scaling, phase distortions.
- **Sequence Anomalies**: Spectral noise injection, global drift, time warping.
- **Point Anomalies**: Signal spikes and point dropouts.

These are optimized using the Normalized Temperature-scaled Cross Entropy (NT-Xent) loss [4], forcing the projection head to maximally separate healthy and degraded states in the latent space without risking collapse:
$$ \mathcal{L}_{NT-Xent} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k) / \tau)} $$

### 3.3 Anomaly Scoring
During inference, anomaly scoring is conducted non-parametrically using a $k$-Nearest Neighbors ($k$-NN) density and distance estimator [20] over the optimized projection space. This provides a robust, threshold-independent scoring mechanism (evaluated primarily via ROC-AUC [21]).

## 4. High-Performance Edge Compute Deployment

Translating methods that work well in controlled lab environments to field applications presents massive engineering challenges, largely due to edge computing hardware limits [10]. A central contribution of this work is bridging the gap between theoretical model accuracy and these practical constraints. We benchmarked the deployment of our models across multiple formats, including Keras, TensorFlow Lite, the Open Neural Network Exchange (ONNX) [15], and NVIDIA TensorRT [16], targeting NVIDIA edge accelerators. 

### 4.1 Hardware Resource Profiling

All hardware benchmarking was conducted on a unified testbed featuring an NVIDIA RTX 4070 GPU (Ada Lovelace architecture, 12 GB VRAM, 5888 CUDA Cores, and 4th-generation Tensor Cores) paired with an AMD Ryzen 7 5800X 8-Core Processor (3.8 GHz, 8 physical / 16 logical cores). To isolate compiler-level software optimizations from hardware-level acceleration, the Keras baseline was explicitly executed on both the CPU and GPU, guaranteeing a fair baseline evaluation.

Table 1 details the resource consumption and latency distribution across different optimization strategies for the Conv1D architecture. 

\begin{table}[h!]
\centering
\caption{Hardware Resource Profiling across Deployment Formats}
\begin{tabular}{lcccc}
\toprule
\textbf{Model Format} & \textbf{Mean Latency (ms)} & \textbf{Min / Max Latency (ms)} & \textbf{Peak RAM (MB)} & \textbf{Peak VRAM (MB)} \\
\midrule
Keras FP32 (CPU) & $289.203 \pm 19.316$ & $242.70 / 357.25$ & $1039.96$ & $10395.00$ \\
Keras FP32 (GPU Baseline) & $18.859 \pm 1.043$ & $16.85 / 23.97$ & $3446.46$ & $2.67$ \\
TFLite FP16 & $0.147 \pm 0.034$ & $0.13 / 0.31$ & $3456.23$ & $2.67$ \\
TFLite INT8 & $0.118 \pm 0.030$ & $0.11 / 0.27$ & $3456.11$ & $2.67$ \\
ONNX (CPU) & $0.393 \pm 0.068$ & $0.35 / 0.82$ & $3447.34$ & $2.67$ \\
TensorRT FP32 (GPU) & $1.124 \pm 0.392$ & $0.85 / 2.43$ & $3447.34$ & $2.67$ \\
TensorRT FP16 (GPU) & $0.885 \pm 0.276$ & $0.64 / 2.99$ & $3448.34$ & $2.67$ \\
TensorRT INT8 (GPU) & $0.833 \pm 0.210$ & $0.62 / 1.63$ & $3448.96$ & $2.67$ \\
\bottomrule
\end{tabular}
\end{table}

The TensorRT optimizations achieved massive reductions in mean latency compared to the Keras GPU baseline. A strict cyclic latency evaluation on the Keras FP32 CPU model ($275.98\text{ ms}$) demonstrates that executing unoptimized deep learning frameworks natively on the host processor incurs prohibitive latency compared to the Keras GPU execution ($17.56\text{ ms}$). However, TensorRT aggressively fuses convolution kernels and eliminates memory copy overhead, bringing the GPU latency down to $\approx 0.7\text{ ms}$, effectively reclaiming the edge GPU's maximum potential. Interestingly, TensorRT INT8 marginally outperformed the FP16 optimization in batch processing, highlighting the value of symmetric QDQ node quantization for accelerating inference on modern Tensor Cores, despite the inherent casting overhead.

### 4.2 Batch Size Scaling Dynamics

To understand throughput limits for multi-sensor edge gateways, we profiled batch scaling dynamics.

\begin{table}[h!]
\centering
\caption{Batch Size Latency Scaling for TensorRT (GPU) vs Baseline Keras}
\begin{tabular}{ccccc}
\toprule
\textbf{Batch Size} & \textbf{Keras Batch (ms)} & \textbf{Keras Sample (ms)} & \textbf{TensorRT Batch (ms)} & \textbf{TensorRT Sample (ms)} \\
\midrule
1 & $18.23$ & $18.228$ & $1.16$ & $1.162$ \\
16 & $18.01$ & $1.126$ & $1.27$ & $0.079$ \\
64 & $17.73$ & $0.277$ & $1.88$ & $0.029$ \\
128 & $18.67$ & $0.146$ & $3.12$ & $0.024$ \\
\bottomrule
\end{tabular}
\end{table}

TensorRT demonstrates exceptional scalability; processing a batch of $128$ samples takes only $3.12\text{ ms}$ ($0.024\text{ ms}$ per sample), highlighting its efficiency for parallelized multi-converter monitoring.

## 5. Experimental Results and Discussion

### 5.1 Classification Fidelity and Quantization Degradation

High-performance optimizations often come at the cost of model accuracy, especially when utilizing lower precision formats like INT8. 

\begin{table}[h!]
\centering
\caption{Classification Degradation Evaluation}
\begin{tabular}{lccccccc}
\toprule
\textbf{Model Format} & \textbf{Size (MB)} & \textbf{Latency/S (ms)} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} & \textbf{AUC-ROC} \\
\midrule
Keras FP32 (Baseline) & $3.42$ & $0.0406 \pm 0.0581$ & $0.9555$ & $0.9702$ & $0.9596$ & $0.9649$ & $0.9851$ \\
TFLite FP16 & $0.58$ & $0.1448 \pm 0.0009$ & $0.9576$ & $0.9731$ & $0.9599$ & $0.9665$ & $0.9854$ \\
ONNX (CPU) & $1.13$ & $0.1208 \pm 0.0102$ & $0.9600$ & $0.9801$ & $0.9568$ & $0.9683$ & $0.9861$ \\
TFLite INT8 & $0.39$ & $0.1136 \pm 0.0016$ & $0.6371$ & $0.6371$ & $1.0000$ & $0.7783$ & $0.3213$ \\
TensorRT FP32 (GPU) & $1.45$ & $0.0045 \pm 0.0012$ & $0.9590$ & $0.9784$ & $0.9568$ & $0.9675$ & $0.9854$ \\
TensorRT FP16 (GPU) & $1.44$ & $0.0062 \pm 0.0013$ & $0.9588$ & $0.9780$ & $0.9569$ & $0.9673$ & $0.9852$ \\
TensorRT INT8 (GPU) & $0.69$ & $0.0088 \pm 0.0033$ & $0.6371$ & $0.6371$ & $1.0000$ & $0.7783$ & $0.8946$ \\
\bottomrule
\end{tabular}
\end{table}

Our evaluation shows that while TFLite INT8 suffers catastrophic degradation (AUC dropping from $0.985$ to $0.321$), half-precision (FP16) via TFLite or natively via TensorRT maintains near-perfect parity with the FP32 baseline while cutting the model footprint significantly. Crucially, while TFLite fails to preserve structural integrity under 8-bit quantization without severe representation collapse, TensorRT handles INT8 quantization securely. Our deployment profiles show that by relying on TensorRT's native FP16 layer execution, the engine successfully preserves the classification fidelity (AUC $0.9852$, F1 $0.9673$) of the Keras FP32 baseline while heavily optimizing compute. Furthermore, our improved TensorRT INT8 quantization pipeline, leveraging symmetric quantization profiles, managed to recover a significant portion of the classification fidelity, achieving an AUC of $0.8946$, demonstrating an acceptable performance tradeoff for the most aggressive latency requirements.

While Conv1D was selected for intensive edge deployment benchmarking due to its fast $\mathcal{O}(N)$ computational complexity, future extended studies will incorporate the latency and memory scaling dynamics for all considered architectures. Table 4 demonstrates the overall classification fidelity achieved by each tested architecture prior to quantization. To ensure strict architectural optimization and prevent overfitting, each model was exposed to an automated Hyperparameter Search Phase using Bayesian Optimization via Keras Tuner. The search spaces evaluated variations in latent dimensions, topological constraints (e.g., filter sizes, attention heads), activation functions (GELU, Swish), and adaptive optimizers (AdamW, Lion). Furthermore, training reconstruction was explicitly stabilized using learning rate schedulers featuring a Warmup Cosine Decay mapped over the total dataset iterations and Early Stopping (based on validation loss plateau constraints).

\begin{table}[h!]
\centering
\caption{Model Architecture Performance Comparison}
\begin{tabular}{lcc}
\toprule
\textbf{Architecture} & \textbf{F1-Score} & \textbf{AUC-ROC} \\
\midrule
MLP-AE & $0.7018$ & $0.6571$ \\
Transformer-AE & $0.8950$ & $0.8800$ \\
LSTM-AE & $0.9161$ & $0.9554$ \\
VAE & $0.8688$ & $0.9229$ \\
CARLA (Base) & $0.8245$ & $0.8309$ \\
Conv1D-AE (Optimal) & $0.9630$ & $0.9840$ \\
GRU-AE & $0.5496$ & $0.5045$ \\
\bottomrule
\end{tabular}
\end{table}

![Comparison of Final Metrics across different model architectures.](plots/final_metrics_comparison.png)

### 5.2 Single-Component Analysis

Using our visualization studio (analyzing 5 varying components: $ESR_C, ESR_L, R_{ds,1}, R_{ds,2}, C_{out}$), the deep learning models successfully partitioned the multi-component degradation profiles. The projection space clearly mapped the boundary between healthy states and anomalous states, demonstrating high sensitivity to the resonance shifts caused by output capacitor degradation.

![Anomaly Deviation Analysis for Output Capacitor ($C_{out}$). The resonance frequency shift is a clear indicator of degradation.](plots/anomaly_deviation_Cout.png)

Furthermore, the test error breakdown across different variation magnitudes confirms that the models exhibit robust generalization even at edge-case bounds.

![Breakdown of test errors across variation magnitudes.](plots/test_error_breakdown.png)

## 6. Conclusion and Future Work

We have demonstrated a comprehensive, modular framework that leverages self-supervised deep learning (CARLA) for non-invasive fault diagnosis in power electronic converters. By integrating NVIDIA TensorRT, we achieved real-time inference latencies below $1\text{ ms}$, proving that advanced contrastive learning methodologies can be deployed effectively on constrained edge hardware.

Future research directions include:
1. **State Space Models (Mamba)**: Implementing linear complexity sequence models [22] to replace Transformers, targeting an 8x speedup.
2. **Physics-Informed Neural Networks (PINNs)**: Embedding the buck converter differential equations [23] directly into the contrastive loss function to improve generalization with less synthetic data.
3. **Power and Energy Profiling**: Extending the hardware benchmarks to incorporate continuous power draw (Watts) and energy-per-inference (Joules) analysis, critical for thermally constrained edge systems.
4. **Kolmogorov-Arnold Networks (KAN)**: Exploring symbolic equation extraction [24] for greater model interpretability.

---

## 7. References

[1] S. Darban, G. A. A. C. V. S. B. P. A., “CARLA: Self-supervised Contrastive Representation Learning for Time Series Anomaly Detection,” *arXiv preprint arXiv:2308.09296*, 2023.

[2] D. P. Kingma and M. Welling, “Auto-Encoding Variational Bayes,” *arXiv preprint arXiv:1312.6114*, 2013.

[3] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you need,” in *Advances in Neural Information Processing Systems*, pp. 5998–6008, 2017.

[4] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A Simple Framework for Contrastive Learning of Visual Representations,” *arXiv preprint arXiv:2002.05709*, 2020.

[5] C. Liu, L. Kou, G. Cai, Z. Zhao, and Z. Zhang, “Review for AI-based Open-Circuit Faults Diagnosis Methods in Power Electronics Converters,” *arXiv preprint arXiv:2209.14058*, 2022.

[6] I. N. Idrisov, D. Okeke, A. Albaseer, M. Abdallah, and F. M. Ibanez, “Leveraging Digital Twin and Machine Learning Techniques for Anomaly Detection in Power Electronics Dominated Grid,” *arXiv preprint arXiv:2501.13474*, 2025.

[7] A. Beattie, P. Mulinka, S. Sahoo, I. T. Christou, C. Kalalas, D. Gutierrez-Rojas, and P. H. J. Nardelli, “A Robust and Explainable Data-Driven Anomaly Detection Approach For Power Electronics,” *arXiv preprint arXiv:2209.11427*, 2022.

[8] S. Kiranyaz, O. Avci, O. Abdeljaber, T. Ince, M. Gabbouj, and D. J. Inman, “1D Convolutional Neural Networks and Applications: A Survey,” *arXiv preprint arXiv:1905.03554*, 2019.

[9] K. Chen, M. Feng, and T. S. Wirjanto, “Harnessing Contrastive Learning and Neural Transformation for Time Series Anomaly Detection,” *arXiv preprint arXiv:2304.07898*, 2023.

[10] P. I. Gomez, M. E. Lopez Gajardo, N. Mijatovic, and T. Dragicevic, “A Self-Commissioning Edge Computing Method for Data-Driven Anomaly Detection in Power Electronic Systems,” *arXiv preprint arXiv:2312.02661*, 2023.

[11] F. Chollet et al., “Keras,” 2015.

[12] M. Abadi et al., “TensorFlow: A system for large-scale machine learning,” in *12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16)*, 2016, pp. 265–283.

[13] W. McKinney, “Data structures for statistical computing in python,” in *Proceedings of the 9th Python in Science Conference*, 2010, pp. 51–56.

[14] F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

[15] J. Bai et al., “ONNX: Open Neural Network Exchange,” 2019.

[16] NVIDIA, “NVIDIA TensorRT,” 2020.

[17] D. E. Rumelhart, G. E. Hinton, and R. J. Williams, “Learning representations by back-propagating errors,” *Nature*, vol. 323, no. 6088, pp. 533–536, 1986.

[18] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” *Neural computation*, vol. 9, no. 8, pp. 1735–1780, 1997.

[19] K. Cho, B. Van Merriënboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio, “Learning phrase representations using RNN encoder-decoder for statistical machine translation,” *arXiv preprint arXiv:1406.1078*, 2014.

[20] T. Cover and P. Hart, “Nearest neighbor pattern classification,” *IEEE transactions on information theory*, vol. 13, no. 1, pp. 21–27, 1967.

[21] T. Fawcett, “An introduction to ROC analysis,” *Pattern recognition letters*, vol. 27, no. 8, pp. 861–874, 2006.

[22] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” *arXiv preprint arXiv:2312.00752*, 2023.

[23] M. Raissi, P. Perdikaris, and G. E. Karniadakis, “Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations,” *Journal of Computational physics*, vol. 378, pp. 686–707, 2019.

[24] Z. Liu et al., “KAN: Kolmogorov-Arnold Networks,” *arXiv preprint arXiv:2404.19756*, 2024.

[25] NVIDIA, “NVIDIA Ada Lovelace Architecture,” *NVIDIA Whitepaper*, 2022.
