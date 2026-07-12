# References

This document lists the key papers and resources that influenced the implementation of the anomaly detection models in this project.

---

## Foundational Papers

### Auto-Encoding Variational Bayes (VAE)
- **Authors**: Kingma, D. P., & Welling, M.
- **arXiv**: 1312.6114
- **Published**: 2013
- **Influence**: Core VAE implementation with reparameterization trick
- **Key Ideas**: Probabilistic latent space, KL divergence regularization

### Attention Is All You Need (Transformer)
- **Authors**: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I.
- **arXiv**: 1706.03762
- **Published**: 2017
- **Influence**: Transformer-AE architecture with multi-head attention
- **Key Ideas**: Self-attention mechanism, positional encoding

### Sequence to Sequence Learning with Neural Networks
- **Authors**: Sutskever, I., Vinyals, O., & Le, Q. V.
- **arXiv**: 1409.3215
- **Published**: 2014
- **Influence**: Encoder-decoder architecture for LSTM-AE
- **Key Ideas**: Sequence embedding and reconstruction

---

## Contrastive Learning for Anomaly Detection

### CARLA: Self-supervised Contrastive Representation Learning for Time Series Anomaly Detection
- **Authors**: Darban, Z. Z., Webb, G. I., Pan, S., Aggarwal, C. C., & Salehi, M.
- **arXiv**: 2308.09296
- **Published**: 2023
- **Influence**: Core CARLA implementation with contrastive autoencoder
- **Key Ideas**: 
  - Synthetic anomaly injection as negative samples
  - NT-Xent contrastive loss with temperature scaling
  - k-NN scoring in projection space
  - Combined reconstruction + contrastive objective

### A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)
- **Authors**: Chen, T., Kornblith, S., Norouzi, M., & Hinton, G.
- **arXiv**: 2002.05709
- **Published**: 2020
- **Influence**: NT-Xent loss formulation, projection head design
- **Key Ideas**: Contrastive learning framework, temperature-scaled cross-entropy

---

## Autoencoder-based Anomaly Detection

### Context-encoding Variational Autoencoder for Unsupervised Anomaly Detection
- **Authors**: Zimmerer, D., Kohl, S. A. A., Petersen, J., Isensee, F., & Maier-Hein, K. H.
- **arXiv**: 1812.05941
- **Published**: 2018
- **Influence**: VAE anomaly scoring combining reconstruction and density-based methods
- **Key Ideas**: Combined reconstruction error with latent space probability for anomaly scoring

### An Encode-then-Decompose Approach to Unsupervised Time Series Anomaly Detection
- **Authors**: Zhang, B., Kieu, T., Qiu, X., Guo, C., Hu, J., Zhou, A., Jensen, C. S., & Yang, B.
- **arXiv**: 2510.18998
- **Published**: 2025
- **Influence**: Robust autoencoder training with contaminated data
- **Key Ideas**: Decomposition of encoded representations, mutual information-based anomaly metric

### RobustTAD: Robust Time Series Anomaly Detection via Decomposition and CNNs
- **Authors**: Gao, J., Song, X., Wen, Q., Wang, P., Sun, L., & Xu, H.
- **arXiv**: 2002.09545
- **Published**: 2020
- **Influence**: CNN-based time series anomaly detection with encoder-decoder architecture
- **Key Ideas**: Seasonal-trend decomposition, multi-scale feature extraction

### Unsupervised Time Series Outlier Detection with Diversity-Driven Convolutional Ensembles
- **Authors**: Campos, D., Kieu, T., Guo, C., Huang, F., Zheng, K., Yang, B., & Jensen, C. S.
- **arXiv**: 2111.11108
- **Published**: 2021
- **Influence**: Convolutional autoencoder ensemble approach for time series
- **Key Ideas**: Diversity-driven training, ensemble methods for anomaly detection

### Trustworthy Anomaly Detection: A Survey
- **Authors**: Yuan, S., & Wu, X.
- **arXiv**: 2202.07787
- **Published**: 2022
- **Influence**: Evaluation methodology and threshold selection approaches
- **Key Ideas**: Interpretability, fairness, and robustness in anomaly detection

---

## Power Electronics & Fault Diagnosis

### Review for AI-based Open-Circuit Faults Diagnosis Methods in Power Electronics Converters
- **Authors**: Liu, C., Kou, L., Cai, G., Zhao, Z., & Zhang, Z.
- **arXiv**: 2209.14058
- **Published**: 2022
- **Influence**: Understanding fault characteristics in power electronics converters
- **Key Ideas**: AI-based fault diagnosis, transient fault features

### Fault Diagnosis for Power Electronics Converters based on Deep Feedforward Network
- **Authors**: Kou, L., Liu, C., Cai, G., & Zhang, Z.
- **arXiv**: 2211.02632
- **Published**: 2022
- **Influence**: Deep learning approach for power converter fault diagnosis
- **Key Ideas**: Wavelet compression, feedforward network for fault classification

---

## Component Tolerances & Degradation Ranges

These sources ground the **normal (healthy) operating envelopes** and the
**anomalous (degraded) ranges** used to (a) generate the training/normal data
(`data/generate_data.py`, reading each converter's
`data/<converter>/component_ranges.json`) and (b) inject synthetic faults on
the transfer function (`src/data/physics_anomaly.py`, `DEFAULT_FAULT_MODES`).
The ranges live in the converter's data folder (not in code) so a new topology
only needs its own `component_ranges.json`; the generic load/classify machinery
is in `src/data/component_ranges.py`. The healthy band per component is *not* a
flat ±5 %: it is dominated by manufacturing tolerance **plus temperature and
ageing**, which differ strongly by component type.

### Small-signal converter model (control-to-output)
- **Reference**: Erickson, R. W., & Maksimović, D. — *Fundamentals of Power Electronics*, 2nd ed., Springer, 2001 (Ch. 8, Table 8.2).
- **Influence**: Canonical `Gvd(s)` used to map component values → pole/zero Bode.
- **Key Ideas**: `ω0 = 1/√(LC)`, `Q = R√(C/L)`, ESR zero `ωz = 1/(Rc·C)`, RHP zero for boost/buck-boost.

### Capacitor value tolerance (E-series / preferred numbers)
- **Standards**: IEC 60063 (E-series preferred numbers); IEC 60384-1 (fixed capacitors, generic spec); tolerance letter codes IEC 60062 (J = ±5 %, K = ±10 %, M = ±20 %).
- **Influence**: Base tolerance of the output capacitor (electrolytic typ. ±20 %, film/MLCC tighter).

### Ceramic (Class 2) temperature + DC-bias + ageing
- **Standards**: EIA RS-198 / IEC 60384-9 (Class-2 codes; e.g. **X7R = ±15 %** over −55…+125 °C).
- **App note**: Waugh, M. D. — "Design solutions for DC bias in multilayer ceramic capacitors," Murata (Class-2 MLCC capacitance can derate by 20–80 % under rated DC bias).
- **Key Ideas**: X7R ageing ≈ 2.5 %/decade-hour; Z5U up to 7 %/decade — so the *effective* healthy spread is well beyond the nameplate tolerance.

### Electrolytic capacitor ageing (C↓, ESR↑) and end-of-life
- **Paper**: Kulkarni, C., & Biswas, G. — "Experimental Studies of Ageing in Electrolytic Capacitors," Annual Conf. of the PHM Society, 2010.
- **Standard**: IEC 60384-4-1 endurance (wear-out) test — a component is a *degradation failure* when **capacitance drops > 30 %** or **ESR / impedance / loss factor rises by > 3×** the initial value. These are the thresholds used for the `Cout` (−30 % ⇒ 0.70×) and `Esr_C` (≥ 3× ⇒ 3.0×) fault onsets.
- **App guides**: Nichicon / Cornell-Dubilier (CDE) Aluminum Electrolytic Capacitor Application Guides — the more conservative *design* end-of-life criteria often quoted are **capacitance −20 %** and/or **ESR ≥ 2× initial**.
- **Key Ideas**: ESR of electrolytics also *decreases* with temperature (≈ 2–3× at cold, ≈ 0.5× at hot) — a reversible healthy variation distinct from ageing; the ×3 fault onset keeps this cold-temperature swing out of the fault band.

### Inductor tolerance and saturation (Isat)
- **App notes**: Coilcraft / Würth Elektronik power-inductor datasheets (typ. **±20 %** tolerance; `Isat` defined as the current at which L drops by 20 % or 30 %); Vishay — "Inductors 101."
- **Key Ideas**: Effective inductance rolls off under DC bias / core saturation; ferrite permeability (hence L) falls as the Curie point is approached.

### MOSFET on-resistance vs. temperature and ageing
- **Datasheets**: Infineon / Vishay power-MOSFET normalized `RDS(on)` vs. junction-temperature curves — positive tempco, typically **≈ 1.7–2.2× from 25 °C to 125 °C** (normal operation).
- **Paper**: Celaya, J. R., et al. — "Towards Prognostics of Power MOSFETs: Accelerated Aging and Precursors of Failure," PHM Society, 2011 (`RDS(on)` rise from die-attach/bond-wire degradation is a recognized ageing precursor).

---

## Additional Resources

### Keras Documentation
- https://keras.io/
- Implementation patterns for custom models and training loops

### Keras Tuner
- https://keras.io/keras_tuner/
- Hyperparameter optimization strategies (Bayesian, Random, Hyperband)

### Scikit-learn Metrics
- https://scikit-learn.org/stable/modules/model_evaluation.html
- Evaluation metrics: ROC-AUC, Precision-Recall, F1-score

---

## BibTeX Citations

```bibtex
% Foundational Papers

@article{kingma2013autoencoding,
  title={Auto-Encoding Variational Bayes},
  author={Kingma, Diederik P. and Welling, Max},
  journal={arXiv preprint arXiv:1312.6114},
  year={2013},
  url={https://arxiv.org/abs/1312.6114}
}

@article{vaswani2017attention,
  title={Attention Is All You Need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, Lukasz and Polosukhin, Illia},
  journal={arXiv preprint arXiv:1706.03762},
  year={2017},
  url={https://arxiv.org/abs/1706.03762}
}

@article{sutskever2014sequence,
  title={Sequence to Sequence Learning with Neural Networks},
  author={Sutskever, Ilya and Vinyals, Oriol and Le, Quoc V.},
  journal={arXiv preprint arXiv:1409.3215},
  year={2014},
  url={https://arxiv.org/abs/1409.3215}
}

% Contrastive Learning

@article{darban2023carla,
  title={CARLA: Self-supervised Contrastive Representation Learning for Time Series Anomaly Detection},
  author={Darban, Zahra Zamanzadeh and Webb, Geoffrey I. and Pan, Shirui and Aggarwal, Charu C. and Salehi, Mahsa},
  journal={arXiv preprint arXiv:2308.09296},
  year={2023},
  url={https://arxiv.org/abs/2308.09296}
}

@article{chen2020simclr,
  title={A Simple Framework for Contrastive Learning of Visual Representations},
  author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  journal={arXiv preprint arXiv:2002.05709},
  year={2020},
  url={https://arxiv.org/abs/2002.05709}
}

% Autoencoder-based Anomaly Detection

@article{zimmerer2018contextvae,
  title={Context-encoding Variational Autoencoder for Unsupervised Anomaly Detection},
  author={Zimmerer, David and Kohl, Simon A. A. and Petersen, Jens and Isensee, Fabian and Maier-Hein, Klaus H.},
  journal={arXiv preprint arXiv:1812.05941},
  year={2018},
  url={https://arxiv.org/abs/1812.05941}
}

@article{zhang2025encodethendecompose,
  title={An Encode-then-Decompose Approach to Unsupervised Time Series Anomaly Detection on Contaminated Training Data},
  author={Zhang, Buang and Kieu, Tung and Qiu, Xiangfei and Guo, Chenjuan and Hu, Jilin and Zhou, Aoying and Jensen, Christian S. and Yang, Bin},
  journal={arXiv preprint arXiv:2510.18998},
  year={2025},
  url={https://arxiv.org/abs/2510.18998}
}

@article{gao2020robusttad,
  title={RobustTAD: Robust Time Series Anomaly Detection via Decomposition and Convolutional Neural Networks},
  author={Gao, Jingkun and Song, Xiaomin and Wen, Qingsong and Wang, Pichao and Sun, Liang and Xu, Huan},
  journal={arXiv preprint arXiv:2002.09545},
  year={2020},
  url={https://arxiv.org/abs/2002.09545}
}

@article{campos2021unsupervised,
  title={Unsupervised Time Series Outlier Detection with Diversity-Driven Convolutional Ensembles},
  author={Campos, David and Kieu, Tung and Guo, Chenjuan and Huang, Feiteng and Zheng, Kai and Yang, Bin and Jensen, Christian S.},
  journal={arXiv preprint arXiv:2111.11108},
  year={2021},
  url={https://arxiv.org/abs/2111.11108}
}

@article{yuan2022trustworthy,
  title={Trustworthy Anomaly Detection: A Survey},
  author={Yuan, Shuhan and Wu, Xintao},
  journal={arXiv preprint arXiv:2202.07787},
  year={2022},
  url={https://arxiv.org/abs/2202.07787}
}

% Power Electronics & Fault Diagnosis

@article{liu2022review,
  title={Review for AI-based Open-Circuit Faults Diagnosis Methods in Power Electronics Converters},
  author={Liu, Chuang and Kou, Lei and Cai, Guowei and Zhao, Zihan and Zhang, Zhe},
  journal={arXiv preprint arXiv:2209.14058},
  year={2022},
  url={https://arxiv.org/abs/2209.14058}
}

@article{kou2022faultdiagnosis,
  title={Fault Diagnosis for Power Electronics Converters based on Deep Feedforward Network and Wavelet Compression},
  author={Kou, Lei and Liu, Chuang and Cai, Guowei and Zhang, Zhe},
  journal={arXiv preprint arXiv:2211.02632},
  year={2022},
  url={https://arxiv.org/abs/2211.02632}
}

% Component Tolerances & Degradation

@book{erickson2001fundamentals,
  title={Fundamentals of Power Electronics},
  author={Erickson, Robert W. and Maksimovi\'c, Dragan},
  edition={2nd},
  publisher={Springer},
  year={2001}
}

@inproceedings{kulkarni2010electrolytic,
  title={Experimental Studies of Ageing in Electrolytic Capacitors},
  author={Kulkarni, Chetan and Biswas, Gautam},
  booktitle={Annual Conference of the PHM Society},
  volume={2},
  year={2010}
}

@inproceedings{celaya2011mosfet,
  title={Towards Prognostics of Power {MOSFETs}: Accelerated Aging and Precursors of Failure},
  author={Celaya, Jose R. and Saxena, Abhinav and Vashchenko, Vladislav and Saha, Sankalita and Goebel, Kai},
  booktitle={Annual Conference of the PHM Society},
  year={2011}
}

@techreport{iec60384_1,
  title={{IEC 60384-1}: Fixed capacitors for use in electronic equipment -- Part 1: Generic specification},
  institution={International Electrotechnical Commission},
  note={See also IEC 60063 (E-series preferred numbers) and EIA RS-198 (ceramic Class-2 codes, e.g. X7R $\pm$15\%)}
}

@misc{waugh_dcbias,
  title={Design Solutions for {DC} Bias in Multilayer Ceramic Capacitors},
  author={Waugh, Mark D.},
  howpublished={Murata Manufacturing application note},
  note={Class-2 MLCC capacitance derates 20--80\% under rated DC bias}
}
```

---

## Citation Format for This Repository

If you use this codebase in your research, please cite the relevant papers above and this repository:

```bibtex
@software{buck_converter_anomaly_detection,
  title = {Buck Converter Anomaly Detection using Autoencoders},
  author = {UC3M Power Electronics Research Group},
  year = {2025},
  url = {https://github.com/your-repo/fault_converters},
  note = {Keras-based autoencoder implementations for power converter fault detection}
}
```
