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
