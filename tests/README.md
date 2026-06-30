# TDD Policies

This module adheres to the following TDD policies:

1. **Behavioral Testing over Implementation Details**: We test the public mathematical interface (e.g., `model.encode`, `model.decode`, `model.project`) rather than asserting specific internal layer array indices or layer counts. This allows architectural flexibility (adding Dropout, Batchnorm, etc. without breaking tests).
2. **Determinism and Speed**: Tests should construct small data sizes (`batch=4, seq=24, features=5`) and use minimal model sizes (e.g., small filters, 32 latent dims) to allow tests to run rapidly locally or in CI without extensive GPU overhead locking.
3. **Loss/Gradient Validities**: Every custom model `train_step` or custom loss calculation MUST include a regression test verifying that `loss > 0` and that gradients flow appropriately (checking if attributes like tracking metrics are updated).
4. **Integration Isolation**: Elements like Anomaly Injectors, Data loaders, Models, and Trainers should be testable independently.
