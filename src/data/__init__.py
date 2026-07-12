"""Data loading, preprocessing, and dataset management."""

from .component_ranges import (
    ComponentRange,
    classify_component,
    classify_variations,
    find_ranges_file,
    load_ranges,
    load_ranges_for,
)
from .dataset import (
    PowerConverterDataset,
    BuckConverterDataset,
    TransferFunctionData,
)
from .loader import (
    DataLoader,
    load_simulation_file,
    load_all_simulations,
)
from .preprocessor import (
    DataPreprocessor,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)
from .generator import (
    SequenceGenerator,
    AugmentedGenerator,
)
from .anomaly_injection import (
    AnomalyInjector,
    AnomalyConfig,
    AnomalyType,
    create_anomaly_injector,
)
from .physics_anomaly import (
    PhysicsAnomalyInjector,
    PhysicsAnomalyType,
    TransferFunctionModel,
    PowerConverter,
    BuckConverter,
    BoostConverter,
    BuckBoostConverter,
    FaultMode,
    DEFAULT_FAULT_MODES,
    available_fault_modes,
    make_converter,
    available_converters,
)

__all__ = [
    # Dataset
    "PowerConverterDataset",
    "BuckConverterDataset",
    "TransferFunctionData",
    # Component ranges (shared normal/anomalous spec)
    "ComponentRange",
    "classify_component",
    "classify_variations",
    "find_ranges_file",
    "load_ranges",
    "load_ranges_for",
    # Loader
    "DataLoader",
    "load_simulation_file",
    "load_all_simulations",
    # Preprocessor
    "DataPreprocessor",
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    # Generator
    "SequenceGenerator",
    "AugmentedGenerator",
    # Anomaly Injection
    "AnomalyInjector",
    "AnomalyConfig",
    "AnomalyType",
    "create_anomaly_injector",
    # Physics-based Anomaly Injection
    "PhysicsAnomalyInjector",
    "PhysicsAnomalyType",
    "TransferFunctionModel",
    "PowerConverter",
    "BuckConverter",
    "BoostConverter",
    "BuckBoostConverter",
    "FaultMode",
    "DEFAULT_FAULT_MODES",
    "available_fault_modes",
    "make_converter",
    "available_converters",
]
