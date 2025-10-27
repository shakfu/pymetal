"""
pymetal - Python bindings for Apple's Metal GPU API

This package provides Python bindings to Metal via metal-cpp and nanobind,
enabling GPU compute and graphics programming from Python.
"""

from ._pymetal import (
    # Device management
    Device,
    create_system_default_device,

    # Command submission
    CommandQueue,
    CommandBuffer,
    CommandBufferStatus,

    # Memory resources
    Buffer,

    # Shader compilation
    Library,
    Function,
    FunctionType,

    # Compute pipeline
    ComputePipelineState,
    ComputeCommandEncoder,

    # Enumerations
    StorageMode,
    CPUCacheMode,
    LoadAction,
    StoreAction,

    # ResourceOptions constants (bitmask values)
    ResourceCPUCacheModeDefaultCache,
    ResourceCPUCacheModeWriteCombined,
    ResourceStorageModeShared,
    ResourceStorageModeManaged,
    ResourceStorageModePrivate,
    ResourceStorageModeMemoryless,
    ResourceHazardTrackingModeUntracked,
)

__version__ = "0.1.0"

__all__ = [
    # Device management
    "Device",
    "create_system_default_device",

    # Command submission
    "CommandQueue",
    "CommandBuffer",
    "CommandBufferStatus",

    # Memory resources
    "Buffer",

    # Shader compilation
    "Library",
    "Function",
    "FunctionType",

    # Compute pipeline
    "ComputePipelineState",
    "ComputeCommandEncoder",

    # Enumerations
    "StorageMode",
    "CPUCacheMode",
    "LoadAction",
    "StoreAction",

    # ResourceOptions constants
    "ResourceCPUCacheModeDefaultCache",
    "ResourceCPUCacheModeWriteCombined",
    "ResourceStorageModeShared",
    "ResourceStorageModeManaged",
    "ResourceStorageModePrivate",
    "ResourceStorageModeMemoryless",
    "ResourceHazardTrackingModeUntracked",
]
