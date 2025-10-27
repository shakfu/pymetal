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
    Texture,
    TextureDescriptor,

    # Shader compilation
    Library,
    Function,
    FunctionType,

    # Compute pipeline
    ComputePipelineState,
    ComputeCommandEncoder,

    # Graphics pipeline
    RenderPipelineState,
    RenderPipelineDescriptor,
    RenderPipelineColorAttachmentDescriptor,
    RenderCommandEncoder,

    # Render pass
    RenderPassDescriptor,
    RenderPassAttachmentDescriptor,
    RenderPassColorAttachmentDescriptor,
    RenderPassDepthAttachmentDescriptor,
    ClearColor,

    # Sampling
    SamplerState,
    SamplerDescriptor,

    # Phase 1 Enumerations
    StorageMode,
    CPUCacheMode,
    LoadAction,
    StoreAction,

    # Phase 2 Enumerations
    PixelFormat,
    PrimitiveType,
    IndexType,
    VertexFormat,
    VertexStepFunction,
    CullMode,
    Winding,
    TextureType,
    SamplerMinMagFilter,
    SamplerMipFilter,
    SamplerAddressMode,
    CompareFunction,
    BlendFactor,
    BlendOperation,

    # ResourceOptions constants (bitmask values)
    ResourceCPUCacheModeDefaultCache,
    ResourceCPUCacheModeWriteCombined,
    ResourceStorageModeShared,
    ResourceStorageModeManaged,
    ResourceStorageModePrivate,
    ResourceStorageModeMemoryless,
    ResourceHazardTrackingModeUntracked,

    # ColorWriteMask constants (bitmask values)
    ColorWriteMaskNone,
    ColorWriteMaskRed,
    ColorWriteMaskGreen,
    ColorWriteMaskBlue,
    ColorWriteMaskAlpha,
    ColorWriteMaskAll,
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
    "Texture",
    "TextureDescriptor",

    # Shader compilation
    "Library",
    "Function",
    "FunctionType",

    # Compute pipeline
    "ComputePipelineState",
    "ComputeCommandEncoder",

    # Graphics pipeline
    "RenderPipelineState",
    "RenderPipelineDescriptor",
    "RenderPipelineColorAttachmentDescriptor",
    "RenderCommandEncoder",

    # Render pass
    "RenderPassDescriptor",
    "RenderPassAttachmentDescriptor",
    "RenderPassColorAttachmentDescriptor",
    "RenderPassDepthAttachmentDescriptor",
    "ClearColor",

    # Sampling
    "SamplerState",
    "SamplerDescriptor",

    # Phase 1 Enumerations
    "StorageMode",
    "CPUCacheMode",
    "LoadAction",
    "StoreAction",

    # Phase 2 Enumerations
    "PixelFormat",
    "PrimitiveType",
    "IndexType",
    "VertexFormat",
    "VertexStepFunction",
    "CullMode",
    "Winding",
    "TextureType",
    "SamplerMinMagFilter",
    "SamplerMipFilter",
    "SamplerAddressMode",
    "CompareFunction",
    "BlendFactor",
    "BlendOperation",

    # ResourceOptions constants
    "ResourceCPUCacheModeDefaultCache",
    "ResourceCPUCacheModeWriteCombined",
    "ResourceStorageModeShared",
    "ResourceStorageModeManaged",
    "ResourceStorageModePrivate",
    "ResourceStorageModeMemoryless",
    "ResourceHazardTrackingModeUntracked",

    # ColorWriteMask constants
    "ColorWriteMaskNone",
    "ColorWriteMaskRed",
    "ColorWriteMaskGreen",
    "ColorWriteMaskBlue",
    "ColorWriteMaskAlpha",
    "ColorWriteMaskAll",
]
