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

    # Vertex descriptors
    VertexDescriptor,
    VertexAttributeDescriptor,
    VertexBufferLayoutDescriptor,

    # Display integration
    MetalLayer,
    MetalDrawable,

    # Phase 2 Advanced: Blit encoder
    BlitCommandEncoder,

    # Phase 2 Advanced: Depth/stencil testing
    DepthStencilState,
    DepthStencilDescriptor,
    StencilDescriptor,

    # Phase 2 Advanced: Memory management
    Heap,
    HeapDescriptor,

    # Phase 2 Advanced: Synchronization
    Fence,

    # Utility structures
    Origin,
    Size,
    Range,

    # Phase 3: Event system
    Event,
    SharedEvent,

    # Phase 3: Argument buffers
    ArgumentEncoder,
    ArgumentDescriptor,

    # Phase 3: Indirect commands
    IndirectCommandBuffer,
    IndirectCommandBufferDescriptor,

    # Phase 3: Binary archive
    BinaryArchive,
    BinaryArchiveDescriptor,

    # Phase 3: Capture/debugging
    CaptureScope,
    CaptureManager,
    shared_capture_manager,

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
    StencilOperation,

    # Phase 3 Enumerations
    DataType,
    BindingAccess,

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

    # IndirectCommandType constants (bitmask values)
    IndirectCommandTypeDraw,
    IndirectCommandTypeDrawIndexed,
    IndirectCommandTypeDrawPatches,
    IndirectCommandTypeDrawIndexedPatches,
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

    # Vertex descriptors
    "VertexDescriptor",
    "VertexAttributeDescriptor",
    "VertexBufferLayoutDescriptor",

    # Display integration
    "MetalLayer",
    "MetalDrawable",

    # Phase 2 Advanced: Blit encoder
    "BlitCommandEncoder",

    # Phase 2 Advanced: Depth/stencil testing
    "DepthStencilState",
    "DepthStencilDescriptor",
    "StencilDescriptor",

    # Phase 2 Advanced: Memory management
    "Heap",
    "HeapDescriptor",

    # Phase 2 Advanced: Synchronization
    "Fence",

    # Utility structures
    "Origin",
    "Size",
    "Range",

    # Phase 3: Event system
    "Event",
    "SharedEvent",

    # Phase 3: Argument buffers
    "ArgumentEncoder",
    "ArgumentDescriptor",

    # Phase 3: Indirect commands
    "IndirectCommandBuffer",
    "IndirectCommandBufferDescriptor",

    # Phase 3: Binary archive
    "BinaryArchive",
    "BinaryArchiveDescriptor",

    # Phase 3: Capture/debugging
    "CaptureScope",
    "CaptureManager",
    "shared_capture_manager",

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
    "StencilOperation",

    # Phase 3 Enumerations
    "DataType",
    "BindingAccess",

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

    # IndirectCommandType constants
    "IndirectCommandTypeDraw",
    "IndirectCommandTypeDrawIndexed",
    "IndirectCommandTypeDrawPatches",
    "IndirectCommandTypeDrawIndexedPatches",
]
