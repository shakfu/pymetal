# Metal-CPP Wrapper Analysis and Recommendations

## Executive Summary

The metal-cpp library provides comprehensive C++ bindings for Apple's Metal GPU API. This analysis identifies which components should be wrapped via nanobind for a useful Python interface, prioritized by importance and interdependency.

## Architecture Overview

Metal follows a **command buffer encoding pattern**:
1. Create GPU device and command queue
2. Allocate resources (buffers, textures)
3. Compile shaders into pipeline states
4. Record commands into encoders
5. Submit command buffers for execution

## Priority 1: Core Foundation (Essential)

These classes form the minimum viable Metal wrapper and are required for any GPU work:

### Device Management
- **`MTL::Device`** - Primary GPU interface
  - `CreateSystemDefaultDevice()` - Get default GPU (global function)
  - `CopyAllDevices()` - Enumerate GPUs
  - `newCommandQueue()` - Create command submission point
  - `newBuffer()` - Allocate GPU memory
  - `newTexture()` - Create image resources
  - `newLibrary()` - Compile shaders
  - Properties: `name`, `registryID`, `maxThreadsPerThreadgroup`

**Why essential**: Entry point to all Metal functionality. Nothing works without a device.

### Command Submission
- **`MTL::CommandQueue`** - Command submission pipeline
  - `commandBuffer()` - Create new command buffer
  - `device()` - Access parent device
  - Property: `label` (debugging)

- **`MTL::CommandBuffer`** - Records GPU work for one frame
  - `renderCommandEncoder(descriptor)` - Begin rendering
  - `computeCommandEncoder()` - Begin compute work
  - `blitCommandEncoder()` - Begin memory operations
  - `commit()` - Submit to GPU
  - `waitUntilCompleted()` - CPU synchronization
  - `addCompletedHandler(callback)` - Async completion
  - Properties: `status`, `error`, `device`, `commandQueue`

**Why essential**: All GPU work flows through command buffers. Core abstraction for work submission.

### Memory Resources
- **`MTL::Buffer`** - Linear GPU memory
  - `contents()` - CPU-accessible pointer (for shared buffers)
  - `didModifyRange(range)` - Notify GPU of CPU writes
  - `length()` - Size in bytes
  - `gpuAddress()` - GPU virtual address
  - Properties: `storageMode`, `cpuCacheMode`, `label`

- **`MTL::Texture`** - 2D/3D image data
  - `replaceRegion(region, level, data)` - Upload data
  - `getBytes(bytes, bytesPerRow, region, level)` - Download data
  - Properties: `textureType`, `pixelFormat`, `width`, `height`, `depth`, `mipmapLevelCount`, `arrayLength`

**Why essential**: Can't do GPU work without memory. Buffers for compute, textures for graphics.

### Shader Compilation
- **`MTL::Library`** - Compiled shader container
  - `newFunction(name)` - Extract shader by name
  - `functionNames()` - List available shaders

- **`MTL::Function`** - Individual shader function
  - Properties: `name`, `functionType` (vertex/fragment/kernel)

**Why essential**: GPU needs shader code to execute. These load and manage compiled shaders.

### Graphics Pipeline
- **`MTL::RenderPipelineState`** - Immutable graphics pipeline configuration
  - Created via `device.newRenderPipelineState(descriptor)`
  - No direct methods (opaque state object)

- **`MTL::RenderPipelineDescriptor`** - Pipeline configuration
  - `vertexFunction`, `fragmentFunction` - Shader stages
  - `colorAttachments` - Render target configuration
  - `depthAttachmentPixelFormat`, `stencilAttachmentPixelFormat`
  - `vertexDescriptor` - Vertex layout
  - Properties: `label`, `sampleCount`, `alphaToCoverageEnabled`

- **`MTL::RenderCommandEncoder`** - Issues draw commands
  - `setRenderPipelineState(state)` - Bind pipeline
  - `setVertexBuffer(buffer, offset, index)` - Bind vertex data
  - `setFragmentBuffer(buffer, offset, index)` - Bind fragment uniforms
  - `setVertexTexture/setFragmentTexture()` - Bind textures
  - `drawPrimitives(type, start, count)` - Non-indexed draw
  - `drawIndexedPrimitives(type, indexCount, indexType, indexBuffer, offset)` - Indexed draw
  - `setViewport(viewport)`, `setScissorRect(rect)` - Viewport control
  - `endEncoding()` - Finalize commands

**Why essential**: Required for any rendering. Draws triangles to screen.

### Compute Pipeline
- **`MTL::ComputePipelineState`** - Immutable compute pipeline
  - Properties: `maxTotalThreadsPerThreadgroup`, `threadExecutionWidth`

- **`MTL::ComputePipelineDescriptor`** - Compute configuration
  - `computeFunction` - Kernel shader
  - Properties: `label`, `threadGroupSizeIsMultipleOfThreadExecutionWidth`

- **`MTL::ComputeCommandEncoder`** - Issues compute commands
  - `setComputePipelineState(state)` - Bind pipeline
  - `setBuffer(buffer, offset, index)` - Bind buffers
  - `setTexture(texture, index)` - Bind textures
  - `setBytes(bytes, length, index)` - Small inline data
  - `dispatchThreadgroups(threadgroups, threadsPerGroup)` - Launch kernel
  - `dispatchThreads(threads, threadsPerGroup)` - Simplified launch
  - `endEncoding()` - Finalize

**Why essential**: Required for compute/AI/data processing workloads.

### Render Target Configuration
- **`MTL::RenderPassDescriptor`** - Defines rendering targets
  - `colorAttachments` - Array of color targets
  - `depthAttachment`, `stencilAttachment` - Depth/stencil targets
  - `renderTargetArrayLength` - For layered rendering
  - `defaultRasterSampleCount` - MSAA sample count

- **`MTL::RenderPassAttachmentDescriptor`** - Per-attachment config
  - `texture` - Target texture
  - `level`, `slice`, `depthPlane` - Subresource selection
  - `loadAction`, `storeAction` - Load/store behavior
  - `clearColor`, `clearDepth`, `clearStencil` - Clear values
  - `resolveTexture` - MSAA resolve target

**Why essential**: Defines where rendering outputs go. Required for render encoders.

---

## Priority 2: Common Usage (Recommended)

Enables practical applications beyond minimal functionality:

### Memory Operations
- **`MTL::BlitCommandEncoder`** - Copy/convert/synchronize
  - `copyFromBuffer(src, srcOffset, dst, dstOffset, size)` - Buffer copy
  - `copyFromTexture(src, ...)` to texture/buffer - Texture operations
  - `generateMipmaps(texture)` - Generate mip levels
  - `fillBuffer(buffer, range, value)` - Initialize memory
  - `synchronizeResource/Texture()` - Explicit synchronization
  - `endEncoding()`

**Why recommended**: Essential for data transfer CPU↔GPU and GPU↔GPU.

### Texture Sampling
- **`MTL::SamplerState`** - Texture sampling configuration
  - Created via `device.newSamplerState(descriptor)`

- **`MTL::SamplerDescriptor`** - Sampling configuration
  - `minFilter`, `magFilter`, `mipFilter` - Filter modes
  - `sAddressMode`, `tAddressMode`, `rAddressMode` - Wrap modes
  - `maxAnisotropy` - Anisotropic filtering
  - `lodMinClamp`, `lodMaxClamp` - LOD control
  - `compareFunction` - For shadow mapping
  - `normalizedCoordinates` - UV [0,1] vs pixel coords

**Why recommended**: Required for any texture reading in shaders.

### Depth/Stencil Testing
- **`MTL::DepthStencilState`** - Depth/stencil test configuration
  - Created via `device.newDepthStencilState(descriptor)`

- **`MTL::DepthStencilDescriptor`** - Test configuration
  - `depthCompareFunction` - Depth test function
  - `depthWriteEnabled` - Enable depth writes
  - `frontFaceStencil`, `backFaceStencil` - Stencil per-face
  - `label`

- **`MTL::StencilDescriptor`** - Per-face stencil operations
  - `stencilCompareFunction` - Test function
  - `stencilFailureOperation`, `depthFailureOperation`, `depthStencilPassOperation`
  - `readMask`, `writeMask`

**Why recommended**: Standard for 3D rendering with depth sorting.

### Vertex Input
- **`MTL::VertexDescriptor`** - Vertex layout specification
  - `layouts` - Buffer stride/step configuration
  - `attributes` - Attribute format/offset

- **`MTL::VertexBufferLayoutDescriptor`** - Buffer layout
  - `stride` - Bytes between vertices
  - `stepFunction` - Per-vertex or per-instance
  - `stepRate` - Instance repetition

- **`MTL::VertexAttributeDescriptor`** - Attribute format
  - `format` - Data type (float3, float4, etc.)
  - `offset` - Bytes from vertex start
  - `bufferIndex` - Which buffer slot

**Why recommended**: Defines vertex structure for rendering.

### Memory Management
- **`MTL::Heap`** - Suballocator for resources
  - `newBuffer(length, options)` - Allocate from heap
  - `newTexture(descriptor)` - Allocate from heap
  - `currentAllocatedSize`, `usedSize` - Memory tracking
  - `maxAvailableSizeWithAlignment()`

- **`MTL::HeapDescriptor`** - Heap configuration
  - `size` - Total heap size
  - `storageMode`, `cpuCacheMode`
  - `type` - Automatic, Placement, Sparse

**Why recommended**: Better memory efficiency for many small allocations.

### Display Integration
- **`CA::MetalLayer`** (QuartzCore) - Native display surface
  - `nextDrawable()` - Get drawable for current frame
  - `device` - GPU to use
  - `pixelFormat` - Color format
  - `drawableSize` - Frame dimensions (CGSize)
  - `displaySyncEnabled` - VSync control
  - `maximumDrawableCount` - Swapchain depth
  - Properties: `framebufferOnly`, `presentsWithTransaction`

- **`CA::MetalDrawable`** - Single frame texture
  - `texture()` - Get render target texture
  - `present()` - Queue for display
  - `presentAtTime(time)` - Scheduled presentation

**Why recommended**: Required for any on-screen rendering.

### Synchronization
- **`MTL::Fence`** - Coarse-grained synchronization
  - Used with encoder methods: `updateFence()`, `waitForFence()`
  - Coordinates work between encoders in same command buffer

**Why recommended**: Enables complex multi-pass rendering.

### Resource Configuration
- **`MTL::TextureDescriptor`** - Texture creation parameters
  - `texture2DDescriptor(format, width, height, mipmapped)` - Common 2D
  - `textureCubeDescriptor(format, size, mipmapped)` - Cubemap
  - Properties: `textureType`, `pixelFormat`, `width`, `height`, `depth`
  - `mipmapLevelCount`, `sampleCount`, `arrayLength`
  - `resourceOptions`, `cpuCacheMode`, `storageMode`, `usage`

**Why recommended**: Fine control over texture allocation.

---

## Priority 3: Advanced Features (Optional)

Specialized capabilities for advanced use cases:

### Ray Tracing
- **`MTL::AccelerationStructure`** - BVH for ray tracing
- **`MTL::AccelerationStructureDescriptor`** - BVH configuration
- **`MTL::PrimitiveAccelerationStructureDescriptor`** - Geometry BVH
- **`MTL::InstanceAccelerationStructureDescriptor`** - Instance BVH
- **`MTL::VisibleFunctionTable`** - Ray tracing shader table
- **`MTL::IntersectionFunctionTable`** - Custom intersection shaders

**Why optional**: Ray tracing is cutting-edge, smaller audience.

### GPU-Driven Rendering
- **`MTL::IndirectCommandBuffer`** - GPU-generated commands
- **`MTL::IndirectCommandBufferDescriptor`** - Configuration
- **`MTL::IndirectRenderCommand`** - GPU-authored draw

**Why optional**: Advanced optimization for expert users.

### Dynamic Linking
- **`MTL::DynamicLibrary`** - Runtime-loaded shaders
- **`MTL::BinaryArchive`** - Cached pipeline compilation

**Why optional**: Complex shader management, limited use cases.

### Event System
- **`MTL::Event`** - Fine-grained synchronization
- **`MTL::SharedEvent`** - Cross-process sync

**Why optional**: Advanced scheduling, most users don't need.

### Argument Buffers
- **`MTL::ArgumentEncoder`** - Indirect resource binding
- **`MTL::ArgumentDescriptor`** - Argument layout

**Why optional**: Performance optimization, adds complexity.

### Debugging/Profiling
- **`MTL::CaptureManager`** - GPU frame capture
- **`MTL::CaptureScope`** - Delimit capture regions
- **`MTL::CounterSet`** - Performance counters

**Why optional**: Development tools, not runtime functionality.

---

## Critical Enumerations and Types

### Must Wrap
```cpp
// Resource configuration
MTL::StorageMode (Shared, Managed, Private, Memoryless)
MTL::CPUCacheMode (DefaultCache, WriteCombined)
MTL::ResourceOptions (combines storage + cache modes)

// Pixel formats (subset of ~100 formats)
MTL::PixelFormat:
  - RGBA8Unorm, BGRA8Unorm (standard color)
  - R32Float, RG32Float, RGBA32Float (HDR/compute)
  - Depth32Float, Depth24Unorm_Stencil8 (depth buffers)

// Primitive types
MTL::PrimitiveType (Point, Line, LineStrip, Triangle, TriangleStrip)

// Texture types
MTL::TextureType (1D, 2D, 3D, Cube, 2DArray, CubeArray)
MTL::TextureUsage (ShaderRead, ShaderWrite, RenderTarget, PixelFormatView)

// Load/store actions
MTL::LoadAction (DontCare, Load, Clear)
MTL::StoreAction (DontCare, Store, MultisampleResolve, StoreAndMultisampleResolve)

// Comparison functions
MTL::CompareFunction (Never, Less, Equal, LessEqual, Greater, NotEqual, GreaterEqual, Always)

// Blend factors/operations
MTL::BlendFactor (Zero, One, SourceColor, OneMinusSourceColor, SourceAlpha, etc.)
MTL::BlendOperation (Add, Subtract, ReverseSubtract, Min, Max)

// Vertex formats
MTL::VertexFormat (Float, Float2, Float3, Float4, Int, UInt, etc.)
MTL::VertexStepFunction (PerVertex, PerInstance)

// Sampler modes
MTL::SamplerMinMagFilter (Nearest, Linear)
MTL::SamplerMipFilter (NotMipmapped, Nearest, Linear)
MTL::SamplerAddressMode (ClampToEdge, MirrorClampToEdge, Repeat, MirrorRepeat, ClampToZero)

// Index types
MTL::IndexType (UInt16, UInt32)

// Function types
MTL::FunctionType (Vertex, Fragment, Kernel)

// Winding/culling
MTL::Winding (Clockwise, CounterClockwise)
MTL::CullMode (None, Front, Back)
```

### Utility Structures
```cpp
MTL::Origin (x, y, z)
MTL::Size (width, height, depth)
MTL::Region (origin, size)
MTL::Viewport (originX, originY, width, height, znear, zfar)
MTL::ScissorRect (x, y, width, height)
MTL::ClearColor (red, green, blue, alpha)
MTL::SizeAndAlign (size, align)
```

---

## Foundation Types to Wrap

Metal uses Objective-C Foundation types via C++ wrappers:

```cpp
NS::String - For labels, shader names, error messages
NS::Error - Error reporting
NS::Array - Collections (for enumeration)
NS::AutoreleasePool - Memory management in tight loops
```

**Recommendation**: Create minimal Python bridges that convert:
- `NS::String` ↔ Python `str`
- `NS::Error` → Python exception
- `NS::Array` → Python `list` or generator
- Handle `NS::AutoreleasePool` internally

---

## Nanobind Wrapping Strategy

### 1. Class Hierarchy
Metal uses reference-counted objects. Nanobind's `nb::class_<T>` with custom holders:

```cpp
// Example pattern
nb::class_<MTL::Device>(m, "Device")
    .def("new_command_queue", &MTL::Device::newCommandQueue)
    .def("new_buffer", &MTL::Device::newBuffer)
    .def_prop_ro("name", &MTL::Device::name);
```

### 2. Enum Wrapping
Use `nb::enum_<T>`:

```cpp
nb::enum_<MTL::PixelFormat>(m, "PixelFormat")
    .value("RGBA8Unorm", MTL::PixelFormat::PixelFormatRGBA8Unorm)
    .value("BGRA8Unorm", MTL::PixelFormat::PixelFormatBGRA8Unorm);
```

### 3. Buffer Protocol
Expose `MTL::Buffer` contents as Python buffer protocol (e.g., numpy arrays):

```cpp
.def_buffer([](MTL::Buffer* buf) {
    return nb::buffer_info(
        buf->contents(),
        buf->length(),
        // ... format descriptor
    );
});
```

### 4. Resource Lifetime
Metal uses retain/release. Nanobind needs custom holder or intrusive_ptr support:

```cpp
// Option 1: intrusive_ptr-style
namespace nanobind::detail {
    template<> struct type_caster<MTL::Device*> {
        // Custom retain/release logic
    };
}

// Option 2: nb::ref wrapper
```

### 5. Callback Handling
For completion handlers, use `nb::cpp_function`:

```cpp
.def("add_completed_handler", [](MTL::CommandBuffer* buf, nb::object callback) {
    buf->addCompletedHandler(^(MTL::CommandBuffer* cb) {
        nb::gil_scoped_acquire guard;
        callback(cb);
    });
});
```

---

## Recommended Phased Implementation

### Phase 1: Minimal Compute (Week 1)
Enable basic GPU compute:
- `MTL::Device` (creation + basic methods)
- `MTL::CommandQueue`, `MTL::CommandBuffer`
- `MTL::Buffer` (with buffer protocol)
- `MTL::Library`, `MTL::Function`
- `MTL::ComputePipelineState`, `MTL::ComputePipelineDescriptor`
- `MTL::ComputeCommandEncoder`
- Core enums: StorageMode, ResourceOptions, LoadAction

**Validation**: Run a simple kernel that squares an array.

### Phase 2: Rendering Basics (Week 2-3)
Add graphics pipeline:
- `MTL::Texture`, `MTL::TextureDescriptor`
- `MTL::SamplerState`, `MTL::SamplerDescriptor`
- `MTL::RenderPipelineState`, `MTL::RenderPipelineDescriptor`
- `MTL::RenderCommandEncoder`
- `MTL::RenderPassDescriptor` + attachment descriptors
- `MTL::VertexDescriptor`
- `CA::MetalLayer`, `CA::MetalDrawable`
- Graphics enums: PixelFormat, PrimitiveType, TextureType

**Validation**: Render a colored triangle to a window.

### Phase 3: Depth/Stencil & Utilities (Week 4)
Complete common rendering:
- `MTL::DepthStencilState`, `MTL::DepthStencilDescriptor`
- `MTL::BlitCommandEncoder`
- `MTL::Fence`
- `MTL::Heap`, `MTL::HeapDescriptor`
- Additional enums: CompareFunction, BlendFactor, CullMode

**Validation**: 3D scene with depth testing and texture mapping.

### Phase 4: Advanced (Ongoing)
Per-demand features:
- Ray tracing structures
- Indirect command buffers
- Argument encoders
- Capture manager

---

## Python API Design Considerations

### Pythonic Patterns

**Resource Management**:
```python
# Use context managers
with device.new_command_queue() as queue:
    with queue.command_buffer() as cmd:
        with cmd.compute_encoder() as encoder:
            encoder.dispatch(...)
```

**Buffer Access**:
```python
# NumPy integration
import numpy as np
buffer = device.new_buffer(1024, StorageMode.Shared)
array = np.asarray(buffer)  # Zero-copy view
array[:] = [1, 2, 3, 4]
```

**Enum Access**:
```python
# Attribute-style access
from pymetal import PixelFormat
texture = device.new_texture(format=PixelFormat.RGBA8Unorm, ...)
```

**Callback Simplicity**:
```python
# Simple Python callbacks
cmd_buffer.add_completed_handler(lambda buf: print("Done!"))
```

### Error Handling
Convert `NS::Error` to Python exceptions:

```python
class MetalError(Exception): pass
class MetalCompileError(MetalError): pass
class MetalResourceError(MetalError): pass
```

### Documentation
Docstrings with type hints:

```python
def new_buffer(self, length: int, options: ResourceOptions) -> Buffer:
    """Create a new buffer resource.

    Args:
        length: Size in bytes
        options: Storage and cache mode configuration

    Returns:
        Allocated buffer object

    Raises:
        MetalResourceError: If allocation fails
    """
```

---

## Testing Strategy

### Unit Tests
Per-class validation:
- Device creation/enumeration
- Buffer allocation/access
- Shader compilation
- Pipeline creation

### Integration Tests
Complete workflows:
- Compute kernel execution
- Triangle rendering
- Texture upload/download
- Multi-pass rendering

### Example Code
Maintain runnable examples:
- `examples/01_device_info.py` - List GPUs
- `examples/02_compute_add.py` - Vector addition
- `examples/03_render_triangle.py` - Basic rendering
- `examples/04_texture_sampling.py` - Texture mapping

---

## Performance Considerations

### Zero-Copy Where Possible
- Buffer protocol for `MTL::Buffer::contents()`
- Avoid unnecessary Python↔C++ conversions

### Batch Operations
- Expose `setBuffers(list)` for binding multiple buffers
- `setTextures(list)` for multiple textures

### GIL Management
- Release GIL during:
  - Command buffer commit
  - Wait operations
  - Texture uploads/downloads
- Acquire GIL only for callbacks

### Memory Efficiency
- Use weak references for encoder↔command buffer
- Proper retain/release for Metal objects
- Avoid creating temporary Python objects in hot paths

---

## Challenges and Solutions

### Challenge 1: Objective-C Blocks
Metal uses Objective-C blocks for callbacks (`^{ ... }`).

**Solution**: Create C++ lambda wrappers that capture Python callables, manage GIL.

### Challenge 2: Reference Counting
Metal uses `NS::Object` retain/release semantics.

**Solution**: Nanobind intrusive_ptr or custom holder with `retain()`/`release()` calls.

### Challenge 3: Metal Shader Language (MSL)
Shaders are strings compiled at runtime.

**Solution**:
- Provide MSL as Python strings
- Consider `.metal` file loading utilities
- Future: Python shader DSL (like Taichi/Warp)

### Challenge 4: Platform Lock-in
Metal is macOS/iOS only.

**Solution**:
- Clear documentation about platform requirements
- Consider future: stub API for cross-platform testing
- Potential: Wrap MoltenVK for Vulkan fallback

### Challenge 5: Type Safety
Many Metal APIs use void pointers and indices.

**Solution**:
- Strong typing in Python API
- Runtime validation of buffer bindings
- Clear error messages

---

## Conclusion

**Minimum Viable Product**: Priority 1 classes (Device → CommandBuffer → Compute/Render encoders → Resources)

**Full-Featured Library**: Priority 1 + Priority 2 (adds display, sampling, memory management)

**Comprehensive Wrapper**: All three priorities (includes ray tracing, advanced features)

**Recommended Start**: Phase 1 (compute only) to validate toolchain, then Phase 2 (graphics) for broader appeal.

The architecture of metal-cpp is well-suited for Python wrapping via nanobind, with clear class boundaries and explicit resource management that maps naturally to Python's object model.
