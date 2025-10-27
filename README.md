# PyMetal

Python bindings for Apple's Metal GPU API, enabling high-performance GPU computing and graphics programming from Python.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![macOS](https://img.shields.io/badge/platform-macOS-lightgrey.svg)](https://www.apple.com/macos/)
[![Metal](https://img.shields.io/badge/Metal-3.0+-orange.svg)](https://developer.apple.com/metal/)

## Overview

PyMetal provides Pythonic access to Apple's Metal API through metal-cpp and nanobind, allowing you to:

- Write and execute Metal compute shaders from Python
- Build complete graphics pipelines with vertex/fragment shaders
- Leverage GPU acceleration for custom algorithms
- Integrate seamlessly with NumPy for zero-copy data transfer
- Access advanced Metal features like events, binary archives, and capture scopes

**Why PyMetal?**
- **Direct Metal Access**: Full control over GPU resources, not a high-level abstraction
- **Zero-Copy NumPy Integration**: Efficient data transfer between Python and GPU
- **Complete API Coverage**: Compute, graphics, advanced synchronization, and debugging
- **Educational**: Clear examples showing GPU programming concepts
- **Performant**: Properly releases GIL for multithreaded Python applications

## Features

### Core Capabilities

#### Phase 1: Compute Pipeline ✓
- Device management and command queues
- Buffer allocation and management
- Shader compilation from Metal Shading Language source
- Compute pipeline creation and execution
- Thread group configuration and dispatch
- Zero-copy NumPy buffer integration

#### Phase 2: Graphics Pipeline ✓
- **Core Graphics**:
  - Texture creation and management
  - Render pipeline state with vertex/fragment shaders
  - Render pass descriptors with color/depth attachments
  - Sampler states for texture filtering
  - Offscreen rendering

- **Advanced Graphics**:
  - Vertex descriptors and buffer layouts
  - Depth/stencil testing
  - Blit command encoder for memory operations
  - Heap-based resource allocation
  - Fence synchronization
  - Metal layer integration for display

#### Phase 3: Advanced Features ✓
- **Event system** for fine-grained synchronization
- **Shared events** for cross-process coordination
- **Argument buffers** for efficient resource binding
- **Indirect command buffers** for GPU-driven rendering
- **Binary archives** for pipeline caching
- **Capture scopes** for Xcode GPU debugging integration

#### Phase 4: Ray Tracing (Planned)
- Ray tracing acceleration structures
- Ray tracing pipelines
- Intersection function tables
- Primitive acceleration structures

*Note: Ray tracing support can be added on-demand. Current implementation focuses on compute and rasterization.*

### Performance Characteristics

PyMetal achieves realistic GPU performance on Apple Silicon:

| Operation | Performance | Notes |
|-----------|-------------|-------|
| **Image Blur** | 4-5× speedup | Over SciPy for large images (1024×1024+) |
| **Matrix Multiply (Naive)** | ~100 GFLOPS | Educational baseline |
| **Matrix Multiply (Optimized)** | ~220 GFLOPS | With tiling and optimizations |
| **Graphics Rendering** | Full speed | Complete pipeline with depth testing |

**Note**: NumPy/SciPy may be faster for standard operations due to Apple's Accelerate framework and AMX coprocessor. PyMetal excels at **custom algorithms** where specialized hardware doesn't exist.

## Installation

### Requirements

- macOS 11.0+ (Big Sur or later)
- Python 3.12+
- Xcode Command Line Tools
- Metal-compatible GPU (all modern Macs)

### Install from Source

```bash
git clone https://github.com/yourusername/pymetal.git
cd pymetal
pip install -e .
```

### Dependencies

PyMetal automatically installs:
- `nanobind` - C++/Python bindings
- `numpy` - Array operations

For examples, you may also want:
```bash
pip install scipy  # For image blur example
```

## Quick Start

### Hello GPU: Vector Addition

```python
import numpy as np
import pymetal as pm

# Initialize device
device = pm.create_system_default_device()
queue = device.new_command_queue()

# Create data
size = 1024
a = np.random.randn(size).astype(np.float32)
b = np.random.randn(size).astype(np.float32)

# Compile shader
shader = """
#include <metal_stdlib>
using namespace metal;

kernel void vector_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    c[id] = a[id] + b[id];
}
"""

library = device.new_library_with_source(shader)
function = library.new_function("vector_add")
pipeline = device.new_compute_pipeline_state(function)

# Create GPU buffers
a_buffer = device.new_buffer(a.nbytes, pm.ResourceStorageModeShared)
b_buffer = device.new_buffer(b.nbytes, pm.ResourceStorageModeShared)
c_buffer = device.new_buffer(a.nbytes, pm.ResourceStorageModeShared)

# Upload data (zero-copy)
np.copyto(np.frombuffer(a_buffer.contents(), dtype=np.float32), a)
np.copyto(np.frombuffer(b_buffer.contents(), dtype=np.float32), b)

# Execute on GPU
cmd_buffer = queue.command_buffer()
encoder = cmd_buffer.compute_command_encoder()
encoder.set_compute_pipeline_state(pipeline)
encoder.set_buffer(a_buffer, 0, 0)
encoder.set_buffer(b_buffer, 0, 1)
encoder.set_buffer(c_buffer, 0, 2)
encoder.dispatch_threadgroups(16, 1, 1, 64, 1, 1)
encoder.end_encoding()
cmd_buffer.commit()
cmd_buffer.wait_until_completed()

# Read result
result = np.frombuffer(c_buffer.contents(), dtype=np.float32, count=size)
print(f"First 5 results: {result[:5]}")
```

### Graphics: Render a Triangle

```python
import pymetal as pm

device = pm.create_system_default_device()
queue = device.new_command_queue()

# Create render target
width, height = 512, 512
color_desc = pm.TextureDescriptor.texture2d_descriptor(
    pm.PixelFormat.RGBA8Unorm, width, height, False
)
color_texture = device.new_texture(color_desc)

# Vertex and fragment shaders
shader = """
#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float4 color;
};

vertex VertexOut vertex_main(uint vertex_id [[vertex_id]]) {
    float2 positions[3] = {
        float2( 0.0,  0.7),
        float2(-0.7, -0.7),
        float2( 0.7, -0.7)
    };
    float4 colors[3] = {
        float4(1.0, 0.0, 0.0, 1.0),  // Red
        float4(0.0, 1.0, 0.0, 1.0),  // Green
        float4(0.0, 0.0, 1.0, 1.0)   // Blue
    };
    VertexOut out;
    out.position = float4(positions[vertex_id], 0.0, 1.0);
    out.color = colors[vertex_id];
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]]) {
    return in.color;
}
"""

library = device.new_library_with_source(shader)
vertex_func = library.new_function("vertex_main")
fragment_func = library.new_function("fragment_main")

# Create render pipeline
pipeline_desc = pm.RenderPipelineDescriptor.render_pipeline_descriptor()
pipeline_desc.vertex_function = vertex_func
pipeline_desc.fragment_function = fragment_func
pipeline_desc.color_attachment(0).pixel_format = pm.PixelFormat.RGBA8Unorm
pipeline = device.new_render_pipeline_state(pipeline_desc)

# Configure render pass
render_pass = pm.RenderPassDescriptor.render_pass_descriptor()
color_att = render_pass.color_attachment(0)
color_att.texture = color_texture
color_att.load_action = pm.LoadAction.Clear
color_att.store_action = pm.StoreAction.Store
color_att.clear_color = pm.ClearColor(0.0, 0.0, 0.0, 1.0)

# Render
cmd_buffer = queue.command_buffer()
encoder = cmd_buffer.render_command_encoder(render_pass)
encoder.set_render_pipeline_state(pipeline)
encoder.draw_primitives(pm.PrimitiveType.Triangle, 0, 3)
encoder.end_encoding()
cmd_buffer.commit()
cmd_buffer.wait_until_completed()
```

## API Guide

### Device Management

```python
# Get default GPU
device = pm.create_system_default_device()

# Device properties
print(device.name)
print(device.max_threads_per_threadgroup)
```

### Memory Management

```python
# Storage modes
pm.ResourceStorageModeShared      # CPU and GPU accessible
pm.ResourceStorageModePrivate     # GPU only (fastest)
pm.ResourceStorageModeManaged     # Explicit sync required
pm.ResourceStorageModeMemoryless  # Tile memory only

# Create buffer
buffer = device.new_buffer(size_in_bytes, pm.ResourceStorageModeShared)

# Access buffer from Python (zero-copy)
buffer_view = np.frombuffer(buffer.contents(), dtype=np.float32)

# Create texture
tex_desc = pm.TextureDescriptor.texture2d_descriptor(
    pm.PixelFormat.RGBA8Unorm,
    width,
    height,
    mipmapped=False
)
texture = device.new_texture(tex_desc)
```

### Shader Compilation

```python
# Compile from source
library = device.new_library_with_source(shader_source_string)
function = library.new_function("kernel_name")

# Create compute pipeline
compute_pipeline = device.new_compute_pipeline_state(function)

# Create graphics pipeline
render_desc = pm.RenderPipelineDescriptor.render_pipeline_descriptor()
render_desc.vertex_function = vertex_function
render_desc.fragment_function = fragment_function
render_pipeline = device.new_render_pipeline_state(render_desc)
```

### Command Execution

```python
# Create command queue (once)
queue = device.new_command_queue()

# Execute commands
cmd_buffer = queue.command_buffer()

# For compute:
encoder = cmd_buffer.compute_command_encoder()
encoder.set_compute_pipeline_state(pipeline)
encoder.set_buffer(buffer, offset, index)
encoder.dispatch_threadgroups(
    grid_w, grid_h, grid_d,      # Number of threadgroups
    threads_w, threads_h, threads_d  # Threads per group
)
encoder.end_encoding()

# For graphics:
encoder = cmd_buffer.render_command_encoder(render_pass)
encoder.set_render_pipeline_state(pipeline)
encoder.draw_primitives(pm.PrimitiveType.Triangle, 0, vertex_count)
encoder.end_encoding()

# Submit and wait
cmd_buffer.commit()
cmd_buffer.wait_until_completed()  # Blocks (GIL is released)
```

### Thread Group Configuration

```python
# Compute thread groups
threads_per_group = 256  # Must be ≤ max_threads_per_threadgroup
num_elements = 100000
num_groups = (num_elements + threads_per_group - 1) // threads_per_group

encoder.dispatch_threadgroups(
    num_groups, 1, 1,        # Grid size
    threads_per_group, 1, 1  # Threads per group
)

# 2D/3D grids
grid_w = (width + 16 - 1) // 16
grid_h = (height + 16 - 1) // 16
encoder.dispatch_threadgroups(
    grid_w, grid_h, 1,
    16, 16, 1  # 16×16 thread groups
)
```

### Synchronization

```python
# Simple: wait for completion
cmd_buffer.wait_until_completed()

# Advanced: use fences
fence = device.new_fence()
encoder.update_fence(fence)
# ... later ...
encoder.wait_for_fence(fence)

# Events (Phase 3)
event = device.new_event()
shared_event = device.new_shared_event()
shared_event.signaled_value = 42
```

### Debugging

```python
# Enable Metal validation
import os
os.environ['METAL_DEVICE_WRAPPER_TYPE'] = '1'
os.environ['MTL_DEBUG_LAYER'] = '1'

# Use capture scopes with Xcode
manager = pm.shared_capture_manager()
scope = manager.new_capture_scope_with_command_queue(queue)
scope.label = "My Debug Capture"
scope.begin_scope()
# ... GPU work ...
scope.end_scope()
# Capture in Xcode: Product > Perform Action > Capture GPU Frame

# Add labels for debugging
buffer.label = "Input Data"
cmd_buffer.label = "Main Rendering Pass"
```

## Examples

See [`examples/README.md`](examples/README.md) for detailed examples:

1. **01_image_blur.py** - Gaussian blur compute shader
2. **02_matrix_multiply_naive.py** - Simple matrix multiplication (educational)
3. **02_matrix_multiply_tiled.py** - Optimized with shared memory tiling
4. **02_matrix_multiply_optimized.py** - Advanced optimizations
5. **03_triangle_rendering.py** - Complete graphics pipeline
6. **04_advanced_features.py** - Events, capture scopes, and more

Run any example:
```bash
python examples/01_image_blur.py
```

## When to Use PyMetal vs Alternatives

### Use PyMetal When:
- ✅ You need **custom GPU algorithms** not available in libraries
- ✅ You want **full control** over GPU resources
- ✅ You're doing **image processing, simulations, or custom compute**
- ✅ You need to **fuse operations** for efficiency
- ✅ You want to **learn GPU programming** on Apple Silicon
- ✅ You need **rasterization or compute pipelines** (ray tracing coming in Phase 4)

### Use NumPy/SciPy When:
- ✅ Standard operations (matrix multiply, FFT, convolution)
- ✅ Prototyping and development speed matters
- ✅ Small datasets where GPU overhead dominates
- ✅ Apple's Accelerate framework provides optimizations

### Hybrid Approach:
Most applications use **both**:
- NumPy for standard linear algebra
- PyMetal for custom kernels and GPU-specific operations
- Example: NumPy for matrix ops, PyMetal for custom activation functions

## Performance Tips

1. **Use Shared Storage Mode** for CPU-GPU data transfer
2. **Batch operations** - submit multiple dispatches per command buffer
3. **Optimize thread group size** - typically 64-256 threads per group
4. **Use shared/threadgroup memory** for data reuse
5. **Profile with Instruments** - Xcode's GPU profiling tools work great
6. **Release GIL** - PyMetal properly releases GIL during blocking operations

## Project Structure

```
pymetal/
├── src/
│   ├── _pymetal.cpp           # Main C++ bindings
│   └── pymetal/
│       └── __init__.py        # Python module exports
├── examples/                  # 6 practical examples
│   ├── 01_image_blur.py
│   ├── 02_matrix_multiply_*.py
│   ├── 03_triangle_rendering.py
│   └── 04_advanced_features.py
├── tests/                     # 41 unit tests
│   ├── test_phase1_compute.py
│   ├── test_phase2_graphics.py
│   ├── test_phase2_advanced.py
│   └── test_phase3_advanced.py
├── thirdparty/
│   └── metal-cpp/             # Apple's Metal C++ headers
├── CMakeLists.txt             # Build configuration
├── pyproject.toml             # Python package metadata
└── README.md                  # This file
```

## Testing

Run the test suite:
```bash
make test
# or
pytest
```

All 41 tests cover:
- Device and buffer management
- Compute pipeline execution
- Graphics pipeline rendering
- Advanced features (events, capture scopes, etc.)
- Memory management and synchronization

## Roadmap / Future Work

### Potential Phase 4 Features (On-Demand)

**Ray Tracing Support:**
- [ ] Acceleration structure creation and management
- [ ] Ray tracing pipeline descriptors
- [ ] Intersection function tables
- [ ] Ray/primitive intersection queries

**Additional Features:**
- [ ] Resource heaps with placement
- [ ] Sparse textures
- [ ] Indirect argument buffers
- [ ] Metal Performance Shaders (MPS) integration
- [ ] Async compute and graphics overlap
- [ ] Multi-GPU support

**Tooling:**
- [ ] Shader debugging utilities
- [ ] Performance profiling helpers
- [ ] Memory leak detection
- [ ] Automatic optimization suggestions

**Language Bindings:**
- [ ] Type stubs for better IDE support
- [ ] Documentation generator from C++ comments
- [ ] Additional high-level abstractions

These features can be implemented as needed. Contributions welcome!

## Contributing

Contributions welcome! Areas of interest:
- **Ray tracing support** (most requested)
- Additional examples and tutorials
- Performance optimizations
- API coverage improvements
- Documentation enhancements
- Bug fixes and testing

## License

[Add your license here]

## Acknowledgments

- Built on Apple's [metal-cpp](https://developer.apple.com/metal/cpp/)
- Uses [nanobind](https://github.com/wjakob/nanobind) for Python bindings
- Inspired by the need for low-level GPU access from Python on macOS

## Resources

- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Metal Best Practices Guide](https://developer.apple.com/documentation/metal/best_practices)
- [Metal Programming Guide](https://developer.apple.com/documentation/metal)
- [PyMetal Examples](examples/README.md)

## Support

- **Issues**: Open an issue on GitHub
- **Examples**: See [`examples/`](examples/) directory
- **Tests**: See [`tests/`](tests/) directory for API usage patterns

---

**Note**: PyMetal is designed for educational and research purposes. For production graphics applications, consider using established game engines or frameworks.
