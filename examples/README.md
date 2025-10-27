# PyMetal Examples

This directory contains practical examples demonstrating PyMetal's capabilities for GPU computing and graphics programming using Apple's Metal API.

## Requirements

- macOS with Metal-compatible GPU
- Python 3.12+
- NumPy
- SciPy (for example 01 only)

Install dependencies:
```bash
pip install numpy scipy
```

## Examples Overview

### 01_image_blur.py - Image Processing with Compute Shaders

Demonstrates GPU-accelerated Gaussian blur with performance comparison against CPU implementation.

**Key Features:**
- Compute shader compilation and execution
- 2D thread grid configuration
- Zero-copy NumPy buffer integration
- CPU vs GPU performance comparison
- Multiple image sizes (256x256 to 1024x1024)

**Run:**
```bash
python examples/01_image_blur.py
```

**Expected Output:**
```
Performance Comparison: CPU vs GPU
Small (256x256):
  CPU: 15.23 ms
  GPU: 2.45 ms
  Speedup: 6.22x
```

---

### 02_matrix_multiply_naive.py - GPU Matrix Multiplication (Naive)

**Educational implementation** showing PyMetal API usage with a simple, unoptimized algorithm.

**Important:** NumPy will be **faster** because it uses Apple's Accelerate framework with the AMX matrix coprocessor. This demo prioritizes code clarity over performance to demonstrate API patterns.

**Key Features:**
- Simple O(N³) matrix multiplication
- Multi-buffer management
- 2D thread group dispatch
- GFLOPS calculation
- Performance comparison showing CPU advantages

**Run:**
```bash
python examples/02_matrix_multiply_naive.py
```

**Expected Output:**
```
Large (512x512) @ Large (512x512):
  NumPy: 0.35 ms (766 GFLOPS)
  Metal: 6.85 ms (39 GFLOPS)
  NumPy wins - optimized Accelerate framework
```

**Why NumPy is Faster:**
- Uses Apple Accelerate BLAS (hand-tuned assembly)
- Leverages AMX matrix coprocessor on Apple Silicon
- Naive GPU implementation doesn't use shared memory or tiling

---

### 02_matrix_multiply_tiled.py - GPU Matrix Multiplication (Optimized)

**Production-quality implementation** using advanced GPU optimization techniques.

**Key Features:**
- Tiled algorithm with threadgroup (shared) memory
- Coalesced memory access patterns
- Reduced global memory bandwidth
- Proper synchronization barriers
- Competitive performance with NumPy for large matrices

**Run:**
```bash
python examples/02_matrix_multiply_tiled.py
```

**Expected Output:**
```
Large (1024x1024) @ Large (1024x1024):
  NumPy (Accelerate): 2.98 ms (719 GFLOPS)
  Metal (Tiled):      X.XX ms (XXX GFLOPS)
  Improved over naive by ~X-Xx
```

**Optimization Techniques:**
- 16×16 tile blocking
- Shared memory caching
- Reduced memory bandwidth by reusing data
- Better cache utilization

Compare with `02_matrix_multiply_naive.py` to see the impact of GPU optimization.

---

### 02_matrix_multiply_optimized.py - Matrix Multiplication (Highly Optimized)

**Highly optimized implementation** with advanced GPU techniques for maximum performance.

**Key Features:**
- Bank conflict avoidance in shared memory
- Loop unrolling with `#pragma unroll`
- Optimal thread group sizing for M1 GPU
- Better instruction pipeline utilization

**Run:**
```bash
python examples/02_matrix_multiply_optimized.py
```

**Expected Output:**
```
Huge (2048x2048):
  NumPy:  21.35 ms (804 GFLOPS)
  Metal:  88.08 ms (195 GFLOPS) - 2.5× slower but improving

Massive (4096x4096):
  NumPy:  158.32 ms (868 GFLOPS)
  Metal:  617.51 ms (222 GFLOPS) - Getting closer!
```

**Optimizations Applied:**
- TILE_SIZE+1 padding prevents bank conflicts
- `#pragma unroll` for better instruction pipelining
- Optimal 16×16 tile size for M1 occupancy
- ~10-20% improvement over basic tiled version

**Performance Progression:**
- Naive: ~100 GFLOPS
- Tiled: ~200 GFLOPS
- Optimized: ~220 GFLOPS

NumPy remains faster due to dedicated AMX matrix hardware, but optimized GPU shows the limits of software optimization.

---

### 03_triangle_rendering.py - Graphics Pipeline and Rendering

Complete graphics pipeline demonstration with offscreen rendering, depth testing, and image output.

**Key Features:**
- Vertex and fragment shaders
- Render pass configuration
- Color and depth attachments
- Depth testing setup
- Triangle rasterization with color interpolation
- Blit encoder for texture-to-buffer copy
- PPM image file output

**Run:**
```bash
python examples/03_triangle_rendering.py
```

**Output:**
- Renders a colored triangle (red, green, blue vertices)
- Saves to `/tmp/pymetal_triangle.ppm`
- View with: `open /tmp/pymetal_triangle.ppm`

**Expected Output:**
```
Rendering 512x512 triangle on Apple M1
Compiling shaders...
Creating render pipeline...
Rendering triangle...
✓ Image saved to: /tmp/pymetal_triangle.ppm
```

---

### 04_advanced_features.py - Phase 3 Advanced Features

Demonstrates advanced Metal features including event system, shared events, binary archives, and capture scopes.

**Key Features:**
- Event-based synchronization
- Shared events with signaled values
- Binary archive API (pipeline caching)
- Capture scopes for GPU debugging
- Multi-pass compute operations
- Fine-grained command synchronization

**Run:**
```bash
python examples/04_advanced_features.py
```

**Expected Output:**
```
=== Event Synchronization Demo ===
Event synchronization verified: all values = 3.0

=== Shared Events Demo ===
Testing shared event signaling...
Initial value: 0
After signal: 100
Final value: 999

=== Capture Scopes Demo ===
Capture scope began - GPU work is now traceable
Computation verified: first 5 results = [0. 2. 4. 6. 8.]
```

## Common Patterns

### Device Initialization
```python
import pymetal as pm
device = pm.create_system_default_device()
queue = device.new_command_queue()
```

### Compute Shader Workflow
```python
# 1. Compile shader
library = device.new_library_with_source(shader_source)
function = library.new_function("kernel_name")
pipeline = device.new_compute_pipeline_state(function)

# 2. Create buffers
buffer = device.new_buffer(size, pm.ResourceStorageModeShared)

# 3. Encode commands
cmd_buffer = queue.command_buffer()
encoder = cmd_buffer.compute_command_encoder()
encoder.set_compute_pipeline_state(pipeline)
encoder.set_buffer(buffer, 0, 0)
encoder.dispatch_threadgroups(grid_w, grid_h, 1, thread_w, thread_h, 1)
encoder.end_encoding()

# 4. Execute
cmd_buffer.commit()
cmd_buffer.wait_until_completed()
```

### Graphics Pipeline Workflow
```python
# 1. Create render targets
color_desc = pm.TextureDescriptor.texture2d_descriptor(
    pm.PixelFormat.RGBA8Unorm, width, height, False
)
color_texture = device.new_texture(color_desc)

# 2. Configure render pass
render_pass = pm.RenderPassDescriptor.render_pass_descriptor()
color_att = render_pass.color_attachment(0)
color_att.texture = color_texture
color_att.load_action = pm.LoadAction.Clear
color_att.store_action = pm.StoreAction.Store

# 3. Create pipeline
pipeline_desc = pm.RenderPipelineDescriptor.render_pipeline_descriptor()
pipeline_desc.vertex_function = vertex_func
pipeline_desc.fragment_function = fragment_func
pipeline = device.new_render_pipeline_state(pipeline_desc)

# 4. Render
cmd_buffer = queue.command_buffer()
encoder = cmd_buffer.render_command_encoder(render_pass)
encoder.set_render_pipeline_state(pipeline)
encoder.draw_primitives(pm.PrimitiveType.Triangle, 0, 3)
encoder.end_encoding()
cmd_buffer.commit()
```

### Zero-Copy NumPy Integration
```python
# Write to GPU buffer from NumPy
data = np.array([1, 2, 3, 4], dtype=np.float32)
buffer = device.new_buffer(data.nbytes, pm.ResourceStorageModeShared)
np.copyto(np.frombuffer(buffer.contents(), dtype=np.float32), data)

# Read from GPU buffer to NumPy
result = np.frombuffer(buffer.contents(), dtype=np.float32, count=4)
```

## Performance Tips

1. **Use Shared Storage Mode** for CPU-GPU data transfer
   ```python
   buffer = device.new_buffer(size, pm.ResourceStorageModeShared)
   ```

2. **Optimize Thread Group Size** based on problem size
   ```python
   threads_per_group = min(16, device.max_threads_per_threadgroup.width)
   ```

3. **Avoid Synchronous Waits** when possible
   ```python
   # Instead of: cmd_buffer.wait_until_completed()
   # Use completion handlers or fence for async operation
   ```

4. **Batch Operations** to reduce command buffer overhead
   ```python
   # Submit multiple operations in one command buffer
   encoder.dispatch_threadgroups(...)  # Operation 1
   encoder.dispatch_threadgroups(...)  # Operation 2
   encoder.end_encoding()
   ```

## Debugging

### Enable Metal API Validation
```bash
export METAL_DEVICE_WRAPPER_TYPE=1
export MTL_DEBUG_LAYER=1
python examples/01_image_blur.py
```

### Use Capture Scopes with Xcode
```python
manager = pm.shared_capture_manager()
scope = manager.new_capture_scope_with_command_queue(queue)
scope.label = "My Debug Capture"
scope.begin_scope()
# ... GPU work ...
scope.end_scope()
```

Then capture in Xcode: Product > Perform Action > Capture GPU Frame

### Check Device Capabilities
```python
device = pm.create_system_default_device()
print(f"Device: {device.name}")
print(f"Max threads per threadgroup: {device.max_threads_per_threadgroup.width}")
print(f"Supports family: {device.supports_family(pm.GPUFamilyApple8)}")
```

## Troubleshooting

**Problem:** Shader compilation fails
- Check Metal Shading Language syntax
- Ensure kernel/vertex/fragment functions are correctly declared
- Verify buffer bindings match `[[buffer(N)]]` indices

**Problem:** Results don't match expected
- Verify thread group size covers entire data range
- Check for race conditions in shared memory
- Ensure proper synchronization between passes

**Problem:** Performance is slower than expected
- Profile thread group configuration
- Check for CPU-GPU transfer bottlenecks
- Consider using Private storage mode for GPU-only data
- Batch multiple operations into single command buffer

## Additional Resources

- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Metal Best Practices Guide](https://developer.apple.com/documentation/metal/best_practices)
- [PyMetal Test Suite](../tests/) - Comprehensive API examples

## Contributing

Found a bug or want to add an example? Please open an issue or pull request on the PyMetal repository.
