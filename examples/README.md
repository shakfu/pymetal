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

### 02_matrix_multiply.py - GPU Matrix Multiplication

GPU-accelerated matrix multiplication with GFLOPS measurement and NumPy comparison.

**Key Features:**
- Matrix multiplication compute kernel
- Multi-buffer management
- 2D thread group dispatch
- GFLOPS calculation
- Result verification
- Scaling from 128x128 to 1024x1024 matrices

**Run:**
```bash
python examples/02_matrix_multiply.py
```

**Expected Output:**
```
Large (512x512) @ Large (512x512):
  NumPy: 45.67 ms
  Metal: 8.23 ms
  Speedup: 5.55x
  Metal: 32.56 GFLOPS
```

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
