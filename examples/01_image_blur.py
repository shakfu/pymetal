"""
Image Blur Demo - Gaussian Blur using Metal Compute Shader

This demo shows how to:
1. Create and initialize GPU buffers
2. Compile and run compute shaders
3. Process image data on the GPU
4. Read results back to CPU

Performance comparison: CPU vs GPU for image processing

Note: GPU shows overhead for small images due to:
- Shader compilation and pipeline setup
- Memory transfer and kernel launch latency
For 256×256, overhead dominates. GPU wins at 512×512+
"""

import numpy as np
import pymetal as pm
import time
from scipy import ndimage


def gaussian_blur_cpu(image, sigma=2.0):
    """CPU-based Gaussian blur using scipy."""
    return ndimage.gaussian_filter(image, sigma=sigma)


def gaussian_blur_gpu(image, sigma=2.0):
    """GPU-based Gaussian blur using Metal compute shader."""
    device = pm.create_system_default_device()
    queue = device.new_command_queue()

    height, width = image.shape

    # Metal shader for Gaussian blur
    shader_source = """
    #include <metal_stdlib>
    using namespace metal;

    // 5x5 Gaussian kernel weights
    constant float gaussian_kernel[25] = {
        1.0/256.0,  4.0/256.0,  6.0/256.0,  4.0/256.0, 1.0/256.0,
        4.0/256.0, 16.0/256.0, 24.0/256.0, 16.0/256.0, 4.0/256.0,
        6.0/256.0, 24.0/256.0, 36.0/256.0, 24.0/256.0, 6.0/256.0,
        4.0/256.0, 16.0/256.0, 24.0/256.0, 16.0/256.0, 4.0/256.0,
        1.0/256.0,  4.0/256.0,  6.0/256.0,  4.0/256.0, 1.0/256.0
    };

    kernel void gaussian_blur(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        constant uint2& dimensions [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]])
    {
        uint width = dimensions.x;
        uint height = dimensions.y;

        if (gid.x >= width || gid.y >= height) return;

        float sum = 0.0;

        // Apply 5x5 kernel
        for (int ky = -2; ky <= 2; ky++) {
            for (int kx = -2; kx <= 2; kx++) {
                int x = int(gid.x) + kx;
                int y = int(gid.y) + ky;

                // Clamp to image boundaries
                x = clamp(x, 0, int(width) - 1);
                y = clamp(y, 0, int(height) - 1);

                uint idx = y * width + x;
                uint kidx = (ky + 2) * 5 + (kx + 2);

                sum += input[idx] * gaussian_kernel[kidx];
            }
        }

        output[gid.y * width + gid.x] = sum;
    }
    """

    # Compile shader
    library = device.new_library_with_source(shader_source)
    function = library.new_function("gaussian_blur")
    pipeline = device.new_compute_pipeline_state(function)

    # Create GPU buffers
    input_data = image.astype(np.float32).flatten()
    dimensions = np.array([width, height], dtype=np.uint32)

    input_buffer = device.new_buffer(input_data.nbytes, pm.ResourceStorageModeShared)
    output_buffer = device.new_buffer(input_data.nbytes, pm.ResourceStorageModeShared)
    dims_buffer = device.new_buffer(dimensions.nbytes, pm.ResourceStorageModeShared)

    # Copy data to GPU
    np.copyto(np.frombuffer(input_buffer.contents(), dtype=np.float32), input_data)
    np.copyto(np.frombuffer(dims_buffer.contents(), dtype=np.uint32), dimensions)

    # Execute compute shader
    cmd_buffer = queue.command_buffer()
    encoder = cmd_buffer.compute_command_encoder()

    encoder.set_compute_pipeline_state(pipeline)
    encoder.set_buffer(input_buffer, 0, 0)
    encoder.set_buffer(output_buffer, 0, 1)
    encoder.set_buffer(dims_buffer, 0, 2)

    # Calculate thread groups
    threads_per_group = min(16, device.max_threads_per_threadgroup.width)
    threadgroup_size = (threads_per_group, threads_per_group, 1)
    grid_size = ((width + threads_per_group - 1) // threads_per_group,
                 (height + threads_per_group - 1) // threads_per_group, 1)

    encoder.dispatch_threadgroups(grid_size[0], grid_size[1], grid_size[2],
                                  threadgroup_size[0], threadgroup_size[1], threadgroup_size[2])
    encoder.end_encoding()

    cmd_buffer.commit()
    cmd_buffer.wait_until_completed()

    # Read result back
    result = np.frombuffer(output_buffer.contents(), dtype=np.float32).copy()
    return result.reshape(height, width)


def main():
    print("=" * 60)
    print("PyMetal Image Blur Demo - Gaussian Blur")
    print("=" * 60)

    # Get device info
    device = pm.create_system_default_device()
    print(f"\nGPU Device: {device.name}")
    print(f"Max threads per threadgroup: {device.max_threads_per_threadgroup.width}")

    # Create test image (grayscale noise)
    sizes = [
        (256, 256, "Small (256x256)"),
        (512, 512, "Medium (512x512)"),
        (1024, 1024, "Large (1024x1024)"),
    ]

    print("\n" + "-" * 60)
    print("Performance Comparison: CPU vs GPU")
    print("-" * 60)

    for height, width, label in sizes:
        # Create random test image
        image = np.random.rand(height, width).astype(np.float32)

        print(f"\n{label}:")

        # CPU timing
        start = time.time()
        cpu_result = gaussian_blur_cpu(image, sigma=2.0)
        cpu_time = time.time() - start
        print(f"  CPU: {cpu_time*1000:.2f} ms")

        # GPU timing (including compilation for first run)
        start = time.time()
        gpu_result = gaussian_blur_gpu(image, sigma=2.0)
        gpu_time = time.time() - start
        print(f"  GPU: {gpu_time*1000:.2f} ms")

        speedup = cpu_time / gpu_time
        print(f"  Speedup: {speedup:.2f}x")

        # Verify results are similar
        diff = np.mean(np.abs(cpu_result - gpu_result))
        print(f"  Mean difference: {diff:.6f}")

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)

    # Show feature usage summary
    print("\nFeatures Demonstrated:")
    print("  ✓ Device query and properties")
    print("  ✓ Shader compilation")
    print("  ✓ Compute pipeline creation")
    print("  ✓ GPU buffer allocation")
    print("  ✓ NumPy zero-copy integration")
    print("  ✓ Compute shader execution")
    print("  ✓ Thread group configuration")
    print("  ✓ CPU-GPU performance comparison")


if __name__ == "__main__":
    main()
