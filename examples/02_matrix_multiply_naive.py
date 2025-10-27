"""
Matrix Multiplication Demo (Naive) - Educational GPU Implementation

IMPORTANT: This is a NAIVE implementation designed for clarity and API demonstration.

This demo shows:
1. Large-scale GPU compute operations
2. Multi-buffer management
3. 2D thread grid configuration
4. Performance comparison with NumPy

PERFORMANCE NOTE:
----------------
This implementation uses a simple O(N³) algorithm that is NOT optimized for GPU:
- No shared/threadgroup memory usage
- No tiling or blocking
- Poor memory access patterns
- Every thread reads directly from global memory

NumPy will be FASTER because it uses:
- Apple Accelerate framework (highly optimized BLAS)
- AMX matrix coprocessor on Apple Silicon
- Cache-optimized algorithms
- Sustained ~700 GFLOPS performance

For production matrix operations, use NumPy or scipy. This demo demonstrates
PyMetal API usage patterns, not optimal GPU performance.

See 02_matrix_multiply_tiled.py for an optimized GPU implementation.
"""

import numpy as np
import pymetal as pm
import time


def matmul_gpu_naive(A, B):
    """
    NAIVE GPU matrix multiplication using Metal compute shader.

    This is an educational implementation that prioritizes code clarity
    over performance. Each thread computes one output element using a
    simple loop over the K dimension.

    Args:
        A: Matrix of shape (M, K)
        B: Matrix of shape (K, N)

    Returns:
        C: Result matrix of shape (M, N)
    """
    device = pm.create_system_default_device()
    queue = device.new_command_queue()

    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Matrix dimensions don't match"

    # Naive matrix multiplication shader
    # WARNING: This is intentionally unoptimized for educational purposes
    shader_source = """
    #include <metal_stdlib>
    using namespace metal;

    // Naive O(N³) matrix multiplication kernel
    // Each thread computes one output element C[row][col]
    kernel void matmul_naive(
        device const float* A [[buffer(0)]],
        device const float* B [[buffer(1)]],
        device float* C [[buffer(2)]],
        constant uint3& dims [[buffer(3)]],  // M, K, N
        uint2 gid [[thread_position_in_grid]])
    {
        uint M = dims.x;
        uint K = dims.y;
        uint N = dims.z;

        uint row = gid.y;
        uint col = gid.x;

        if (row >= M || col >= N) return;

        // Naive inner loop - reads from global memory every iteration
        // No use of shared/threadgroup memory
        // Causes poor cache utilization and memory bandwidth waste
        float sum = 0.0;
        for (uint k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];  // Uncoalesced reads
        }

        C[row * N + col] = sum;
    }
    """

    # Compile shader
    library = device.new_library_with_source(shader_source)
    function = library.new_function("matmul_naive")
    pipeline = device.new_compute_pipeline_state(function)

    # Prepare data
    A_flat = A.astype(np.float32).flatten()
    B_flat = B.astype(np.float32).flatten()
    dims = np.array([M, K, N], dtype=np.uint32)

    # Create buffers
    A_buffer = device.new_buffer(A_flat.nbytes, pm.ResourceStorageModeShared)
    B_buffer = device.new_buffer(B_flat.nbytes, pm.ResourceStorageModeShared)
    C_buffer = device.new_buffer(M * N * 4, pm.ResourceStorageModeShared)  # float32 = 4 bytes
    dims_buffer = device.new_buffer(dims.nbytes, pm.ResourceStorageModeShared)

    # Upload data
    np.copyto(np.frombuffer(A_buffer.contents(), dtype=np.float32), A_flat)
    np.copyto(np.frombuffer(B_buffer.contents(), dtype=np.float32), B_flat)
    np.copyto(np.frombuffer(dims_buffer.contents(), dtype=np.uint32), dims)

    # Execute
    cmd_buffer = queue.command_buffer()
    encoder = cmd_buffer.compute_command_encoder()

    encoder.set_compute_pipeline_state(pipeline)
    encoder.set_buffer(A_buffer, 0, 0)
    encoder.set_buffer(B_buffer, 0, 1)
    encoder.set_buffer(C_buffer, 0, 2)
    encoder.set_buffer(dims_buffer, 0, 3)

    # Configure thread groups
    threads_per_group = 16
    grid_w = (N + threads_per_group - 1) // threads_per_group
    grid_h = (M + threads_per_group - 1) // threads_per_group

    encoder.dispatch_threadgroups(grid_w, grid_h, 1,
                                  threads_per_group, threads_per_group, 1)
    encoder.end_encoding()

    cmd_buffer.commit()
    cmd_buffer.wait_until_completed()

    # Read result
    result = np.frombuffer(C_buffer.contents(), dtype=np.float32, count=M*N).copy()
    return result.reshape(M, N)


def main():
    print("=" * 60)
    print("PyMetal Matrix Multiplication Demo (NAIVE)")
    print("=" * 60)
    print()
    print("NOTE: This is an educational implementation showing API usage.")
    print("NumPy will be faster due to Apple Accelerate optimizations.")
    print("See 02_matrix_multiply_tiled.py for an optimized GPU version.")
    print()

    device = pm.create_system_default_device()
    print(f"\nGPU Device: {device.name}")

    # Test different matrix sizes
    sizes = [
        (128, 128, "Small (128x128)"),
        (256, 256, "Medium (256x256)"),
        (512, 512, "Large (512x512)"),
        (1024, 1024, "Very Large (1024x1024)"),
    ]

    print("\n" + "-" * 60)
    print("Performance Comparison: NumPy vs Metal GPU")
    print("-" * 60)

    for size, _, label in sizes:
        # Create random matrices
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)

        print(f"\n{label} @ {label}:")

        # NumPy timing
        start = time.time()
        numpy_result = A @ B
        numpy_time = time.time() - start
        print(f"  NumPy: {numpy_time*1000:.2f} ms")

        # GPU timing (naive implementation)
        start = time.time()
        gpu_result = matmul_gpu_naive(A, B)
        gpu_time = time.time() - start
        print(f"  Metal: {gpu_time*1000:.2f} ms")

        speedup = numpy_time / gpu_time
        print(f"  Speedup: {speedup:.2f}x")

        # Verify correctness
        max_diff = np.max(np.abs(numpy_result - gpu_result))
        print(f"  Max difference: {max_diff:.6f}")

        # Calculate throughput
        ops = 2 * size ** 3  # Matrix multiply is 2*N^3 operations
        gflops_numpy = ops / (numpy_time * 1e9)
        gflops_metal = ops / (gpu_time * 1e9)
        print(f"  NumPy: {gflops_numpy:.2f} GFLOPS")
        print(f"  Metal: {gflops_metal:.2f} GFLOPS")

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)

    print("\nFeatures Demonstrated:")
    print("  ✓ Large-scale compute operations")
    print("  ✓ 2D thread grid configuration")
    print("  ✓ Multiple buffer management")
    print("  ✓ Performance profiling")
    print("  ✓ Result verification")
    print("  ✓ GFLOPS measurement")


if __name__ == "__main__":
    main()
