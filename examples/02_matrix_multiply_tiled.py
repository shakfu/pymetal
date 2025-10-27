"""
Matrix Multiplication Demo (Optimized) - Tiled GPU Implementation

This is an OPTIMIZED implementation using advanced GPU techniques for performance.

This demo shows:
1. Shared/threadgroup memory for caching
2. Tiled matrix multiplication with blocking
3. Coalesced memory access patterns
4. Proper thread group configuration
5. Competitive performance with optimized CPU libraries

OPTIMIZATION TECHNIQUES:
-----------------------
- Threadgroup memory: Cache tiles in fast on-chip memory
- Tiling/blocking: Reduce global memory bandwidth by reusing data
- Coalesced access: All threads in a warp access contiguous memory
- Proper synchronization: threadgroup_barrier ensures correctness

This implementation should show GPU advantages for larger matrices (512x512+)
while still being outperformed by NumPy's Accelerate framework for smaller sizes.

Compare with 02_matrix_multiply_naive.py to see the performance difference.
"""

import numpy as np
import pymetal as pm
import time


def matmul_gpu_tiled(A, B, tile_size=16):
    """
    Optimized GPU matrix multiplication using tiled algorithm with shared memory.

    Uses a 2D tiling approach where each thread group loads tiles of A and B
    into shared memory, computes partial products, and accumulates the result.

    Args:
        A: Matrix of shape (M, K)
        B: Matrix of shape (K, N)
        tile_size: Size of square tiles (default 16x16)

    Returns:
        C: Result matrix of shape (M, N)
    """
    device = pm.create_system_default_device()
    queue = device.new_command_queue()

    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Matrix dimensions don't match"

    # Optimized tiled matrix multiplication shader
    shader_source = f"""
    #include <metal_stdlib>
    using namespace metal;

    // Tiled matrix multiplication with shared memory
    // Each threadgroup computes a {tile_size}x{tile_size} tile of output
    kernel void matmul_tiled(
        device const float* A [[buffer(0)]],
        device const float* B [[buffer(1)]],
        device float* C [[buffer(2)]],
        constant uint3& dims [[buffer(3)]],  // M, K, N
        uint2 gid [[thread_position_in_grid]],
        uint2 tid [[thread_position_in_threadgroup]],
        uint2 bid [[threadgroup_position_in_grid]])
    {{
        uint M = dims.x;
        uint K = dims.y;
        uint N = dims.z;

        // Shared memory tiles - fast on-chip memory
        threadgroup float As[{tile_size}][{tile_size}];
        threadgroup float Bs[{tile_size}][{tile_size}];

        // Global output position
        uint row = gid.y;
        uint col = gid.x;

        float sum = 0.0;

        // Number of tiles needed to cover K dimension
        uint num_tiles = (K + {tile_size} - 1) / {tile_size};

        // Loop over tiles in K dimension
        for (uint t = 0; t < num_tiles; t++) {{
            // Load tile of A into shared memory
            uint a_row = bid.y * {tile_size} + tid.y;
            uint a_col = t * {tile_size} + tid.x;

            if (a_row < M && a_col < K) {{
                As[tid.y][tid.x] = A[a_row * K + a_col];
            }} else {{
                As[tid.y][tid.x] = 0.0;
            }}

            // Load tile of B into shared memory
            uint b_row = t * {tile_size} + tid.y;
            uint b_col = bid.x * {tile_size} + tid.x;

            if (b_row < K && b_col < N) {{
                Bs[tid.y][tid.x] = B[b_row * N + b_col];
            }} else {{
                Bs[tid.y][tid.x] = 0.0;
            }}

            // Synchronize to ensure tile is fully loaded
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Compute partial dot product using shared memory
            for (uint k = 0; k < {tile_size}; k++) {{
                sum += As[tid.y][k] * Bs[k][tid.x];
            }}

            // Synchronize before loading next tile
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        // Write result to global memory
        if (row < M && col < N) {{
            C[row * N + col] = sum;
        }}
    }}
    """

    # Compile shader
    library = device.new_library_with_source(shader_source)
    function = library.new_function("matmul_tiled")
    pipeline = device.new_compute_pipeline_state(function)

    # Prepare data
    A_flat = A.astype(np.float32).flatten()
    B_flat = B.astype(np.float32).flatten()
    dims = np.array([M, K, N], dtype=np.uint32)

    # Create buffers
    A_buffer = device.new_buffer(A_flat.nbytes, pm.ResourceStorageModeShared)
    B_buffer = device.new_buffer(B_flat.nbytes, pm.ResourceStorageModeShared)
    C_buffer = device.new_buffer(M * N * 4, pm.ResourceStorageModeShared)
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

    # Configure thread groups - tile_size x tile_size threads per group
    grid_w = (N + tile_size - 1) // tile_size
    grid_h = (M + tile_size - 1) // tile_size

    encoder.dispatch_threadgroups(grid_w, grid_h, 1,
                                  tile_size, tile_size, 1)
    encoder.end_encoding()

    cmd_buffer.commit()
    cmd_buffer.wait_until_completed()

    # Read result
    result = np.frombuffer(C_buffer.contents(), dtype=np.float32, count=M*N).copy()
    return result.reshape(M, N)


def main():
    print("=" * 70)
    print("PyMetal Matrix Multiplication Demo (OPTIMIZED - Tiled)")
    print("=" * 70)
    print()
    print("This implementation uses:")
    print("  - Shared/threadgroup memory for tile caching")
    print("  - Tiled algorithm to reduce memory bandwidth")
    print("  - Coalesced memory access patterns")
    print("  - Proper synchronization barriers")
    print()

    device = pm.create_system_default_device()
    print(f"GPU Device: {device.name}")

    # Test different matrix sizes
    sizes = [
        (128, 128, "Small (128x128)"),
        (256, 256, "Medium (256x256)"),
        (512, 512, "Large (512x512)"),
        (1024, 1024, "Very Large (1024x1024)"),
        (2048, 2048, "Huge (2048x2048)"),
        (4096, 4096, "Massive (4096x4096)"),
    ]

    print("\n" + "-" * 70)
    print("Performance Comparison: NumPy (Accelerate) vs Metal GPU (Tiled)")
    print("-" * 70)

    for size, _, label in sizes:
        # Create random matrices
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)

        print(f"\n{label} @ {label}:")

        # NumPy timing (uses Apple Accelerate with AMX)
        start = time.time()
        numpy_result = A @ B
        numpy_time = time.time() - start
        print(f"  NumPy (Accelerate): {numpy_time*1000:.2f} ms")

        # GPU timing (optimized tiled implementation)
        start = time.time()
        gpu_result = matmul_gpu_tiled(A, B, tile_size=16)
        gpu_time = time.time() - start
        print(f"  Metal (Tiled):      {gpu_time*1000:.2f} ms")

        speedup = numpy_time / gpu_time
        if speedup >= 1.0:
            print(f"  Speedup: {speedup:.2f}x (GPU wins)")
        else:
            print(f"  Speedup: {speedup:.2f}x (NumPy wins)")

        # Verify correctness
        max_diff = np.max(np.abs(numpy_result - gpu_result))
        rel_error = max_diff / np.max(np.abs(numpy_result))
        print(f"  Max difference: {max_diff:.6f} (relative: {rel_error:.2e})")

        # Calculate throughput
        ops = 2 * size ** 3  # Matrix multiply is 2*N^3 operations
        gflops_numpy = ops / (numpy_time * 1e9)
        gflops_metal = ops / (gpu_time * 1e9)
        print(f"  NumPy: {gflops_numpy:.2f} GFLOPS")
        print(f"  Metal: {gflops_metal:.2f} GFLOPS")

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)

    print("\nOptimization Techniques Demonstrated:")
    print("  ✓ Threadgroup (shared) memory usage")
    print("  ✓ Tiled/blocked matrix multiplication")
    print("  ✓ Coalesced memory access patterns")
    print("  ✓ Proper barrier synchronization")
    print("  ✓ Reduced global memory bandwidth")
    print("  ✓ Improved cache utilization")
    print()
    print("Performance Analysis:")
    print("  - Small matrices: NumPy wins (overhead + AMX coprocessor)")
    print("  - Large matrices: GPU competitive (parallelism pays off)")
    print("  - NumPy uses Apple Accelerate with dedicated matrix hardware")
    print("  - Further GPU optimization possible with async copies, etc.")


if __name__ == "__main__":
    main()
