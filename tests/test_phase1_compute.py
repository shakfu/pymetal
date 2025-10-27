"""
Phase 1 Validation Tests: Basic Metal Compute

Tests basic GPU compute functionality including:
- Device creation
- Buffer allocation
- Shader compilation
- Compute pipeline execution
- Array squaring kernel
"""

import pytest
import numpy as np
import pymetal as pm


def test_device_creation():
    """Test that we can create a Metal device."""
    device = pm.create_system_default_device()
    assert device is not None
    assert len(device.name) > 0
    print(f"Device: {device.name}")


def test_command_queue_creation():
    """Test command queue creation."""
    device = pm.create_system_default_device()
    queue = device.new_command_queue()
    assert queue is not None
    assert queue.device is device


def test_buffer_creation():
    """Test buffer allocation and CPU access."""
    device = pm.create_system_default_device()

    # Create buffer with 1KB
    buffer = device.new_buffer(1024, pm.ResourceStorageModeShared)
    assert buffer is not None
    assert buffer.length == 1024
    assert buffer.gpu_address > 0


def test_buffer_numpy_access():
    """Test buffer access via numpy arrays."""
    device = pm.create_system_default_device()

    # Allocate buffer
    size = 16
    buffer = device.new_buffer(size * 4, pm.ResourceStorageModeShared)

    # Get numpy view
    data = buffer.contents()
    assert data.shape == (size * 4,)

    # Write some data
    float_view = np.frombuffer(data, dtype=np.float32)
    float_view[:] = np.arange(size, dtype=np.float32)

    # Verify
    assert float_view[0] == 0.0
    assert float_view[15] == 15.0


def test_shader_compilation():
    """Test Metal shader compilation."""
    device = pm.create_system_default_device()

    # Simple kernel that does nothing
    shader_source = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void dummy_kernel(device float* data [[buffer(0)]],
                            uint idx [[thread_position_in_grid]]) {
        // Do nothing
    }
    """

    library = device.new_library_with_source(shader_source)
    assert library is not None


def test_shader_compilation_error():
    """Test that shader compilation errors are reported."""
    device = pm.create_system_default_device()

    # Invalid shader source
    shader_source = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void bad_kernel(INVALID_TYPE* data [[buffer(0)]]) {
    }
    """

    with pytest.raises(RuntimeError, match="shader compilation failed"):
        library = device.new_library_with_source(shader_source)


def test_function_extraction():
    """Test extracting functions from library."""
    device = pm.create_system_default_device()

    shader_source = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void test_kernel(device float* data [[buffer(0)]]) {
    }
    """

    library = device.new_library_with_source(shader_source)
    function = library.new_function("test_kernel")

    assert function is not None
    assert function.name == "test_kernel"
    assert function.function_type == pm.FunctionType.Kernel


def test_compute_pipeline_creation():
    """Test compute pipeline state creation."""
    device = pm.create_system_default_device()

    shader_source = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void simple_kernel(device float* data [[buffer(0)]]) {
    }
    """

    library = device.new_library_with_source(shader_source)
    function = library.new_function("simple_kernel")
    pipeline = device.new_compute_pipeline_state(function)

    assert pipeline is not None
    assert pipeline.max_total_threads_per_threadgroup > 0
    assert pipeline.thread_execution_width > 0


def test_compute_square_array():
    """
    Complete end-to-end test: Square an array on the GPU.

    This validates the entire Phase 1 implementation:
    - Device and queue creation
    - Buffer allocation
    - Shader compilation
    - Pipeline creation
    - Encoder setup
    - Dispatch and execution
    - Result verification
    """
    device = pm.create_system_default_device()
    queue = device.new_command_queue()

    # Input data
    size = 64
    input_data = np.arange(size, dtype=np.float32)

    # Allocate GPU buffer
    buffer = device.new_buffer(size * 4, pm.ResourceStorageModeShared)

    # Copy input to GPU
    gpu_data = buffer.contents()
    np.frombuffer(gpu_data, dtype=np.float32)[:] = input_data

    # Compile shader
    shader_source = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void square_kernel(device float* data [[buffer(0)]],
                             uint idx [[thread_position_in_grid]]) {
        data[idx] = data[idx] * data[idx];
    }
    """

    library = device.new_library_with_source(shader_source)
    function = library.new_function("square_kernel")
    pipeline = device.new_compute_pipeline_state(function)

    # Create command buffer and encoder
    cmd_buffer = queue.command_buffer()
    encoder = cmd_buffer.compute_command_encoder()

    # Set pipeline and buffer
    encoder.set_compute_pipeline_state(pipeline)
    encoder.set_buffer(buffer, 0, 0)

    # Dispatch
    threads_per_group = min(size, pipeline.thread_execution_width)
    num_groups = (size + threads_per_group - 1) // threads_per_group

    encoder.dispatch_threads(
        size, 1, 1,
        threads_per_group, 1, 1
    )

    encoder.end_encoding()

    # Execute and wait
    cmd_buffer.commit()
    cmd_buffer.wait_until_completed()

    # Verify results
    result = np.frombuffer(buffer.contents(), dtype=np.float32)
    expected = input_data ** 2

    np.testing.assert_allclose(result, expected, rtol=1e-5)
    print(f"Successfully squared {size} elements on GPU")
    print(f"Sample: {input_data[:5]} -> {result[:5]}")


def test_command_buffer_status():
    """Test command buffer status tracking."""
    device = pm.create_system_default_device()
    queue = device.new_command_queue()
    cmd_buffer = queue.command_buffer()

    # Initially not enqueued
    status = cmd_buffer.status
    assert status in [pm.CommandBufferStatus.NotEnqueued, pm.CommandBufferStatus.Enqueued]

    # After commit, should be at least committed
    cmd_buffer.commit()
    cmd_buffer.wait_until_completed()

    status = cmd_buffer.status
    assert status == pm.CommandBufferStatus.Completed


def test_buffer_labels():
    """Test setting debug labels on buffers."""
    device = pm.create_system_default_device()
    buffer = device.new_buffer(1024, pm.ResourceStorageModeShared)

    buffer.label = "test_buffer"
    assert buffer.label == "test_buffer"


def test_dispatch_threadgroups():
    """Test dispatch_threadgroups API variant."""
    device = pm.create_system_default_device()
    queue = device.new_command_queue()

    size = 256
    buffer = device.new_buffer(size * 4, pm.ResourceStorageModeShared)

    shader_source = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void add_one(device float* data [[buffer(0)]],
                       uint idx [[thread_position_in_grid]]) {
        data[idx] += 1.0;
    }
    """

    library = device.new_library_with_source(shader_source)
    function = library.new_function("add_one")
    pipeline = device.new_compute_pipeline_state(function)

    # Initialize data
    gpu_data = buffer.contents()
    np.frombuffer(gpu_data, dtype=np.float32)[:] = 0.0

    # Dispatch using threadgroups
    cmd_buffer = queue.command_buffer()
    encoder = cmd_buffer.compute_command_encoder()

    encoder.set_compute_pipeline_state(pipeline)
    encoder.set_buffer(buffer, 0, 0)

    threads_per_group = 64
    num_groups = size // threads_per_group

    encoder.dispatch_threadgroups(
        num_groups, 1, 1,
        threads_per_group, 1, 1
    )

    encoder.end_encoding()
    cmd_buffer.commit()
    cmd_buffer.wait_until_completed()

    # Verify
    result = np.frombuffer(buffer.contents(), dtype=np.float32)
    assert np.all(result == 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
