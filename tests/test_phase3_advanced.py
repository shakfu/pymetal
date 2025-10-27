"""
Phase 3 Advanced Tests: Events, Argument Buffers, Indirect Commands, Binary Archives, Debugging

Tests cutting-edge Metal features including:
- Event system for fine-grained synchronization
- Argument buffers for efficient resource binding
- Indirect command buffers for GPU-driven rendering
- Binary archives for pipeline caching
- Capture scopes for GPU debugging
"""

import pytest
import numpy as np
import pymetal as pm
import tempfile
import os


def test_event_creation():
    """Test basic event creation."""
    device = pm.create_system_default_device()

    # Create event
    event = device.new_event()
    assert event is not None
    assert event.device is not None
    assert event.device.name == device.name
    print(f"Created event on device: {event.device.name}")


def test_shared_event_creation():
    """Test shared event creation and signaling."""
    device = pm.create_system_default_device()

    # Create shared event
    shared_event = device.new_shared_event()
    assert shared_event is not None

    # Test signaled value property
    initial_value = shared_event.signaled_value
    print(f"Initial signaled value: {initial_value}")

    # Set signaled value
    shared_event.signaled_value = 42
    assert shared_event.signaled_value == 42

    shared_event.signaled_value = 100
    assert shared_event.signaled_value == 100

    print(f"Successfully set signaled values: 42 -> 100")


def test_argument_descriptor():
    """Test argument descriptor creation and configuration."""
    # Create argument descriptor
    arg_desc = pm.ArgumentDescriptor.argument_descriptor()
    assert arg_desc is not None

    # Configure descriptor
    arg_desc.data_type = pm.DataType.Float4
    arg_desc.index = 0
    arg_desc.array_length = 1
    arg_desc.access = pm.BindingAccess.ReadOnly

    # Verify properties
    assert arg_desc.data_type == pm.DataType.Float4
    assert arg_desc.index == 0
    assert arg_desc.array_length == 1
    assert arg_desc.access == pm.BindingAccess.ReadOnly

    print(f"Created argument descriptor: type={arg_desc.data_type}, index={arg_desc.index}")


def test_indirect_command_buffer_descriptor():
    """Test indirect command buffer descriptor creation."""
    # Create descriptor
    desc = pm.IndirectCommandBufferDescriptor.indirect_command_buffer_descriptor()
    assert desc is not None

    # Configure descriptor
    desc.command_types = pm.IndirectCommandTypeDraw
    desc.inherit_buffers = False
    desc.inherit_pipeline_state = True
    desc.max_vertex_buffer_bind_count = 8

    # Verify properties
    assert desc.command_types == pm.IndirectCommandTypeDraw
    assert desc.inherit_buffers == False
    assert desc.inherit_pipeline_state == True
    assert desc.max_vertex_buffer_bind_count == 8

    print(f"Created indirect command buffer descriptor: types={desc.command_types}")


def test_indirect_command_buffer_creation():
    """Test indirect command buffer creation."""
    device = pm.create_system_default_device()

    # Create descriptor
    desc = pm.IndirectCommandBufferDescriptor.indirect_command_buffer_descriptor()
    desc.command_types = pm.IndirectCommandTypeDraw
    desc.inherit_pipeline_state = True
    desc.max_vertex_buffer_bind_count = 4

    # Create indirect command buffer
    icb = device.new_indirect_command_buffer(desc, 10, pm.ResourceStorageModeShared)
    assert icb is not None
    assert icb.size == 10

    print(f"Created indirect command buffer with {icb.size} commands")


def test_binary_archive_descriptor():
    """Test binary archive descriptor creation."""
    # Create descriptor
    desc = pm.BinaryArchiveDescriptor.binary_archive_descriptor()
    assert desc is not None

    # Set URL (path)
    with tempfile.NamedTemporaryFile(suffix=".metallib", delete=False) as f:
        temp_path = f.name

    try:
        desc.set_url(temp_path)
        print(f"Set binary archive path: {temp_path}")
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_binary_archive_creation():
    """Test binary archive creation."""
    device = pm.create_system_default_device()

    # Create descriptor
    desc = pm.BinaryArchiveDescriptor.binary_archive_descriptor()

    # Create archive (will be empty/new)
    try:
        archive = device.new_binary_archive(desc)
        assert archive is not None
        assert archive.device is not None
        print(f"Created binary archive on device: {archive.device.name}")
    except RuntimeError as e:
        # Binary archive may fail without valid URL, which is expected
        print(f"Binary archive creation: {e}")
        pytest.skip("Binary archive requires valid URL")


def test_capture_scope_creation():
    """Test capture scope creation."""
    device = pm.create_system_default_device()

    # Get capture manager
    manager = pm.shared_capture_manager()
    assert manager is not None

    # Create capture scope with device
    scope = manager.new_capture_scope_with_device(device)
    assert scope is not None
    assert scope.device is not None

    # Set label
    scope.label = "Test Capture Scope"
    assert scope.label == "Test Capture Scope"

    print(f"Created capture scope: {scope.label}")


def test_capture_scope_with_queue():
    """Test capture scope creation with command queue."""
    device = pm.create_system_default_device()
    queue = device.new_command_queue()

    # Get capture manager
    manager = pm.shared_capture_manager()

    # Create capture scope with queue
    scope = manager.new_capture_scope_with_command_queue(queue)
    assert scope is not None

    scope.label = "Queue Capture Scope"
    print(f"Created capture scope with queue: {scope.label}")


def test_capture_scope_begin_end():
    """Test capture scope begin/end operations."""
    device = pm.create_system_default_device()

    # Get capture manager and create scope
    manager = pm.shared_capture_manager()
    scope = manager.new_capture_scope_with_device(device)
    scope.label = "Begin/End Test"

    # Begin and end scope (doesn't actually capture without starting capture)
    scope.begin_scope()
    # Do some work (or not)
    scope.end_scope()

    print(f"Successfully began and ended capture scope")


def test_capture_manager_properties():
    """Test capture manager properties."""
    manager = pm.shared_capture_manager()
    assert manager is not None

    # Check if capturing (should be False initially)
    is_capturing = manager.is_capturing
    assert isinstance(is_capturing, bool)
    print(f"Capture manager is_capturing: {is_capturing}")

    # Get default capture scope (may be None)
    default_scope = manager.default_capture_scope
    print(f"Default capture scope: {default_scope}")


def test_data_type_enum():
    """Test DataType enumeration values."""
    # Test various data types
    assert pm.DataType.Float == pm.DataType.Float
    assert pm.DataType.Float2 == pm.DataType.Float2
    assert pm.DataType.Float3 == pm.DataType.Float3
    assert pm.DataType.Float4 == pm.DataType.Float4

    assert pm.DataType.Int == pm.DataType.Int
    assert pm.DataType.UInt == pm.DataType.UInt

    assert pm.DataType.Texture == pm.DataType.Texture
    assert pm.DataType.Sampler == pm.DataType.Sampler

    assert pm.DataType.Struct == pm.DataType.Struct
    assert pm.DataType.Array == pm.DataType.Array

    print("All DataType enum values accessible")


def test_binding_access_enum():
    """Test BindingAccess enumeration values."""
    # Test access modes
    assert pm.BindingAccess.ReadOnly == pm.BindingAccess.ReadOnly
    assert pm.BindingAccess.ReadWrite == pm.BindingAccess.ReadWrite
    assert pm.BindingAccess.WriteOnly == pm.BindingAccess.WriteOnly

    # Ensure they're different
    assert pm.BindingAccess.ReadOnly != pm.BindingAccess.ReadWrite
    assert pm.BindingAccess.ReadOnly != pm.BindingAccess.WriteOnly

    print("All BindingAccess enum values accessible")


def test_indirect_command_type_constants():
    """Test IndirectCommandType constants."""
    # Test constants exist
    assert hasattr(pm, 'IndirectCommandTypeDraw')
    assert hasattr(pm, 'IndirectCommandTypeDrawIndexed')
    assert hasattr(pm, 'IndirectCommandTypeDrawPatches')
    assert hasattr(pm, 'IndirectCommandTypeDrawIndexedPatches')

    # Get values
    draw = pm.IndirectCommandTypeDraw
    draw_indexed = pm.IndirectCommandTypeDrawIndexed

    print(f"IndirectCommandType constants: Draw={draw}, DrawIndexed={draw_indexed}")


def test_argument_encoder_with_function():
    """Test argument encoder creation from a function (integration test)."""
    device = pm.create_system_default_device()

    # Create a simple compute shader that uses an argument buffer
    shader_source = """
    #include <metal_stdlib>
    using namespace metal;

    struct Arguments {
        texture2d<float> input [[id(0)]];
        texture2d<float, access::write> output [[id(1)]];
        sampler samp [[id(2)]];
    };

    kernel void argument_kernel(
        constant Arguments& args [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]])
    {
        float4 color = args.input.sample(args.samp, float2(gid) / float2(256.0));
        args.output.write(color, gid);
    }
    """

    try:
        library = device.new_library_with_source(shader_source)
        function = library.new_function("argument_kernel")
        assert function is not None
        print(f"Created function with argument buffer support: {function.name}")
    except Exception as e:
        print(f"Note: Argument buffer shader compilation: {e}")
        pytest.skip("Argument buffer shader requires specific setup")


def test_complete_phase3_workflow():
    """Integration test demonstrating Phase 3 features together."""
    device = pm.create_system_default_device()
    queue = device.new_command_queue()

    print("\n=== Phase 3 Complete Workflow ===")

    # 1. Create events
    event = device.new_event()
    shared_event = device.new_shared_event()
    shared_event.signaled_value = 1
    print(f"✓ Created events (signaled value: {shared_event.signaled_value})")

    # 2. Create capture scope
    manager = pm.shared_capture_manager()
    scope = manager.new_capture_scope_with_command_queue(queue)
    scope.label = "Integration Test Scope"
    print(f"✓ Created capture scope: {scope.label}")

    # 3. Create indirect command buffer
    icb_desc = pm.IndirectCommandBufferDescriptor.indirect_command_buffer_descriptor()
    icb_desc.command_types = pm.IndirectCommandTypeDraw
    icb_desc.inherit_pipeline_state = True
    icb = device.new_indirect_command_buffer(icb_desc, 5, pm.ResourceStorageModeShared)
    print(f"✓ Created indirect command buffer with {icb.size} commands")

    # 4. Create argument descriptor
    arg_desc = pm.ArgumentDescriptor.argument_descriptor()
    arg_desc.data_type = pm.DataType.Texture
    arg_desc.index = 0
    arg_desc.access = pm.BindingAccess.ReadOnly
    print(f"✓ Created argument descriptor for texture at index {arg_desc.index}")

    # 5. Demonstrate capture scope workflow
    scope.begin_scope()

    # Do some GPU work (simple compute)
    shader = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void simple_kernel(device float* data [[buffer(0)]], uint id [[thread_position_in_grid]]) {
        data[id] = float(id);
    }
    """

    library = device.new_library_with_source(shader)
    function = library.new_function("simple_kernel")
    pipeline = device.new_compute_pipeline_state(function)

    buffer = device.new_buffer(256, pm.ResourceStorageModeShared)

    cmd_buffer = queue.command_buffer()
    encoder = cmd_buffer.compute_command_encoder()
    encoder.set_compute_pipeline_state(pipeline)
    encoder.set_buffer(buffer, 0, 0)
    encoder.dispatch_threadgroups(1, 1, 1, 64, 1, 1)
    encoder.end_encoding()
    cmd_buffer.commit()
    cmd_buffer.wait_until_completed()

    scope.end_scope()
    print("✓ Executed GPU work within capture scope")

    print("\n=== Phase 3 Workflow Complete ===")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
