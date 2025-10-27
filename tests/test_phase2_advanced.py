"""
Phase 2 Advanced Tests: GPU Memory Operations, Depth/Stencil, Heaps, Fences

Tests advanced Metal features including:
- BlitCommandEncoder for GPU memory transfers
- Depth/stencil testing
- Heap memory management
- Fence synchronization
"""

import pytest
import numpy as np
import pymetal as pm


def test_blit_buffer_copy():
    """Test buffer-to-buffer copy using blit encoder."""
    device = pm.create_system_default_device()
    queue = device.new_command_queue()

    # Create source buffer with data
    src_data = np.array([1, 2, 3, 4, 5], dtype=np.uint32)
    src_buffer = device.new_buffer(
        src_data.nbytes,
        pm.ResourceStorageModeShared
    )
    # Copy data into buffer
    np.copyto(np.frombuffer(src_buffer.contents(), dtype=np.uint32), src_data)

    # Create destination buffer
    dst_buffer = device.new_buffer(
        src_data.nbytes,
        pm.ResourceStorageModeShared
    )

    # Copy using blit encoder
    cmd_buffer = queue.command_buffer()
    blit_encoder = cmd_buffer.blit_command_encoder()
    blit_encoder.copy_from_buffer(
        src_buffer, 0,  # source, offset
        dst_buffer, 0,  # destination, offset
        src_data.nbytes  # size
    )
    blit_encoder.end_encoding()

    cmd_buffer.commit()
    cmd_buffer.wait_until_completed()

    # Verify data was copied
    dst_view = np.frombuffer(dst_buffer.contents(), dtype=np.uint32)
    assert np.array_equal(src_data, dst_view)
    print(f"Successfully copied buffer: {src_data} -> {dst_view}")


def test_blit_fill_buffer():
    """Test filling a buffer with a constant value."""
    device = pm.create_system_default_device()
    queue = device.new_command_queue()

    # Create buffer
    size = 16
    buffer = device.new_buffer(size, pm.ResourceStorageModeShared)

    # Fill with value 0x42
    cmd_buffer = queue.command_buffer()
    blit_encoder = cmd_buffer.blit_command_encoder()
    blit_encoder.fill_buffer(buffer, pm.Range(0, size), 0x42)
    blit_encoder.end_encoding()

    cmd_buffer.commit()
    cmd_buffer.wait_until_completed()

    # Verify all bytes are 0x42
    data = np.frombuffer(buffer.contents(), dtype=np.uint8)
    assert np.all(data == 0x42)
    print(f"Successfully filled buffer with 0x42: {data[:8]}...")


def test_depth_stencil_descriptor():
    """Test depth/stencil state creation."""
    device = pm.create_system_default_device()

    # Create stencil descriptors
    front_stencil = pm.StencilDescriptor.stencil_descriptor()
    front_stencil.stencil_compare_function = pm.CompareFunction.Always
    front_stencil.stencil_failure_operation = pm.StencilOperation.Keep
    front_stencil.depth_failure_operation = pm.StencilOperation.Keep
    front_stencil.depth_stencil_pass_operation = pm.StencilOperation.Replace
    front_stencil.read_mask = 0xFF
    front_stencil.write_mask = 0xFF

    # Create depth/stencil descriptor
    desc = pm.DepthStencilDescriptor.depth_stencil_descriptor()
    desc.depth_compare_function = pm.CompareFunction.Less
    desc.depth_write_enabled = True
    desc.front_face_stencil = front_stencil
    desc.back_face_stencil = front_stencil
    desc.label = "Test Depth/Stencil"

    # Create state
    state = device.new_depth_stencil_state(desc)
    assert state is not None
    assert state.label == "Test Depth/Stencil"
    print(f"Created depth/stencil state: {state.label}")


def test_heap_creation():
    """Test heap creation and buffer allocation from heap."""
    device = pm.create_system_default_device()

    # Create heap descriptor
    heap_size = 1024 * 1024  # 1MB
    heap_desc = pm.HeapDescriptor.heap_descriptor()
    heap_desc.size = heap_size
    heap_desc.storage_mode = pm.StorageMode.Private
    heap_desc.cpu_cache_mode = pm.CPUCacheMode.DefaultCache

    # Create heap
    heap = device.new_heap(heap_desc)
    assert heap is not None
    assert heap.size == heap_size
    print(f"Created heap: size={heap.size}, used={heap.used_size}")

    # Allocate buffer from heap
    buffer = heap.new_buffer(1024, pm.ResourceStorageModePrivate)
    assert buffer is not None
    assert buffer.length == 1024
    print(f"Allocated buffer from heap: length={buffer.length}")


def test_fence_synchronization():
    """Test fence creation."""
    device = pm.create_system_default_device()

    # Create fence
    fence = device.new_fence()
    assert fence is not None
    assert fence.device is not None
    print(f"Created fence on device: {fence.device.name}")


def test_utility_structures():
    """Test utility structures (Origin, Size, Range)."""
    # Test Origin
    origin = pm.Origin(10, 20, 30)
    assert origin.x == 10
    assert origin.y == 20
    assert origin.z == 30

    # Test Size
    size = pm.Size(100, 200, 1)
    assert size.width == 100
    assert size.height == 200
    assert size.depth == 1

    # Test Range
    range_obj = pm.Range(0, 1024)
    assert range_obj.location == 0
    assert range_obj.length == 1024

    print("All utility structures created successfully")


def test_depth_render_with_depth_stencil():
    """Test rendering with depth/stencil state."""
    device = pm.create_system_default_device()
    queue = device.new_command_queue()

    width, height = 256, 256

    # Create color attachment
    color_desc = pm.TextureDescriptor.texture2d_descriptor(
        pm.PixelFormat.RGBA8Unorm,
        width, height,
        False
    )
    color_texture = device.new_texture(color_desc)

    # Create depth attachment
    depth_desc = pm.TextureDescriptor.texture2d_descriptor(
        pm.PixelFormat.Depth32Float,
        width, height,
        False
    )
    depth_texture = device.new_texture(depth_desc)

    # Create depth/stencil state
    ds_desc = pm.DepthStencilDescriptor.depth_stencil_descriptor()
    ds_desc.depth_compare_function = pm.CompareFunction.Less
    ds_desc.depth_write_enabled = True
    depth_stencil_state = device.new_depth_stencil_state(ds_desc)

    # Create simple shader
    shader_source = """
    #include <metal_stdlib>
    using namespace metal;

    struct VertexOut {
        float4 position [[position]];
        float4 color;
    };

    vertex VertexOut vs_main(uint vid [[vertex_id]]) {
        VertexOut out;
        float2 positions[3] = {
            float2(0.0, 0.5),
            float2(-0.5, -0.5),
            float2(0.5, -0.5)
        };
        out.position = float4(positions[vid], 0.5, 1.0);  // depth = 0.5
        out.color = float4(1.0, 0.0, 0.0, 1.0);
        return out;
    }

    fragment float4 fs_main(VertexOut in [[stage_in]]) {
        return in.color;
    }
    """

    library = device.new_library_with_source(shader_source)
    vs = library.new_function("vs_main")
    fs = library.new_function("fs_main")

    # Create pipeline
    pipeline_desc = pm.RenderPipelineDescriptor.render_pipeline_descriptor()
    pipeline_desc.vertex_function = vs
    pipeline_desc.fragment_function = fs
    pipeline_desc.color_attachment(0).pixel_format = pm.PixelFormat.RGBA8Unorm
    pipeline_desc.depth_attachment_pixel_format = pm.PixelFormat.Depth32Float

    pipeline = device.new_render_pipeline_state(pipeline_desc)

    # Create render pass with depth attachment
    render_pass = pm.RenderPassDescriptor.render_pass_descriptor()

    color_attachment = render_pass.color_attachment(0)
    color_attachment.texture = color_texture
    color_attachment.load_action = pm.LoadAction.Clear
    color_attachment.store_action = pm.StoreAction.Store
    color_attachment.clear_color = pm.ClearColor(0.0, 0.0, 0.0, 1.0)

    depth_attachment = render_pass.depth_attachment
    depth_attachment.texture = depth_texture
    depth_attachment.load_action = pm.LoadAction.Clear
    depth_attachment.store_action = pm.StoreAction.Store
    depth_attachment.clear_depth = 1.0

    # Render
    cmd_buffer = queue.command_buffer()
    encoder = cmd_buffer.render_command_encoder(render_pass)

    encoder.set_render_pipeline_state(pipeline)
    encoder.set_depth_stencil_state(depth_stencil_state)
    encoder.draw_primitives(pm.PrimitiveType.Triangle, 0, 3)
    encoder.end_encoding()

    cmd_buffer.commit()
    cmd_buffer.wait_until_completed()

    print(f"Successfully rendered with depth testing to {width}x{height} texture")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
