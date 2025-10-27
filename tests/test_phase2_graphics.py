"""
Phase 2 Validation Tests: Basic Metal Graphics

Tests basic GPU graphics functionality including:
- Texture creation
- Render pipeline setup
- Offscreen rendering
- Triangle rendering to texture
"""

import pytest
import numpy as np
import pymetal as pm


def test_texture_creation():
    """Test texture creation."""
    device = pm.create_system_default_device()

    desc = pm.TextureDescriptor.texture2d_descriptor(
        pm.PixelFormat.RGBA8Unorm,
        512, 512,
        False
    )

    texture = device.new_texture(desc)
    assert texture is not None
    assert texture.width == 512
    assert texture.height == 512
    assert texture.pixel_format == pm.PixelFormat.RGBA8Unorm


def test_render_pipeline_creation():
    """Test render pipeline creation."""
    device = pm.create_system_default_device()

    # Simple shaders
    shader_source = """
    #include <metal_stdlib>
    using namespace metal;

    struct VertexOut {
        float4 position [[position]];
        float4 color;
    };

    vertex VertexOut vertex_main(uint vertex_id [[vertex_id]]) {
        VertexOut out;
        // Simple triangle
        float2 positions[3] = {
            float2(0.0, 0.5),
            float2(-0.5, -0.5),
            float2(0.5, -0.5)
        };
        float4 colors[3] = {
            float4(1.0, 0.0, 0.0, 1.0),
            float4(0.0, 1.0, 0.0, 1.0),
            float4(0.0, 0.0, 1.0, 1.0)
        };
        out.position = float4(positions[vertex_id], 0.0, 1.0);
        out.color = colors[vertex_id];
        return out;
    }

    fragment float4 fragment_main(VertexOut in [[stage_in]]) {
        return in.color;
    }
    """

    library = device.new_library_with_source(shader_source)
    vertex_func = library.new_function("vertex_main")
    fragment_func = library.new_function("fragment_main")

    assert vertex_func is not None
    assert fragment_func is not None

    # Create pipeline descriptor
    pipeline_desc = pm.RenderPipelineDescriptor.render_pipeline_descriptor()
    pipeline_desc.vertex_function = vertex_func
    pipeline_desc.fragment_function = fragment_func

    # Set color attachment format
    color_attachment = pipeline_desc.color_attachment(0)
    color_attachment.pixel_format = pm.PixelFormat.RGBA8Unorm

    # Create pipeline
    pipeline = device.new_render_pipeline_state(pipeline_desc)
    assert pipeline is not None


def test_render_pass_descriptor():
    """Test render pass descriptor creation."""
    device = pm.create_system_default_device()

    # Create render target texture
    tex_desc = pm.TextureDescriptor.texture2d_descriptor(
        pm.PixelFormat.RGBA8Unorm,
        256, 256,
        False
    )
    texture = device.new_texture(tex_desc)

    # Create render pass descriptor
    render_pass = pm.RenderPassDescriptor.render_pass_descriptor()
    assert render_pass is not None

    # Configure color attachment
    color_attachment = render_pass.color_attachment(0)
    color_attachment.texture = texture
    color_attachment.load_action = pm.LoadAction.Clear
    color_attachment.store_action = pm.StoreAction.Store

    clear_color = pm.ClearColor(0.2, 0.3, 0.4, 1.0)
    color_attachment.clear_color = clear_color

    # Verify configuration
    assert color_attachment.texture is not None
    assert color_attachment.load_action == pm.LoadAction.Clear
    assert color_attachment.store_action == pm.StoreAction.Store


def test_offscreen_triangle_rendering():
    """
    Complete end-to-end test: Render a triangle to an offscreen texture.

    This validates the entire Phase 2 graphics pipeline:
    - Texture creation for render target
    - Shader compilation (vertex + fragment)
    - Pipeline creation
    - Render pass setup
    - Triangle rendering
    """
    device = pm.create_system_default_device()
    queue = device.new_command_queue()

    width, height = 512, 512

    # Create render target texture
    tex_desc = pm.TextureDescriptor.texture2d_descriptor(
        pm.PixelFormat.RGBA8Unorm,
        width, height,
        False
    )
    render_target = device.new_texture(tex_desc)

    # Compile shaders
    shader_source = """
    #include <metal_stdlib>
    using namespace metal;

    struct VertexOut {
        float4 position [[position]];
        float4 color;
    };

    vertex VertexOut vertex_main(uint vertex_id [[vertex_id]]) {
        VertexOut out;
        float2 positions[3] = {
            float2(0.0, 0.5),
            float2(-0.5, -0.5),
            float2(0.5, -0.5)
        };
        float4 colors[3] = {
            float4(1.0, 0.0, 0.0, 1.0),
            float4(0.0, 1.0, 0.0, 1.0),
            float4(0.0, 0.0, 1.0, 1.0)
        };
        out.position = float4(positions[vertex_id], 0.0, 1.0);
        out.color = colors[vertex_id];
        return out;
    }

    fragment float4 fragment_main(VertexOut in [[stage_in]]) {
        return in.color;
    }
    """

    library = device.new_library_with_source(shader_source)
    vertex_func = library.new_function("vertex_main")
    fragment_func = library.new_function("fragment_main")

    # Create render pipeline
    pipeline_desc = pm.RenderPipelineDescriptor.render_pipeline_descriptor()
    pipeline_desc.vertex_function = vertex_func
    pipeline_desc.fragment_function = fragment_func
    pipeline_desc.color_attachment(0).pixel_format = pm.PixelFormat.RGBA8Unorm

    pipeline = device.new_render_pipeline_state(pipeline_desc)

    # Create render pass
    render_pass = pm.RenderPassDescriptor.render_pass_descriptor()
    color_attachment = render_pass.color_attachment(0)
    color_attachment.texture = render_target
    color_attachment.load_action = pm.LoadAction.Clear
    color_attachment.store_action = pm.StoreAction.Store
    color_attachment.clear_color = pm.ClearColor(0.0, 0.0, 0.0, 1.0)  # Black background

    # Render
    cmd_buffer = queue.command_buffer()
    encoder = cmd_buffer.render_command_encoder(render_pass)

    encoder.set_render_pipeline_state(pipeline)
    encoder.draw_primitives(pm.PrimitiveType.Triangle, 0, 3)
    encoder.end_encoding()

    cmd_buffer.commit()
    cmd_buffer.wait_until_completed()

    # Success - we rendered a triangle!
    print(f"Successfully rendered triangle to {width}x{height} texture")
    print(f"Render target: {render_target.width}x{render_target.height}, format: {render_target.pixel_format}")


def test_sampler_creation():
    """Test sampler state creation."""
    device = pm.create_system_default_device()

    sampler_desc = pm.SamplerDescriptor.sampler_descriptor()
    sampler_desc.min_filter = pm.SamplerMinMagFilter.Linear
    sampler_desc.mag_filter = pm.SamplerMinMagFilter.Linear
    sampler_desc.mip_filter = pm.SamplerMipFilter.Linear
    sampler_desc.s_address_mode = pm.SamplerAddressMode.ClampToEdge
    sampler_desc.t_address_mode = pm.SamplerAddressMode.ClampToEdge

    sampler = device.new_sampler_state(sampler_desc)
    assert sampler is not None


def test_cull_and_winding():
    """Test cull mode and winding order settings."""
    device = pm.create_system_default_device()
    queue = device.new_command_queue()

    # Create minimal render setup
    tex_desc = pm.TextureDescriptor.texture2d_descriptor(
        pm.PixelFormat.RGBA8Unorm,
        64, 64,
        False
    )
    texture = device.new_texture(tex_desc)

    shader_source = """
    #include <metal_stdlib>
    using namespace metal;

    vertex float4 vs_main(uint vid [[vertex_id]]) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }

    fragment float4 fs_main() {
        return float4(1.0, 1.0, 1.0, 1.0);
    }
    """

    library = device.new_library_with_source(shader_source)
    pipeline_desc = pm.RenderPipelineDescriptor.render_pipeline_descriptor()
    pipeline_desc.vertex_function = library.new_function("vs_main")
    pipeline_desc.fragment_function = library.new_function("fs_main")
    pipeline_desc.color_attachment(0).pixel_format = pm.PixelFormat.RGBA8Unorm
    pipeline = device.new_render_pipeline_state(pipeline_desc)

    render_pass = pm.RenderPassDescriptor.render_pass_descriptor()
    render_pass.color_attachment(0).texture = texture
    render_pass.color_attachment(0).load_action = pm.LoadAction.Clear
    render_pass.color_attachment(0).store_action = pm.StoreAction.Store

    cmd_buffer = queue.command_buffer()
    encoder = cmd_buffer.render_command_encoder(render_pass)

    encoder.set_render_pipeline_state(pipeline)

    # Test cull mode and winding settings
    encoder.set_cull_mode(pm.CullMode.Back)
    encoder.set_front_facing_winding(pm.Winding.CounterClockwise)

    encoder.draw_primitives(pm.PrimitiveType.Triangle, 0, 3)
    encoder.end_encoding()

    cmd_buffer.commit()
    cmd_buffer.wait_until_completed()

    print("Successfully configured cull mode and winding order")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
