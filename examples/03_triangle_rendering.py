"""
Triangle Rendering Demo - Offscreen Graphics with Depth Testing

This demo shows:
1. Complete graphics pipeline setup
2. Vertex and fragment shaders
3. Render pass configuration
4. Depth buffer and depth testing
5. Multiple render targets
6. Saving rendered output to image file
"""

import numpy as np
import pymetal as pm


def save_image_as_ppm(filename, pixels, width, height):
    """Save RGBA pixels as PPM image (simple format, no dependencies)."""
    with open(filename, 'w') as f:
        f.write(f'P3\n{width} {height}\n255\n')
        for y in range(height):
            for x in range(width):
                idx = (y * width + x) * 4
                r = int(pixels[idx] * 255)
                g = int(pixels[idx + 1] * 255)
                b = int(pixels[idx + 2] * 255)
                f.write(f'{r} {g} {b} ')
            f.write('\n')


def render_triangle():
    """Render a colored triangle with depth testing to an offscreen texture."""
    device = pm.create_system_default_device()
    queue = device.new_command_queue()

    width, height = 512, 512

    print(f"Rendering {width}x{height} triangle on {device.name}")

    # Create color render target
    color_desc = pm.TextureDescriptor.texture2d_descriptor(
        pm.PixelFormat.RGBA8Unorm,
        width, height,
        False
    )
    color_texture = device.new_texture(color_desc)

    # Create depth render target
    depth_desc = pm.TextureDescriptor.texture2d_descriptor(
        pm.PixelFormat.Depth32Float,
        width, height,
        False
    )
    depth_texture = device.new_texture(depth_desc)

    # Vertex and fragment shaders
    shader_source = """
    #include <metal_stdlib>
    using namespace metal;

    struct VertexOut {
        float4 position [[position]];
        float4 color;
    };

    vertex VertexOut vertex_main(uint vertex_id [[vertex_id]]) {
        // Triangle vertices in normalized device coordinates
        float2 positions[3] = {
            float2( 0.0,  0.7),   // Top
            float2(-0.7, -0.7),   // Bottom-left
            float2( 0.7, -0.7)    // Bottom-right
        };

        // RGB colors for each vertex
        float4 colors[3] = {
            float4(1.0, 0.0, 0.0, 1.0),  // Red
            float4(0.0, 1.0, 0.0, 1.0),  // Green
            float4(0.0, 0.0, 1.0, 1.0)   // Blue
        };

        VertexOut out;
        out.position = float4(positions[vertex_id], 0.5, 1.0);  // depth = 0.5
        out.color = colors[vertex_id];
        return out;
    }

    fragment float4 fragment_main(VertexOut in [[stage_in]]) {
        return in.color;
    }
    """

    # Compile shaders
    print("Compiling shaders...")
    library = device.new_library_with_source(shader_source)
    vertex_func = library.new_function("vertex_main")
    fragment_func = library.new_function("fragment_main")

    # Create depth/stencil state
    depth_stencil_desc = pm.DepthStencilDescriptor.depth_stencil_descriptor()
    depth_stencil_desc.depth_compare_function = pm.CompareFunction.Less
    depth_stencil_desc.depth_write_enabled = True
    depth_stencil_state = device.new_depth_stencil_state(depth_stencil_desc)

    # Create render pipeline
    print("Creating render pipeline...")
    pipeline_desc = pm.RenderPipelineDescriptor.render_pipeline_descriptor()
    pipeline_desc.vertex_function = vertex_func
    pipeline_desc.fragment_function = fragment_func

    # Configure color attachment
    color_attachment = pipeline_desc.color_attachment(0)
    color_attachment.pixel_format = pm.PixelFormat.RGBA8Unorm

    # Configure depth attachment
    pipeline_desc.depth_attachment_pixel_format = pm.PixelFormat.Depth32Float

    pipeline = device.new_render_pipeline_state(pipeline_desc)

    # Create render pass
    render_pass = pm.RenderPassDescriptor.render_pass_descriptor()

    # Configure color attachment
    color_att = render_pass.color_attachment(0)
    color_att.texture = color_texture
    color_att.load_action = pm.LoadAction.Clear
    color_att.store_action = pm.StoreAction.Store
    color_att.clear_color = pm.ClearColor(0.1, 0.1, 0.15, 1.0)  # Dark blue background

    # Configure depth attachment
    depth_att = render_pass.depth_attachment
    depth_att.texture = depth_texture
    depth_att.load_action = pm.LoadAction.Clear
    depth_att.store_action = pm.StoreAction.Store
    depth_att.clear_depth = 1.0

    # Render!
    print("Rendering triangle...")
    cmd_buffer = queue.command_buffer()
    encoder = cmd_buffer.render_command_encoder(render_pass)

    encoder.set_render_pipeline_state(pipeline)
    encoder.set_depth_stencil_state(depth_stencil_state)
    encoder.draw_primitives(pm.PrimitiveType.Triangle, 0, 3)

    encoder.end_encoding()
    cmd_buffer.commit()
    cmd_buffer.wait_until_completed()

    print("Rendering complete!")

    return color_texture, width, height


def read_texture_to_buffer(texture, device, queue, width, height):
    """Read texture contents to CPU memory using blit encoder."""
    # Create staging buffer
    bytes_per_row = width * 4  # RGBA8
    buffer_size = bytes_per_row * height
    staging_buffer = device.new_buffer(buffer_size, pm.ResourceStorageModeShared)

    # Blit texture to buffer
    cmd_buffer = queue.command_buffer()
    blit_encoder = cmd_buffer.blit_command_encoder()

    origin = pm.Origin(0, 0, 0)
    size = pm.Size(width, height, 1)

    blit_encoder.copy_from_texture_to_buffer(
        texture, 0, 0,  # slice, level
        origin, size,
        staging_buffer, 0,  # buffer, offset
        bytes_per_row,
        bytes_per_row * height
    )

    blit_encoder.end_encoding()
    cmd_buffer.commit()
    cmd_buffer.wait_until_completed()

    # Read buffer contents
    pixels = np.frombuffer(staging_buffer.contents(), dtype=np.uint8, count=buffer_size)
    return pixels / 255.0  # Normalize to [0, 1]


def main():
    print("=" * 60)
    print("PyMetal Triangle Rendering Demo")
    print("=" * 60)
    print()

    # Render triangle
    color_texture, width, height = render_triangle()

    # Read back pixel data
    print("Reading back rendered image...")
    device = pm.create_system_default_device()
    queue = device.new_command_queue()
    pixels = read_texture_to_buffer(color_texture, device, queue, width, height)

    # Save to file
    output_file = "/tmp/pymetal_triangle.ppm"
    print(f"Saving image to {output_file}...")
    save_image_as_ppm(output_file, pixels, width, height)

    print(f"\n✓ Image saved to: {output_file}")
    print("  You can view it with: open /tmp/pymetal_triangle.ppm")

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)

    print("\nFeatures Demonstrated:")
    print("  -Graphics pipeline setup")
    print("  -Vertex and fragment shaders")
    print("  -Render target creation (color + depth)")
    print("  -Depth testing configuration")
    print("  -Render pass with multiple attachments")
    print("  -Triangle rasterization")
    print("  -Color interpolation across triangle")
    print("  -Blit encoder for texture-to-buffer copy")
    print("  -Offscreen rendering")
    print("  -Reading GPU results back to CPU")


if __name__ == "__main__":
    main()
