#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/function.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
using namespace nb::literals;

// ============================================================================
// Helper: NS::String <-> Python str conversions
// ============================================================================

std::string ns_string_to_std(NS::String* nsstr) {
    if (!nsstr) return "";
    return std::string(nsstr->utf8String());
}

NS::String* std_string_to_ns(const std::string& str) {
    return NS::String::string(str.c_str(), NS::UTF8StringEncoding);
}

// ============================================================================
// Phase 1: Core Enumerations
// ============================================================================

void wrap_enums(nb::module_& m) {
    // StorageMode
    nb::enum_<MTL::StorageMode>(m, "StorageMode")
        .value("Shared", MTL::StorageModeShared)
        .value("Managed", MTL::StorageModeManaged)
        .value("Private", MTL::StorageModePrivate)
        .value("Memoryless", MTL::StorageModeMemoryless)
        .export_values();

    // CPUCacheMode
    nb::enum_<MTL::CPUCacheMode>(m, "CPUCacheMode")
        .value("DefaultCache", MTL::CPUCacheModeDefaultCache)
        .value("WriteCombined", MTL::CPUCacheModeWriteCombined)
        .export_values();

    // ResourceOptions (bitmask - not an enum, just constants)
    // These are NS::UInteger constants that can be OR'd together
    m.attr("ResourceCPUCacheModeDefaultCache") = static_cast<uint64_t>(MTL::ResourceCPUCacheModeDefaultCache);
    m.attr("ResourceCPUCacheModeWriteCombined") = static_cast<uint64_t>(MTL::ResourceCPUCacheModeWriteCombined);
    m.attr("ResourceStorageModeShared") = static_cast<uint64_t>(MTL::ResourceStorageModeShared);
    m.attr("ResourceStorageModeManaged") = static_cast<uint64_t>(MTL::ResourceStorageModeManaged);
    m.attr("ResourceStorageModePrivate") = static_cast<uint64_t>(MTL::ResourceStorageModePrivate);
    m.attr("ResourceStorageModeMemoryless") = static_cast<uint64_t>(MTL::ResourceStorageModeMemoryless);
    m.attr("ResourceHazardTrackingModeUntracked") = static_cast<uint64_t>(MTL::ResourceHazardTrackingModeUntracked);

    // LoadAction
    nb::enum_<MTL::LoadAction>(m, "LoadAction")
        .value("DontCare", MTL::LoadActionDontCare)
        .value("Load", MTL::LoadActionLoad)
        .value("Clear", MTL::LoadActionClear)
        .export_values();

    // StoreAction
    nb::enum_<MTL::StoreAction>(m, "StoreAction")
        .value("DontCare", MTL::StoreActionDontCare)
        .value("Store", MTL::StoreActionStore)
        .value("MultisampleResolve", MTL::StoreActionMultisampleResolve)
        .value("StoreAndMultisampleResolve", MTL::StoreActionStoreAndMultisampleResolve)
        .value("Unknown", MTL::StoreActionUnknown)
        .value("CustomSampleDepthStore", MTL::StoreActionCustomSampleDepthStore)
        .export_values();
}

// ============================================================================
// Phase 1: MTL::Device
// ============================================================================

void wrap_device(nb::module_& m) {
    nb::class_<MTL::Device>(m, "Device")
        .def("new_command_queue",
            [](MTL::Device* self) {
                return self->newCommandQueue();
            },
            nb::rv_policy::reference,
            "Create a new command queue")

        .def("new_buffer",
            [](MTL::Device* self, size_t length, NS::UInteger options) {
                return self->newBuffer(length, options);
            },
            "length"_a, "options"_a,
            nb::rv_policy::reference,
            "Create a new buffer with specified length and options")

        .def("new_buffer_with_data",
            [](MTL::Device* self, const void* pointer, size_t length, NS::UInteger options) {
                return self->newBuffer(pointer, length, options);
            },
            "data"_a, "length"_a, "options"_a,
            nb::rv_policy::reference,
            "Create a new buffer initialized with data")

        .def("new_library_with_source",
            [](MTL::Device* self, const std::string& source) {
                NS::Error* error = nullptr;
                NS::String* src = std_string_to_ns(source);
                MTL::CompileOptions* options = nullptr;

                MTL::Library* lib = self->newLibrary(src, options, &error);

                if (error) {
                    std::string error_msg = ns_string_to_std(error->localizedDescription());
                    throw std::runtime_error("Metal shader compilation failed: " + error_msg);
                }

                return lib;
            },
            "source"_a,
            nb::rv_policy::reference,
            "Compile a new library from Metal shader source code")

        .def("new_compute_pipeline_state",
            [](MTL::Device* self, MTL::Function* function) {
                NS::Error* error = nullptr;
                MTL::ComputePipelineState* state = self->newComputePipelineState(function, &error);

                if (error) {
                    std::string error_msg = ns_string_to_std(error->localizedDescription());
                    throw std::runtime_error("Failed to create compute pipeline: " + error_msg);
                }

                return state;
            },
            "function"_a,
            nb::rv_policy::reference,
            "Create a compute pipeline state from a kernel function")

        .def_prop_ro("name",
            [](MTL::Device* self) {
                return ns_string_to_std(self->name());
            },
            "Device name")

        .def_prop_ro("max_threads_per_threadgroup",
            [](MTL::Device* self) {
                return self->maxThreadsPerThreadgroup();
            },
            "Maximum threads per threadgroup")

        // Phase 2: Texture methods
        .def("new_texture",
            [](MTL::Device* self, MTL::TextureDescriptor* descriptor) {
                return self->newTexture(descriptor);
            },
            "descriptor"_a,
            nb::rv_policy::reference,
            "Create a new texture")

        .def("new_sampler_state",
            [](MTL::Device* self, MTL::SamplerDescriptor* descriptor) {
                return self->newSamplerState(descriptor);
            },
            "descriptor"_a,
            nb::rv_policy::reference,
            "Create a new sampler state")

        .def("new_render_pipeline_state",
            [](MTL::Device* self, MTL::RenderPipelineDescriptor* descriptor) {
                NS::Error* error = nullptr;
                MTL::RenderPipelineState* state = self->newRenderPipelineState(descriptor, &error);

                if (error) {
                    std::string error_msg = ns_string_to_std(error->localizedDescription());
                    throw std::runtime_error("Failed to create render pipeline: " + error_msg);
                }

                return state;
            },
            "descriptor"_a,
            nb::rv_policy::reference,
            "Create a render pipeline state from a descriptor");

    // Global function to create default device
    m.def("create_system_default_device",
        []() {
            return MTL::CreateSystemDefaultDevice();
        },
        nb::rv_policy::reference,
        "Create the default Metal device");
}

// ============================================================================
// Phase 1: MTL::CommandQueue
// ============================================================================

void wrap_command_queue(nb::module_& m) {
    nb::class_<MTL::CommandQueue>(m, "CommandQueue")
        .def("command_buffer",
            [](MTL::CommandQueue* self) {
                return self->commandBuffer();
            },
            nb::rv_policy::reference,
            "Create a new command buffer")

        .def_prop_ro("device",
            [](MTL::CommandQueue* self) {
                return self->device();
            },
            nb::rv_policy::reference,
            "The device this queue was created from")

        .def_prop_rw("label",
            [](MTL::CommandQueue* self) {
                return ns_string_to_std(self->label());
            },
            [](MTL::CommandQueue* self, const std::string& label) {
                self->setLabel(std_string_to_ns(label));
            },
            "Debug label for this queue");
}

// ============================================================================
// Phase 1: MTL::CommandBuffer
// ============================================================================

void wrap_command_buffer(nb::module_& m) {
    nb::class_<MTL::CommandBuffer>(m, "CommandBuffer")
        .def("compute_command_encoder",
            [](MTL::CommandBuffer* self) {
                return self->computeCommandEncoder();
            },
            nb::rv_policy::reference,
            "Create a compute command encoder")

        // Phase 2: Render command encoder
        .def("render_command_encoder",
            [](MTL::CommandBuffer* self, MTL::RenderPassDescriptor* descriptor) {
                return self->renderCommandEncoder(descriptor);
            },
            "descriptor"_a,
            nb::rv_policy::reference,
            "Create a render command encoder")

        .def("commit",
            [](MTL::CommandBuffer* self) {
                self->commit();
            },
            "Submit the command buffer for execution")

        .def("wait_until_completed",
            [](MTL::CommandBuffer* self) {
                nb::gil_scoped_release release;
                self->waitUntilCompleted();
            },
            "Wait until the command buffer has completed execution")

        .def("wait_until_scheduled",
            [](MTL::CommandBuffer* self) {
                nb::gil_scoped_release release;
                self->waitUntilScheduled();
            },
            "Wait until the command buffer has been scheduled")

        .def_prop_ro("status",
            [](MTL::CommandBuffer* self) {
                return self->status();
            },
            "Current status of the command buffer")

        .def_prop_rw("label",
            [](MTL::CommandBuffer* self) {
                return ns_string_to_std(self->label());
            },
            [](MTL::CommandBuffer* self, const std::string& label) {
                self->setLabel(std_string_to_ns(label));
            },
            "Debug label for this command buffer");

    // CommandBufferStatus enum
    nb::enum_<MTL::CommandBufferStatus>(m, "CommandBufferStatus")
        .value("NotEnqueued", MTL::CommandBufferStatusNotEnqueued)
        .value("Enqueued", MTL::CommandBufferStatusEnqueued)
        .value("Committed", MTL::CommandBufferStatusCommitted)
        .value("Scheduled", MTL::CommandBufferStatusScheduled)
        .value("Completed", MTL::CommandBufferStatusCompleted)
        .value("Error", MTL::CommandBufferStatusError)
        .export_values();
}

// ============================================================================
// Phase 1: MTL::Buffer
// ============================================================================

void wrap_buffer(nb::module_& m) {
    nb::class_<MTL::Buffer>(m, "Buffer", nb::is_weak_referenceable())
        .def("contents",
            [](MTL::Buffer* self) {
                void* ptr = self->contents();
                size_t size = self->length();

                // Return as numpy-compatible buffer
                return nb::ndarray<nb::numpy, uint8_t>(
                    static_cast<uint8_t*>(ptr),
                    {size},
                    nb::handle()
                );
            },
            nb::rv_policy::reference_internal,
            "Get CPU-accessible pointer to buffer contents as numpy array")

        .def("did_modify_range",
            [](MTL::Buffer* self, size_t offset, size_t length) {
                NS::Range range(offset, length);
                self->didModifyRange(range);
            },
            "offset"_a, "length"_a,
            "Notify Metal that the CPU modified a range of the buffer")

        .def_prop_ro("length",
            [](MTL::Buffer* self) {
                return self->length();
            },
            "Buffer size in bytes")

        .def_prop_ro("gpu_address",
            [](MTL::Buffer* self) {
                return self->gpuAddress();
            },
            "GPU virtual address of the buffer")

        .def_prop_rw("label",
            [](MTL::Buffer* self) {
                return ns_string_to_std(self->label());
            },
            [](MTL::Buffer* self, const std::string& label) {
                self->setLabel(std_string_to_ns(label));
            },
            "Debug label for this buffer");
}

// ============================================================================
// Phase 1: MTL::Library and MTL::Function
// ============================================================================

void wrap_library(nb::module_& m) {
    nb::class_<MTL::Library>(m, "Library")
        .def("new_function",
            [](MTL::Library* self, const std::string& name) {
                NS::String* func_name = std_string_to_ns(name);
                return self->newFunction(func_name);
            },
            "name"_a,
            nb::rv_policy::reference,
            "Get a function by name from the library")

        .def_prop_rw("label",
            [](MTL::Library* self) {
                return ns_string_to_std(self->label());
            },
            [](MTL::Library* self, const std::string& label) {
                self->setLabel(std_string_to_ns(label));
            },
            "Debug label for this library");

    nb::class_<MTL::Function>(m, "Function")
        .def_prop_ro("name",
            [](MTL::Function* self) {
                return ns_string_to_std(self->name());
            },
            "Function name")

        .def_prop_ro("function_type",
            [](MTL::Function* self) {
                return self->functionType();
            },
            "Function type (vertex, fragment, kernel)");

    nb::enum_<MTL::FunctionType>(m, "FunctionType")
        .value("Vertex", MTL::FunctionTypeVertex)
        .value("Fragment", MTL::FunctionTypeFragment)
        .value("Kernel", MTL::FunctionTypeKernel)
        .export_values();
}

// ============================================================================
// Phase 1: MTL::ComputePipelineState
// ============================================================================

void wrap_compute_pipeline(nb::module_& m) {
    nb::class_<MTL::ComputePipelineState>(m, "ComputePipelineState")
        .def_prop_ro("max_total_threads_per_threadgroup",
            [](MTL::ComputePipelineState* self) {
                return self->maxTotalThreadsPerThreadgroup();
            },
            "Maximum number of threads per threadgroup")

        .def_prop_ro("thread_execution_width",
            [](MTL::ComputePipelineState* self) {
                return self->threadExecutionWidth();
            },
            "Thread execution width (SIMD width)");
}

// ============================================================================
// Phase 1: MTL::ComputeCommandEncoder
// ============================================================================

void wrap_compute_encoder(nb::module_& m) {
    nb::class_<MTL::ComputeCommandEncoder>(m, "ComputeCommandEncoder")
        .def("set_compute_pipeline_state",
            [](MTL::ComputeCommandEncoder* self, MTL::ComputePipelineState* state) {
                self->setComputePipelineState(state);
            },
            "state"_a,
            "Set the active compute pipeline state")

        .def("set_buffer",
            [](MTL::ComputeCommandEncoder* self, MTL::Buffer* buffer, size_t offset, uint32_t index) {
                self->setBuffer(buffer, offset, index);
            },
            "buffer"_a, "offset"_a, "index"_a,
            "Bind a buffer at the specified index")

        .def("set_bytes",
            [](MTL::ComputeCommandEncoder* self, nb::bytes data, uint32_t index) {
                self->setBytes(data.c_str(), data.size(), index);
            },
            "data"_a, "index"_a,
            "Set small inline data at the specified index")

        .def("dispatch_threadgroups",
            [](MTL::ComputeCommandEncoder* self,
               uint32_t threadgroups_x, uint32_t threadgroups_y, uint32_t threadgroups_z,
               uint32_t threads_x, uint32_t threads_y, uint32_t threads_z) {
                MTL::Size threadgroups(threadgroups_x, threadgroups_y, threadgroups_z);
                MTL::Size threads_per_group(threads_x, threads_y, threads_z);
                self->dispatchThreadgroups(threadgroups, threads_per_group);
            },
            "threadgroups_x"_a, "threadgroups_y"_a, "threadgroups_z"_a,
            "threads_x"_a, "threads_y"_a, "threads_z"_a,
            "Dispatch compute work with specified threadgroup configuration")

        .def("dispatch_threads",
            [](MTL::ComputeCommandEncoder* self,
               uint32_t threads_x, uint32_t threads_y, uint32_t threads_z,
               uint32_t threads_per_group_x, uint32_t threads_per_group_y, uint32_t threads_per_group_z) {
                MTL::Size threads(threads_x, threads_y, threads_z);
                MTL::Size threads_per_group(threads_per_group_x, threads_per_group_y, threads_per_group_z);
                self->dispatchThreads(threads, threads_per_group);
            },
            "threads_x"_a, "threads_y"_a, "threads_z"_a,
            "threads_per_group_x"_a, "threads_per_group_y"_a, "threads_per_group_z"_a,
            "Dispatch compute work with specified total thread count")

        .def("end_encoding",
            [](MTL::ComputeCommandEncoder* self) {
                self->endEncoding();
            },
            "Finish encoding commands");
}

// ============================================================================
// Phase 2: Graphics Enumerations
// ============================================================================

void wrap_graphics_enums(nb::module_& m) {
    // PixelFormat (subset of common formats)
    nb::enum_<MTL::PixelFormat>(m, "PixelFormat")
        .value("Invalid", MTL::PixelFormatInvalid)
        .value("RGBA8Unorm", MTL::PixelFormatRGBA8Unorm)
        .value("RGBA8Unorm_sRGB", MTL::PixelFormatRGBA8Unorm_sRGB)
        .value("BGRA8Unorm", MTL::PixelFormatBGRA8Unorm)
        .value("BGRA8Unorm_sRGB", MTL::PixelFormatBGRA8Unorm_sRGB)
        .value("Depth32Float", MTL::PixelFormatDepth32Float)
        .value("Stencil8", MTL::PixelFormatStencil8)
        .value("Depth24Unorm_Stencil8", MTL::PixelFormatDepth24Unorm_Stencil8)
        .value("Depth32Float_Stencil8", MTL::PixelFormatDepth32Float_Stencil8)
        .value("R32Float", MTL::PixelFormatR32Float)
        .value("RG32Float", MTL::PixelFormatRG32Float)
        .value("RGBA32Float", MTL::PixelFormatRGBA32Float)
        .value("R16Float", MTL::PixelFormatR16Float)
        .value("RG16Float", MTL::PixelFormatRG16Float)
        .value("RGBA16Float", MTL::PixelFormatRGBA16Float)
        .export_values();

    // PrimitiveType
    nb::enum_<MTL::PrimitiveType>(m, "PrimitiveType")
        .value("Point", MTL::PrimitiveTypePoint)
        .value("Line", MTL::PrimitiveTypeLine)
        .value("LineStrip", MTL::PrimitiveTypeLineStrip)
        .value("Triangle", MTL::PrimitiveTypeTriangle)
        .value("TriangleStrip", MTL::PrimitiveTypeTriangleStrip)
        .export_values();

    // IndexType
    nb::enum_<MTL::IndexType>(m, "IndexType")
        .value("UInt16", MTL::IndexTypeUInt16)
        .value("UInt32", MTL::IndexTypeUInt32)
        .export_values();

    // VertexFormat
    nb::enum_<MTL::VertexFormat>(m, "VertexFormat")
        .value("Float", MTL::VertexFormatFloat)
        .value("Float2", MTL::VertexFormatFloat2)
        .value("Float3", MTL::VertexFormatFloat3)
        .value("Float4", MTL::VertexFormatFloat4)
        .value("Int", MTL::VertexFormatInt)
        .value("Int2", MTL::VertexFormatInt2)
        .value("Int3", MTL::VertexFormatInt3)
        .value("Int4", MTL::VertexFormatInt4)
        .value("UInt", MTL::VertexFormatUInt)
        .value("UInt2", MTL::VertexFormatUInt2)
        .value("UInt3", MTL::VertexFormatUInt3)
        .value("UInt4", MTL::VertexFormatUInt4)
        .export_values();

    // VertexStepFunction
    nb::enum_<MTL::VertexStepFunction>(m, "VertexStepFunction")
        .value("PerVertex", MTL::VertexStepFunctionPerVertex)
        .value("PerInstance", MTL::VertexStepFunctionPerInstance)
        .export_values();

    // CullMode
    nb::enum_<MTL::CullMode>(m, "CullMode")
        .value("None", MTL::CullModeNone)
        .value("Front", MTL::CullModeFront)
        .value("Back", MTL::CullModeBack)
        .export_values();

    // Winding
    nb::enum_<MTL::Winding>(m, "Winding")
        .value("Clockwise", MTL::WindingClockwise)
        .value("CounterClockwise", MTL::WindingCounterClockwise)
        .export_values();

    // TextureType
    nb::enum_<MTL::TextureType>(m, "TextureType")
        .value("Type1D", MTL::TextureType1D)
        .value("Type2D", MTL::TextureType2D)
        .value("Type3D", MTL::TextureType3D)
        .value("TypeCube", MTL::TextureTypeCube)
        .value("Type2DArray", MTL::TextureType2DArray)
        .export_values();

    // SamplerMinMagFilter
    nb::enum_<MTL::SamplerMinMagFilter>(m, "SamplerMinMagFilter")
        .value("Nearest", MTL::SamplerMinMagFilterNearest)
        .value("Linear", MTL::SamplerMinMagFilterLinear)
        .export_values();

    // SamplerMipFilter
    nb::enum_<MTL::SamplerMipFilter>(m, "SamplerMipFilter")
        .value("NotMipmapped", MTL::SamplerMipFilterNotMipmapped)
        .value("Nearest", MTL::SamplerMipFilterNearest)
        .value("Linear", MTL::SamplerMipFilterLinear)
        .export_values();

    // SamplerAddressMode
    nb::enum_<MTL::SamplerAddressMode>(m, "SamplerAddressMode")
        .value("ClampToEdge", MTL::SamplerAddressModeClampToEdge)
        .value("MirrorClampToEdge", MTL::SamplerAddressModeMirrorClampToEdge)
        .value("Repeat", MTL::SamplerAddressModeRepeat)
        .value("MirrorRepeat", MTL::SamplerAddressModeMirrorRepeat)
        .value("ClampToZero", MTL::SamplerAddressModeClampToZero)
        .export_values();

    // CompareFunction
    nb::enum_<MTL::CompareFunction>(m, "CompareFunction")
        .value("Never", MTL::CompareFunctionNever)
        .value("Less", MTL::CompareFunctionLess)
        .value("Equal", MTL::CompareFunctionEqual)
        .value("LessEqual", MTL::CompareFunctionLessEqual)
        .value("Greater", MTL::CompareFunctionGreater)
        .value("NotEqual", MTL::CompareFunctionNotEqual)
        .value("GreaterEqual", MTL::CompareFunctionGreaterEqual)
        .value("Always", MTL::CompareFunctionAlways)
        .export_values();

    // BlendFactor
    nb::enum_<MTL::BlendFactor>(m, "BlendFactor")
        .value("Zero", MTL::BlendFactorZero)
        .value("One", MTL::BlendFactorOne)
        .value("SourceColor", MTL::BlendFactorSourceColor)
        .value("OneMinusSourceColor", MTL::BlendFactorOneMinusSourceColor)
        .value("SourceAlpha", MTL::BlendFactorSourceAlpha)
        .value("OneMinusSourceAlpha", MTL::BlendFactorOneMinusSourceAlpha)
        .value("DestinationColor", MTL::BlendFactorDestinationColor)
        .value("OneMinusDestinationColor", MTL::BlendFactorOneMinusDestinationColor)
        .value("DestinationAlpha", MTL::BlendFactorDestinationAlpha)
        .value("OneMinusDestinationAlpha", MTL::BlendFactorOneMinusDestinationAlpha)
        .export_values();

    // BlendOperation
    nb::enum_<MTL::BlendOperation>(m, "BlendOperation")
        .value("Add", MTL::BlendOperationAdd)
        .value("Subtract", MTL::BlendOperationSubtract)
        .value("ReverseSubtract", MTL::BlendOperationReverseSubtract)
        .value("Min", MTL::BlendOperationMin)
        .value("Max", MTL::BlendOperationMax)
        .export_values();

    // ColorWriteMask
    m.attr("ColorWriteMaskNone") = static_cast<uint64_t>(MTL::ColorWriteMaskNone);
    m.attr("ColorWriteMaskRed") = static_cast<uint64_t>(MTL::ColorWriteMaskRed);
    m.attr("ColorWriteMaskGreen") = static_cast<uint64_t>(MTL::ColorWriteMaskGreen);
    m.attr("ColorWriteMaskBlue") = static_cast<uint64_t>(MTL::ColorWriteMaskBlue);
    m.attr("ColorWriteMaskAlpha") = static_cast<uint64_t>(MTL::ColorWriteMaskAlpha);
    m.attr("ColorWriteMaskAll") = static_cast<uint64_t>(MTL::ColorWriteMaskAll);
}

// ============================================================================
// Phase 2: MTL::Texture and MTL::TextureDescriptor
// ============================================================================

void wrap_texture(nb::module_& m) {
    nb::class_<MTL::TextureDescriptor>(m, "TextureDescriptor")
        .def_static("texture2d_descriptor",
            [](MTL::PixelFormat format, uint32_t width, uint32_t height, bool mipmapped) {
                return MTL::TextureDescriptor::texture2DDescriptor(format, width, height, mipmapped);
            },
            "format"_a, "width"_a, "height"_a, "mipmapped"_a,
            nb::rv_policy::reference,
            "Create a 2D texture descriptor")

        .def_prop_rw("texture_type",
            [](MTL::TextureDescriptor* self) { return self->textureType(); },
            [](MTL::TextureDescriptor* self, MTL::TextureType type) { self->setTextureType(type); })

        .def_prop_rw("pixel_format",
            [](MTL::TextureDescriptor* self) { return self->pixelFormat(); },
            [](MTL::TextureDescriptor* self, MTL::PixelFormat format) { self->setPixelFormat(format); })

        .def_prop_rw("width",
            [](MTL::TextureDescriptor* self) { return self->width(); },
            [](MTL::TextureDescriptor* self, uint32_t w) { self->setWidth(w); })

        .def_prop_rw("height",
            [](MTL::TextureDescriptor* self) { return self->height(); },
            [](MTL::TextureDescriptor* self, uint32_t h) { self->setHeight(h); })

        .def_prop_rw("depth",
            [](MTL::TextureDescriptor* self) { return self->depth(); },
            [](MTL::TextureDescriptor* self, uint32_t d) { self->setDepth(d); })

        .def_prop_rw("mipmap_level_count",
            [](MTL::TextureDescriptor* self) { return self->mipmapLevelCount(); },
            [](MTL::TextureDescriptor* self, uint32_t count) { self->setMipmapLevelCount(count); })

        .def_prop_rw("sample_count",
            [](MTL::TextureDescriptor* self) { return self->sampleCount(); },
            [](MTL::TextureDescriptor* self, uint32_t count) { self->setSampleCount(count); })

        .def_prop_rw("storage_mode",
            [](MTL::TextureDescriptor* self) { return self->storageMode(); },
            [](MTL::TextureDescriptor* self, MTL::StorageMode mode) { self->setStorageMode(mode); });

    nb::class_<MTL::Texture>(m, "Texture")
        .def_prop_ro("texture_type",
            [](MTL::Texture* self) { return self->textureType(); })

        .def_prop_ro("pixel_format",
            [](MTL::Texture* self) { return self->pixelFormat(); })

        .def_prop_ro("width",
            [](MTL::Texture* self) { return self->width(); })

        .def_prop_ro("height",
            [](MTL::Texture* self) { return self->height(); })

        .def_prop_ro("depth",
            [](MTL::Texture* self) { return self->depth(); })

        .def_prop_ro("mipmap_level_count",
            [](MTL::Texture* self) { return self->mipmapLevelCount(); })

        .def_prop_ro("sample_count",
            [](MTL::Texture* self) { return self->sampleCount(); })

        .def_prop_ro("array_length",
            [](MTL::Texture* self) { return self->arrayLength(); })

        .def("replace_region",
            [](MTL::Texture* self, nb::bytes data, uint32_t bytes_per_row,
               uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint32_t level) {
                MTL::Region region(x, y, width, height);
                self->replaceRegion(region, level, data.c_str(), bytes_per_row);
            },
            "data"_a, "bytes_per_row"_a, "x"_a, "y"_a, "width"_a, "height"_a, "level"_a = 0,
            "Upload data to a region of the texture")

        .def_prop_rw("label",
            [](MTL::Texture* self) { return ns_string_to_std(self->label()); },
            [](MTL::Texture* self, const std::string& label) { self->setLabel(std_string_to_ns(label)); });
}

// ============================================================================
// Phase 2: MTL::SamplerState and MTL::SamplerDescriptor
// ============================================================================

void wrap_sampler(nb::module_& m) {
    nb::class_<MTL::SamplerDescriptor>(m, "SamplerDescriptor")
        .def_static("sampler_descriptor",
            []() { return MTL::SamplerDescriptor::alloc()->init(); },
            nb::rv_policy::reference,
            "Create a new sampler descriptor")

        .def_prop_rw("min_filter",
            [](MTL::SamplerDescriptor* self) { return self->minFilter(); },
            [](MTL::SamplerDescriptor* self, MTL::SamplerMinMagFilter filter) { self->setMinFilter(filter); })

        .def_prop_rw("mag_filter",
            [](MTL::SamplerDescriptor* self) { return self->magFilter(); },
            [](MTL::SamplerDescriptor* self, MTL::SamplerMinMagFilter filter) { self->setMagFilter(filter); })

        .def_prop_rw("mip_filter",
            [](MTL::SamplerDescriptor* self) { return self->mipFilter(); },
            [](MTL::SamplerDescriptor* self, MTL::SamplerMipFilter filter) { self->setMipFilter(filter); })

        .def_prop_rw("s_address_mode",
            [](MTL::SamplerDescriptor* self) { return self->sAddressMode(); },
            [](MTL::SamplerDescriptor* self, MTL::SamplerAddressMode mode) { self->setSAddressMode(mode); })

        .def_prop_rw("t_address_mode",
            [](MTL::SamplerDescriptor* self) { return self->tAddressMode(); },
            [](MTL::SamplerDescriptor* self, MTL::SamplerAddressMode mode) { self->setTAddressMode(mode); })

        .def_prop_rw("r_address_mode",
            [](MTL::SamplerDescriptor* self) { return self->rAddressMode(); },
            [](MTL::SamplerDescriptor* self, MTL::SamplerAddressMode mode) { self->setRAddressMode(mode); })

        .def_prop_rw("max_anisotropy",
            [](MTL::SamplerDescriptor* self) { return self->maxAnisotropy(); },
            [](MTL::SamplerDescriptor* self, uint32_t max) { self->setMaxAnisotropy(max); });

    nb::class_<MTL::SamplerState>(m, "SamplerState")
        .def_prop_ro("label",
            [](MTL::SamplerState* self) { return ns_string_to_std(self->label()); });
}

// ============================================================================
// Phase 2: MTL::RenderPipelineDescriptor and MTL::RenderPipelineState
// ============================================================================

void wrap_render_pipeline(nb::module_& m) {
    // RenderPipelineColorAttachmentDescriptor
    nb::class_<MTL::RenderPipelineColorAttachmentDescriptor>(m, "RenderPipelineColorAttachmentDescriptor")
        .def_prop_rw("pixel_format",
            [](MTL::RenderPipelineColorAttachmentDescriptor* self) { return self->pixelFormat(); },
            [](MTL::RenderPipelineColorAttachmentDescriptor* self, MTL::PixelFormat format) {
                self->setPixelFormat(format);
            })

        .def_prop_rw("blending_enabled",
            [](MTL::RenderPipelineColorAttachmentDescriptor* self) { return self->blendingEnabled(); },
            [](MTL::RenderPipelineColorAttachmentDescriptor* self, bool enabled) {
                self->setBlendingEnabled(enabled);
            })

        .def_prop_rw("source_rgb_blend_factor",
            [](MTL::RenderPipelineColorAttachmentDescriptor* self) { return self->sourceRGBBlendFactor(); },
            [](MTL::RenderPipelineColorAttachmentDescriptor* self, MTL::BlendFactor factor) {
                self->setSourceRGBBlendFactor(factor);
            })

        .def_prop_rw("destination_rgb_blend_factor",
            [](MTL::RenderPipelineColorAttachmentDescriptor* self) { return self->destinationRGBBlendFactor(); },
            [](MTL::RenderPipelineColorAttachmentDescriptor* self, MTL::BlendFactor factor) {
                self->setDestinationRGBBlendFactor(factor);
            })

        .def_prop_rw("rgb_blend_operation",
            [](MTL::RenderPipelineColorAttachmentDescriptor* self) { return self->rgbBlendOperation(); },
            [](MTL::RenderPipelineColorAttachmentDescriptor* self, MTL::BlendOperation op) {
                self->setRgbBlendOperation(op);
            })

        .def_prop_rw("source_alpha_blend_factor",
            [](MTL::RenderPipelineColorAttachmentDescriptor* self) { return self->sourceAlphaBlendFactor(); },
            [](MTL::RenderPipelineColorAttachmentDescriptor* self, MTL::BlendFactor factor) {
                self->setSourceAlphaBlendFactor(factor);
            })

        .def_prop_rw("destination_alpha_blend_factor",
            [](MTL::RenderPipelineColorAttachmentDescriptor* self) { return self->destinationAlphaBlendFactor(); },
            [](MTL::RenderPipelineColorAttachmentDescriptor* self, MTL::BlendFactor factor) {
                self->setDestinationAlphaBlendFactor(factor);
            })

        .def_prop_rw("alpha_blend_operation",
            [](MTL::RenderPipelineColorAttachmentDescriptor* self) { return self->alphaBlendOperation(); },
            [](MTL::RenderPipelineColorAttachmentDescriptor* self, MTL::BlendOperation op) {
                self->setAlphaBlendOperation(op);
            })

        .def_prop_rw("write_mask",
            [](MTL::RenderPipelineColorAttachmentDescriptor* self) { return self->writeMask(); },
            [](MTL::RenderPipelineColorAttachmentDescriptor* self, MTL::ColorWriteMask mask) {
                self->setWriteMask(mask);
            });

    // RenderPipelineDescriptor
    nb::class_<MTL::RenderPipelineDescriptor>(m, "RenderPipelineDescriptor")
        .def_static("render_pipeline_descriptor",
            []() { return MTL::RenderPipelineDescriptor::alloc()->init(); },
            nb::rv_policy::reference,
            "Create a new render pipeline descriptor")

        .def_prop_rw("vertex_function",
            [](MTL::RenderPipelineDescriptor* self) { return self->vertexFunction(); },
            [](MTL::RenderPipelineDescriptor* self, MTL::Function* func) {
                self->setVertexFunction(func);
            },
            nb::rv_policy::reference)

        .def_prop_rw("fragment_function",
            [](MTL::RenderPipelineDescriptor* self) { return self->fragmentFunction(); },
            [](MTL::RenderPipelineDescriptor* self, MTL::Function* func) {
                self->setFragmentFunction(func);
            },
            nb::rv_policy::reference)

        .def("color_attachment",
            [](MTL::RenderPipelineDescriptor* self, uint32_t index) {
                return self->colorAttachments()->object(index);
            },
            "index"_a,
            nb::rv_policy::reference,
            "Get color attachment descriptor at index")

        .def_prop_rw("depth_attachment_pixel_format",
            [](MTL::RenderPipelineDescriptor* self) { return self->depthAttachmentPixelFormat(); },
            [](MTL::RenderPipelineDescriptor* self, MTL::PixelFormat format) {
                self->setDepthAttachmentPixelFormat(format);
            })

        .def_prop_rw("stencil_attachment_pixel_format",
            [](MTL::RenderPipelineDescriptor* self) { return self->stencilAttachmentPixelFormat(); },
            [](MTL::RenderPipelineDescriptor* self, MTL::PixelFormat format) {
                self->setStencilAttachmentPixelFormat(format);
            })

        .def_prop_rw("label",
            [](MTL::RenderPipelineDescriptor* self) { return ns_string_to_std(self->label()); },
            [](MTL::RenderPipelineDescriptor* self, const std::string& label) {
                self->setLabel(std_string_to_ns(label));
            })

        .def_prop_rw("vertex_descriptor",
            [](MTL::RenderPipelineDescriptor* self) { return self->vertexDescriptor(); },
            [](MTL::RenderPipelineDescriptor* self, MTL::VertexDescriptor* desc) {
                self->setVertexDescriptor(desc);
            },
            nb::rv_policy::reference,
            "The vertex descriptor");

    // RenderPipelineState
    nb::class_<MTL::RenderPipelineState>(m, "RenderPipelineState")
        .def_prop_ro("label",
            [](MTL::RenderPipelineState* self) { return ns_string_to_std(self->label()); });
}

// ============================================================================
// Phase 2: MTL::RenderPassDescriptor and Attachments
// ============================================================================

void wrap_render_pass(nb::module_& m) {
    // ClearColor structure
    nb::class_<MTL::ClearColor>(m, "ClearColor")
        .def(nb::init<double, double, double, double>(),
            "red"_a, "green"_a, "blue"_a, "alpha"_a,
            "Create a clear color")
        .def_rw("red", &MTL::ClearColor::red)
        .def_rw("green", &MTL::ClearColor::green)
        .def_rw("blue", &MTL::ClearColor::blue)
        .def_rw("alpha", &MTL::ClearColor::alpha);

    // RenderPassAttachmentDescriptor
    nb::class_<MTL::RenderPassAttachmentDescriptor>(m, "RenderPassAttachmentDescriptor")
        .def_prop_rw("texture",
            [](MTL::RenderPassAttachmentDescriptor* self) { return self->texture(); },
            [](MTL::RenderPassAttachmentDescriptor* self, MTL::Texture* tex) {
                self->setTexture(tex);
            },
            nb::rv_policy::reference)

        .def_prop_rw("load_action",
            [](MTL::RenderPassAttachmentDescriptor* self) { return self->loadAction(); },
            [](MTL::RenderPassAttachmentDescriptor* self, MTL::LoadAction action) {
                self->setLoadAction(action);
            })

        .def_prop_rw("store_action",
            [](MTL::RenderPassAttachmentDescriptor* self) { return self->storeAction(); },
            [](MTL::RenderPassAttachmentDescriptor* self, MTL::StoreAction action) {
                self->setStoreAction(action);
            });

    // RenderPassColorAttachmentDescriptor
    nb::class_<MTL::RenderPassColorAttachmentDescriptor, MTL::RenderPassAttachmentDescriptor>(
        m, "RenderPassColorAttachmentDescriptor")
        .def_prop_rw("clear_color",
            [](MTL::RenderPassColorAttachmentDescriptor* self) { return self->clearColor(); },
            [](MTL::RenderPassColorAttachmentDescriptor* self, const MTL::ClearColor& color) {
                self->setClearColor(color);
            });

    // RenderPassDepthAttachmentDescriptor
    nb::class_<MTL::RenderPassDepthAttachmentDescriptor, MTL::RenderPassAttachmentDescriptor>(
        m, "RenderPassDepthAttachmentDescriptor")
        .def_prop_rw("clear_depth",
            [](MTL::RenderPassDepthAttachmentDescriptor* self) { return self->clearDepth(); },
            [](MTL::RenderPassDepthAttachmentDescriptor* self, double depth) {
                self->setClearDepth(depth);
            });

    // RenderPassDescriptor
    nb::class_<MTL::RenderPassDescriptor>(m, "RenderPassDescriptor")
        .def_static("render_pass_descriptor",
            []() { return MTL::RenderPassDescriptor::renderPassDescriptor(); },
            nb::rv_policy::reference,
            "Create a new render pass descriptor")

        .def("color_attachment",
            [](MTL::RenderPassDescriptor* self, uint32_t index) {
                return self->colorAttachments()->object(index);
            },
            "index"_a,
            nb::rv_policy::reference,
            "Get color attachment at index")

        .def_prop_ro("depth_attachment",
            [](MTL::RenderPassDescriptor* self) { return self->depthAttachment(); },
            nb::rv_policy::reference)

        .def_prop_ro("stencil_attachment",
            [](MTL::RenderPassDescriptor* self) { return self->stencilAttachment(); },
            nb::rv_policy::reference);
}

// ============================================================================
// Phase 2: MTL::VertexDescriptor and related classes
// ============================================================================

void wrap_vertex_descriptor(nb::module_& m) {
    // VertexAttributeDescriptor
    nb::class_<MTL::VertexAttributeDescriptor>(m, "VertexAttributeDescriptor")
        .def_prop_rw("format",
            [](MTL::VertexAttributeDescriptor* self) { return self->format(); },
            [](MTL::VertexAttributeDescriptor* self, MTL::VertexFormat format) {
                self->setFormat(format);
            })

        .def_prop_rw("offset",
            [](MTL::VertexAttributeDescriptor* self) { return self->offset(); },
            [](MTL::VertexAttributeDescriptor* self, uint32_t offset) {
                self->setOffset(offset);
            })

        .def_prop_rw("buffer_index",
            [](MTL::VertexAttributeDescriptor* self) { return self->bufferIndex(); },
            [](MTL::VertexAttributeDescriptor* self, uint32_t index) {
                self->setBufferIndex(index);
            });

    // VertexBufferLayoutDescriptor
    nb::class_<MTL::VertexBufferLayoutDescriptor>(m, "VertexBufferLayoutDescriptor")
        .def_prop_rw("stride",
            [](MTL::VertexBufferLayoutDescriptor* self) { return self->stride(); },
            [](MTL::VertexBufferLayoutDescriptor* self, uint32_t stride) {
                self->setStride(stride);
            })

        .def_prop_rw("step_function",
            [](MTL::VertexBufferLayoutDescriptor* self) { return self->stepFunction(); },
            [](MTL::VertexBufferLayoutDescriptor* self, MTL::VertexStepFunction func) {
                self->setStepFunction(func);
            })

        .def_prop_rw("step_rate",
            [](MTL::VertexBufferLayoutDescriptor* self) { return self->stepRate(); },
            [](MTL::VertexBufferLayoutDescriptor* self, uint32_t rate) {
                self->setStepRate(rate);
            });

    // VertexDescriptor
    nb::class_<MTL::VertexDescriptor>(m, "VertexDescriptor")
        .def_static("vertex_descriptor",
            []() { return MTL::VertexDescriptor::vertexDescriptor(); },
            nb::rv_policy::reference,
            "Create a new vertex descriptor")

        .def("attribute",
            [](MTL::VertexDescriptor* self, uint32_t index) {
                return self->attributes()->object(index);
            },
            "index"_a,
            nb::rv_policy::reference,
            "Get vertex attribute descriptor at index")

        .def("layout",
            [](MTL::VertexDescriptor* self, uint32_t index) {
                return self->layouts()->object(index);
            },
            "index"_a,
            nb::rv_policy::reference,
            "Get vertex buffer layout descriptor at index")

        .def("reset",
            [](MTL::VertexDescriptor* self) {
                self->reset();
            },
            "Reset the vertex descriptor to default state");
}

// ============================================================================
// Phase 2: CA::MetalLayer and CA::MetalDrawable
// ============================================================================

void wrap_metal_layer(nb::module_& m) {
    // MetalDrawable
    nb::class_<CA::MetalDrawable>(m, "MetalDrawable")
        .def_prop_ro("texture",
            [](CA::MetalDrawable* self) { return self->texture(); },
            nb::rv_policy::reference,
            "The texture to render into")

        .def_prop_ro("layer",
            [](CA::MetalDrawable* self) { return self->layer(); },
            nb::rv_policy::reference,
            "The layer that owns this drawable")

        .def("present",
            [](CA::MetalDrawable* self) {
                self->present();
            },
            "Present the drawable to the screen");

    // MetalLayer
    nb::class_<CA::MetalLayer>(m, "MetalLayer")
        .def_static("layer",
            []() { return CA::MetalLayer::layer(); },
            nb::rv_policy::reference,
            "Create a new Metal layer")

        .def_prop_rw("device",
            [](CA::MetalLayer* self) { return self->device(); },
            [](CA::MetalLayer* self, MTL::Device* device) {
                self->setDevice(device);
            },
            nb::rv_policy::reference,
            "The Metal device to use")

        .def_prop_rw("pixel_format",
            [](CA::MetalLayer* self) { return self->pixelFormat(); },
            [](CA::MetalLayer* self, MTL::PixelFormat format) {
                self->setPixelFormat(format);
            },
            "The pixel format of the layer")

        .def_prop_rw("framebuffer_only",
            [](CA::MetalLayer* self) { return self->framebufferOnly(); },
            [](CA::MetalLayer* self, bool framebuffer_only) {
                self->setFramebufferOnly(framebuffer_only);
            },
            "Whether the drawable can only be used as a framebuffer")

        .def_prop_rw("drawable_size",
            [](CA::MetalLayer* self) {
                auto size = self->drawableSize();
                return std::make_pair(size.width, size.height);
            },
            [](CA::MetalLayer* self, std::pair<double, double> size) {
                CGSize cg_size;
                cg_size.width = size.first;
                cg_size.height = size.second;
                self->setDrawableSize(cg_size);
            },
            "The size of the drawable in pixels")

        .def("next_drawable",
            [](CA::MetalLayer* self) {
                return self->nextDrawable();
            },
            nb::rv_policy::reference,
            "Get the next drawable for rendering");
}

// ============================================================================
// Phase 2: Update RenderPipelineDescriptor for vertex descriptor
// ============================================================================

void add_vertex_descriptor_to_pipeline(nb::module_& m) {
    auto pipeline_class = nb::type<MTL::RenderPipelineDescriptor>();
    // Note: We can't add methods to already-defined classes in nanobind,
    // so we need to add this in the original wrap_render_pipeline function
}

// ============================================================================
// Phase 2: MTL::RenderCommandEncoder
// ============================================================================

void wrap_render_encoder(nb::module_& m) {
    nb::class_<MTL::RenderCommandEncoder>(m, "RenderCommandEncoder")
        .def("set_render_pipeline_state",
            [](MTL::RenderCommandEncoder* self, MTL::RenderPipelineState* state) {
                self->setRenderPipelineState(state);
            },
            "state"_a,
            "Set the active render pipeline state")

        .def("set_vertex_buffer",
            [](MTL::RenderCommandEncoder* self, MTL::Buffer* buffer, uint32_t offset, uint32_t index) {
                self->setVertexBuffer(buffer, offset, index);
            },
            "buffer"_a, "offset"_a, "index"_a,
            "Bind a vertex buffer at the specified index")

        .def("set_fragment_buffer",
            [](MTL::RenderCommandEncoder* self, MTL::Buffer* buffer, uint32_t offset, uint32_t index) {
                self->setFragmentBuffer(buffer, offset, index);
            },
            "buffer"_a, "offset"_a, "index"_a,
            "Bind a fragment buffer at the specified index")

        .def("set_fragment_texture",
            [](MTL::RenderCommandEncoder* self, MTL::Texture* texture, uint32_t index) {
                self->setFragmentTexture(texture, index);
            },
            "texture"_a, "index"_a,
            "Bind a fragment texture at the specified index")

        .def("set_fragment_sampler_state",
            [](MTL::RenderCommandEncoder* self, MTL::SamplerState* sampler, uint32_t index) {
                self->setFragmentSamplerState(sampler, index);
            },
            "sampler"_a, "index"_a,
            "Bind a fragment sampler at the specified index")

        .def("set_cull_mode",
            [](MTL::RenderCommandEncoder* self, MTL::CullMode mode) {
                self->setCullMode(mode);
            },
            "mode"_a,
            "Set the cull mode")

        .def("set_front_facing_winding",
            [](MTL::RenderCommandEncoder* self, MTL::Winding winding) {
                self->setFrontFacingWinding(winding);
            },
            "winding"_a,
            "Set the front-facing winding order")

        .def("draw_primitives",
            [](MTL::RenderCommandEncoder* self, MTL::PrimitiveType type, uint32_t start, uint32_t count) {
                self->drawPrimitives(type, start, count);
            },
            "type"_a, "start"_a, "count"_a,
            "Draw primitives")

        .def("draw_indexed_primitives",
            [](MTL::RenderCommandEncoder* self, MTL::PrimitiveType type, uint32_t index_count,
               MTL::IndexType index_type, MTL::Buffer* index_buffer, uint32_t index_buffer_offset) {
                self->drawIndexedPrimitives(type, index_count, index_type, index_buffer, index_buffer_offset);
            },
            "type"_a, "index_count"_a, "index_type"_a, "index_buffer"_a, "index_buffer_offset"_a,
            "Draw indexed primitives")

        .def("end_encoding",
            [](MTL::RenderCommandEncoder* self) {
                self->endEncoding();
            },
            "Finish encoding commands");
}

// ============================================================================
// Module Definition
// ============================================================================

NB_MODULE(_pymetal, m) {
    m.doc() = "Python bindings for Metal GPU API via metal-cpp";

    // Wrap all Phase 1 components
    wrap_enums(m);
    wrap_device(m);
    wrap_command_queue(m);
    wrap_command_buffer(m);
    wrap_buffer(m);
    wrap_library(m);
    wrap_compute_pipeline(m);
    wrap_compute_encoder(m);

    // Wrap all Phase 2 components
    wrap_graphics_enums(m);
    wrap_texture(m);
    wrap_sampler(m);
    wrap_render_pipeline(m);
    wrap_render_pass(m);
    wrap_render_encoder(m);

    // Wrap optional features
    wrap_vertex_descriptor(m);
    wrap_metal_layer(m);
}