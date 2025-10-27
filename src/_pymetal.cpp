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
            "Maximum threads per threadgroup");

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
}