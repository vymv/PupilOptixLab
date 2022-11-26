#pragma once

#include "dx12_device.h"
#include <optix.h>

#include "optix_wrap/module.h"
#include "optix_wrap/pipeline.h"

#include <cuda_runtime.h>
#include <sstream>
#include <memory>

inline void CudaCheck(cudaError_t error, const char *call, const char *file, unsigned int line) {
    if (error != cudaSuccess) {
        std::wstringstream ss;
        ss << "CUDA call (" << call << " ) failed with error: '"
           << cudaGetErrorString(error) << "' (" << file << ":" << line << ")\n";

        OutputDebugString(ss.str().c_str());
        assert(false);
    }
}
inline void CudaCheck(CUresult error, const char *call, const char *file, unsigned int line) {
    if (error != cudaSuccess) {
        std::wstringstream ss;
        ss << "CUDA call (" << call << " ) failed with error: '"
           << error << "' (" << file << ":" << line << ")\n";

        OutputDebugString(ss.str().c_str());
        assert(false);
    }
}
inline void OptixCheck(OptixResult res, const char *call, const char *file, unsigned int line) {
    if (res != OPTIX_SUCCESS) {
        std::wstringstream ss;
        ss << "Optix call '" << call << "' failed: " << file << ':' << line << ")\n";
        OutputDebugString(ss.str().c_str());
        assert(false);
    }
}
inline void OptixCheckLog(
    OptixResult res,
    const char *log,
    size_t sizeof_log,
    size_t sizeof_log_returned,
    const char *call,
    const char *file,
    unsigned int line) {
    if (res != OPTIX_SUCCESS) {
        std::wstringstream ss;
        ss << "Optix call '" << call << "' failed: " << file << ':' << line << ")\nLog:\n"
           << log << (sizeof_log_returned > sizeof_log ? "<TRUNCATED>" : "") << '\n';
        OutputDebugString(ss.str().c_str());
        assert(false);
    }
}
#define CUDA_CHECK(call) CudaCheck(call, #call, __FILE__, __LINE__)
#define OPTIX_CHECK(call) OptixCheck(call, #call, __FILE__, __LINE__)
#define OPTIX_CHECK_LOG(call)                                  \
    do {                                                       \
        char LOG[400];                                         \
        size_t LOG_SIZE = sizeof(LOG);                         \
        OptixCheckLog(call, LOG, sizeof(LOG), LOG_SIZE, #call, \
                      __FILE__, __LINE__);                     \
    } while (false)

namespace device {
struct CudaDx12SharedTexture {
    Microsoft::WRL::ComPtr<ID3D12Resource> dx12_resource;
    cudaExternalMemory_t cuda_ext_memory;
    cudaSurfaceObject_t cuda_surf_obj;
    uint64_t fence_value;
};

struct SharedFrameResource {
    std::unique_ptr<CudaDx12SharedTexture> frame[DX12::NUM_OF_FRAMES];
};

class Optix {
public:
    uint32_t cuda_device_id = 0;
    uint32_t cuda_node_mask = 0;

    cudaStream_t cuda_stream = nullptr;
    cudaExternalSemaphore_t cuda_semaphore = nullptr;

    OptixDeviceContext context = nullptr;

    Optix() = delete;
    Optix(DX12 *dx12_backend) noexcept;
    ~Optix() noexcept = default;

    [[nodiscard]] std::unique_ptr<SharedFrameResource>
    CreateSharedFrameResource() noexcept;

    void InitPipeline(const optix_wrap::PipelineDesc &desc) noexcept;
    void InitScene() noexcept;

    void Run() noexcept;

private:
    DX12 *m_dx12_backend = nullptr;
    SharedFrameResource *m_frame_resource = nullptr;
    std::unique_ptr<optix_wrap::Pipeline> pipeline = nullptr;

    void InitCuda() noexcept;

    [[nodiscard]] std::unique_ptr<CudaDx12SharedTexture>
    CreateSharedResourceWithDX12() noexcept;
};
}// namespace device