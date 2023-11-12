#pragma once

#include "../indirect/global.h"
#include "probeCopyBorder.h"
#include "probeIntegrate.h"
// #include "test.h"
#include "type.h"

#include "system/pass.h"
#include "system/buffer.h"
#include "render/camera.h"
#include "resource/scene.h"
#include "optix/pass.h"

#include "world/world.h"

#include "cuda/stream.h"

#include "stb/stb_image_write.h"
#include "util/timer.h"
#include <memory>
#include <mutex>

namespace Pupil::ddgi::probe {
struct SBTTypes : public optix::EmptySBT {
    using RayGenDataType = Pupil::ddgi::probe::RayGenData;
    using MissDataType = Pupil::ddgi::probe::MissData;
    using HitGroupDataType = Pupil::ddgi::probe::HitGroupData;
};

class ProbePass : public Pass {
public:
    ProbePass(std::string_view name = "DDGI Probe Pass") noexcept;
    virtual void OnRun() noexcept override;
    virtual void Inspector() noexcept override;
    // virtual void BeforeRunning() noexcept override;
    // virtual void AfterRunning() noexcept override;

    void SetScene(world::World *) noexcept;

private:
    void BindingEventCallback() noexcept;
    void InitOptixPipeline() noexcept;

    OptixLaunchParams m_optix_launch_params;
    UpdateParams m_update_params;
    std::unique_ptr<cuda::Stream> m_stream;
    std::unique_ptr<optix::Pass<SBTTypes, OptixLaunchParams>> m_optix_pass;
    // size_t m_output_pixel_num = 0;

    std::atomic_bool m_dirty = true;
    world::CameraHelper *m_world_camera = nullptr;

    Timer m_timer;

    Buffer *m_rayradiance = nullptr;
    Buffer *m_rayhitposition = nullptr;
    Buffer *m_raydirection = nullptr;
    Buffer *m_rayhitnormal = nullptr;
    Buffer *m_probeirradiance = nullptr;
    Buffer *m_probedepth = nullptr;
    Buffer *m_probeposition = nullptr;

    CUdeviceptr m_randomoriention_cuda_memory = 0;
    CUdeviceptr m_probepos_cuda_memory = 0;
    CUdeviceptr m_zeroradiance_cuda_memory = 0;
    CUdeviceptr m_zerodepth_cuda_memory = 0;
    std::vector<float3> m_probepos;

    // int m_probesidelength = 64;
    // int m_irradiancerays_perprobe = 64;
    // int m_probecountperside = 2;
    // float3 m_probestep;

    float m_hysteresis = 0.95f;
    float m_maxdistance;

    unsigned int m_frame_cnt = 0;
};
}// namespace Pupil::ddgi::probe