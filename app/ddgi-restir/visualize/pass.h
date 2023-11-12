#pragma once

#include "../indirect/global.h"
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

namespace Pupil::ddgi::visualize {
struct SBTTypes : public optix::EmptySBT {
    using RayGenDataType = Pupil::ddgi::visualize::RayGenData;
    using MissDataType = Pupil::ddgi::visualize::MissData;
    using HitGroupDataType = Pupil::ddgi::visualize::HitGroupData;
};

class VisualizePass : public Pass {
public:
    VisualizePass(std::string_view name = "DDGI Visualize Pass") noexcept;
    virtual void OnRun() noexcept override;
    virtual void Inspector() noexcept override;
    // virtual void BeforeRunning() noexcept override;
    // virtual void AfterRunning() noexcept override;

    void SetScene(world::World *) noexcept;

private:
    void BindingEventCallback() noexcept;
    void InitOptixPipeline() noexcept;

    OptixLaunchParams m_optix_launch_params;
    std::unique_ptr<cuda::Stream> m_stream;
    std::unique_ptr<optix::Pass<SBTTypes, OptixLaunchParams>> m_optix_pass;
    // size_t m_output_pixel_num = 0;

    std::atomic_bool m_dirty = true;
    world::CameraHelper *m_world_camera = nullptr;

    Timer m_timer;
};
}// namespace Pupil::ddgi::visualize