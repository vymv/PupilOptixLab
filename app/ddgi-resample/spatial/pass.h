#pragma once

#include "../indirect/global.h"
#include "system/pass.h"
#include "resource/scene.h"
#include "optix/pass.h"
#include "system/buffer.h"
#include "world/world.h"

#include "type.h"

#include "cuda/stream.h"

#include "util/timer.h"

#include <memory>
#include <mutex>

namespace Pupil::ddgi::spatial {

struct SBTTypes : public optix::EmptySBT {
    using RayGenDataType = Pupil::ddgi::spatial::RayGenData;
    using MissDataType = Pupil::ddgi::spatial::MissData;
    using HitGroupDataType = Pupil::ddgi::spatial::HitGroupData;
};

class SpatialPass : public Pass {
public:
    SpatialPass(std::string_view name = "DDGI Spatial Pass") noexcept;
    virtual void OnRun() noexcept override;
    virtual void Inspector() noexcept override;

    // virtual void BeforeRunning() noexcept override {
    // }
    // virtual void AfterRunning() noexcept override {
    // }

    void SetScene(world::World *world) noexcept;

private:
    void BindingEventCallback() noexcept;
    void InitOptixPipeline() noexcept;

    OptixLaunchParams m_optix_launch_params;
    std::unique_ptr<cuda::Stream> m_stream;
    std::unique_ptr<optix::Pass<SBTTypes, OptixLaunchParams>> m_optix_pass;

    std::atomic_bool m_dirty = true;
    world::CameraHelper *m_world_camera = nullptr;
    int m_spatial_radius;

    Timer m_timer;
};
}// namespace Pupil::ddgi::spatial