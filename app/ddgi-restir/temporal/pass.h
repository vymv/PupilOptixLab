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

namespace Pupil::ddgi::temporal {

struct SBTTypes : public optix::EmptySBT {
};

class TemporalPass : public Pass {
public:
    TemporalPass(std::string_view name = "DDGI Temporal Pass") noexcept;
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

    Timer m_timer;
};
}// namespace Pupil::ddgi::temporal