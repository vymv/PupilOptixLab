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

namespace Pupil::ddgi::shadow {
struct SBTTypes : public Pupil::optix::EmptySBT {
};

class ShadowRayPass : public Pass {
public:
    ShadowRayPass(std::string_view name = "ShadowRay Pass") noexcept;
    virtual void OnRun() noexcept override;
    virtual void Inspector() noexcept override;

    void SetScene(world::World *) noexcept;

private:
    void BindingEventCallback() noexcept;
    void InitOptixPipeline() noexcept;

    ShadowRayPassLaunchParams m_params;
    std::unique_ptr<Pupil::cuda::Stream> m_stream;
    std::unique_ptr<Pupil::optix::Pass<SBTTypes, ShadowRayPassLaunchParams>> m_optix_pass;
    size_t m_output_pixel_num = 0;

    Pupil::Timer m_timer;
};
};// namespace Pupil::ddgi::shadow