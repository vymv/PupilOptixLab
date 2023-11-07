#pragma once

#include "../reservoir.h"
#include "render/geometry.h"
#include "render/camera.h"
#include "render/emitter.h"
#include "render/material/optix_material.h"
namespace Pupil::ddgi::shadow {
struct ShadowRayPassLaunchParams {
    struct {
        struct
        {
            unsigned int width;
            unsigned int height;
        } frame;
    } config;

    unsigned int type;
    OptixTraversableHandle handle;
    Pupil::cuda::ConstArrayView<float4> position;
    Pupil::cuda::ConstArrayView<float4> normal;
    Pupil::cuda::ConstArrayView<float4> albedo;

    Pupil::cuda::RWArrayView<Reservoir> reservoirs;

    Pupil::cuda::RWArrayView<float4> frame_buffer;
};
};// namespace Pupil::ddgi::shadow