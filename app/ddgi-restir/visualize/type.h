#pragma once

#include "render/geometry.h"
#include "render/camera.h"
#include "render/emitter.h"
#include "render/material/optix_material.h"

namespace Pupil::ddgi::visualize {

struct OptixLaunchParams {
    struct
    {
        struct
        {
            unsigned int width;
            unsigned int height;
        } frame;
    } config;

    cuda::ConstArrayView<float3> probe_position;
    cuda::ConstArrayView<float4> input_buffer;
    cuda::RWArrayView<float4> visualize_buffer;

    float probe_visualize_size;
    int probe_count;
    int highlight_index;
    mat4x4 proj_view;
};

struct RayGenData {
};
struct MissData {
};
struct HitGroupData {
    optix::material::Material mat;
    optix::Geometry geo;
    int emitter_index_offset = -1;
};

}// namespace Pupil::ddgi::visualize