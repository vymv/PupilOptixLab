#pragma once
#include "../reservoir.h"
#include "render/geometry.h"
#include "render/camera.h"
#include "render/emitter.h"
#include "render/material/optix_material.h"

namespace Pupil::ddgi::temporal {
// OptixLaunchParams是在optix管线中全局的常量，可以自由定义，但尽量保持结构体占用的内存较小，
// 在.cu文件中需要按照如下声明，且常量名必须为optix_launch_params
// extern "C" {
// __constant__ OptixLaunchParams optix_launch_params;
// }
struct OptixLaunchParams {
    struct {
        struct
        {
            unsigned int width;
            unsigned int height;
        } frame;
    } config;

    // int spatial_radius;
    unsigned int random_seed;
    Pupil::cuda::ConstArrayView<float4> position_buffer;
    Pupil::cuda::ConstArrayView<float4> prev_position;

    Pupil::cuda::RWArrayView<Reservoir> reservoirs;
    Pupil::cuda::RWArrayView<Reservoir> prev_frame_reservoirs;

    struct {
        mat4x4 prev_proj_view;
    } camera;
};

}// namespace Pupil::ddgi::temporal