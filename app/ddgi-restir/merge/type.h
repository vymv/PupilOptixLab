#pragma once
#include "../reservoir.h"
#include "render/geometry.h"
#include "render/camera.h"
#include "render/emitter.h"
#include "render/material/optix_material.h"

namespace Pupil::ddgi::merge {
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

    cuda::ConstArrayView<float4> direct_buffer;
    cuda::ConstArrayView<float4> indirect_buffer;
    cuda::RWArrayView<float4> output_buffer;
    bool is_pathtracer;
};

struct RayGenData {
};
struct MissData {
};
struct HitGroupData {
};

}// namespace Pupil::ddgi::merge