#pragma once
#include "../reservoir.h"
#include "render/geometry.h"
#include "render/camera.h"
#include "render/emitter.h"
#include "render/material/optix_material.h"

namespace Pupil::ddgi::shading {
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

    OptixTraversableHandle handle;
    cuda::RWArrayView<Reservoir> final_reservoirs;
    cuda::RWArrayView<float4> frame_buffer;
    cuda::ConstArrayView<float3> albedo_buffer;
    cuda::ConstArrayView<float3> position_buffer;
    cuda::ConstArrayView<float3> normal_buffer;
    cuda::ConstArrayView<float3> emission_buffer;
};

// 下面三个是和SBT绑定的结构体，
// 分别对应在__raygen__xxx、__miss__xxx、__closesthit__xxx中可以访问的数据
// 可以自定义结构体内容，也可以为空
struct RayGenData {
};
struct MissData {
};
struct HitGroupData {
    optix::material::Material mat;
    optix::Geometry geo;
    int emitter_index_offset = -1;
};

}// namespace Pupil::ddgi::shading