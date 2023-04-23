#pragma once

#include "material/optix_material.h"
#include "optix/geometry.h"
#include "optix/scene/camera.h"
#include "optix/scene/emitter.h"

namespace Pupil::ddgi::render
{
// OptixLaunchParams是在optix管线中全局的常量，可以自由定义，但尽量保持结构体占用的内存较小，
// 在.cu文件中需要按照如下声明，且常量名必须为optix_launch_params
// extern "C" {
// __constant__ OptixLaunchParams optix_launch_params;
// }
struct OptixLaunchParams
{
    struct
    {
        unsigned int max_depth;

        struct
        {
            unsigned int width;
            unsigned int height;
        } frame;
    } config;
    unsigned int random_seed;
    unsigned int spp;

    // cuda::ConstDataView和ConstArrayView可以理解为指针
    // 可以直接用optix::Camera，使用cuda指针是为了减小结构体体积
    cuda::ConstDataView<optix::Camera> camera;
    optix::EmitterGroup emitters;

    cuda::RWArrayView<float4> frame_buffer;
    cuda::ConstArrayView<float4> probeirradiance;

    // 可以理解为场景的bvh，用来发射光线和场景求交
    OptixTraversableHandle handle;

    float3 probeStartPosition;
    float3 probeStep;
    int3 probeCount;
    uint2 probeirradiancesize;
    int probeSideLength;
};

// 下面三个是和SBT绑定的结构体，
// 分别对应在__raygen__xxx、__miss__xxx、__closesthit__xxx中可以访问的数据
// 可以自定义结构体内容，也可以为空
struct RayGenData
{
};
struct MissData
{
};
struct HitGroupData
{
    optix::material::Material mat;
    optix::Geometry geo;
    int emitter_index_offset = -1;
};

} // namespace Pupil::ddgi::render