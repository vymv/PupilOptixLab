#pragma once

#include "material/optix_material.h"
#include "optix/geometry.h"
#include "optix/scene/camera.h"
#include "optix/scene/emitter.h"

namespace Pupil::ddgi::probe
{

struct OptixLaunchParams
{
    struct
    {
        struct
        {
            unsigned int width;
            unsigned int height;
        } frame;
    } config;
    unsigned int random_seed;

    cuda::ConstDataView<util::Mat3> randomOrientation;
    cuda::ConstArrayView<float3> probepos;

    optix::EmitterGroup emitters;

    cuda::RWArrayView<float4> rayradiance;
    cuda::RWArrayView<float3> raydirection;
    cuda::RWArrayView<float3> rayhitposition;
    cuda::RWArrayView<float3> rayhitnormal;
    //  cuda::RWArrayView<float4> normal;

    OptixTraversableHandle handle;
};

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

} // namespace Pupil::ddgi::probe