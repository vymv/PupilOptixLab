#pragma once

#include "render/geometry.h"
#include "render/camera.h"
#include "render/emitter.h"
#include "render/material/optix_material.h"

namespace Pupil::ddgi::probe {

struct OptixLaunchParams {
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

    float3 probeStartPosition;
    float3 probeStep;
    int3 probeCount;
    uint2 probeirradiancesize;
    int probeSideLength;
    cuda::ConstArrayView<float4> probeirradiance;
    cuda::ConstArrayView<float4> probedepth;
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

}// namespace Pupil::ddgi::probe