#pragma once

#include "optix/geometry.h"
#include "optix/scene/camera.h"
#include "optix/scene/emitter.h"
#include "material/optix_material.h"

namespace Pupil::ddgi::generateray {
struct OptixLaunchParams {
    struct {
        struct {
            unsigned int width;
            unsigned int height;
        } frame;
    } config;
    unsigned int random_seed;

    cuda::ConstDataView<optix::Camera> camera;
    optix::EmitterGroup emitters;

    cuda::RWArrayView<float4> raydirection;

    OptixTraversableHandle handle;
    unsigned int ray_per_probe;
};

struct RayGenData {};
struct MissData {};
struct HitGroupData {
    optix::material::Material mat;
    optix::Geometry geo;
    int emitter_index_offset = -1;
};

}// namespace Pupil::ddgi::generateray