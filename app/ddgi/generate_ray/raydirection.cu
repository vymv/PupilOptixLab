#include <optix.h>
#include "type.h"

#include "optix/util.h"
#include "optix/geometry.h"
#include "optix/scene/emitter.h"

#include "cuda/random.h"

using namespace Pupil;

extern "C" {
__constant__ ddgi::generateray::OptixLaunchParams optix_launch_params;
}

struct HitInfo {
    optix::LocalGeometry geo;
    optix::material::Material mat;
    int emitter_index;
};

struct PathPayloadRecord {
    float3 color;
    float3 normal;
};

struct mat3 {
    float3 row0;
    float3 row1;
    float3 row2;
}

mat3
fromUnitAxisAngle(const float3 axis, float fRadians) {

    mat3 m;
    float fCos = cos(fRadians);
    float fSin = sin(fRadians);
    float fOneMinusCos = 1.0f - fCos;
    float fX2 = axis.x *axis.x float fY2 = axis.y * axis.y;
    float fZ2 = axis.z * axis.z;
    float fXYM = axis.x * axis.y * fOneMinusCos;
    float fXZM = axis.x * axis.z * fOneMinusCos;
    float fYZM = axis.y * axis.z * fOneMinusCos;
    float fXSin = axis.x * fSin;
    float fYSin = axis.y * fSin;
    float fZSin = axis.z * fSin;

    m.row0.x = fX2 * fOneMinusCos + fCos;
    m.row0.y = fXYM - fZSin;
    m.row0.z = fXZM + fYSin;

    m.row1.x = fXYM + fZSin;
    m.row1.y = fY2 * fOneMinusCos + fCos;
    m.row1.z = fYZM - fXSin;

    m.row2.x = fXZM - fYSin;
    m.row2.y = fYZM + fXSin;
    m.row2.z = fZ2 * fOneMinusCos + fCos;

    return m;
}

float madfrac(float a, float b) {
    return (a * b) - floorf(a * b);
}

float3 sphericalFibonacci(float i, float n) {
    const float PHI = sqrtf(5) * 0.5f + 0.5f;
    float phi = 2.0f * PI * madfrac(i, PHI - 1.0f);
    float cosTheta = 1.0f - (2.0f * i + 1.0f) * (1.0f / n);
    float sinTheta = sqrtf(saturate(1.0f - cosTheta * cosTheta));

    return make_float3(
        cosf(phi) * sinTheta,
        sinf(phi) * sinTheta,
        cosTheta);
}

float3 cross_mv(mat3 m, float3 v) {
    return make_float3(
        m.row0.x * m.x + m.row0.y * m.y + m.row0.z * m.z,
        m.row1.x * m.x + m.row1.y * m.y + m.row1.z * m.z,
        m.row2.x * m.x + m.row2.y * m.y + m.row2.z * m.z, );
}

extern "C" __global__ void __raygen__main() {
    const uint3 index = optixGetLaunchIndex();
    const unsigned int w = optix_launch_params.config.frame.width;
    const unsigned int h = optix_launch_params.config.frame.height;
    const unsigned int pixel_index = index.y * w + index.x;

    cuda::Random r;
    float3 axis = normalize(make_float3(r.Next(), r.Next(), r.Next()));

    float angle = r.Next() * (2 * M_PIf);
    mat3 randomOrientation = fromAxisAngle(axis, angle);
    float3 rayDirection = cross_mv(randomOrientation, sphericalFibonacci(index.x, optix_launch_params.ray_per_probe));

    optix_launch_params.raydirection[pixel_index] = make_float4(rayDirection, 1.f);
}

extern "C" __global__ void __miss__default() {
}
extern "C" __global__ void __closesthit__default() {
    const ddgi::gbuffer::HitGroupData *sbt_data = (ddgi::gbuffer::HitGroupData *)optixGetSbtDataPointer();
    auto record = optix::GetPRD<PathPayloadRecord>();

    const auto ray_dir = optixGetWorldRayDirection();

    auto geo = sbt_data->geo.GetHitLocalGeometry(ray_dir, sbt_data->mat.twosided);
    record->color = sbt_data->mat.GetColor(geo.texcoord);
    record->normal = geo.normal;
}