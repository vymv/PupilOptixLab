#include <optix.h>
#include "type.h"

#include "optix/util.h"
#include "optix/geometry.h"
#include "optix/scene/emitter.h"

#include "cuda/random.h"

using namespace Pupil;

extern "C" {
__constant__ ddgi::probe::OptixLaunchParams optix_launch_params;
}
struct Mat3 {

    float re[4][4];
};
struct HitInfo {
    optix::LocalGeometry geo;
    optix::material::Material mat;
    int emitter_index;
};

struct PathPayloadRecord {
    float3 radiance;
    float3 env_radiance;
    float env_pdf;
    cuda::Random random;
    HitInfo hit;
    float3 color;
};

__device__ float3 sphericalFibonacci(float i, float n) {
    const float PHI = sqrt(5.0) * 0.5f + 0.5f;

    float madfrac = (i * (PHI - 1)) - floor(i * (PHI - 1));
    float phi = 2.0f * M_PIf * (madfrac < 0.0f ? 0.0 : (madfrac > 1.0f ? 1.0 : madfrac));
    float cosTheta = 1.0f - (2.0f * i + 1.0f) * (1.0f / n);
    float sinTheta = sqrt(1.0f - cosTheta * cosTheta < 0.0f ? 0.0 : (1.0f - cosTheta * cosTheta > 1.0f ? 1.0 : 1.0f - cosTheta * cosTheta));

    return make_float3(
        cos(phi) * sinTheta,
        sin(phi) * sinTheta,
        cosTheta);
}

__device__ Mat3 fromAxisAngle(const float3 axis, float fRadians) {

    Mat3 m;
    float fCos = cos(fRadians);
    float fSin = sin(fRadians);
    float fOneMinusCos = 1.0f - fCos;
    float fX2 = axis.x * axis.x;
    float fY2 = axis.y * axis.y;
    float fZ2 = axis.z * axis.z;
    float fXYM = axis.x * axis.y * fOneMinusCos;
    float fXZM = axis.x * axis.z * fOneMinusCos;
    float fYZM = axis.y * axis.z * fOneMinusCos;
    float fXSin = axis.x * fSin;
    float fYSin = axis.y * fSin;
    float fZSin = axis.z * fSin;

    m.re[0][0] = fX2 * fOneMinusCos + fCos;
    m.re[0][1] = fXYM - fZSin;
    m.re[0][2] = fXZM + fYSin;

    m.re[1][0] = fXYM + fZSin;
    m.re[1][1] = fY2 * fOneMinusCos + fCos;
    m.re[1][2] = fYZM - fXSin;

    m.re[2][0] = fXZM - fYSin;
    m.re[2][1] = fYZM + fXSin;
    m.re[2][2] = fZ2 * fOneMinusCos + fCos;

    return m;
}

extern "C" __global__ void __raygen__main() {

    const uint3 index = optixGetLaunchIndex();
    const unsigned int w = optix_launch_params.config.frame.width;
    //const unsigned int h = optix_launch_params.config.frame.height;
    const unsigned int pixel_index = index.y * w + index.x;
    const unsigned int rayid = index.x;
    const unsigned int probeid = index.y;

    PathPayloadRecord record{};
    uint32_t u0, u1;
    optix::PackPointer(&record, u0, u1);
    record.random.Init(4, pixel_index, optix_launch_params.random_seed);

    float3 ray_origin = optix_launch_params.probepos[probeid];
    auto &m = *optix_launch_params.randomOrientation.operator->();
    float3 sf = sphericalFibonacci(float(rayid), float(w));

    float3 ray_direction = make_float3(m.re[0][0] * sf.x + m.re[0][1] * sf.y + m.re[0][2] * sf.z,
                                       m.re[1][0] * sf.x + m.re[1][1] * sf.y + m.re[1][2] * sf.z,
                                       m.re[2][0] * sf.x + m.re[2][1] * sf.y + m.re[2][2] * sf.z);

    //printf("%d,(%f,%f,%f)\n", probeid, ray_origin.x, ray_origin.y, ray_origin.z);
    float3 result = make_float3(0.f);
    optixTrace(optix_launch_params.handle,
               ray_origin, ray_direction,
               1e-5f, 1e16f, 0.f,
               255, OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
               0, 2, 0,
               u0, u1);
    // optix_launch_params.rayradiance[pixel_index] = make_float4(record.color, 1.f);
    // return;
    //printf("%d,%f,%f,%f\n", probeid, record.hit.geo.position.x, record.hit.geo.position.y, record.hit.geo.position.z);

    auto local_hit = record.hit;
    if (record.hit.emitter_index >= 0) {
        auto &emitter = optix_launch_params.emitters.areas[local_hit.emitter_index];
        auto emission = emitter.GetRadiance(local_hit.geo.texcoord);
        record.radiance += emission;
    }
    // direct light sampling
    {
        auto &emitter = optix_launch_params.emitters.SelectOneEmiiter(record.random.Next());
        auto emitter_sample_record = emitter.SampleDirect(local_hit.geo, record.random.Next2());

        if (!optix::IsZero(emitter_sample_record.pdf)) {
            bool occluded =
                optix::Emitter::TraceShadowRay(
                    optix_launch_params.handle,
                    local_hit.geo.position, emitter_sample_record.wi,
                    0.001f, emitter_sample_record.distance - 0.001f);

            if (!occluded) {
                float3 wi = optix::ToLocal(emitter_sample_record.wi, local_hit.geo.normal);
                float3 wo = optix::ToLocal(-ray_direction, local_hit.geo.normal);
                auto [f, pdf] = record.hit.mat.Eval(wi, wo, local_hit.geo.texcoord);
                if (!optix::IsZero(f)) {
                    float NoL = dot(local_hit.geo.normal, emitter_sample_record.wi);
                    emitter_sample_record.pdf *= emitter.select_probability;
                    record.radiance += emitter_sample_record.radiance * f * NoL / emitter_sample_record.pdf;
                }
            }
        }
    }

    record.radiance += record.env_radiance;
    result += record.radiance;

    optix_launch_params.rayradiance[pixel_index] = make_float4(result, 1.f);
    //optix_launch_params.rayradiance[pixel_index] = make_float4(record.color, 1.f);
    //optix_launch_params.rayradiance[pixel_index] = make_float4(ray_origin, 1.f);
    //optix_launch_params.rayradiance[pixel_index] = make_float4(ray_direction, 1.f);
}

extern "C" __global__ void __miss__default() {

    auto record = optix::GetPRD<PathPayloadRecord>();
    if (optix_launch_params.emitters.env) {
        optix::LocalGeometry temp;
        temp.position = optixGetWorldRayDirection();
        float3 scatter_pos = make_float3(0.f);
        auto env_emit_record = optix_launch_params.emitters.env->Eval(temp, scatter_pos);
        record->env_radiance = env_emit_record.radiance;
        record->env_pdf = env_emit_record.pdf;
    }
    record->hit.emitter_index = -1;
    record->color = make_float3(1.0, 0.0, 1.0);
}
extern "C" __global__ void __closesthit__default() {

    const ddgi::probe::HitGroupData *sbt_data = (ddgi::probe::HitGroupData *)optixGetSbtDataPointer();
    auto record = optix::GetPRD<PathPayloadRecord>();

    const auto ray_dir = optixGetWorldRayDirection();
    const auto ray_o = optixGetWorldRayOrigin();

    record->hit.geo = sbt_data->geo.GetHitLocalGeometry(ray_dir, sbt_data->mat.twosided);
    if (sbt_data->emitter_index_offset >= 0) {
        record->hit.emitter_index = sbt_data->emitter_index_offset + optixGetPrimitiveIndex();
    } else {
        record->hit.emitter_index = -1;
    }

    record->hit.mat = sbt_data->mat;

    record->color = make_float3(record->hit.geo.texcoord, 1.0);
}

// 只是返回一个bool值，返回1表示被遮挡了
extern "C" __global__ void __closesthit__shadow() {
    optixSetPayload_0(1u);
}

extern "C" __global__ void __miss__shadow() {
    // optixSetPayload_0(0u);
}
