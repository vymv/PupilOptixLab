#include <optix.h>
#include "type.h"

#include "optix/util.h"
#include "optix/geometry.h"
#include "optix/scene/emitter.h"

#include "cuda/random.h"

using namespace Pupil;

extern "C" {
__constant__ ddgi::gbuffer::OptixLaunchParams optix_launch_params;
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

extern "C" __global__ void __raygen__main() {

    // 每根ray的直接光照
    float3 ray_direction = normalize(make_float3(
        dot(camera.camera_to_world.r0, d),
        dot(camera.camera_to_world.r1, d),
        dot(camera.camera_to_world.r2, d)));

    float3 ray_origin = make_float3(
        camera.camera_to_world.r0.w,
        camera.camera_to_world.r1.w,
        camera.camera_to_world.r2.w);

    // primary ray
    optixTrace(optix_launch_params.handle,
               ray_origin, ray_direction,
               0.001f, 1e16f, 0.f,
               255, OPTIX_RAY_FLAG_NONE,
               0, 2, 0,
               u0, u1);

    // 获取emitter
    if (record.hit.emitter_index >= 0) {
        auto &emitter = optix_launch_params.emitters.areas[local_hit.emitter_index];
        auto emission = emitter.GetRadiance(local_hit.geo.texcoord);
        record.radiance += emission;
    }

    // 直接光照
    auto &emitter = optix_launch_params.emitters.SelectOneEmiiter(record.random.Next());
    auto emitter_sample_record = emitter.SampleDirect(local_hit.geo, record.random.Next2());

    if (!optix::IsZero(emitter_sample_record.pdf)) {

        // shadow ray
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

    optix_launch_params.albedo[pixel_index] = make_float4(record.color, 1.f);
    optix_launch_params.normal[pixel_index] = make_float4(record.normal * 0.5f + 0.5f, 1.f);
}

extern "C" __global__ void __miss__default() {
}
extern "C" __global__ void __closesthit__default() {

    //

    const ddgi::gbuffer::HitGroupData *sbt_data = (ddgi::gbuffer::HitGroupData *)optixGetSbtDataPointer();
    auto record = optix::GetPRD<PathPayloadRecord>();

    const auto ray_dir = optixGetWorldRayDirection();

    auto geo = sbt_data->geo.GetHitLocalGeometry(ray_dir, sbt_data->mat.twosided);
    record->color = sbt_data->mat.GetColor(geo.texcoord);
    record->normal = geo.normal;
}