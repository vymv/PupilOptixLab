#include "type.h"
#include <optix.h>

//#include "../indirect/indirect.h"
#include "../indirect/probemath.h"
#include "render/geometry.h"
#include "render/emitter.h"
#include "optix/util.h"

#include "cuda/random.h"

using namespace Pupil;

extern "C" {
__constant__ ddgi::render::OptixLaunchParams optix_launch_params;
}

struct PathPayloadRecord {
    cuda::Random random;
    unsigned int pixel_index;
    bool hit_flag;
    bool is_emitter;

    optix::material::Material::LocalBsdf bsdf;
    float3 color;
    float3 pos;
    float3 normal;
    float depth;
};

extern "C" __global__ void __raygen__main() {
     const uint3 index = optixGetLaunchIndex();
    const unsigned int w = optix_launch_params.config.frame.width;
    const unsigned int h = optix_launch_params.config.frame.height;
    const unsigned int pixel_index = index.y * w + index.x;
    auto &camera = optix_launch_params.camera;

    PathPayloadRecord record{};
    uint32_t u0, u1;
    optix::PackPointer(&record, u0, u1);
    record.color = make_float3(0.f);
    record.random.Init(4, pixel_index, optix_launch_params.random_seed);
    record.pixel_index = pixel_index;
    record.hit_flag = false;
    record.is_emitter = false;
    record.depth = 1e16f;

    const float2 subpixel_jitter = make_float2(0.5f);
    // const float2 subpixel_jitter = record.random.Next2();
    const float2 subpixel =
        make_float2(
            (static_cast<float>(index.x) + subpixel_jitter.x) / static_cast<float>(w),
            (static_cast<float>(index.y) + subpixel_jitter.y) / static_cast<float>(h));
    const float4 point_on_film = make_float4(subpixel, 0.f, 1.f);

    float4 d = camera.sample_to_camera * point_on_film;

    d /= d.w;
    d.w = 0.f;
    d = normalize(d);

    float3 ray_direction = normalize(make_float3(camera.camera_to_world * d));

    float3 ray_origin = make_float3(
        camera.camera_to_world.r0.w,
        camera.camera_to_world.r1.w,
        camera.camera_to_world.r2.w);

    optix_launch_params.reservoirs[pixel_index].Init();

    optixTrace(optix_launch_params.handle,
               ray_origin, ray_direction,
               0.001f, 1e16f, 0.f,
               255, OPTIX_RAY_FLAG_NONE,
               0, 2, 0,
               u0, u1);

    optix_launch_params.reservoirs[pixel_index].CalcW();

    optix_launch_params.position_buffer[pixel_index] = make_float4(record.pos, record.hit_flag ? 1.f : 0.f);
    optix_launch_params.albedo_buffer[pixel_index] = make_float4(record.color, record.is_emitter ? 1.f : 0.f);
    optix_launch_params.normal_buffer[pixel_index] = make_float4(record.normal, record.depth);
    optix_launch_params.bsdf_buffer[pixel_index] = record.bsdf;
}

extern "C" __global__ void __miss__default() {
    auto record = optix::GetPRD<PathPayloadRecord>();
    optix::material::Material::LocalBsdf bsdf;
    bsdf.type = EMatType::Unknown;
    record->bsdf = bsdf;
}

extern "C" __global__ void __closesthit__default() {
    const ddgi::render::HitGroupData *sbt_data = (ddgi::render::HitGroupData *)optixGetSbtDataPointer();
    auto record = optix::GetPRD<PathPayloadRecord>();

    const auto ray_dir = optixGetWorldRayDirection();

    optix::LocalGeometry geo;
    sbt_data->geo.GetHitLocalGeometry(geo, ray_dir, sbt_data->mat.twosided);
    record->color = sbt_data->mat.GetColor(geo.texcoord);
    record->bsdf = sbt_data->mat.GetLocalBsdf(geo.texcoord);
    record->normal = geo.normal;
    record->pos = geo.position;
    // +Z points -view
    record->depth = -(optix_launch_params.camera.view * make_float4(geo.position, 1.f)).z;
    record->hit_flag = true;

    if (sbt_data->emitter_index_offset >= 0) {
        record->is_emitter = true;
        auto emitter_index = sbt_data->emitter_index_offset + optixGetPrimitiveIndex();
        auto &emitter = optix_launch_params.emitters.areas[emitter_index];
        auto emission = emitter.GetRadiance(geo.texcoord);

        record->color = emission;
        return;
    }
    
    for (unsigned int i = 0u; i < optix_launch_params.M; i++) {

        // 每次循环都随机取一个光源面片
        float r1 = record->random.Next();
        auto &emitter = optix_launch_params.emitters.SelectOneEmiiter(r1);
        // auto emitter = optix_launch_params.emitters.areas[1];
        // printf("emitter_num: %d\n", optix_launch_params.emitters.areas.GetNum());
        // emitter.select_probability = 1.0f;
        optix::EmitterSampleRecord emitter_sample_record;

        // 计算其直接光照
        float2 r2 = record->random.Next2();
        emitter.SampleDirect(emitter_sample_record, geo, r2);

        // 填写Reservoir样本
        Reservoir::Sample x_i;
        x_i.pos = emitter_sample_record.pos;
        x_i.distance = emitter_sample_record.distance;
        x_i.normal = emitter_sample_record.normal;
        x_i.emission = emitter_sample_record.radiance;
        x_i.radiance = make_float3(0.f);
        // x_i.emission = make_float3(0.f);

        // 计算样本的权重
        float w_i = 0.f;
        float3 wi = optix::ToLocal(emitter_sample_record.wi, geo.normal); // shading point to light source
        float3 wo = optix::ToLocal(-ray_dir, geo.normal); // shading point to eye
        optix::BsdfSamplingRecord bsdf_sample_record;
        bsdf_sample_record.wo = wo;
        bsdf_sample_record.wi = wi;
        record->bsdf.Eval(bsdf_sample_record);

        // float3 f = make_float3(0.f);
        // if (wi.z > 0.f && wo.z > 0.f) {
        //     f = record->color * M_1_PIf;
        // }
        float3 f = bsdf_sample_record.f;

        if (!optix::IsZero(f)) {
            float NoL = dot(geo.normal, emitter_sample_record.wi);
            emitter_sample_record.pdf *= emitter.select_probability;
            if (emitter_sample_record.pdf > 0.f) {
                x_i.radiance += emitter_sample_record.radiance * f * NoL;
                // x_i.emission += emitter_sample_record.radiance * f * NoL;
                w_i = 1.f / emitter_sample_record.pdf;
            }
        }

        x_i.p_hat = optix::GetLuminance(x_i.radiance);
        x_i.emitter_rand = make_float3(r1, r2.x, r2.y);
        w_i *= x_i.p_hat;
        // x_i.emission = x_i.radiance;
        optix_launch_params.reservoirs[record->pixel_index].Update(x_i, w_i, record->random);
    }
}
extern "C" __global__ void __miss__shadow() {
    
}
extern "C" __global__ void __closesthit__shadow() {
    optixSetPayload_0(1u);
}
