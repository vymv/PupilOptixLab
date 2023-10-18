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
__constant__ ddgi::shading::OptixLaunchParams optix_launch_params;
}


extern "C" __global__ void __raygen__main() {
    const uint3 index = optixGetLaunchIndex();
    const unsigned int w = optix_launch_params.config.frame.width;
    const unsigned int h = optix_launch_params.config.frame.height;
    const unsigned int pixel_index = index.y * w + index.x;
    float3 emission = optix_launch_params.emission_buffer[pixel_index];

    if(emission.x > 0 || emission.y > 0 || emission.z > 0){
        optix_launch_params.frame_buffer[pixel_index] = make_float4(emission, 1.f);
        return;
    }

    auto &reservoir = optix_launch_params.final_reservoirs[pixel_index];
    float3 position = optix_launch_params.position_buffer[pixel_index];
    float3 ray_origin = reservoir.y.pos;
    float3 ray_direction = normalize(position - ray_origin);
    unsigned int occluded = 0;
    optixTrace(optix_launch_params.handle,
               ray_origin, ray_direction, 
               0.001f, 
               length(position - ray_origin) - 0.001f, 
               0.f, 
               255, 
               OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, 
               1, 2, 1, 
               occluded);

    float3 result  = make_float3(0.0f);
    if(!occluded) {
        float3 albedo = optix_launch_params.albedo_buffer[pixel_index];
        float3 normal = optix_launch_params.normal_buffer[pixel_index];
        float3 lightdir = position - reservoir.y.pos;
        float NdotL = dot(normal, -lightdir);
        
        if(NdotL > 0.0f) {
            result = reservoir.W * reservoir.y.radiance * albedo * M_1_PIf * NdotL;
        }
        // if (reservoir.y.sample_type == 0)
        //     result = make_float3(0,1,0);
        // else
        //     result = make_float3(1,0,0);
    }

    optix_launch_params.frame_buffer[pixel_index] = make_float4(result, 1.f);
}

extern "C" __global__ void __miss__default() {
    // auto record = optix::GetPRD<PathPayloadRecord>();
    // if (optix_launch_params.emitters.env) {
    //     // optix::LocalGeometry temp;
    //     // temp.position = optixGetWorldRayDirection();
    //     // float3 scatter_pos = make_float3(0.f);
    //     // auto env_emit_record = optix_launch_params.emitters.env->Eval(temp, scatter_pos);
    //     // record->env_radiance = env_emit_record.radiance;
    //     // record->env_pdf = env_emit_record.pdf;
    //     auto &env = *optix_launch_params.emitters.env.GetDataPtr();

    //     const auto ray_dir = normalize(optixGetWorldRayDirection());
    //     const auto ray_o = optixGetWorldRayOrigin();

    //     optix::LocalGeometry env_local;
    //     env_local.position = ray_o + ray_dir;
    //     optix::EmitEvalRecord emit_record;
    //     env.Eval(emit_record, env_local, ray_o);
    //     record->env_radiance = emit_record.radiance;
    //     record->env_pdf = emit_record.pdf;
    // }

    // record->miss = true;
}

extern "C" __global__ void __closesthit__default() {
    // const ddgi::render::HitGroupData *sbt_data = (ddgi::render::HitGroupData *)optixGetSbtDataPointer();
    // auto record = optix::GetPRD<PathPayloadRecord>();

    // const auto ray_dir = optixGetWorldRayDirection();
    // const auto ray_o = optixGetWorldRayOrigin();

    // sbt_data->geo.GetHitLocalGeometry(record->hit.geo, ray_dir, sbt_data->mat.twosided);
    // if (sbt_data->emitter_index_offset >= 0) {
    //     record->hit.emitter_index = sbt_data->emitter_index_offset + optixGetPrimitiveIndex();
    // } else {
    //     record->hit.emitter_index = -1;
    // }

    // record->hit.bsdf = sbt_data->mat.GetLocalBsdf(record->hit.geo.texcoord);
    // record->albedo = sbt_data->mat.GetColor(record->hit.geo.texcoord);
}
extern "C" __global__ void __miss__shadow() {
    
}
extern "C" __global__ void __closesthit__shadow() {
    optixSetPayload_0(1u);
}
