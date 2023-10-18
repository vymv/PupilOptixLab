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

struct HitInfo {
    optix::LocalGeometry geo;
    optix::material::Material::LocalBsdf bsdf;
    int emitter_index;
};

struct PathPayloadRecord {
    float3 radiance;
    float3 env_radiance;
    float env_pdf;
    cuda::Random random;

    float3 throughput;

    HitInfo hit;
    float3 albedo;

    unsigned int depth;
    bool miss;
};

// uniform
__device__ int3 getBaseGridCoord(float3 probeStartPosition, float3 probeStep, int3 probeCount, float3 X) {
    return clamp(make_int3((X - probeStartPosition) / probeStep), make_int3(0, 0, 0), probeCount - make_int3(1));
}

// uniform
__device__ int gridCoordToProbeIndex(int3 probeCount, int3 probeCoords) {
    return int(probeCoords.x + probeCoords.y * probeCount.x + probeCoords.z * probeCount.x * probeCount.y);
    // return int(probeCoords.x * probeCount.y * probeCount.z + probeCoords.y * probeCount.z + probeCoords.z);
}

// uniform
__device__ float3 gridCoordToPosition(float3 probeStartPosition, float3 probeStep, int3 c) {
    return probeStep * make_float3(c) + probeStartPosition;
}

__device__ float4 bilinearInterpolation(const float4 *textureData,float2 texCoord, int fullTextureWidth, int fullTextureHeight){
    
    int x0 = floor(texCoord.x);
    int y0 = floor(texCoord.y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float u = texCoord.x - x0;
    float v = texCoord.y - y0;

    x0 = clamp(x0, 0, fullTextureWidth - 1);
    y0 = clamp(y0, 0, fullTextureHeight - 1);
    x1 = clamp(x1, 0, fullTextureWidth - 1);
    y1 = clamp(y1, 0, fullTextureHeight - 1);

    int index00 = y0 * fullTextureWidth + x0;
    int index01 = y0 * fullTextureWidth + x1;
    int index10 = y1 * fullTextureWidth + x0;
    int index11 = y1 * fullTextureWidth + x1;

    float4 pixel00 = textureData[index00];
    float4 pixel01 = textureData[index01];
    float4 pixel10 = textureData[index10];
    float4 pixel11 = textureData[index11];

    float4 result = (1 - u) * (1 - v) * pixel00 +
                u * (1 - v) * pixel01 +
                (1 - u) * v * pixel10 +
                u * v * pixel11;
    return result;

}

__device__ float2 textureCoordFromDirection(float3 dir, int probeIndex, int fullTextureWidth, int fullTextureHeight,
                                          int probeSideLength) {
    float2 normalizedOctCoord = octEncode(normalize(dir));
    float2 normalizedOctCoordZeroOne = (normalizedOctCoord + make_float2(1.0f)) * 0.5f;

    // Length of a probe side, plus one pixel on each edge for the border
    float probeWithBorderSide = (float)probeSideLength + 2.0f;

    float2 octCoordNormalizedToTextureDimensions = normalizedOctCoordZeroOne * (float)probeSideLength;
    int probesPerRow = (fullTextureWidth - 2) / (int)probeWithBorderSide;

    // Add (2,2) back to texCoord within larger texture. Compensates for 1 pix
    // border around texture and further 1 pix border around top left probe.
    float2 probeTopLeftPosition = make_float2((probeIndex % probesPerRow) * probeWithBorderSide,
                                              (probeIndex / probesPerRow) * probeWithBorderSide) +
                                  make_float2(2.0f);

    return probeTopLeftPosition + octCoordNormalizedToTextureDimensions;
}


__device__ float3 ComputeIndirectDiffuse(const float3 lightDir, const float3 wsPosition, const float3 viewPosition,
                                  const float4 *probeirradiance, const float4 *probedepth, float3 probeStartPosition,
                                  float3 probeStep, int3 probeCount, uint2 probeirradiancesize, int probeSideLength,
                                  float energyConservation) {

    const float epsilon = 1e-6;

    if (dot(lightDir, lightDir) < 0.01) {
        return make_float3(0.0f);
    }

    int3 baseGridCoord = getBaseGridCoord(probeStartPosition, probeStep, probeCount, wsPosition);
    float3 baseProbePos = gridCoordToPosition(probeStartPosition, probeStep, baseGridCoord);

    float3 sumIrradiance = make_float3(0.0f);
    float sumWeight = 0.0f;

    //  alpha is how far from the floor(currentVertex) position. on [0, 1] for each axis.
    float3 alpha = clamp((wsPosition - baseProbePos) / probeStep, make_float3(0), make_float3(1));

    for (int i = 0; i < 8; ++i) {
        float weight = 1.0;
        int3 offset = make_int3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        int3 probeGridCoord = clamp(baseGridCoord + offset, make_int3(0), probeCount - make_int3(1));
        int probeIndex = gridCoordToProbeIndex(probeCount, probeGridCoord);
        float3 probePos = gridCoordToPosition(probeStartPosition, probeStep, probeGridCoord);
        
        // // Smooth backface test
        // {
        //     float3 trueDirectionToProbe = normalize(probePos - wsPosition);
        //     // weight *= max(0.0001, dot(trueDirectionToProbe, wsN));
        //     weight *= pow(max(0.0001f, (dot(trueDirectionToProbe, wsN) + 1.0) * 0.5), 2) + 0.2;
        // }

        // // Moment visibility test (chebyshev)
        // {
        //     float normalBias = 0.05f;
        //     float3 w_o = normalize(viewPosition - wsPosition); 
        //     float3 probeToPoint = wsPosition - probePos + (wsN + 3.0 * w_o) * normalBias;
     
        //     float3 dir = normalize(probeToPoint);
        //     float2 texCoord = textureCoordFromDirection(dir, probeIndex, probeirradiancesize.x, probeirradiancesize.y,
        //                                               probeSideLength);
        //     // float4 temp = probedepth[texCoord.x + texCoord.y * probeirradiancesize.x];
        //     float4 temp = bilinearInterpolation(probedepth, texCoord, probeirradiancesize.x, probeirradiancesize.y);
        //     float mean = temp.x;
        //     float variance = abs(pow(temp.x, 2) - temp.y);

        //     float distToProbe = length(probeToPoint);
        //     float chebyshevWeight = variance / (variance + pow(max(distToProbe - mean, 0.0), 2));
        //     chebyshevWeight = max(pow(chebyshevWeight, 3), 0.0f);

        //     weight *= (distToProbe <= mean) ? 1.0 : chebyshevWeight;
        // }

        // // Avoid zero
        // weight = max(0.000001, weight);

        // const float crushThreshold = 0.2;
        // if (weight < crushThreshold) {
        //     weight *= weight * weight * (1.0 / pow(crushThreshold, 2));
        // }

        // Trilinear
        float3 trilinear = (1.0 - alpha) * (1 - make_float3(offset)) + alpha * make_float3(offset);
        weight *= trilinear.x * trilinear.y * trilinear.z;

        float2 texCoord = textureCoordFromDirection(normalize(-lightDir), probeIndex, probeirradiancesize.x,
                                                  probeirradiancesize.y, probeSideLength);

        // float4 irradiance = probeirradiance[texCoord.x + texCoord.y * probeirradiancesize.x];
        float4 irradiance = bilinearInterpolation(probeirradiance, texCoord, probeirradiancesize.x, probeirradiancesize.y);

        sumIrradiance += weight * make_float3(irradiance.x, irradiance.y, irradiance.z);
        sumWeight += weight;
    }

    float3 netIrradiance = sumIrradiance / sumWeight;
    netIrradiance *= energyConservation;
    float3 indirect = 2.0 * M_PIf * netIrradiance;

    return indirect;
}

__device__ float3 ComputeIndirect(const float3 lightDir, const float3 wsPosition, const float3 rayorigin,
                                  const float4 *probeirradiance, const float4 *probedepth, float3 probeStartPosition,
                                  float3 probeStep, int3 probeCount, uint2 probeirradiancesize, int probeSideLength,
                                  float energyConservation) {

    return ComputeIndirectDiffuse(lightDir, wsPosition, rayorigin, probeirradiance, probedepth, probeStartPosition, probeStep, probeCount, probeirradiancesize, probeSideLength, energyConservation);
}

extern "C" __global__ void __raygen__main() {
    const uint3 index = optixGetLaunchIndex();
    const unsigned int w = optix_launch_params.config.frame.width;
    const unsigned int h = optix_launch_params.config.frame.height;
    const unsigned int pixel_index = index.y * w + index.x;
        auto &camera = *optix_launch_params.camera.GetDataPtr();

    PathPayloadRecord record{};
    uint32_t u0, u1;
    optix::PackPointer(&record, u0, u1);
    float3 result = make_float3(0.f);

    record.miss = false;
    //record.depth = 0u;
    record.throughput = make_float3(1.f);
    record.radiance = make_float3(0.f);
    record.env_radiance = make_float3(0.f);
    record.random.Init(4, pixel_index, optix_launch_params.random_seed);

    const float2 subpixel_jitter = make_float2(record.random.Next(), record.random.Next());
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

    float3 camera_pos = make_float3(
        camera.camera_to_world.r0.w,
        camera.camera_to_world.r1.w,
        camera.camera_to_world.r2.w);

    optixTrace(optix_launch_params.handle,
               camera_pos, ray_direction,
               0.001f, 1e16f, 0.f,
               255, OPTIX_RAY_FLAG_NONE,
               0, 2, 0,
               u0, u1);

    auto primary_local_hit = record.hit;
    float3 primary_albedo = record.albedo;
    
    if(record.miss){
        result = record.env_radiance;
        optix_launch_params.frame_buffer[pixel_index] = make_float4(result, 1.f);
        return;
    }
    if (record.hit.emitter_index >= 0) {
        auto &emitter = optix_launch_params.emitters.areas[primary_local_hit.emitter_index];
        auto emission = emitter.GetRadiance(primary_local_hit.geo.texcoord);
        result = emission;

        optix_launch_params.frame_buffer[pixel_index] = make_float4(result, 1.f);
        optix_launch_params.albedo_buffer[pixel_index] = primary_albedo;
        optix_launch_params.position_buffer[pixel_index] = primary_local_hit.geo.position;
        optix_launch_params.normal_buffer[pixel_index] = primary_local_hit.geo.normal;
        optix_launch_params.emission_buffer[pixel_index] = emission;
        return;
    }
    optix_launch_params.reservoirs[pixel_index].Init();
    

    // 1. Generate candidate
    // 1.1 Emissive surface
    for(int i = 0; i < optix_launch_params.num_emission; i++){

        // Generate emitter candidate
        float r = record.random.Next();
        float2 r2 = record.random.Next2();
        auto &emitter = optix_launch_params.emitters.SelectOneEmiiter(r);
        optix::EmitterSampleRecord emitter_sample_record;
        emitter.SampleDirect(emitter_sample_record, record.hit.geo, r2);
        
        // Update reservior
        Reservoir::Sample x_i;
        x_i.pos = emitter_sample_record.pos;
        x_i.normal = emitter_sample_record.normal;
        x_i.emitter_rand = make_float3(r, r2);
        x_i.radiance = emitter_sample_record.radiance;
        x_i.sample_type = 0;
        float w_i = 0.0;
        // p
        if (emitter_sample_record.pdf > 0.f) {
            w_i = 1.0 / emitter_sample_record.pdf;
        }
        // p_hat  Le + Lddgi
        x_i.p_hat = optix::GetLuminance(emitter_sample_record.radiance);
        w_i = w_i * x_i.p_hat;
        
        optix_launch_params.reservoirs[pixel_index].Update(x_i, w_i, record.random);
        // printf("%f\n",optix_launch_params.reservoirs[pixel_index].w_sum);
    }

    // 1.2 Secondary vertex (bsdf sampling)
    for(int i = 0; i < optix_launch_params.num_secondary; i++)
    {
        // Generate secondary candidate
        float3 wo = optix::ToLocal(-ray_direction, primary_local_hit.geo.normal);
        optix::BsdfSamplingRecord bsdf_sample_record;
        bsdf_sample_record.wo = optix::ToLocal(-ray_direction, primary_local_hit.geo.normal);
        bsdf_sample_record.sampler = &record.random;
        record.hit.bsdf.Sample(bsdf_sample_record);

        float3 ray_origin = record.hit.geo.position;
        ray_direction = optix::ToWorld(bsdf_sample_record.wi, primary_local_hit.geo.normal);
        record.miss = false;

        optixTrace(optix_launch_params.handle,
                    ray_origin, ray_direction,
                    0.001f, 1e16f, 0.f,
                    255, OPTIX_RAY_FLAG_NONE,
                    0, 2, 0,
                    u0, u1);

        // Update reservior
        Reservoir::Sample x_i;
        float w_i = 0.0f;
        if (record.hit.emitter_index < 0 && !record.miss) {
            float3 Lddgi = ComputeIndirect(normalize(primary_local_hit.geo.position - record.hit.geo.position),
                                            record.hit.geo.position, camera_pos,
                                            optix_launch_params.probeirradiance.GetDataPtr(),
                                            optix_launch_params.probedepth.GetDataPtr(),
                                            optix_launch_params.probeStartPosition,
                                            optix_launch_params.probeStep,
                                            optix_launch_params.probeCount,
                                            optix_launch_params.probeirradiancesize,
                                            optix_launch_params.probeSideLength,
                                            1.0f);
            
            // bsdf
            // optix::BsdfSamplingRecord eval_record;
            // eval_record.wi = optix::ToLocal(record.hit.geo.normal, record.hit.geo.normal);
            // eval_record.wo = optix::ToLocal(record.hit.geo.normal, record.hit.geo.normal);
            // eval_record.sampler = &record.random;
            // record.hit.bsdf.Eval(eval_record);
            // float3 f = eval_record.f;
            Lddgi = Lddgi * record.albedo * M_1_PIf;

            x_i.pos = record.hit.geo.position;
            x_i.normal = record.hit.geo.normal;
            x_i.albedo = record.albedo;
            x_i.emitter_rand = make_float3(-1.0f);
            x_i.radiance = Lddgi;
            x_i.p_hat = optix::GetLuminance(Lddgi);
            x_i.sample_type = 0;

            // p
            if(bsdf_sample_record.pdf > 0.f){
                w_i = 1.0 / bsdf_sample_record.pdf;
            }
            // p_hat  Le + Lddgi 
            w_i = w_i * x_i.p_hat;
        }
        optix_launch_params.reservoirs[pixel_index].Update(x_i, w_i, record.random);
    }

    optix_launch_params.reservoirs[pixel_index].CalcW();

    // // 2. Spatial reuse
    // Reservoir reservoir;
    // reservoir.Init();
    // unsigned int M = 0;
    // for (auto i = 0u; i < 5; ++i) {
    //     float r = optix_launch_params.spatial_radius * record.random.Next();
    //     float theta = M_PIf * 2.f * record.random.Next();
    //     // 随机一个方向的neighbour pixel
    //     int2 neighbor_pixel = make_int2(index.x + r * cos(theta), index.y + r * sin(theta));
    //     if (neighbor_pixel.x < 0 || neighbor_pixel.x >= w || neighbor_pixel.y < 0 || neighbor_pixel.y >= h)
    //         continue;
    //     const unsigned int neighbor_pixel_index = neighbor_pixel.y * w + neighbor_pixel.x;

    //     // 取出邻居neighbor
    //     auto &neighbor_reservoir = optix_launch_params.reservoirs[neighbor_pixel_index];
    //     Reservoir::Sample x_i = neighbor_reservoir.y;

    //     // 构建新的样本
    //     // w_i = p_hat / p_n_hat * w_n_sum
    //     //     = p_hat * M_n * w_n_sum / (p_n_hat * M_n)
    //     //     = p_hat * M_n * W_n
    //     if((x_i.emitter_rand.x < 0.0f) && (x_i.emitter_rand.y < 0.0f) && (x_i.emitter_rand.z < 0.0f)){
    //         float3 Lddgi = ComputeIndirect(normalize(primary_local_hit.geo.position - x_i.pos),
    //                                                 x_i.pos, camera_pos,
    //                                                 optix_launch_params.probeirradiance.GetDataPtr(),
    //                                                 optix_launch_params.probedepth.GetDataPtr(),
    //                                                 optix_launch_params.probeStartPosition,
    //                                                 optix_launch_params.probeStep,
    //                                                 optix_launch_params.probeCount,
    //                                                 optix_launch_params.probeirradiancesize,
    //                                                 optix_launch_params.probeSideLength,
    //                                                 1.0f);
    //         x_i.radiance = Lddgi;
    //         x_i.p_hat = optix::GetLuminance(Lddgi);
    //         Lddgi = Lddgi * x_i.albedo * M_1_PIf;

    //     }else{
    //         auto &emitter = optix_launch_params.emitters.SelectOneEmiiter(x_i.emitter_rand.x);
    //         optix::EmitterSampleRecord emitter_sample_record;
    //         emitter.SampleDirect(emitter_sample_record, primary_local_hit.geo, make_float2(x_i.emitter_rand.y,x_i.emitter_rand.z));
    //         x_i.radiance = emitter_sample_record.radiance;
    //         x_i.p_hat = optix::GetLuminance(emitter_sample_record.radiance);
            
    //     }
           

    //     float w_i = x_i.p_hat * neighbor_reservoir.M * neighbor_reservoir.W;
    //     reservoir.Update(x_i, w_i, record.random);
    //     M += neighbor_reservoir.M - 1;  
    // }
    // reservoir.M = M;
    // reservoir.CalcW();
    // if (reservoir.W > 0.f) {
    //     optix_launch_params.reservoirs[pixel_index].Combine(reservoir, record.random);
    // }




    optix_launch_params.frame_buffer[pixel_index] = make_float4(result, 1.f);
    optix_launch_params.albedo_buffer[pixel_index] = primary_albedo;
    optix_launch_params.position_buffer[pixel_index] = primary_local_hit.geo.position;
    optix_launch_params.normal_buffer[pixel_index] = primary_local_hit.geo.normal;
    optix_launch_params.emission_buffer[pixel_index] = make_float3(0.f);
}

extern "C" __global__ void __miss__default() {
    auto record = optix::GetPRD<PathPayloadRecord>();
    if (optix_launch_params.emitters.env) {
        // optix::LocalGeometry temp;
        // temp.position = optixGetWorldRayDirection();
        // float3 scatter_pos = make_float3(0.f);
        // auto env_emit_record = optix_launch_params.emitters.env->Eval(temp, scatter_pos);
        // record->env_radiance = env_emit_record.radiance;
        // record->env_pdf = env_emit_record.pdf;
        auto &env = *optix_launch_params.emitters.env.GetDataPtr();

        const auto ray_dir = normalize(optixGetWorldRayDirection());
        const auto ray_o = optixGetWorldRayOrigin();

        optix::LocalGeometry env_local;
        env_local.position = ray_o + ray_dir;
        optix::EmitEvalRecord emit_record;
        env.Eval(emit_record, env_local, ray_o);
        record->env_radiance = emit_record.radiance;
        record->env_pdf = emit_record.pdf;
    }

    record->miss = true;
}

extern "C" __global__ void __closesthit__default() {
    const ddgi::render::HitGroupData *sbt_data = (ddgi::render::HitGroupData *)optixGetSbtDataPointer();
    auto record = optix::GetPRD<PathPayloadRecord>();

    const auto ray_dir = optixGetWorldRayDirection();
    const auto ray_o = optixGetWorldRayOrigin();

    sbt_data->geo.GetHitLocalGeometry(record->hit.geo, ray_dir, sbt_data->mat.twosided);
    if (sbt_data->emitter_index_offset >= 0) {
        record->hit.emitter_index = sbt_data->emitter_index_offset + optixGetPrimitiveIndex();
    } else {
        record->hit.emitter_index = -1;
    }

    record->hit.bsdf = sbt_data->mat.GetLocalBsdf(record->hit.geo.texcoord);
    record->albedo = sbt_data->mat.GetColor(record->hit.geo.texcoord);
}
extern "C" __global__ void __miss__shadow() {
    
}
extern "C" __global__ void __closesthit__shadow() {
    optixSetPayload_0(1u);
}
