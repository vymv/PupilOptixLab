#include "type.h"
#include <optix.h>

// #include "../indirect/indirect.h"
#include "../indirect/probemath.h"
#include "render/geometry.h"
#include "render/emitter.h"
#include "render/material/optix_material.h"
#include "optix/util.h"

#include "cuda/random.h"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

using namespace Pupil;

extern "C" {
__constant__ ddgi::probe::OptixLaunchParams optix_launch_params;
}
struct Mat3 {

    float re[4][4];
};
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
    HitInfo hit;
    float3 albedo;
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
    // TODO bilinear filtering
    return probeTopLeftPosition + octCoordNormalizedToTextureDimensions;
}

__device__ float3 ComputeIndirect(const float3 lightDir, const float3 wsPosition, const float3 probePos,
                                  const float4 *probeirradiance, const float4 *probedepth, float3 probeStartPosition,
                                  float3 probeStep, int3 probeCount, uint2 probeirradiancesize, int probeSideLength,
                                  float energyConservation) {

    const float epsilon = 1e-6;

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
        //     float3 w_o = normalize(rayorigin - wsPosition); // 相机位置-世界坐标位置
        //     float3 probeToPoint = wsPosition - probePos + (wsN + 3.0 * w_o) * normalBias;
            
        //     float2 texCoord = textureCoordFromDirection(normalize(probeToPoint), probeIndex, probeirradiancesize.x, probeirradiancesize.y,
        //                                               probeSideLength);
        //     float4 temp = bilinearInterpolation(probedepth, texCoord, probeirradiancesize.x, probeirradiancesize.y);
        //     // float4 temp = probedepth[texCoord.x + texCoord.y * probeirradiancesize.x];
        //     float mean = temp.x;
        //     float variance = fabs(pow(temp.x, 2) - temp.y);

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
        float4 irradiance = bilinearInterpolation(probeirradiance, texCoord, probeirradiancesize.x, probeirradiancesize.y);

        sumIrradiance += weight * make_float3(irradiance.x, irradiance.y, irradiance.z);
        sumWeight += weight;
    }

    float3 netIrradiance = sumIrradiance / sumWeight;
    netIrradiance *= energyConservation;
    float3 indirect = 2.0 * M_PIf * netIrradiance;

    return indirect;
}

__device__ float3 sphericalFibonacci(float i, float n) {
    const float PHI = sqrt(5.0) * 0.5f + 0.5f;

    float madfrac = (i * (PHI - 1)) - floor(i * (PHI - 1));
    float phi = 2.0f * M_PIf * (madfrac < 0.0f ? 0.0 : (madfrac > 1.0f ? 1.0 : madfrac));
    float cosTheta = 1.0f - (2.0f * i + 1.0f) * (1.0f / n);
    float sinTheta = sqrt(1.0f - cosTheta * cosTheta < 0.0f ? 0.0 : (1.0f - cosTheta * cosTheta > 1.0f ? 1.0 : 1.0f - cosTheta * cosTheta));

    return make_float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
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


__device__ float getClampInvertDistance(float3 origin, float3 probeStep, float3 dir){

    float3 faceNormals[6] = {
        make_float3(1, 0, 0),  // +X 面的法向量
        make_float3(-1, 0, 0), // -X 面的法向量
        make_float3(0, 1, 0),  // +Y 面的法向量
        make_float3(0, -1, 0), // -Y 面的法向量
        make_float3(0, 0, 1),  // +Z 面的法向量
        make_float3(0, 0, -1)  // -Z 面的法向量
    };  

    float minDistance = 1e16f;
    for (int i = 0; i < 6; i++) {
        float3 planeNormal = faceNormals[i];
        float step = 0;
        if(i >= 0 && i < 2)
            step = probeStep.x;
        else if (i >= 2 && i < 4)
            step = probeStep.y;
        else if (i >= 4 && i < 6)
            step = probeStep.z;

        float3 planePoint = origin + planeNormal * 0.5 * step;

        float distance = length(planePoint - origin) / dot(planePoint - origin, dir);
        if(distance < minDistance)
            minDistance = distance;
    }


    return minDistance;
}

extern "C" __global__ void __raygen__main() {

    const uint3 index = optixGetLaunchIndex();
    const unsigned int w = optix_launch_params.config.frame.width;
    // const unsigned int h = optix_launch_params.config.frame.height;
    const unsigned int pixel_index = index.y * w + index.x;
    const unsigned int rayid = index.x;
    const unsigned int probeid = index.y;

    PathPayloadRecord record{};
    uint32_t u0, u1;
    optix::PackPointer(&record, u0, u1);
    record.random.Init(4, pixel_index, optix_launch_params.random_seed);

    float3 probePosition = optix_launch_params.probepos[probeid]; // probe position

    // identity
    auto &m = *optix_launch_params.randomOrientation.operator->();

    float3 sf = sphericalFibonacci(float(rayid), float(w));

    float3 ray_direction = make_float3(m.re[0][0] * sf.x + m.re[0][1] * sf.y + m.re[0][2] * sf.z,
                                       m.re[1][0] * sf.x + m.re[1][1] * sf.y + m.re[1][2] * sf.z,
                                       m.re[2][0] * sf.x + m.re[2][1] * sf.y + m.re[2][2] * sf.z);

    optixTrace(optix_launch_params.handle, probePosition, ray_direction, 1e-5f, 1e16f, 0.f, 255,
               OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, 0, 2, 0, u0, u1);

    float3 result = make_float3(0.f);
    auto local_hit = record.hit;
    if (record.hit.emitter_index >= 0) {
        auto &emitter = optix_launch_params.emitters.areas[local_hit.emitter_index];
        auto emission = emitter.GetRadiance(local_hit.geo.texcoord);
        result = emission;
        // // attenuation
        // float distance = length(probePosition - local_hit.geo.position);

        // // invert distance
        // float3 probeToPoint = local_hit.geo.position - probePosition;
        // float2 texCoord = textureCoordFromDirection(normalize(-probeToPoint), probeid, optix_launch_params.probeirradiancesize.x, optix_launch_params.probeirradiancesize.y,
        //                                               optix_launch_params.probeSideLength);
        // float4 temp = bilinearInterpolation(optix_launch_params.probedepth.GetDataPtr(), texCoord, optix_launch_params.probeirradiancesize.x, optix_launch_params.probeirradiancesize.y);
        // float mean = temp.x;

        // float invertDistance = mean;
        // float maxInvertDistance = getClampInvertDistance(probePosition, optix_launch_params.probeStep, normalize(-probeToPoint));
        // if(mean > maxInvertDistance){
        //     invertDistance = maxInvertDistance;
        // }
        
        // if(distance + invertDistance < 1e-3){
        //     result = emission;
        // }else{
        //     result = emission / pow(distance + invertDistance, 2.0);
        // }

        optix_launch_params.rayradiance[pixel_index] = make_float4(result, 1.f);
        optix_launch_params.rayhitposition[pixel_index] = record.hit.geo.position;
        optix_launch_params.rayhitnormal[pixel_index] = record.hit.geo.normal;
        optix_launch_params.raydirection[pixel_index] = ray_direction;
        return;
    }

    result += record.env_radiance;

    // diffuse indirect
    float3 indirectlight = ComputeIndirect(normalize(probePosition - record.hit.geo.position), record.hit.geo.position, probePosition,
        optix_launch_params.probeirradiance.GetDataPtr(), optix_launch_params.probedepth.GetDataPtr(),
        optix_launch_params.probeStartPosition, optix_launch_params.probeStep, optix_launch_params.probeCount,
        optix_launch_params.probeirradiancesize, optix_launch_params.probeSideLength, 0.95f);
    
    // bsdf
    // optix::BsdfSamplingRecord eval_record;
    // eval_record.wi = optix::ToLocal(local_hit.geo.normal, local_hit.geo.normal);
    // eval_record.wo = optix::ToLocal(local_hit.geo.normal, local_hit.geo.normal);
    // eval_record.sampler = &record.random;
    // local_hit.bsdf.Eval(eval_record);
    // float3 f = eval_record.f;
    result = indirectlight * record.albedo * M_1_PIf;
 
    optix_launch_params.rayradiance[pixel_index] = make_float4(result, 1.f);
    optix_launch_params.rayhitposition[pixel_index] = record.hit.geo.position;
    optix_launch_params.rayhitnormal[pixel_index] = record.hit.geo.normal;
    optix_launch_params.raydirection[pixel_index] = ray_direction;
}

extern "C" __global__ void __miss__default() {

    auto record = optix::GetPRD<PathPayloadRecord>();
    if (optix_launch_params.emitters.env) {
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
    record->hit.emitter_index = -1;
    optix::material::Material::LocalBsdf local_bsdf;
    local_bsdf.type = EMatType::Unknown;
    record->hit.bsdf = local_bsdf;
}
extern "C" __global__ void __closesthit__default() {

    const ddgi::probe::HitGroupData *sbt_data = (ddgi::probe::HitGroupData *)optixGetSbtDataPointer();
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

// 只是返回一个bool值，返回1表示被遮挡了
extern "C" __global__ void __closesthit__shadow() {
    optixSetPayload_0(1u);
}

extern "C" __global__ void __miss__shadow() {
    // optixSetPayload_0(0u);
}
