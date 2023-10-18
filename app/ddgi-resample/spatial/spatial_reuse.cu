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
__constant__ ddgi::spatial::OptixLaunchParams optix_launch_params;
}

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

    cuda::Random random;
    random.Init(4, pixel_index, optix_launch_params.random_seed);
    float3 position = optix_launch_params.position_buffer[pixel_index];
    auto &camera = *optix_launch_params.camera.GetDataPtr();
    float3 camera_pos = make_float3(
        camera.camera_to_world.r0.w,
        camera.camera_to_world.r1.w,
        camera.camera_to_world.r2.w);

    Reservoir reservoir;
    reservoir.Init();
    unsigned int M = 0;
    for (auto i = 0u; i < 5; ++i) {
        float r = optix_launch_params.spatial_radius * random.Next();
        float theta = M_PIf * 2.f * random.Next();
        // 随机一个方向的neighbour pixel
        int2 neighbor_pixel = make_int2(index.x + r * cos(theta), index.y + r * sin(theta));
        if (neighbor_pixel.x < 0 || neighbor_pixel.x >= w || neighbor_pixel.y < 0 || neighbor_pixel.y >= h)
            continue;
        const unsigned int neighbor_pixel_index = neighbor_pixel.y * w + neighbor_pixel.x;

        // 取出邻居neighbor
        auto &neighbor_reservoir = optix_launch_params.reservoirs[neighbor_pixel_index];
        Reservoir::Sample x_i = neighbor_reservoir.y;

        // 构建新的样本
        // w_i = p_hat / p_n_hat * w_n_sum
        //     = p_hat * M_n * w_n_sum / (p_n_hat * M_n)
        //     = p_hat * M_n * W_n
        if((x_i.emitter_rand.x < 0.0f) && (x_i.emitter_rand.y < 0.0f) && (x_i.emitter_rand.z < 0.0f)){
            float3 Lddgi = ComputeIndirect(normalize(position - x_i.pos),
                                                    x_i.pos, camera_pos,
                                                    optix_launch_params.probeirradiance.GetDataPtr(),
                                                    optix_launch_params.probedepth.GetDataPtr(),
                                                    optix_launch_params.probeStartPosition,
                                                    optix_launch_params.probeStep,
                                                    optix_launch_params.probeCount,
                                                    optix_launch_params.probeirradiancesize,
                                                    optix_launch_params.probeSideLength,
                                                    1.0f);
            Lddgi = Lddgi * x_i.albedo * M_1_PIf;
            x_i.radiance = Lddgi;
            x_i.p_hat = optix::GetLuminance(Lddgi);
            x_i.sample_type = 1;
            

        }else{
            auto &emitter = optix_launch_params.emitters.SelectOneEmiiter(x_i.emitter_rand.x);
            optix::EmitterSampleRecord emitter_sample_record;
            optix::LocalGeometry primary_local_hit;
            primary_local_hit.position = position;
            emitter.SampleDirect(emitter_sample_record, primary_local_hit, make_float2(x_i.emitter_rand.y,x_i.emitter_rand.z));
            x_i.radiance = emitter_sample_record.radiance;
            x_i.p_hat = optix::GetLuminance(emitter_sample_record.radiance);
            x_i.sample_type = 1;
        }
           

        float w_i = x_i.p_hat * neighbor_reservoir.M * neighbor_reservoir.W;
        reservoir.Update(x_i, w_i, random);
        M += neighbor_reservoir.M - 1;  
    }
    reservoir.M = M;
    optix_launch_params.final_reservoirs[pixel_index] = optix_launch_params.reservoirs[pixel_index];
    reservoir.CalcW();
    if (reservoir.W > 0.f) {
        optix_launch_params.final_reservoirs[pixel_index].Combine(reservoir, random);
    }

}

