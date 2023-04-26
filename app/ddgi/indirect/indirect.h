
#pragma once
#include "cuda/data_view.h"
#include "cuda/vec_math.h"
#include "probemath.h"
#include <stdio.h>

// uniform
__device__ int3 getBaseGridCoord(float3 probeStartPosition, float3 probeStep, int3 probeCount, float3 X)
{
    return clamp(make_int3((X - probeStartPosition) / probeStep), make_int3(0, 0, 0), probeCount - make_int3(1));
}

// uniform
__device__ int gridCoordToProbeIndex(int3 probeCount, int3 probeCoords)
{
    return int(probeCoords.x + probeCoords.y * probeCount.x + probeCoords.z * probeCount.x * probeCount.y);
}

// uniform
__device__ float3 gridCoordToPosition(float3 probeStartPosition, float3 probeStep, int3 c)
{
    return probeStep * make_float3(c) + probeStartPosition;
}

__device__ int2 textureCoordFromDirection(float3 dir, int probeIndex, int fullTextureWidth, int fullTextureHeight,
                                          int probeSideLength)
{
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

    return make_int2(probeTopLeftPosition + octCoordNormalizedToTextureDimensions);
}

__device__ float3 ComputeIndirect(const float3 wsN, const float3 wsPosition, const float3 rayorigin,
                                  const float4 *probeirradiance, const float4 *probedepth, float3 probeStartPosition,
                                  float3 probeStep, int3 probeCount, uint2 probeirradiancesize, int probeSideLength,
                                  float energyConservation)
{

    const float epsilon = 1e-6;
    // gbuffer_WS_NORMAL_buffer
    // gbuffer_WS_POSITION_buffer
    // gbuffer_WS_RAY_ORIGIN_buffer
    // probe irradiance buffer

    if (dot(wsN, wsN) < 0.01)
    {
        return make_float3(0.0f);
    }

    int3 baseGridCoord = getBaseGridCoord(probeStartPosition, probeStep, probeCount, wsPosition);
    float3 baseProbePos = gridCoordToPosition(probeStartPosition, probeStep, baseGridCoord);

    float3 sumIrradiance = make_float3(0.0f);
    float sumWeight = 0.0f;

    //  alpha is how far from the floor(currentVertex) position. on [0, 1] for each axis.
    float3 alpha = clamp((wsPosition - baseProbePos) / probeStep, make_float3(0), make_float3(1));

    for (int i = 0; i < 8; ++i)
    {
        float weight = 1.0;
        int3 offset = make_int3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        int3 probeGridCoord = clamp(baseGridCoord + offset, make_int3(0), probeCount - make_int3(1));
        int probeIndex = gridCoordToProbeIndex(probeCount, probeGridCoord);
        float3 probePos = gridCoordToPosition(probeStartPosition, probeStep, probeGridCoord);

        // Smooth backface test
        {
            float3 trueDirectionToProbe = normalize(probePos - wsPosition);
            // weight *= max(0.0001, dot(trueDirectionToProbe, wsN));
            weight *= pow(max(0.0001, (dot(trueDirectionToProbe, wsN) + 1.0) * 0.5), 2) + 0.2;
        }

        // Moment visibility test (chebyshev)
        {
            float normalBias = 0.05f;
            float3 w_o = normalize(rayorigin - wsPosition);
            float3 probeToPoint = wsPosition - probePos + (wsN + 3.0 * w_o) * normalBias;
            float3 dir = normalize(-probeToPoint);
            int2 texCoord = textureCoordFromDirection(-dir, probeIndex, probeirradiancesize.x, probeirradiancesize.y,
                                                      probeSideLength);
            float4 temp = probedepth[texCoord.x + texCoord.y * probeirradiancesize.x];
            float mean = temp.x;
            float variance = abs(pow(temp.x, 2) - temp.y);

            float distToProbe = length(probeToPoint);
            float chebyshevWeight = variance / (variance + pow(max(distToProbe - mean, 0.0), 2));
            chebyshevWeight = max(pow(chebyshevWeight, 3), 0.0);

            weight *= (distToProbe <= mean) ? 1.0 : chebyshevWeight;
        }

        // Avoid zero
        weight = max(0.000001, weight);

        const float crushThreshold = 0.2;
        if (weight < crushThreshold)
        {
            weight *= weight * weight * (1.0 / pow(crushThreshold, 2));
        }

        // Trilinear
        float3 trilinear = (1.0 - alpha) * (1 - make_float3(offset)) + alpha * make_float3(offset);
        weight *= trilinear.x * trilinear.y * trilinear.z;

        int2 texCoord = textureCoordFromDirection(normalize(wsN), probeIndex, probeirradiancesize.x,
                                                  probeirradiancesize.y, probeSideLength);
        float4 irradiance = probeirradiance[texCoord.x + texCoord.y * probeirradiancesize.x];

        sumIrradiance += weight * make_float3(irradiance.x, irradiance.y, irradiance.z);
        sumWeight += weight;
    }

    float3 netIrradiance = sumIrradiance / sumWeight;
    netIrradiance *= energyConservation;
    float3 indirect = 2.0 * M_PIf * netIrradiance;
    // indirect = make_float3(0.5f);
    // printf("%f,%f,%f", indirect.x, indirect.y, indirect.z);
    //     return indirect;
    return indirect;
}