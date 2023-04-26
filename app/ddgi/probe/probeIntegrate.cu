#include "../indirect/probemath.h"
#include "cuda/vec_math.h"
#include "probeIntegrate.h"

__device__ float2 normalizedOctCoord(int2 fragCoord, int probeSideLength)
{
    int probeWithBorderSide = probeSideLength + 2;

    float2 octFragCoord = make_float2((fragCoord.x - 2) % probeWithBorderSide, (fragCoord.y - 2) % probeWithBorderSide);
    // Add back the half pixel to get pixel center normalized coordinates
    return (octFragCoord + make_float2(0.5)) * (2.0f / float(probeSideLength)) - make_float2(1.0f);
}

__global__ void ChangeAlpha(float4 *probeirradiance_show, const float4 *probeirradiance, uint2 size)
{
    int pixel_x = threadIdx.x + blockIdx.x * blockDim.x;
    int pixel_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (pixel_x >= size.x)
        return;

    if (pixel_y >= size.y)
        return;
    int pixel_index = pixel_x + size.x * pixel_y;
    probeirradiance_show[pixel_index] = make_float4(probeirradiance[pixel_index].x, probeirradiance[pixel_index].y,
                                                    probeirradiance[pixel_index].z, 1.0);
}
// probeRayGbuffer -> probeTexture
__global__ void UpdateProbe(float4 *probeirradiance, float4 *probedepth, const float4 *raygbuffer,
                            const float3 *rayorigin, const float3 *raydirection, const float3 *rayhitposition,
                            const float3 *rayhitnormal, uint2 size, int raysPerProbe, int probeSideLength,
                            float maxDistance, float hysteresis, float depthSharpness, bool irradiance)
{
    // __global__ void UpdateProbe(float4 *probeirradiance, const float4 *raygbuffer, const float3 *rayorigin,
    //                             const float3 *raydirection, const float3 *rayhitposition, const float3 *rayhitnormal,
    //                             uint2 size, int raysPerProbe, int probeSideLength, float maxDistance, float
    //                             hysteresis, float depthSharpness, bool irradiance)
    // {

    float4 *output = nullptr;
    if (irradiance)
        output = probeirradiance;
    else
        output = probedepth;

    const float epsilon = 1e-6;
    int pixel_x = threadIdx.x + blockIdx.x * blockDim.x;
    int pixel_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (pixel_x >= size.x)
        return;

    if (pixel_y >= size.y)
        return;
    int pixel_index = pixel_x + size.x * pixel_y;

    const float energyConservation = 0.95f;

    float4 result = make_float4(0.0f);
    float3 oldvaule = make_float3(output[pixel_index].x, output[pixel_index].y, output[pixel_index].z);
    float3 newvaule = make_float3(0.0f);

    // 计算probeId
    int probeWithBorderSide = probeSideLength + 2;
    int probesPerSide = (size.x - 2) / probeWithBorderSide;
    int probeId = int(pixel_x / probeWithBorderSide) + probesPerSide * int(pixel_y / probeWithBorderSide);

    if (probeId == -1)
    {
        output[pixel_index] = make_float4(0.0f);
        return;
    }

    // 计算direction
    float3 texelDirection = octDecode(normalizedOctCoord(make_int2(pixel_x, pixel_y), probeSideLength));
    for (int r = 0; r < raysPerProbe; ++r)
    {

        // 取出参数
        int2 coord = make_int2(r, probeId);
        int rayIndex = raysPerProbe * coord.y + coord.x;
        float3 rayHitRadiance =
            make_float3(raygbuffer[rayIndex].x, raygbuffer[rayIndex].y, raygbuffer[rayIndex].z) * energyConservation;
        float3 rayDirection = raydirection[rayIndex];
        float3 rayHitLocation = rayhitposition[rayIndex];
        float3 probeLocation = rayorigin[rayIndex];
        float3 rayHitNormal = rayhitnormal[rayIndex];

        // 计算距离
        rayHitLocation = rayHitLocation + rayHitNormal * 0.01f;
        float rayProbeDistance = min(maxDistance, length(probeLocation - rayHitLocation));

        if (dot(rayHitNormal, rayHitNormal) < epsilon)
        {
            rayProbeDistance = maxDistance;
        }

        // Weight
        float weight = 0.0;
        if (irradiance)
        {
            weight = max(0.0, dot(texelDirection, rayDirection));
        }
        else
        {
            weight = pow(max(0.0, dot(texelDirection, rayDirection)), depthSharpness);
        }

        // Accumulate
        if (weight >= epsilon)
        {
            if (irradiance)
            {
                result = result + make_float4(rayHitRadiance * weight, weight);
            }
            else
            {
                result = result + make_float4(rayProbeDistance * weight, rayProbeDistance * rayProbeDistance * weight,
                                              0.0, weight);
            }
        }
    }
    if (result.w > epsilon)
    {
        newvaule = make_float3(result.x, result.y, result.z) / result.w;
        float srcfactor = 1.0f - hysteresis;
        result = make_float4(newvaule * srcfactor + oldvaule * (1.0f - srcfactor), srcfactor);
    }
    output[pixel_index] = result;
}

void UpdateProbeCPU(cudaStream_t stream, Pupil::ddgi::probe::UpdateParams update_params, uint2 size, int raysPerProbe,
                    int probeSideLength, float maxDistance, float hysteresis, float depthSharpness, bool irradiance)
{

    constexpr int block_size_x = 32;
    constexpr int block_size_y = 32;
    int grid_size_x = (size.x + block_size_x - 1) / block_size_x;
    int grid_size_y = (size.y + block_size_y - 1) / block_size_y;
    UpdateProbe<<<dim3(grid_size_x, grid_size_y), dim3(block_size_x, block_size_y), 0, stream>>>(
        update_params.probeirradiance.GetDataPtr(), update_params.probedepth.GetDataPtr(),
        update_params.rayradiance.GetDataPtr(), update_params.rayorgin.GetDataPtr(),
        update_params.raydirection.GetDataPtr(), update_params.rayhitposition.GetDataPtr(),
        update_params.rayhitnormal.GetDataPtr(), size, raysPerProbe, probeSideLength, maxDistance, hysteresis,
        depthSharpness, irradiance);
    // UpdateProbe<<<dim3(grid_size_x, grid_size_y), dim3(block_size_x, block_size_y), 0, stream>>>(
    //     update_params.probeirradiance.GetDataPtr(), update_params.rayradiance.GetDataPtr(),
    //     update_params.rayorgin.GetDataPtr(), update_params.raydirection.GetDataPtr(),
    //     update_params.rayhitposition.GetDataPtr(), update_params.rayhitnormal.GetDataPtr(), size, raysPerProbe,
    //     probeSideLength, maxDistance, hysteresis, depthSharpness, irradiance);
}

void ChangeAlphaCPU(cudaStream_t stream, Pupil::cuda::RWArrayView<float4> &probeirradiance_show,
                    Pupil::cuda::ConstArrayView<float4> probeirradiance, uint2 size)
{

    constexpr int block_size_x = 32;
    constexpr int block_size_y = 32;
    int grid_size_x = (size.x + block_size_x - 1) / block_size_x;
    int grid_size_y = (size.y + block_size_y - 1) / block_size_y;
    ChangeAlpha<<<dim3(grid_size_x, grid_size_y), dim3(block_size_x, block_size_y), 0, stream>>>(
        probeirradiance_show.GetDataPtr(), probeirradiance.GetDataPtr(), size);
}
