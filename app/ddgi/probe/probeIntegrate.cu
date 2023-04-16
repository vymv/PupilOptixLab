#include "cuda/vec_math.h"
#include "probeIntegrate.h"

__device__ float2 octEncode(float3 v)
{
    float l1norm = fabsf(v.x) + fabsf(v.y) + fabsf(v.z);
    float2 result = make_float2(v.x, v.y) * (1.0f / l1norm);
    if (v.z < 0.0f)
    {
        result = (make_float2(1.0f) - make_float2(fabsf(result.y), fabsf(result.x))) *
                 (make_float2(result.x >= 0 ? 1.0 : 0.0, result.y >= 0 ? 1.0 : 0.0));
    }
    return result;
}

__device__ float3 octDecode(float2 o)
{
    float3 v = make_float3(o.x, o.y, 1.0f - fabsf(o.x) - fabsf(o.y));
    if (v.z < 0.0f)
    {
        float2 xy = (make_float2(1.0f) - make_float2(fabsf(v.y), fabsf(v.x))) *
                    (make_float2(v.x >= 0 ? 1.0 : 0.0, v.y >= 0 ? 1.0 : 0.0));
        v.x = xy.x;
        v.y = xy.y;
    }
    return normalize(v);
}

__device__ float2 normalizedOctCoord(int2 fragCoord, int probeSideLength)
{
    int probeWithBorderSide = probeSideLength + 2;

    float2 octFragCoord = make_float2((fragCoord.x - 2) % probeWithBorderSide, (fragCoord.y - 2) % probeWithBorderSide);
    // Add back the half pixel to get pixel center normalized coordinates
    return (octFragCoord + make_float2(0.5)) * (2.0f / float(probeSideLength)) - make_float2(1.0f);
}

// probeRayGbuffer -> probeTexture
__global__ void UpdateProbe(const float4 *rayGbuffer, float4 *probeIrradiance, uint2 size, int raysPerProbe,
                            int probeSideLength)
{

    int pixel_x = threadIdx.x + blockIdx.x * blockDim.x;
    int pixel_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (pixel_x >= size.x)
        return;
    if (pixel_y >= size.y)
        return;
    int pixel_index = pixel_x + size.x * pixel_y;

    for (int r = 0; r < raysPerProbe; ++r)
    {

        // 计算probeId
        int probeWithBorderSide = probeSideLength + 2;
        int probesPerSide = (size.x - 2) / probeWithBorderSide;
        int probeId = int(pixel_x / probeWithBorderSide) + probesPerSide * int(pixel_y / probeWithBorderSide);

        // 计算direction
        float3 texelDirection = octDecode(normalizedOctCoord(make_int2(pixel_x, pixel_y), probeSideLength));

        // hit radiance
        int2 coord = make_int2(r, probeId);
        float4 rayHitRadiance = probeIrradiance[raysPerProbe * coord.y + coord.x];

        // float weight = max(0.0, dot(texelDirection, rayDirection));
    }

    probeIrradiance[pixel_index] = make_float4(0.5);
}

// void UpdateProbeCPU(cudaStream_t stream, Pupil::cuda::ConstArrayView<float4> rayGbuffer,
//                     Pupil::cuda::RWArrayView<float4> &probeIrradiance, uint2 size, int raysPerProbe,
//                     int probeSideLength)
void UpdateProbeCPU(cudaStream_t stream, Pupil::ddgi::probe::UpdateParams update_params, uint2 size, int raysPerProbe,
                    int probeSideLength)
{

    constexpr int block_size_x = 32;
    constexpr int block_size_y = 32;
    int grid_size_x = (size.x + block_size_x - 1) / block_size_x;
    int grid_size_y = (size.y + block_size_y - 1) / block_size_y;
    UpdateProbe<<<dim3(grid_size_x, grid_size_y), dim3(block_size_x, block_size_y), 0, stream>>>(
        update_params.rayradiance.GetDataPtr(), update_params.probeirradiance.GetDataPtr(), size, raysPerProbe,
        probeSideLength);
    // UpdateProbe<<<dim3(grid_size_x, grid_size_y), dim3(block_size_x, block_size_y), 0, stream>>>(
    //     rayGbuffer.GetDataPtr(), probeIrradiance.GetDataPtr(), size, raysPerProbe, probeSideLength);
}
