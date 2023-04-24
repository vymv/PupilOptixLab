#pragma once
#include "cuda/data_view.h"
#include "cuda/vec_math.h"
#include <stdio.h>

// void UpdateProbeCPU(cudaStream_t stream, Pupil::cuda::ConstArrayView<float4> rayGbuffer,
//                     Pupil::cuda::RWArrayView<float4> &probeIrradiance, uint2 size, int raysPerProbe,
//                     int probeSideLength);
namespace Pupil::ddgi::probe
{
struct UpdateParams
{
    cuda::ConstArrayView<float4> rayradiance;
    cuda::ConstArrayView<float3> rayhitposition;
    cuda::ConstArrayView<float3> rayorgin;
    cuda::ConstArrayView<float3> raydirection;
    cuda::ConstArrayView<float3> rayhitnormal;

    cuda::RWArrayView<float4> probeirradiance;
};
} // namespace Pupil::ddgi::probe
// void UpdateProbeCPU(cudaStream_t stream, Pupil::cuda::ConstArrayView<float4> rayGbuffer,
//                     Pupil::cuda::RWArrayView<float4> &probeIrradiance, uint2 size, int raysPerProbe,
//                     int probeSideLength);
void UpdateProbeCPU(cudaStream_t stream, Pupil::ddgi::probe::UpdateParams update_params, uint2 size, int raysPerProbe,
                    int probeSideLengt, float maxDistance, float hysteresis);

void ChangeAlphaCPU(cudaStream_t stream, Pupil::cuda::RWArrayView<float4> &probeirradiance_show,
                    Pupil::cuda::ConstArrayView<float4> probeirradiance, uint2 size);