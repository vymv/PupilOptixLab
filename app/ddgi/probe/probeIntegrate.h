#pragma once

#include "cuda/data_view.h"
#include "cuda/vec_math.h"

void UpdateProbeCPU(cudaStream_t stream, Pupil::cuda::RWArrayView<float4> &rayGbuffer,
                    Pupil::cuda::RWArrayView<float4> &probeIrradiance, uint2 size, int raysPerProbe,
                    int probeSideLength);
