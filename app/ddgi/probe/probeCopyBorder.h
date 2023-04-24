#pragma once
#include "cuda/data_view.h"
#include "cuda/vec_math.h"
#include <stdio.h>

void CopyBorderCPU(cudaStream_t stream, Pupil::cuda::RWArrayView<float4> rayradiance, uint2 size, int probeSideLength);