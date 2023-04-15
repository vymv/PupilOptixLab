#pragma once
#include "cuda/data_view.h"
#include "cuda/vec_math.h"

void CudaSetColor(cudaStream_t stream, Pupil::cuda::RWArrayView<float4> &output_image, uint2 size,
                  unsigned int frame_cnt);
