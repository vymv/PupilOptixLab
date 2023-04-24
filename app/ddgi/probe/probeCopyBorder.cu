
#include "cuda/vec_math.h"
#include "probeCopyBorder.h"

__global__ void CopyBorder(float4 *probeirradiance, uint2 size, int probeSideLength)
{
    int border = 2;
    int pixel_x = threadIdx.x + blockIdx.x * blockDim.x;
    int pixel_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (pixel_x > size.x - 1 || pixel_y > size.y - 1 || pixel_x < 0 || pixel_y < 0)
        return;

    int pixel_index = pixel_x + size.x * pixel_y;
    if (pixel_x == size.x - 1 || pixel_y == size.y - 1 || pixel_x == 0 || pixel_y == 0)
    {
        probeirradiance[pixel_index] = make_float4(0.0f);
        return;
    }

    if (pixel_x % (probeSideLength + border) == 0)
    {
        probeirradiance[pixel_index] = probeirradiance[(pixel_x - 1) + size.x * pixel_y];
    }
    if (pixel_x % (probeSideLength + border) == 1)
    {
        probeirradiance[pixel_index] = probeirradiance[(pixel_x + 1) + size.x * pixel_y];
    }
    if (pixel_y % (probeSideLength + border) == 0)
    {
        probeirradiance[pixel_index] = probeirradiance[pixel_x + size.x * (pixel_y - 1)];
    }
    if (pixel_y % (probeSideLength + border) == 1)
    {
        probeirradiance[pixel_index] = probeirradiance[pixel_x + size.x * (pixel_y + 1)];
    }
}

void CopyBorderCPU(cudaStream_t stream, Pupil::cuda::RWArrayView<float4> probeirradiance, uint2 size,
                   int probeSideLength)
{

    constexpr int block_size_x = 32;
    constexpr int block_size_y = 32;
    int grid_size_x = (size.x + block_size_x - 1) / block_size_x;
    int grid_size_y = (size.y + block_size_y - 1) / block_size_y;

    CopyBorder<<<dim3(grid_size_x, grid_size_y), dim3(block_size_x, block_size_y), 0, stream>>>(
        probeirradiance.GetDataPtr(), size, probeSideLength);
}
