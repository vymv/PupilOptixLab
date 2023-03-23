#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
#define CUDA_HOST __host__
#define CUDA_DEVICE __device__
#define CUDA_HOSTDEVICE __host__ __device__
#define CUDA_INLINE __forceinline__
#define CONST_STATIC_INIT(...)
#else
#define PUPIL_OPTIX_LAUNCHER_SIDE
#define CUDA_HOST
#define CUDA_DEVICE
#define CUDA_HOSTDEVICE
#define CUDA_INLINE inline
#define CONST_STATIC_INIT(...) = __VA_ARGS__
#endif
