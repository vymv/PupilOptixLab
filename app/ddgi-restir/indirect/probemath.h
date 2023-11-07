#include "cuda/vec_math.h"
#include "cuda/preprocessor.h"
#include <optix.h>

#include <vector_functions.h>
#include <vector_types.h>

CUDA_INLINE CUDA_DEVICE float2 octEncode(float3 v)
{
    float l1norm = fabsf(v.x) + fabsf(v.y) + fabsf(v.z);
    float2 result = make_float2(v.x, v.y) * (1.0f / l1norm);
    if (v.z < 0.0f)
    {
        result = (make_float2(1.0f) - make_float2(fabsf(result.y), fabsf(result.x))) *
                 (make_float2(result.x >= 0 ? 1.0 : -1.0, result.y >= 0 ? 1.0 : -1.0));
    }
    return result;
}

CUDA_INLINE CUDA_DEVICE float3 octDecode(float2 o)
{
    float3 v = make_float3(o.x, o.y, 1.0f - fabsf(o.x) - fabsf(o.y));
    if (v.z < 0.0f)
    {
        float2 xy = (make_float2(1.0f) - make_float2(fabsf(v.y), fabsf(v.x))) *
                    (make_float2(v.x >= 0 ? 1.0 : -1.0, v.y >= 0 ? 1.0 : -1.0));
        v.x = xy.x;
        v.y = xy.y;
    }
    return normalize(v);
}