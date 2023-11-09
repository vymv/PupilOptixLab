#include "type.h"
#include <optix.h>

//#include "../indirect/indirect.h"
#include "../indirect/probemath.h"
#include "render/geometry.h"
#include "render/emitter.h"
#include "optix/util.h"

#include "cuda/random.h"

using namespace Pupil;

extern "C" {
__constant__ ddgi::merge::OptixLaunchParams optix_launch_params;
}


extern "C" __global__ void __raygen__main() {
    const uint3 index = optixGetLaunchIndex();
    const unsigned int w = optix_launch_params.config.frame.width;
    const unsigned int h = optix_launch_params.config.frame.height;
    const unsigned int pixel_index = index.y * w + index.x;

    float4 direct_color = optix_launch_params.direct_buffer[pixel_index];
    optix_launch_params.output_buffer[pixel_index] = direct_color;
    // optix_launch_params.output_buffer[pixel_index] = make_float4(1.0f);

    if(!optix_launch_params.is_pathtracer){
        float4 indirect_color = optix_launch_params.indirect_buffer[pixel_index];
        optix_launch_params.output_buffer[pixel_index] += indirect_color;
    }

}

extern "C" __global__ void __miss__default() {

}

extern "C" __global__ void __closesthit__default() {

}
