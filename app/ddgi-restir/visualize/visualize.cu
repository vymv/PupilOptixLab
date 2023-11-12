#include "type.h"
#include <optix.h>

// #include "../indirect/indirect.h"
#include "../indirect/probemath.h"
#include "render/geometry.h"
#include "render/emitter.h"
#include "render/material/optix_material.h"
#include "optix/util.h"

#include "cuda/random.h"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

using namespace Pupil;

extern "C" {
__constant__ ddgi::visualize::OptixLaunchParams optix_launch_params;
}

struct HitInfo {
    optix::LocalGeometry geo;
    optix::material::Material::LocalBsdf bsdf;
    int emitter_index;
};


__device__ int2 worldToScreen(float4 pos, mat4x4 proj_view, int w, int h){

    float4 clip_pos = proj_view * pos;
    float3 ndc_pos = make_float3(clip_pos.x, clip_pos.y, clip_pos.z) / clip_pos.w;
    float2 screen_pos = make_float2(ndc_pos.x, ndc_pos.y) * 0.5f + 0.5f;

    return make_int2(floor(screen_pos.x * w), floor(screen_pos.y * h));
}

extern "C" __global__ void __raygen__main() {

    const uint3 index = optixGetLaunchIndex();
    const unsigned int w = optix_launch_params.config.frame.width;
    const unsigned int h = optix_launch_params.config.frame.height;
    const unsigned int pixel_index = index.y * w + index.x;

    optix_launch_params.visualize_buffer[pixel_index] = optix_launch_params.input_buffer[pixel_index];

    for(int i = 0; i < optix_launch_params.probe_count; i++) {
         // 计算probe屏幕坐标
        int2 probe_screen_pos = worldToScreen(make_float4(optix_launch_params.probe_position[i],1), optix_launch_params.proj_view, w, h);

        // 是否在半径之内s
        float distance_x = int(index.x) - probe_screen_pos.x;
        float distance_y = int(index.y) - probe_screen_pos.y;
        if(distance_x * distance_x + distance_y * distance_y < pow(optix_launch_params.probe_visualize_size, 2)) {
            if(i!= optix_launch_params.highlight_index)
                optix_launch_params.visualize_buffer[pixel_index] = make_float4(optix_launch_params.probe_position[i], 1.0);
            else
                optix_launch_params.visualize_buffer[pixel_index] = make_float4(1.0, 0.0, 0.0, 1.0);
        }
        // if(abs(distance_x) < optix_launch_params.probe_visualize_size && abs(distance_y) < optix_launch_params.probe_visualize_size) {
        //     optix_launch_params.visualize_buffer[pixel_index] = make_float4(optix_launch_params.probe_position[i], 1.0);
        // }
    }
}

extern "C" __global__ void __miss__default() {

}
extern "C" __global__ void __closesthit__default() {

}
