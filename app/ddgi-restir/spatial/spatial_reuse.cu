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
__constant__ ddgi::spatial::OptixLaunchParams optix_launch_params;
}

extern "C" __global__ void __raygen__main() {
    const uint3 index = optixGetLaunchIndex();
    const unsigned int w = optix_launch_params.config.frame.width;
    const unsigned int h = optix_launch_params.config.frame.height;
    const unsigned int pixel_index = index.y * w + index.x;

    float4 pos_with_flag = optix_launch_params.position_buffer[pixel_index];
    if (pos_with_flag.w <= 0.f) {
        optix_launch_params.final_reservoirs[pixel_index] = optix_launch_params.reservoirs[pixel_index];
        return;
    }
    float3 pos = make_float3(pos_with_flag);

    cuda::Random random;
    random.Init(4, pixel_index, optix_launch_params.random_seed);

    auto normal_depth = optix_launch_params.normal_buffer[pixel_index];
    auto normal = make_float3(normal_depth);
    auto albedo_flag = optix_launch_params.albedo_buffer[pixel_index];
    if (albedo_flag.w > 0.f) {
        optix_launch_params.final_reservoirs[pixel_index] = optix_launch_params.reservoirs[pixel_index];
        return;
    }

    float3 wo = optix::ToLocal(-normalize(pos - optix_launch_params.camera.pos), normal);

    auto albedo = make_float3(albedo_flag);

    Reservoir reservoir;
    reservoir.Init();
    unsigned int M = 0;

    // 取5个neighbour
    for (auto i = 0u; i < 5; ++i) {
        float r = optix_launch_params.spatial_radius * random.Next();
        float theta = M_PIf * 2.f * random.Next();
        // 随机一个方向的neighbour pixel
        int2 neighbor_pixel = make_int2(index.x + r * cos(theta), index.y + r * sin(theta));
        if (neighbor_pixel.x < 0 || neighbor_pixel.x >= w || neighbor_pixel.y < 0 || neighbor_pixel.y >= h)
            continue;
        const unsigned int neighbor_pixel_index = neighbor_pixel.y * w + neighbor_pixel.x;

        // 去掉一些特殊情况
        auto neighbor_normal_depth = optix_launch_params.normal_buffer[neighbor_pixel_index];
        if (dot(normal, make_float3(neighbor_normal_depth)) < 0.906307787f)
            continue;
        if (normal_depth.w * 0.9f > neighbor_normal_depth.w || normal_depth.w * 1.1f < neighbor_normal_depth.w)
            continue;

        // 取出neighbour reservoir中的样本，采样的光源点还是neighbour中的，但方位和材质都用自己的
        auto &neighbor_reservoir = optix_launch_params.reservoirs[neighbor_pixel_index];
        float3 wi = optix::ToLocal(normalize(neighbor_reservoir.y.pos - pos), normal);

        float3 f = make_float3(0.f);
        if (wi.z > 0.f && wo.z > 0.f) {
            f = albedo * M_1_PIf;
        }

        Reservoir::Sample x_i = neighbor_reservoir.y;
        x_i.radiance = make_float3(0.f);
        x_i.p_hat = 0.f;
        if (!optix::IsZero(f)) {
            x_i.radiance = x_i.emission * f * wi.z; // 原本是 x_i.radiance += emitter_sample_record.radiance * f * NoL;
            // phat
            auto &emitter = optix_launch_params.emitters.SelectOneEmiiter(x_i.emitter_rand.x);
            optix::EmitterSampleRecord emitter_sample_record;
            optix::LocalGeometry geo;
            geo.position = pos;
            geo.normal = normal;
            emitter.SampleDirect(emitter_sample_record, geo, make_float2(x_i.emitter_rand.y, x_i.emitter_rand.z));
            float NoL = dot(geo.normal, emitter_sample_record.wi);
            if(emitter_sample_record.pdf < 1e-5)
                emitter_sample_record.radiance = make_float3(0.0f);
            x_i.radiance += emitter_sample_record.radiance * f * NoL;
            x_i.p_hat = optix::GetLuminance(emitter_sample_record.radiance);
        }

        if(x_i.p_hat > 0){
            float w_i = x_i.p_hat * neighbor_reservoir.M * neighbor_reservoir.W;
            reservoir.Update(x_i, w_i, random); // 五个neighbour update五次
            M += neighbor_reservoir.M - 1;  
        }
    }
    reservoir.M = M; // 五个neighbour的M累加
    optix_launch_params.final_reservoirs[pixel_index] = optix_launch_params.reservoirs[pixel_index];
    reservoir.CalcW();
    if (reservoir.W > 0.f) {
        optix_launch_params.final_reservoirs[pixel_index].Combine(reservoir, random); // 和原本的reservoir合并
    }
    return;
}

extern "C" __global__ void __miss__default() {
}
extern "C" __global__ void __closesthit__default() {
}