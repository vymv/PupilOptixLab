#pragma once

#include <vector_types.h>
#include "cuda/preprocessor.h"
#include "cuda/random.h"

struct Reservoir {
    struct Sample {
        float3 pos;
        float3 normal;
        // float3 emission;
        // float distance;
        // float3 radiance;
        float3 radiance;
        float p_hat;
        float3 emitter_rand;
        float3 albedo;
        int sample_type;
    };

    Sample y;
    float w_sum;
    float p_sum;
    float W;
    unsigned int M;

    CUDA_HOSTDEVICE Reservoir() noexcept { Init(); }

    CUDA_HOSTDEVICE void Init() noexcept {
        w_sum = 0.f;
        p_sum = 0.f;
        W = 0.f;
        M = 0u;
        y.p_hat = 0.f;
        y.radiance = make_float3(0.f);
    }

    // CUDA_HOSTDEVICE void Update(
    //     const Sample &x_i, float w_i,
    //     Pupil::cuda::Random &random) noexcept {
    //     w_sum += w_i;
    //     M += 1;
    //     // if (random.Next() < w_i / w_sum)
    //     //     y = x_i;
    //     if (M == 1) {
    //         if (w_i != 0.f)// 第一个样本一定选
    //             y = x_i;
    //         else// 第一个样本选择的概率为0，去掉该样本
    //             M = 0;
    //     } else if (random.Next() * w_sum < w_i)
    //         y = x_i;
    // }
    
    CUDA_HOSTDEVICE void Update(
        const Sample &x_i, float w_i, float p_i,
        Pupil::cuda::Random &random) noexcept {
        w_sum += w_i;
        p_sum += p_i;
        M += 1;
        // if (random.Next() < w_i / w_sum)
        //     y = x_i;
        if (M == 1) {
            if (w_i != 0.f)// 第一个样本一定选
                y = x_i;
            else// 第一个样本选择的概率为0，去掉该样本
                M = 0;
        } else if (random.Next() * w_sum < w_i)
            y = x_i;
    }

    CUDA_HOSTDEVICE void CalcW() noexcept {
        // W = (y.p_hat == 0.f || M == 0) ? 0.f : w_sum / (y.p_hat * p_sum); 
        W = (y.p_hat == 0.f || M == 0) ? 0.f : w_sum / (y.p_hat * M);
    }
    CUDA_HOSTDEVICE void CalcMISW() noexcept {
        W = (y.p_hat == 0.f || M == 0) ? 0.f : w_sum / (y.p_hat * p_sum); 
        // W = (y.p_hat == 0.f || M == 0) ? 0.f : w_sum / (y.p_hat * M);
    }

    CUDA_HOSTDEVICE void Combine(const Reservoir &other, Pupil::cuda::Random &random) noexcept {
        //   w_mis * x_i.p_hat / x_i.p
        // = w_mis * other.y.p_hat * other.W
        // = other.y.p_hat / sum_phat * other.y.p_hat * other.W
        // = other.y.p_hat * other.y.p_hat * x_i.W, sum_old_phat累加，之后再除，M倍的权重
        Update(other.y, other.y.p_hat * other.y.p_hat * other.W * other.M, other.p_sum, random);
        // Update(other.y, other.y.p_hat * other.W * other.M, random);
        M += other.M - 1;
        // CalcW();
        CalcMISW();
    }
};