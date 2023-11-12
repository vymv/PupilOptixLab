#include "pass.h"
#include "../indirect/global.h"
#include "imgui.h"

#include "cuda/context.h"
#include "optix/context.h"
#include "optix/module.h"

#include "system/system.h"
#include "system/gui/gui.h"
#include "util/event.h"

extern "C" char ddgi_probe_pass_embedded_ptx_code[];

namespace Pupil {
extern uint32_t g_window_w;
extern uint32_t g_window_h;
}// namespace Pupil

namespace {
double m_time_cnt = 1.f;
int m_probeperside = 2;
}// namespace

namespace Pupil::ddgi::probe {
ProbePass::ProbePass(std::string_view name) noexcept : Pass(name) {
    auto optix_ctx = util::Singleton<optix::Context>::instance();
    auto cuda_ctx = util::Singleton<cuda::Context>::instance();
    m_stream = std::make_unique<cuda::Stream>();
    m_optix_pass =
        std::make_unique<optix::Pass<SBTTypes, OptixLaunchParams>>(optix_ctx->context, m_stream->GetStream());

    InitOptixPipeline();
    BindingEventCallback();
}

float3 sphericalFibonacci(float i, float n) {
    const float PHI = std::sqrt(5) * 0.5f + 0.5f;

    float madfrac = (i * (PHI - 1)) - std::floor(i * (PHI - 1));

    float phi = 2.0f * M_PIf * std::clamp(madfrac, 0.0f, 1.0f);
    float cosTheta = 1.0f - (2.0f * i + 1.0f) * (1.0f / n);
    float sinTheta = std::sqrt(std::clamp(1.0f - cosTheta * cosTheta, 0.0f, 1.0f));

    return make_float3(std::cos(phi) * sinTheta, std::sin(phi) * sinTheta, cosTheta);
}

util::Mat3 fromAxisAngle(const float3 axis, float fRadians) {

    util::Mat3 m;
    float fCos = cos(fRadians);
    float fSin = sin(fRadians);
    float fOneMinusCos = 1.0f - fCos;
    float fX2 = axis.x * axis.x;
    float fY2 = axis.y * axis.y;
    float fZ2 = axis.z * axis.z;
    float fXYM = axis.x * axis.y * fOneMinusCos;
    float fXZM = axis.x * axis.z * fOneMinusCos;
    float fYZM = axis.y * axis.z * fOneMinusCos;
    float fXSin = axis.x * fSin;
    float fYSin = axis.y * fSin;
    float fZSin = axis.z * fSin;

    m.re[0][0] = fX2 * fOneMinusCos + fCos;
    m.re[0][1] = fXYM - fZSin;
    m.re[0][2] = fXZM + fYSin;

    m.re[1][0] = fXYM + fZSin;
    m.re[1][1] = fY2 * fOneMinusCos + fCos;
    m.re[1][2] = fYZM - fXSin;

    m.re[2][0] = fXZM - fYSin;
    m.re[2][1] = fYZM + fXSin;
    m.re[2][2] = fZ2 * fOneMinusCos + fCos;

    return m;
}

void ProbePass::OnRun() noexcept {

    if (!is_pathtracer) {
        // random orientation
        const float3 axis =
            make_float3(std::rand() / float(RAND_MAX), std::rand() / float(RAND_MAX), std::rand() / float(RAND_MAX));
        util::Mat3 m = fromAxisAngle(normalize(axis), std::rand() / float(RAND_MAX) * 2.0 * M_PIf);

        CUDA_FREE(m_randomoriention_cuda_memory);
        m_randomoriention_cuda_memory = cuda::CudaMemcpyToDevice(&m, sizeof(m));
        m_optix_launch_params.randomOrientation.SetData(m_randomoriention_cuda_memory);
        m_timer.Start();
        {
            if (m_dirty) {
                m_dirty = false;
                m_optix_launch_params.energyConservation = m_energyconservation;
            }

            m_optix_launch_params.rayhitnormal.SetData(m_rayhitnormal->cuda_ptr,
                                                       m_raygbuffersize.w * m_raygbuffersize.h);
            m_optix_launch_params.raydirection.SetData(m_raydirection->cuda_ptr,
                                                       m_raygbuffersize.w * m_raygbuffersize.h);
            m_optix_launch_params.rayhitposition.SetData(m_rayhitposition->cuda_ptr,
                                                         m_raygbuffersize.w * m_raygbuffersize.h);
            m_optix_launch_params.rayradiance.SetData(m_rayradiance->cuda_ptr, m_raygbuffersize.w * m_raygbuffersize.h);
            // auto &frame_buffer = util::Singleton<GuiPass>::instance()->GetCurrentRenderOutputBuffer().shared_buffer;
            // m_optix_launch_params.rayradiance.SetData(frame_buffer.cuda_ptr, raygbuffersize.w * raygbuffersize.h);
            auto buf_mngr = util::Singleton<BufferManager>::instance();
            if (m_frame_cnt != 0) {
                auto probeirradiancebuffer = buf_mngr->GetBuffer("ddgi_probeirradiance");
                m_optix_launch_params.probeirradiance.SetData(probeirradiancebuffer->cuda_ptr,
                                                              m_probeirradiancesize.w * m_probeirradiancesize.h);

                auto probedepthbuffer = buf_mngr->GetBuffer("ddgi_probedepth");
                m_optix_launch_params.probedepth.SetData(probedepthbuffer->cuda_ptr,
                                                         m_probeirradiancesize.w * m_probeirradiancesize.h);
            } else {
                std::vector<float4> probezerovector;
                probezerovector.resize(m_probeirradiancesize.w * m_probeirradiancesize.h);
                probezerovector.assign(probezerovector.size(), make_float4(0, 0, 0, 0));

                CUDA_FREE(m_zeroradiance_cuda_memory);
                m_zeroradiance_cuda_memory =
                    cuda::CudaMemcpyToDevice(probezerovector.data(), probezerovector.size() * sizeof(float4));
                m_optix_launch_params.probeirradiance.SetData(m_zeroradiance_cuda_memory, probezerovector.size());

                CUDA_FREE(m_zerodepth_cuda_memory);
                m_zerodepth_cuda_memory =
                    cuda::CudaMemcpyToDevice(probezerovector.data(), probezerovector.size() * sizeof(float4));
                m_optix_launch_params.probedepth.SetData(m_zerodepth_cuda_memory, probezerovector.size());
            }

            m_optix_pass->Run(m_optix_launch_params, m_optix_launch_params.config.frame.width,
                              m_optix_launch_params.config.frame.height);// probeTrace
            m_optix_pass->Synchronize();

            ++m_optix_launch_params.random_seed;

            // 积分
            // 输入
            // gbuffer

            auto raygbuffer = buf_mngr->GetBuffer("ddgi_rayradiance");
            m_update_params.rayradiance.SetData(raygbuffer->cuda_ptr,
                                                m_irradiancerays_perprobe * std::pow(m_probecountperside, 3));
            // hitposition
            auto rayhitpositionbuffer = buf_mngr->GetBuffer("ddgi_rayhitposition");
            m_update_params.rayhitposition.SetData(rayhitpositionbuffer->cuda_ptr,
                                                   m_irradiancerays_perprobe * std::pow(m_probecountperside, 3));
            // hitnormal
            auto rayhitnormalbuffer = buf_mngr->GetBuffer("ddgi_rayhitnormal");
            m_update_params.rayhitnormal.SetData(rayhitnormalbuffer->cuda_ptr,
                                                 m_irradiancerays_perprobe * std::pow(m_probecountperside, 3));
            // direction
            auto raydirectionbuffer = buf_mngr->GetBuffer("ddgi_raydirection");
            m_update_params.raydirection.SetData(raydirectionbuffer->cuda_ptr,
                                                 m_irradiancerays_perprobe * std::pow(m_probecountperside, 3));

            // // origin
            // CUDA_FREE(m_probepos_cuda_memory);
            // m_probepos_cuda_memory = cuda::CudaMemcpyToDevice(m_probepos.data(), m_probepos.size() * sizeof(float3));
            // m_update_params.rayorigin.SetData(m_probepos_cuda_memory, m_probepos.size());

            // 输出
            // auto &frame_buffer = util::Singleton<GuiPass>::instance()->GetCurrentRenderOutputBuffer().shared_buffer;
            // m_update_params.probeirradiance.SetData(frame_buffer.cuda_ptr, size.h * size.w);
            auto probeirradiancebuffer = buf_mngr->GetBuffer("ddgi_probeirradiance");
            auto probedepthbuffer = buf_mngr->GetBuffer("ddgi_probedepth");

            m_update_params.probeirradiance.SetData(probeirradiancebuffer->cuda_ptr,
                                                    m_probeirradiancesize.h * m_probeirradiancesize.w);
            m_update_params.probedepth.SetData(probedepthbuffer->cuda_ptr,
                                               m_probeirradiancesize.h * m_probeirradiancesize.w);

            // irradiance
            UpdateProbeCPU(m_stream->GetStream(), m_update_params,
                           make_uint2(m_probeirradiancesize.w, m_probeirradiancesize.h), m_irradiancerays_perprobe,
                           m_probesidelength, m_maxdistance, m_frame_cnt == 0, m_hysteresis, m_depthSharpness,
                           true);

            m_stream->Synchronize();

            // depth
            UpdateProbeCPU(m_stream->GetStream(), m_update_params,
                           make_uint2(m_probeirradiancesize.w, m_probeirradiancesize.h), m_irradiancerays_perprobe,
                           m_probesidelength, m_maxdistance, m_frame_cnt == 0, m_hysteresis, m_depthSharpness,
                           false);

            m_stream->Synchronize();

            // Copy Border
            cuda::RWArrayView<float4> probeirradiance;
            probeirradiance.SetData(probeirradiancebuffer->cuda_ptr, m_probeirradiancesize.h *
                                                                         m_probeirradiancesize.w);
            CopyBorderCPU(m_stream->GetStream(), probeirradiance,
                          make_uint2(m_probeirradiancesize.w, m_probeirradiancesize.h), m_probesidelength);
            m_stream->Synchronize();

            cuda::RWArrayView<float4> probedepth;
            probedepth.SetData(probedepthbuffer->cuda_ptr, m_probeirradiancesize.h * m_probeirradiancesize.w);
            CopyBorderCPU(m_stream->GetStream(), probedepth, make_uint2(m_probeirradiancesize.w, m_probeirradiancesize.h),
                          m_probesidelength);
            m_stream->Synchronize();

            auto *frame_buffer = buf_mngr->GetBuffer(buf_mngr->DEFAULT_FINAL_RESULT_BUFFER_NAME);
            m_frame_cnt++;
        }
        m_timer.Stop();
        m_time_cnt = m_timer.ElapsedMilliseconds();
    }
}

void ProbePass::InitOptixPipeline() noexcept {
    auto module_mngr = util::Singleton<optix::ModuleManager>::instance();

    auto sphere_module = module_mngr->GetModule(optix::EModuleBuiltinType::SpherePrimitive);
    auto pt_module = module_mngr->GetModule(ddgi_probe_pass_embedded_ptx_code);

    optix::PipelineDesc pipeline_desc;
    {
        // for mesh(triangle) geo
        optix::RayTraceProgramDesc forward_ray_desc{
            .module_ptr = pt_module,
            .ray_gen_entry = "__raygen__main",
            .miss_entry = "__miss__default",
            .hit_group = { .ch_entry = "__closesthit__default" },
        };
        pipeline_desc.ray_trace_programs.push_back(forward_ray_desc);
        optix::RayTraceProgramDesc shadow_ray_desc{
            .module_ptr = pt_module,
            .miss_entry = "__miss__shadow",
            .hit_group = { .ch_entry = "__closesthit__shadow" },
        };
        pipeline_desc.ray_trace_programs.push_back(shadow_ray_desc);
    }

    {
        optix::RayTraceProgramDesc forward_ray_desc{
            .module_ptr = pt_module,
            .hit_group = { .ch_entry = "__closesthit__default",
                           .intersect_module = sphere_module },
        };
        pipeline_desc.ray_trace_programs.push_back(forward_ray_desc);
        optix::RayTraceProgramDesc shadow_ray_desc{
            .module_ptr = pt_module,
            .hit_group = { .ch_entry = "__closesthit__shadow",
                           .intersect_module = sphere_module },
        };
        pipeline_desc.ray_trace_programs.push_back(shadow_ray_desc);
    }
    {
        auto mat_programs = Pupil::resource::GetMaterialProgramDesc();
        pipeline_desc.callable_programs.insert(
            pipeline_desc.callable_programs.end(),
            mat_programs.begin(), mat_programs.end());
    }
    m_optix_pass->InitPipeline(pipeline_desc);
}

void ProbePass::SetScene(world::World *world) noexcept {

    // m_world_camera = world->optix_scene->camera.get();
    m_frame_cnt = 0;

    m_optix_launch_params.config.frame.width = m_irradiancerays_perprobe;
    m_optix_launch_params.config.frame.height = std::pow(m_probecountperside, 3);

    m_optix_launch_params.random_seed = 0;

    m_optix_launch_params.probeStartPosition = m_probestartpos;
    m_optix_launch_params.probeStep = m_probestep;
    m_optix_launch_params.probeCount = make_int3(m_probecountperside, m_probecountperside, m_probecountperside);
    m_optix_launch_params.probeirradiancesize = make_uint2(m_probeirradiancesize.w, m_probeirradiancesize.h);
    m_optix_launch_params.probeSideLength = m_probesidelength;
    m_optix_launch_params.energyConservation = m_energyconservation;

    auto buf_mngr = util::Singleton<BufferManager>::instance();
    BufferDesc rayradiance_buf_desc = {
        .name = "ddgi_rayradiance",
        .flag = EBufferFlag::AllowDisplay,
        .width = static_cast<uint32_t>(m_irradiancerays_perprobe),
        .height = static_cast<uint32_t>(std::pow(m_probecountperside, 3)),
        .stride_in_byte = sizeof(float) * 4
    };

    m_rayradiance = buf_mngr->AllocBuffer(rayradiance_buf_desc);

    BufferDesc rayhitposition_buf_desc = {
        .name = "ddgi_rayhitposition",
        .flag = EBufferFlag::AllowDisplay,
        .width = static_cast<uint32_t>(m_irradiancerays_perprobe),
        .height = static_cast<uint32_t>(std::pow(m_probecountperside, 3)),
        .stride_in_byte = sizeof(float) * 3
    };

    m_rayhitposition = buf_mngr->AllocBuffer(rayhitposition_buf_desc);

    BufferDesc raydirection_buf_desc = {
        .name = "ddgi_raydirection",
        .flag = EBufferFlag::AllowDisplay,
        .width = static_cast<uint32_t>(m_irradiancerays_perprobe),
        .height = static_cast<uint32_t>(std::pow(m_probecountperside, 3)),
        .stride_in_byte = sizeof(float) * 3
    };

    m_raydirection = buf_mngr->AllocBuffer(raydirection_buf_desc);

    BufferDesc rayhitnormal_buf_desc = {
        .name = "ddgi_rayhitnormal",
        .flag = EBufferFlag::AllowDisplay,
        .width = static_cast<uint32_t>(m_irradiancerays_perprobe),
        .height = static_cast<uint32_t>(std::pow(m_probecountperside, 3)),
        .stride_in_byte = sizeof(float) * 3
    };

    m_rayhitnormal = buf_mngr->AllocBuffer(rayhitnormal_buf_desc);

    BufferDesc probeirradiance_buf_desc = {
        .name = "ddgi_probeirradiance",
        .flag = EBufferFlag::AllowDisplay,
        .width = static_cast<uint32_t>(m_probeirradiancesize.w),
        .height = static_cast<uint32_t>(m_probeirradiancesize.h),
        .stride_in_byte = sizeof(float) * 4
    };

    m_probeirradiance = buf_mngr->AllocBuffer(probeirradiance_buf_desc);

    BufferDesc probedepth_buf_desc = {
        .name = "ddgi_probedepth",
        .flag = EBufferFlag::AllowDisplay,
        .width = static_cast<uint32_t>(m_probeirradiancesize.w),
        .height = static_cast<uint32_t>(m_probeirradiancesize.h),
        .stride_in_byte = sizeof(float) * 4
    };

    m_probedepth = buf_mngr->AllocBuffer(probedepth_buf_desc);

    BufferDesc probeposition_buf_desc = {
        .name = "ddgi_probeposition",
        .flag = EBufferFlag::AllowDisplay,
        .width = static_cast<uint32_t>(m_probecountperside * m_probecountperside * m_probecountperside),
        .height = static_cast<uint32_t>(1),
        .stride_in_byte = sizeof(float) * 3
    };
    m_probeposition = buf_mngr->AllocBuffer(probeposition_buf_desc);

    // 确定probe位置
    util::AABB aabb = world->GetAABB();
    float3 min = float3(aabb.min.x, aabb.min.y, aabb.min.z);
    float3 max = float3(aabb.max.x, aabb.max.y, aabb.max.z);
    // min.y = 1.0f;
    float shrink = 0.7f;
    float3 center = (min + max) / 2.0f;
    min = center - (center - min) * shrink;
    max = center + (max - center) * shrink;

    m_probestartpos = min;

    m_probestep =
        make_float3((max.x - min.x) / float(m_probecountperside - 1), (max.y - min.y) / float(m_probecountperside - 1),
                    (max.z - min.z) / float(m_probecountperside - 1));
    for (int i = 0; i < m_probecountperside; i++) {
        for (int j = 0; j < m_probecountperside; j++) {
            for (int k = 0; k < m_probecountperside; k++) {
                m_probepos.push_back(min + make_float3(k * m_probestep.x, j * m_probestep.y, i * m_probestep.z));
            }
        }
    }
    // m_probestartpos = make_float3(-0.918999970, 0.298500001, -0.938499928);
    // m_probestep = make_float3(1.81799996, 1.39299989, 1.82699990);
    // m_probepos.push_back(make_float3(-0.918999970, 0.298500001, -0.938499928));
    // m_probepos.push_back(make_float3(0.898999989, 0.298500001, -0.938499928));
    // m_probepos.push_back(make_float3(-0.918999970, 1.69149995, -0.938499928));
    // m_probepos.push_back(make_float3(0.898999989, 1.69149995, -0.938499928));
    // m_probepos.push_back(make_float3(-0.918999970, 0.298500001, 0.888499975));
    // m_probepos.push_back(make_float3(0.898999989, 0.298500001, 0.888499975));
    // m_probepos.push_back(make_float3(-0.918999970, 1.69149995, 0.888499975));
    // m_probepos.push_back(make_float3(0.898999989, 1.69149995, 0.888499975));

    float3 boundingboxlength = max - min;
    m_maxdistance = length(boundingboxlength / make_float3(m_probecountperside)) * 1.5f;

    CUDA_FREE(m_probepos_cuda_memory);
    m_probepos_cuda_memory = cuda::CudaMemcpyToDevice(m_probepos.data(), m_probepos.size() * sizeof(float3));

    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void *>(m_probeposition->cuda_ptr),
        reinterpret_cast<void *>(m_probepos_cuda_memory),
        m_probecountperside * m_probecountperside * m_probecountperside * sizeof(float3),
        cudaMemcpyKind::cudaMemcpyDeviceToDevice, m_stream->GetStream()));

    CUDA_CHECK(cudaStreamSynchronize(m_stream->GetStream()));

    // m_optix_launch_params.probepos.SetData(m_probepos_cuda_memory, m_probepos.size());
    m_optix_launch_params.probepos.SetData(m_probeposition->cuda_ptr, m_probepos.size());
    m_update_params.rayorigin.SetData(m_probeposition->cuda_ptr, m_probepos.size());

    m_optix_launch_params.rayradiance.SetData(0, 0);
    m_optix_launch_params.rayhitposition.SetData(0, 0);
    m_optix_launch_params.raydirection.SetData(0, 0);
    m_optix_launch_params.rayhitnormal.SetData(0, 0);
    m_update_params.probeirradiance.SetData(0, 0);
    m_update_params.probedepth.SetData(0, 0);

    m_optix_launch_params.handle = world->GetIASHandle(2, true);
    m_optix_launch_params.emitters = world->emitters->GetEmitterGroup();

    optix::SBTDesc<SBTTypes> desc{};
    desc.ray_gen_data = { .program = "__raygen__main", .data = SBTTypes::RayGenDataType{} };
    {
        int emitter_index_offset = 0;
        using HitGroupDataRecord = optix::ProgDataDescPair<SBTTypes::HitGroupDataType>;
        for (auto &&ro : world->GetRenderobjects()) {
            HitGroupDataRecord hit_default_data{};
            hit_default_data.program = "__closesthit__default";
            hit_default_data.data.mat = ro->mat;
            hit_default_data.data.geo = ro->geo;
            if (ro->is_emitter) {
                hit_default_data.data.emitter_index_offset = emitter_index_offset;
                emitter_index_offset += ro->sub_emitters_num;
            }

            desc.hit_datas.push_back(hit_default_data);

            HitGroupDataRecord hit_shadow_data{};
            hit_shadow_data.program = "__closesthit__shadow";
            desc.hit_datas.push_back(hit_shadow_data);
        }
    }
    {
        optix::ProgDataDescPair<SBTTypes::MissDataType> miss_data = { .program = "__miss__default",
                                                                      .data = SBTTypes::MissDataType{} };
        desc.miss_datas.push_back(miss_data);
        optix::ProgDataDescPair<SBTTypes::MissDataType> miss_shadow_data = { .program = "__miss__shadow",
                                                                             .data = SBTTypes::MissDataType{} };
        desc.miss_datas.push_back(miss_shadow_data);
    }
    {
        auto mat_programs = Pupil::resource::GetMaterialProgramDesc();
        for (auto &mat_prog : mat_programs) {
            if (mat_prog.cc_entry) {
                optix::ProgDataDescPair<SBTTypes::CallablesDataType> cc_data = {
                    .program = mat_prog.cc_entry
                };
                desc.callables_datas.push_back(cc_data);
            }
            if (mat_prog.dc_entry) {
                optix::ProgDataDescPair<SBTTypes::CallablesDataType> dc_data = {
                    .program = mat_prog.dc_entry
                };
                desc.callables_datas.push_back(dc_data);
            }
        }
    }
    m_optix_pass->InitSBT(desc);

    m_dirty = true;
}

void ProbePass::BindingEventCallback() noexcept {
    EventBinder<EWorldEvent::CameraChange>([this](void *) { m_dirty = true; });

    EventBinder<ESystemEvent::SceneLoad>([this](void *p) { SetScene((world::World *)p); });
}

void ProbePass::Inspector() noexcept {
    ImGui::InputFloat("energy conservation", &m_energyconservation, 0.05f);
    ImGui::InputInt("probe perside", &m_probecountperside, 2);
    if (m_energyconservation < 0.0f) {
        m_energyconservation = 0.0f;
    } else if (m_energyconservation > 1.0f) {
        m_energyconservation = 1.0f;
    }

    if (m_optix_launch_params.energyConservation != m_energyconservation) {
        m_dirty = true;
    }
}
}// namespace Pupil::ddgi::probe