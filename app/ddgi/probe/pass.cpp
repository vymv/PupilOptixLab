#include "pass.h"
#include "../indirect/global.h"
#include "imgui.h"

#include "cuda/context.h"
#include "optix/context.h"
#include "optix/module.h"

#include "system/gui.h"
#include "system/system.h"
#include "util/event.h"

extern "C" char ddgi_probe_pass_embedded_ptx_code[];

namespace Pupil
{
extern uint32_t g_window_w;
extern uint32_t g_window_h;
} // namespace Pupil

namespace
{
double m_time_cnt = 1.f;
}

namespace Pupil::ddgi::probe
{
ProbePass::ProbePass(std::string_view name) noexcept : Pass(name)
{
    auto optix_ctx = util::Singleton<optix::Context>::instance();
    auto cuda_ctx = util::Singleton<cuda::Context>::instance();
    m_stream = std::make_unique<cuda::Stream>();
    m_optix_pass =
        std::make_unique<optix::Pass<SBTTypes, OptixLaunchParams>>(optix_ctx->context, m_stream->GetStream());

    InitOptixPipeline();
    BindingEventCallback();
}

float3 sphericalFibonacci(float i, float n)
{
    const float PHI = std::sqrt(5) * 0.5f + 0.5f;

    float madfrac = (i * (PHI - 1)) - std::floor(i * (PHI - 1));

    float phi = 2.0f * M_PIf * std::clamp(madfrac, 0.0f, 1.0f);
    float cosTheta = 1.0f - (2.0f * i + 1.0f) * (1.0f / n);
    float sinTheta = std::sqrt(std::clamp(1.0f - cosTheta * cosTheta, 0.0f, 1.0f));

    return make_float3(std::cos(phi) * sinTheta, std::sin(phi) * sinTheta, cosTheta);
}

util::Mat3 fromAxisAngle(const float3 axis, float fRadians)
{

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

void ProbePass::BeforeRunning() noexcept
{

    // random orientation
    const float3 axis =
        make_float3(std::rand() / float(RAND_MAX), std::rand() / float(RAND_MAX), std::rand() / float(RAND_MAX));
    util::Mat3 m = fromAxisAngle(normalize(axis), std::rand() / float(RAND_MAX) * 2.0 * M_PIf);

    CUDA_FREE(m_randomoriention_cuda_memory);
    m_randomoriention_cuda_memory = cuda::CudaMemcpyToDevice(&m, sizeof(m));
    m_optix_launch_params.randomOrientation.SetData(m_randomoriention_cuda_memory);
}

void ProbePass::Run() noexcept
{
    m_timer.Start();
    {

        // if (m_firstframe)
        // {
        //     Pupil::EventDispatcher<Pupil::ECanvasEvent::Resize>(raygbuffersize);
        //     m_firstframe = false;
        // }
        m_optix_launch_params.rayhitnormal.SetData(m_rayhitnormal->cuda_res.ptr,
                                                   m_raygbuffersize.w * m_raygbuffersize.h);
        m_optix_launch_params.raydirection.SetData(m_raydirection->cuda_res.ptr,
                                                   m_raygbuffersize.w * m_raygbuffersize.h);
        m_optix_launch_params.rayhitposition.SetData(m_rayhitposition->cuda_res.ptr,
                                                     m_raygbuffersize.w * m_raygbuffersize.h);
        m_optix_launch_params.rayradiance.SetData(m_rayradiance->cuda_res.ptr, m_raygbuffersize.w * m_raygbuffersize.h);
        // auto &frame_buffer = util::Singleton<GuiPass>::instance()->GetCurrentRenderOutputBuffer().shared_buffer;
        // m_optix_launch_params.rayradiance.SetData(frame_buffer.cuda_ptr, raygbuffersize.w * raygbuffersize.h);

        m_optix_pass->Run(m_optix_launch_params, m_optix_launch_params.config.frame.width,
                          m_optix_launch_params.config.frame.height);
        m_optix_pass->Synchronize();

        ++m_optix_launch_params.random_seed;
        // {
        //     std::string temp = "D:/Code/PupilOptixLab/probehitradiance" + std::to_string(m_frame_cnt) + ".hdr";
        //     std::filesystem::path path1{temp};

        //     auto image1 = new float[raygbuffersize.w * raygbuffersize.h * 4];
        //     memset(image1, 0, raygbuffersize.w * raygbuffersize.h * 4);
        //     cuda::CudaMemcpyToHost(image1, m_rayradiance->cuda_res.ptr,
        //                            raygbuffersize.w * raygbuffersize.h * 4 * sizeof(float));

        //     stbi_flip_vertically_on_write(true);
        //     stbi_write_hdr(path1.string().c_str(), raygbuffersize.w, raygbuffersize.h, 4, image1);
        //     delete[] image1;

        //     Pupil::Log::Info("image was saved successfully in [{}].\n", path1.string());
        // }

        // if (m_firstframe)
        // {
        //     Pupil::EventDispatcher<Pupil::ECanvasEvent::Resize>(size);
        //     m_firstframe = false;
        // }

        // 积分
        // 输入
        // gbuffer
        auto buf_mngr = util::Singleton<BufferManager>::instance();
        auto raygbuffer = buf_mngr->GetBuffer("ddgi_rayradiance");
        m_update_params.rayradiance.SetData(raygbuffer->cuda_res.ptr,
                                            m_irradiancerays_perprobe * std::pow(m_probecountperside, 3));
        // hitposition
        auto rayhitpositionbuffer = buf_mngr->GetBuffer("ddgi_rayhitposition");
        m_update_params.rayhitposition.SetData(rayhitpositionbuffer->cuda_res.ptr,
                                               m_irradiancerays_perprobe * std::pow(m_probecountperside, 3));
        // hitnormal
        auto rayhitnormalbuffer = buf_mngr->GetBuffer("ddgi_rayhitnormal");
        m_update_params.rayhitnormal.SetData(rayhitnormalbuffer->cuda_res.ptr,
                                             m_irradiancerays_perprobe * std::pow(m_probecountperside, 3));
        // direction
        auto raydirectionbuffer = buf_mngr->GetBuffer("ddgi_raydirection");
        m_update_params.raydirection.SetData(raydirectionbuffer->cuda_res.ptr,
                                             m_irradiancerays_perprobe * std::pow(m_probecountperside, 3));

        // origin
        CUDA_FREE(m_probepos_cuda_memory);
        m_probepos_cuda_memory = cuda::CudaMemcpyToDevice(m_probepos.data(), m_probepos.size() * sizeof(float3));
        m_update_params.rayorgin.SetData(m_probepos_cuda_memory, m_probepos.size());

        // 输出
        // auto &frame_buffer = util::Singleton<GuiPass>::instance()->GetCurrentRenderOutputBuffer().shared_buffer;
        // m_update_params.probeirradiance.SetData(frame_buffer.cuda_ptr, size.h * size.w);
        auto probeirradiancebuffer = buf_mngr->GetBuffer("ddgi_probeirradiance");
        m_update_params.probeirradiance.SetData(probeirradiancebuffer->cuda_res.ptr,
                                                m_probeirradiancesize.h * m_probeirradiancesize.w);

        UpdateProbeCPU(m_stream->GetStream(), m_update_params,
                       make_uint2(m_probeirradiancesize.w, m_probeirradiancesize.h), m_irradiancerays_perprobe,
                       m_probesidelength, m_maxdistance, m_firstframe ? 0.0f : m_hysteresis);

        m_stream->Synchronize();

        // {
        //     std::string tmp = "D:/Code/PupilOptixLab/probeirradiance" + std::to_string(m_frame_cnt) + ".hdr";
        //     std::filesystem::path path{tmp};

        //     auto image = new float[size.w * size.h * 4];
        //     memset(image, 0, size.w * size.h * 4);
        //     cuda::CudaMemcpyToHost(image, frame_buffer.cuda_ptr, size.w * size.h * 4 * sizeof(float));

        //     stbi_flip_vertically_on_write(true);
        //     stbi_write_hdr(path.string().c_str(), size.w, size.h, 4, image);
        //     delete[] image;

        //     Pupil::Log::Info("image was saved successfully in [{}].\n", path.string());
        //     ++m_frame_cnt;
        // }

        auto &frame_buffer = util::Singleton<GuiPass>::instance()->GetCurrentRenderOutputBuffer().shared_buffer;
        if (m_show_type == 0)
        {
            // pt
        }
        else if (m_show_type == 1)
        { // probeirradiance

            if (show_type_changed)
            {
                Pupil::EventDispatcher<Pupil::ECanvasEvent::Resize>(m_probeirradiancesize);
                show_type_changed = false;
            }

            auto buf_mngr = util::Singleton<BufferManager>::instance();
            auto probeirradiance = buf_mngr->GetBuffer("ddgi_probeirradiance");
            cudaMemcpyAsync((void *)frame_buffer.cuda_ptr, (void *)probeirradiance->cuda_res.ptr,
                            m_probeirradiancesize.w * m_probeirradiancesize.h * sizeof(float4),
                            cudaMemcpyKind::cudaMemcpyDeviceToDevice, m_stream->GetStream());
            cudaStreamSynchronize(m_stream->GetStream());
        }
        else if (m_show_type == 2)
        { // rayGbuffer
            if (show_type_changed)
            {
                Pupil::EventDispatcher<Pupil::ECanvasEvent::Resize>(m_raygbuffersize);
                show_type_changed = false;
            }

            auto buf_mngr = util::Singleton<BufferManager>::instance();
            auto rayGbuffer = buf_mngr->GetBuffer("ddgi_rayradiance");
            cudaMemcpyAsync((void *)frame_buffer.cuda_ptr, (void *)rayGbuffer->cuda_res.ptr,
                            m_raygbuffersize.w * m_raygbuffersize.h * sizeof(float4),
                            cudaMemcpyKind::cudaMemcpyDeviceToDevice, m_stream->GetStream());
            cudaStreamSynchronize(m_stream->GetStream());
        }
    }
    m_timer.Stop();
    m_time_cnt = m_timer.ElapsedMilliseconds();
}

void ProbePass::AfterRunning() noexcept
{
}

void ProbePass::InitOptixPipeline() noexcept
{
    auto module_mngr = util::Singleton<optix::ModuleManager>::instance();

    auto sphere_module = module_mngr->GetModule(OPTIX_PRIMITIVE_TYPE_SPHERE);
    auto pt_module = module_mngr->GetModule(ddgi_probe_pass_embedded_ptx_code);

    optix::PipelineDesc pipeline_desc;
    {
        // for mesh(triangle) geo
        optix::ProgramDesc desc{.module_ptr = pt_module,
                                .ray_gen_entry = "__raygen__main",
                                .hit_miss = "__miss__default",
                                .shadow_miss = "__miss__shadow",
                                .hit_group = {.ch_entry = "__closesthit__default"},
                                .shadow_grop = {.ch_entry = "__closesthit__shadow"}};
        pipeline_desc.programs.push_back(desc);
    }

    {
        // for sphere geo
        optix::ProgramDesc desc{.module_ptr = pt_module,
                                .hit_group = {.ch_entry = "__closesthit__default", .intersect_module = sphere_module},
                                .shadow_grop = {.ch_entry = "__closesthit__shadow", .intersect_module = sphere_module}};
        pipeline_desc.programs.push_back(desc);
    }
    m_optix_pass->InitPipeline(pipeline_desc);
}

void ProbePass::SetScene(World *world) noexcept
{

    // m_world_camera = world->optix_scene->camera.get();

    m_optix_launch_params.config.frame.width = m_irradiancerays_perprobe;
    m_optix_launch_params.config.frame.height = std::pow(m_probecountperside, 3);

    m_optix_launch_params.random_seed = 0;

    auto buf_mngr = util::Singleton<BufferManager>::instance();
    BufferDesc rayradiance_buf_desc = {
        .type = EBufferType::Cuda,
        .name = "ddgi_rayradiance",
        .size = static_cast<uint64_t>(m_irradiancerays_perprobe * std::pow(m_probecountperside, 3) * sizeof(float4))};

    m_rayradiance = buf_mngr->AllocBuffer(rayradiance_buf_desc);

    BufferDesc rayhitposition_buf_desc = {
        .type = EBufferType::Cuda,
        .name = "ddgi_rayhitposition",
        .size = static_cast<uint64_t>(m_irradiancerays_perprobe * std::pow(m_probecountperside, 3) * sizeof(float3))};

    m_rayhitposition = buf_mngr->AllocBuffer(rayhitposition_buf_desc);

    BufferDesc raydirection_buf_desc = {
        .type = EBufferType::Cuda,
        .name = "ddgi_raydirection",
        .size = static_cast<uint64_t>(m_irradiancerays_perprobe * std::pow(m_probecountperside, 3) * sizeof(float3))};

    m_raydirection = buf_mngr->AllocBuffer(raydirection_buf_desc);

    BufferDesc rayhitnormal_buf_desc = {
        .type = EBufferType::Cuda,
        .name = "ddgi_rayhitnormal",
        .size = static_cast<uint64_t>(m_irradiancerays_perprobe * std::pow(m_probecountperside, 3) * sizeof(float3))};

    m_rayhitnormal = buf_mngr->AllocBuffer(rayhitnormal_buf_desc);

    BufferDesc probeirradiance_buf_desc = {
        .type = EBufferType::Cuda,
        .name = "ddgi_probeirradiance",
        .size = static_cast<uint64_t>(m_probeirradiancesize.w * m_probeirradiancesize.h * sizeof(float4))};

    m_probeirradiance = buf_mngr->AllocBuffer(probeirradiance_buf_desc);

    // 确定probe位置
    float3 min = make_float3(world->scene->aabb.min.x, world->scene->aabb.min.y, world->scene->aabb.min.z);
    float3 max = make_float3(world->scene->aabb.max.x, world->scene->aabb.max.y, world->scene->aabb.max.z);
    // min.y = 1.0f;
    float shrink = 0.9f;
    float3 center = (min + max) / 2.0f;
    min = center - (center - min) * shrink;
    max = center + (max - center) * shrink;

    m_probestartpos = min;

    m_probestep =
        make_float3((max.x - min.x) / float(m_probecountperside - 1), (max.y - min.y) / float(m_probecountperside - 1),
                    (max.z - min.z) / float(m_probecountperside - 1));
    for (int i = 0; i < m_probecountperside; i++)
    {
        for (int j = 0; j < m_probecountperside; j++)
        {
            for (int k = 0; k < m_probecountperside; k++)
            {
                m_probepos.push_back(min + make_float3(k * m_probestep.x, j * m_probestep.y, i * m_probestep.z));
            }
        }
    }

    float3 boundingboxlength = max - min;
    m_maxdistance = length(boundingboxlength / make_float3(m_probecountperside)) * 1.5f;

    CUDA_FREE(m_probepos_cuda_memory);
    m_probepos_cuda_memory = cuda::CudaMemcpyToDevice(m_probepos.data(), m_probepos.size() * sizeof(float3));
    m_optix_launch_params.probepos.SetData(m_probepos_cuda_memory, m_probepos.size());

    m_optix_launch_params.rayradiance.SetData(0, 0);
    m_optix_launch_params.rayhitposition.SetData(0, 0);
    m_optix_launch_params.raydirection.SetData(0, 0);
    m_optix_launch_params.rayhitnormal.SetData(0, 0);
    m_update_params.probeirradiance.SetData(0, 0);
    m_optix_launch_params.handle = world->optix_scene->ias_handle;
    m_optix_launch_params.emitters = world->optix_scene->emitters->GetEmitterGroup();

    SetSBT(world->scene.get());

    m_dirty = true;
}

void ProbePass::SetSBT(scene::Scene *scene) noexcept
{
    optix::SBTDesc<SBTTypes> desc{};
    desc.ray_gen_data = {.program_name = "__raygen__main", .data = SBTTypes::RayGenDataType{}};
    {
        int emitter_index_offset = 0;
        using HitGroupDataRecord = decltype(desc)::Pair<SBTTypes::HitGroupDataType>;
        for (auto &&shape : scene->shapes)
        {
            HitGroupDataRecord hit_default_data{};
            hit_default_data.program_name = "__closesthit__default";
            hit_default_data.data.mat.LoadMaterial(shape.mat);
            hit_default_data.data.geo.LoadGeometry(shape);
            if (shape.is_emitter)
            {
                hit_default_data.data.emitter_index_offset = emitter_index_offset;
                emitter_index_offset += shape.sub_emitters_num;
            }

            desc.hit_datas.push_back(hit_default_data);

            HitGroupDataRecord hit_shadow_data{};
            hit_shadow_data.program_name = "__closesthit__shadow";
            desc.hit_datas.push_back(hit_shadow_data);
        }
    }
    {
        decltype(desc)::Pair<SBTTypes::MissDataType> miss_data = {.program_name = "__miss__default",
                                                                  .data = SBTTypes::MissDataType{}};
        desc.miss_datas.push_back(miss_data);
        decltype(desc)::Pair<SBTTypes::MissDataType> miss_shadow_data = {.program_name = "__miss__shadow",
                                                                         .data = SBTTypes::MissDataType{}};
        desc.miss_datas.push_back(miss_shadow_data);
    }
    m_optix_pass->InitSBT(desc);
}

void ProbePass::BindingEventCallback() noexcept
{
    EventBinder<EWorldEvent::CameraChange>([this](void *) { m_dirty = true; });

    EventBinder<ESystemEvent::SceneLoad>([this](void *p) { SetScene((World *)p); });
}

void ProbePass::Inspector() noexcept
{
    // constexpr auto show_type = std::array{"render result", "albedo", "normal"};
    // constexpr auto show_type = std::array{"render result", "probeirradiance", "rayGbuffer"};
    // ImGui::Combo("result", &m_show_type, show_type.data(), (int)show_type.size());

    ImGui::Text("Rendering average %.3lf ms/frame (%.1lf FPS)", m_time_cnt, 1000.0f / m_time_cnt);
}
} // namespace Pupil::ddgi::probe