#include "pass.h"
#include "imgui.h"

#include "cuda/context.h"
#include "optix/context.h"
#include "optix/module.h"

#include "system/gui.h"
#include "system/system.h"
#include "util/event.h"

extern "C" char ddgi_gbuffer_pass_embedded_ptx_code[];

namespace Pupil
{
extern uint32_t g_window_w;
extern uint32_t g_window_h;
} // namespace Pupil

namespace
{
double m_time_cnt = 1.f;
}

namespace Pupil::ddgi::gbuffer
{
GBufferPass::GBufferPass(std::string_view name) noexcept : Pass(name)
{
    auto optix_ctx = util::Singleton<optix::Context>::instance();
    auto cuda_ctx = util::Singleton<cuda::Context>::instance();
    m_stream = std::make_unique<cuda::Stream>();
    m_optix_pass =
        std::make_unique<optix::Pass<SBTTypes, OptixLaunchParams>>(optix_ctx->context, m_stream->GetStream());
    InitOptixPipeline();
    BindingEventCallback();
}

void GBufferPass::Run() noexcept
{
    m_timer.Start();
    {
        if (m_dirty)
        {
            m_optix_launch_params.camera.SetData(m_world_camera->GetCudaMemory());
            m_optix_launch_params.random_seed = 0;
            m_dirty = false;
        }

        m_optix_pass->Run(m_optix_launch_params, m_optix_launch_params.config.frame.width,
                          m_optix_launch_params.config.frame.height);
        m_optix_pass->Synchronize();

        ++m_optix_launch_params.random_seed;
    }
    m_timer.Stop();
    m_time_cnt = m_timer.ElapsedMilliseconds();
}

void GBufferPass::InitOptixPipeline() noexcept
{
    auto module_mngr = util::Singleton<optix::ModuleManager>::instance();

    auto sphere_module = module_mngr->GetModule(OPTIX_PRIMITIVE_TYPE_SPHERE);
    auto pt_module = module_mngr->GetModule(ddgi_gbuffer_pass_embedded_ptx_code);

    optix::PipelineDesc pipeline_desc;
    {
        // for mesh(triangle) geo
        optix::ProgramDesc desc{
            .module_ptr = pt_module,
            .ray_gen_entry = "__raygen__main",
            .hit_miss = "__miss__default",
            // .shadow_miss = "__miss__shadow",
            .hit_group = {.ch_entry = "__closesthit__default"},
            // .shadow_grop = { .ch_entry = "__closesthit__shadow" }
        };
        pipeline_desc.programs.push_back(desc);
    }

    {
        // for sphere geo
        optix::ProgramDesc desc{
            .module_ptr = pt_module,
            .hit_group = {.ch_entry = "__closesthit__default", .intersect_module = sphere_module},
            // .shadow_grop = { .ch_entry = "__closesthit__shadow",
            //                  .intersect_module = sphere_module }
        };
        pipeline_desc.programs.push_back(desc);
    }
    m_optix_pass->InitPipeline(pipeline_desc);
}

void GBufferPass::SetScene(World *world) noexcept
{
    m_world_camera = world->camera.get();
    m_optix_launch_params.config.frame.width = world->scene->sensor.film.w;
    m_optix_launch_params.config.frame.height = world->scene->sensor.film.h;

    m_optix_launch_params.random_seed = 0;

    m_output_pixel_num = m_optix_launch_params.config.frame.width * m_optix_launch_params.config.frame.height;

    auto buf_mngr = util::Singleton<BufferManager>::instance();
    BufferDesc albedo_buf_desc = {
        .type = EBufferType::Cuda, .name = "ddgi_albedo", .size = m_output_pixel_num * sizeof(float4)};
    m_albedo = buf_mngr->AllocBuffer(albedo_buf_desc);
    m_optix_launch_params.albedo.SetData(m_albedo->cuda_res.ptr, m_output_pixel_num);

    BufferDesc normal_buf_desc = {
        .type = EBufferType::Cuda, .name = "ddgi_normal", .size = m_output_pixel_num * sizeof(float4)};
    m_normal = buf_mngr->AllocBuffer(normal_buf_desc);
    m_optix_launch_params.normal.SetData(m_normal->cuda_res.ptr, m_output_pixel_num);

    m_optix_launch_params.handle = world->optix_scene->ias_handle;
    m_optix_launch_params.emitters = world->optix_scene->emitters->GetEmitterGroup();

    SetSBT(world->scene.get());

    m_dirty = true;
}

void GBufferPass::SetSBT(scene::Scene *scene) noexcept
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
                emitter_index_offset += shape.sub_emitters_num; // 前面的所有shape一共有多少emitter
            }

            desc.hit_datas.push_back(hit_default_data);

            HitGroupDataRecord hit_shadow_data{};
            hit_shadow_data.program_name = "__closesthit__default";
            desc.hit_datas.push_back(hit_shadow_data);
        }
    }
    {
        decltype(desc)::Pair<SBTTypes::MissDataType> miss_data = {.program_name = "__miss__default",
                                                                  .data = SBTTypes::MissDataType{}};
        desc.miss_datas.push_back(miss_data);
        decltype(desc)::Pair<SBTTypes::MissDataType> miss_shadow_data = {.program_name = "__miss__default",
                                                                         .data = SBTTypes::MissDataType{}};
        desc.miss_datas.push_back(miss_shadow_data);
    }
    m_optix_pass->InitSBT(desc);
}

void GBufferPass::BindingEventCallback() noexcept
{
    EventBinder<EWorldEvent::CameraChange>([this](void *) { m_dirty = true; });

    EventBinder<ESystemEvent::SceneLoad>([this](void *p) { SetScene((World *)p); });
}

void GBufferPass::Inspector() noexcept
{
    ImGui::Text("Rendering average %.3lf ms/frame (%.1lf FPS)", m_time_cnt, 1000.0f / m_time_cnt);
}
} // namespace Pupil::ddgi::gbuffer