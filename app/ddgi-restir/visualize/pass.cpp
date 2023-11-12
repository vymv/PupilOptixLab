#include "pass.h"
#include "../indirect/global.h"
#include "imgui.h"

#include "cuda/context.h"
#include "optix/context.h"
#include "optix/module.h"

#include "system/system.h"
#include "system/gui/gui.h"
#include "util/event.h"

extern "C" char ddgi_visualize_pass_embedded_ptx_code[];

namespace Pupil {
extern uint32_t g_window_w;
extern uint32_t g_window_h;
}// namespace Pupil

namespace {
double m_time_cnt = 1.f;
float m_probe_visualize_size = 10.0f;
int highlight_index = -1;

}// namespace

namespace Pupil::ddgi::visualize {
VisualizePass::VisualizePass(std::string_view name) noexcept : Pass(name) {
    auto optix_ctx = util::Singleton<optix::Context>::instance();
    auto cuda_ctx = util::Singleton<cuda::Context>::instance();
    m_stream = std::make_unique<cuda::Stream>();
    m_optix_pass =
        std::make_unique<optix::Pass<SBTTypes, OptixLaunchParams>>(optix_ctx->context, m_stream->GetStream());

    InitOptixPipeline();
    BindingEventCallback();
}

void VisualizePass::OnRun() noexcept {

    if (!is_pathtracer && m_enable_visualize) {

        if (m_dirty) {
            auto proj = m_world_camera->GetProjectionMatrix();
            auto view = m_world_camera->GetViewMatrix();
            m_optix_launch_params.proj_view = Pupil::ToCudaType(proj * view);
            m_optix_launch_params.probe_visualize_size = m_probe_visualize_size;
            m_optix_launch_params.highlight_index = highlight_index;
        }

        m_optix_pass->Run(m_optix_launch_params, m_optix_launch_params.config.frame.width,
                          m_optix_launch_params.config.frame.height);
        m_optix_pass->Synchronize();

        m_time_cnt = m_timer.ElapsedMilliseconds();

    } else {
        auto buf_mngr = util::Singleton<BufferManager>::instance();
        auto *frame_buffer = buf_mngr->GetBuffer(buf_mngr->DEFAULT_FINAL_RESULT_BUFFER_NAME);
        auto *result_buffer = buf_mngr->GetBuffer("result buffer");
        CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<void *>(frame_buffer->cuda_ptr),
            reinterpret_cast<void *>(result_buffer->cuda_ptr),
            m_optix_launch_params.config.frame.width * m_optix_launch_params.config.frame.height * sizeof(float4),
            cudaMemcpyKind::cudaMemcpyDeviceToDevice, m_stream->GetStream()));

        CUDA_CHECK(cudaStreamSynchronize(m_stream->GetStream()));
    }
}

void VisualizePass::InitOptixPipeline() noexcept {
    auto module_mngr = util::Singleton<optix::ModuleManager>::instance();

    auto sphere_module = module_mngr->GetModule(optix::EModuleBuiltinType::SpherePrimitive);
    auto pt_module = module_mngr->GetModule(ddgi_visualize_pass_embedded_ptx_code);

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
    }
    // mesh的求交使用built-in的三角形求交模块（默认）
    // 球的求交使用built-in的球求交模块(这里使用module_mngr->GetModule(OPTIX_PRIMITIVE_TYPE_SPHERE)生成)
    {
        optix::RayTraceProgramDesc forward_ray_desc{
            .module_ptr = pt_module,
            .hit_group = { .ch_entry = "__closesthit__default",
                           .intersect_module = sphere_module },
        };
        pipeline_desc.ray_trace_programs.push_back(forward_ray_desc);
    }
    {
        auto mat_programs = Pupil::resource::GetMaterialProgramDesc();
        pipeline_desc.callable_programs.insert(
            pipeline_desc.callable_programs.end(),
            mat_programs.begin(), mat_programs.end());
    }
    m_optix_pass->InitPipeline(pipeline_desc);
}

void VisualizePass::SetScene(world::World *world) noexcept {

    m_world_camera = world->camera.get();
    m_optix_launch_params.config.frame.width = world->scene->sensor.film.w;
    m_optix_launch_params.config.frame.height = world->scene->sensor.film.h;

    auto buf_mngr = util::Singleton<BufferManager>::instance();
    auto *frame_buffer = buf_mngr->GetBuffer(buf_mngr->DEFAULT_FINAL_RESULT_BUFFER_NAME);
    m_optix_launch_params.visualize_buffer.SetData(frame_buffer->cuda_ptr, m_optix_launch_params.config.frame.width * m_optix_launch_params.config.frame.height);

    auto pos_buf = buf_mngr->GetBuffer("ddgi_probeposition");
    m_optix_launch_params.probe_position.SetData(pos_buf->cuda_ptr, m_probecountperside * m_probecountperside * m_probecountperside);
    m_optix_launch_params.probe_count = m_probecountperside * m_probecountperside * m_probecountperside;

    auto result_buf = buf_mngr->GetBuffer("result buffer");
    m_optix_launch_params.input_buffer.SetData(result_buf->cuda_ptr, m_optix_launch_params.config.frame.width * m_optix_launch_params.config.frame.height);

    m_optix_launch_params.probe_visualize_size = m_probe_visualize_size;
    m_optix_launch_params.highlight_index = highlight_index;

    // 将场景数据绑定到sbt中
    optix::SBTDesc<SBTTypes> desc{};
    desc.ray_gen_data = {
        .program = "__raygen__main"
    };
    {
        int emitter_index_offset = 0;
        using HitGroupDataRecord = optix::ProgDataDescPair<SBTTypes::HitGroupDataType>;
        for (auto &&ro : world->GetRenderobjects()) {
            HitGroupDataRecord hit_default_data{};
            hit_default_data.program = "__closesthit__default";
            desc.hit_datas.push_back(hit_default_data);
        }
    }
    {
        optix::ProgDataDescPair<SBTTypes::MissDataType> miss_data = {
            .program = "__miss__default"
        };
        desc.miss_datas.push_back(miss_data);
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

void VisualizePass::BindingEventCallback() noexcept {
    EventBinder<EWorldEvent::CameraChange>([this](void *) { m_dirty = true; });

    EventBinder<ESystemEvent::SceneLoad>([this](void *p) { SetScene((world::World *)p); });
}

void VisualizePass::Inspector() noexcept {

    ImGui::Checkbox("enable", &m_enable_visualize);
    if (m_optix_launch_params.probe_visualize_size != m_probe_visualize_size) {
        m_dirty = true;
    }
    ImGui::InputFloat("size", &m_probe_visualize_size, 0.05f);
    ImGui::InputInt("high light probe", &highlight_index);
}
}// namespace Pupil::ddgi::visualize