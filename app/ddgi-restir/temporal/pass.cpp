#include "pass.h"
#include "imgui.h"
#include "cuda/context.h"
#include "optix/context.h"
#include "optix/module.h"

#include "util/event.h"
#include "system/gui/gui.h"
#include "system/system.h"

// 构建期通过CMake将.cu代码编译并嵌入到.c文件中，
// 代码指令由char类型保存，只需要声明extern即可获取
extern "C" char ddgi_temporal_pass_embedded_ptx_code[];

namespace Pupil {
extern uint32_t g_window_w;
extern uint32_t g_window_h;
}// namespace Pupil

namespace {

// int m_spatial_radius = 30;
bool m_on = false;
Pupil::world::World *m_world;
mat4x4 m_camera_proj_view;

}// namespace
extern mat4x4 prev_camera_proj_view;

namespace Pupil::ddgi::temporal {
TemporalPass::TemporalPass(std::string_view name) noexcept : Pass(name) {
    auto optix_ctx = util::Singleton<optix::Context>::instance();
    auto cuda_ctx = util::Singleton<cuda::Context>::instance();
    m_stream = std::make_unique<cuda::Stream>();
    m_optix_pass =
        std::make_unique<optix::Pass<SBTTypes, OptixLaunchParams>>(optix_ctx->context, m_stream->GetStream());
    InitOptixPipeline();
    BindingEventCallback();
}

void TemporalPass::OnRun() noexcept {

    if (!is_pathtracer) {
        m_timer.Start();
        {
            if (m_on) {

                auto buf_mngr = util::Singleton<Pupil::BufferManager>::instance();
                m_optix_launch_params.camera.prev_proj_view = prev_camera_proj_view;

                m_optix_pass->Run(m_optix_launch_params, m_optix_launch_params.config.frame.width, m_optix_launch_params.config.frame.height);

                auto reservoir_buf = buf_mngr->GetBuffer("screen reservoir");
                auto prev_reservoir_buf = buf_mngr->GetBuffer("prev screen reservoir");

                auto pos_buf = buf_mngr->GetBuffer("gbuffer position");
                auto prev_pos_buf = buf_mngr->GetBuffer("prev gbuffer position");

                CUDA_CHECK(cudaMemcpyAsync(
                    reinterpret_cast<void *>(prev_reservoir_buf->cuda_ptr),
                    reinterpret_cast<void *>(reservoir_buf->cuda_ptr),
                    m_optix_launch_params.config.frame.height * m_optix_launch_params.config.frame.width * sizeof(Reservoir), cudaMemcpyKind::cudaMemcpyDeviceToDevice, m_stream->GetStream()));
                CUDA_CHECK(cudaMemcpyAsync(
                    reinterpret_cast<void *>(prev_pos_buf->cuda_ptr),
                    reinterpret_cast<void *>(pos_buf->cuda_ptr),
                    m_optix_launch_params.config.frame.height * m_optix_launch_params.config.frame.width * sizeof(float4), cudaMemcpyKind::cudaMemcpyDeviceToDevice, m_stream->GetStream()));

                m_optix_launch_params.prev_frame_reservoirs.SetData(prev_reservoir_buf->cuda_ptr, m_optix_launch_params.config.frame.height * m_optix_launch_params.config.frame.width);
                m_optix_launch_params.prev_position.SetData(prev_pos_buf->cuda_ptr, m_optix_launch_params.config.frame.height * m_optix_launch_params.config.frame.width);

                m_optix_pass->Synchronize();
            }
        }
        m_timer.Stop();
    }
}

void TemporalPass::InitOptixPipeline() noexcept {
    auto module_mngr = util::Singleton<optix::ModuleManager>::instance();
    auto pt_module = module_mngr->GetModule(ddgi_temporal_pass_embedded_ptx_code);

    optix::PipelineDesc pipeline_desc;
    {
        // for mesh(triangle) geo
        optix::RayTraceProgramDesc forward_ray_desc{
            .module_ptr = pt_module,
            .ray_gen_entry = "__raygen__main",
        };
        pipeline_desc.ray_trace_programs.push_back(forward_ray_desc);
    }
    m_optix_pass->InitPipeline(pipeline_desc);
}

void TemporalPass::SetScene(world::World *world) noexcept {
    m_world = world;
    // 对于场景初始化参数
    m_world_camera = world->camera.get();
    m_optix_launch_params.config.frame.width = world->scene->sensor.film.w;
    m_optix_launch_params.config.frame.height = world->scene->sensor.film.h;
    auto m_output_pixel_num = m_optix_launch_params.config.frame.width * m_optix_launch_params.config.frame.height;

    auto buf_mngr = util::Singleton<BufferManager>::instance();

    BufferDesc prev_reservoir_buf_desc = {
        .name = "prev screen reservoir",
        .flag = EBufferFlag::None,
        .width = m_optix_launch_params.config.frame.width,
        .height = m_optix_launch_params.config.frame.height,
        .stride_in_byte = sizeof(Reservoir)
    };
    auto prev_reservoir_buf = buf_mngr->AllocBuffer(prev_reservoir_buf_desc);
    m_optix_launch_params.prev_frame_reservoirs.SetData(prev_reservoir_buf->cuda_ptr, m_output_pixel_num);

    BufferDesc prev_position_buf_desc = {
        .name = "prev gbuffer position",
        .flag = EBufferFlag::None,
        .width = m_optix_launch_params.config.frame.width,
        .height = m_optix_launch_params.config.frame.height,
        .stride_in_byte = sizeof(float4)
    };
    auto prev_position_buf = buf_mngr->AllocBuffer(prev_position_buf_desc);
    m_optix_launch_params.prev_position.SetData(prev_position_buf->cuda_ptr, m_output_pixel_num);

    auto reservoir_buf = buf_mngr->GetBuffer("screen reservoir");
    m_optix_launch_params.reservoirs.SetData(reservoir_buf->cuda_ptr, m_output_pixel_num);

    auto position_buf = buf_mngr->GetBuffer("gbuffer position");
    m_optix_launch_params.position_buffer.SetData(position_buf->cuda_ptr, m_output_pixel_num);

    m_optix_launch_params.random_seed = 2;

    auto proj = m_world_camera->GetProjectionMatrix();
    auto view = m_world_camera->GetViewMatrix();
    m_camera_proj_view = Pupil::ToCudaType(proj * view);

    // 将场景数据绑定到sbt中
    optix::SBTDesc<SBTTypes> desc{};
    desc.ray_gen_data = {
        .program = "__raygen__main"
    };
    m_optix_pass->InitSBT(desc);

    m_dirty = true;
}

void TemporalPass::BindingEventCallback() noexcept {
    EventBinder<EWorldEvent::CameraChange>([this](void *) {
        if (m_world_camera) {
            auto proj = m_world_camera->GetProjectionMatrix();
            auto view = m_world_camera->GetViewMatrix();
            m_camera_proj_view = Pupil::ToCudaType(proj * view);
        }
    });

    EventBinder<EWorldEvent::RenderInstanceUpdate>([this](void *) {
        m_dirty = true;
    });

    EventBinder<ESystemEvent::SceneLoad>([this](void *p) {
        SetScene((world::World *)p);
    });
}

void TemporalPass::Inspector() noexcept {
    // ImGui::Text("cost: %d ms", (int)m_time_cost);
    // ImGui::Checkbox("use Spatial reuse", &m_flag);
    ImGui::Checkbox("use temporal reuse", &m_on);
    // ImGui::InputInt("spatial radius", &m_spatial_radius);
    // m_spatial_radius = clamp(m_spatial_radius, 0, 50);
    // if (m_optix_launch_params.spatial_radius != m_spatial_radius) {
    //     m_dirty = true;
    // }
}
}// namespace Pupil::ddgi::temporal