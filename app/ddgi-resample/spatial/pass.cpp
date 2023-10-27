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
extern "C" char ddgi_spatial_pass_embedded_ptx_code[];

namespace Pupil {
extern uint32_t g_window_w;
extern uint32_t g_window_h;
}// namespace Pupil

namespace {

int m_spatial_radius = 30;
bool m_on = true;
Pupil::world::World *m_world;
}// namespace

namespace Pupil::ddgi::spatial {
SpatialPass::SpatialPass(std::string_view name) noexcept : Pass(name) {
    auto optix_ctx = util::Singleton<optix::Context>::instance();
    auto cuda_ctx = util::Singleton<cuda::Context>::instance();
    m_stream = std::make_unique<cuda::Stream>();
    m_optix_pass =
        std::make_unique<optix::Pass<SBTTypes, OptixLaunchParams>>(optix_ctx->context, m_stream->GetStream());
    InitOptixPipeline();
    BindingEventCallback();
}

void SpatialPass::OnRun() noexcept {

    if (!is_pathtracer) {
        m_timer.Start();
        {
            if (m_on) {
                m_optix_launch_params.spatial_radius = m_spatial_radius;
                m_optix_launch_params.camera.SetData(m_world_camera->GetCudaMemory());

                auto buf_mngr = util::Singleton<BufferManager>::instance();

                auto probeirradiancebuffer = buf_mngr->GetBuffer("ddgi_probeirradiance");
                m_optix_launch_params.probeirradiance.SetData(probeirradiancebuffer->cuda_ptr,
                                                              m_probeirradiancesize.w * m_probeirradiancesize.h);
                auto probedepthbuffer = buf_mngr->GetBuffer("ddgi_probedepth");
                m_optix_launch_params.probedepth.SetData(probedepthbuffer->cuda_ptr,
                                                         m_probeirradiancesize.w * m_probeirradiancesize.h);

                m_optix_pass->Run(m_optix_launch_params, m_optix_launch_params.config.frame.width, m_optix_launch_params.config.frame.height);
                m_optix_pass->Synchronize();
                m_optix_launch_params.random_seed += 3;
            } else {
                // just copy
                auto buf_mngr = util::Singleton<Pupil::BufferManager>::instance();
                auto reservoir_buf = buf_mngr->GetBuffer("screen reservoir");
                auto final_reservoir_buf = buf_mngr->GetBuffer("final screen reservoir");

                CUDA_CHECK(cudaMemcpyAsync(
                    reinterpret_cast<void *>(final_reservoir_buf->cuda_ptr),
                    reinterpret_cast<void *>(reservoir_buf->cuda_ptr),
                    m_optix_launch_params.config.frame.height * m_optix_launch_params.config.frame.width * sizeof(Reservoir),
                    cudaMemcpyKind::cudaMemcpyDeviceToDevice, m_stream->GetStream()));

                CUDA_CHECK(cudaStreamSynchronize(m_stream->GetStream()));
            }
        }
        m_timer.Stop();
    }
}

void SpatialPass::InitOptixPipeline() noexcept {
    auto module_mngr = util::Singleton<optix::ModuleManager>::instance();
    auto pt_module = module_mngr->GetModule(ddgi_spatial_pass_embedded_ptx_code);

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

void SpatialPass::SetScene(world::World *world) noexcept {
    m_world = world;
    // 对于场景初始化参数
    m_world_camera = world->camera.get();
    m_optix_launch_params.config.frame.width = world->scene->sensor.film.w;
    m_optix_launch_params.config.frame.height = world->scene->sensor.film.h;
    auto m_output_pixel_num = m_optix_launch_params.config.frame.width * m_optix_launch_params.config.frame.height;

    auto buf_mngr = util::Singleton<BufferManager>::instance();

    BufferDesc final_reservoir_buf_desc = {
        .name = "final screen reservoir",
        .flag = EBufferFlag::None,
        .width = m_optix_launch_params.config.frame.width,
        .height = m_optix_launch_params.config.frame.height,
        .stride_in_byte = sizeof(Reservoir)
    };
    auto final_reservoir_buf = buf_mngr->AllocBuffer(final_reservoir_buf_desc);
    m_optix_launch_params.final_reservoirs.SetData(final_reservoir_buf->cuda_ptr, m_output_pixel_num);

    auto reservoir_buf = buf_mngr->GetBuffer("screen reservoir");
    m_optix_launch_params.reservoirs.SetData(reservoir_buf->cuda_ptr, m_output_pixel_num);

    auto position_buf = buf_mngr->GetBuffer("gbuffer position");
    m_optix_launch_params.position_buffer.SetData(position_buf->cuda_ptr, m_output_pixel_num);

    m_optix_launch_params.random_seed = 2;
    m_optix_launch_params.emitters = world->emitters->GetEmitterGroup();
    m_spatial_radius = 30;

    m_optix_launch_params.probeStartPosition = m_probestartpos;
    m_optix_launch_params.probeStep = m_probestep;
    m_optix_launch_params.probeCount = make_int3(m_probecountperside, m_probecountperside, m_probecountperside);
    m_optix_launch_params.probeirradiancesize = make_uint2(m_probeirradiancesize.w, m_probeirradiancesize.h);
    m_optix_launch_params.probeSideLength = m_probesidelength;
    m_optix_launch_params.probeirradiance.SetData(0, 0);
    m_optix_launch_params.probedepth.SetData(0, 0);

    // 将场景数据绑定到sbt中
    optix::SBTDesc<SBTTypes> desc{};
    desc.ray_gen_data = {
        .program = "__raygen__main"
    };
    m_optix_pass->InitSBT(desc);

    m_dirty = true;
}

void SpatialPass::BindingEventCallback() noexcept {
    EventBinder<EWorldEvent::CameraChange>([this](void *) {
        m_dirty = true;
    });

    EventBinder<EWorldEvent::RenderInstanceUpdate>([this](void *) {
        m_dirty = true;
    });

    EventBinder<ESystemEvent::SceneLoad>([this](void *p) {
        SetScene((world::World *)p);
    });
}

void SpatialPass::Inspector() noexcept {
    // ImGui::Text("cost: %d ms", (int)m_time_cost);
    // ImGui::Checkbox("use Spatial reuse", &m_flag);
    ImGui::Checkbox("use spatial reuse", &m_on);
    ImGui::InputInt("spatial radius", &m_spatial_radius);
    m_spatial_radius = clamp(m_spatial_radius, 0, 50);
    if (m_optix_launch_params.spatial_radius != m_spatial_radius) {
        m_dirty = true;
    }
}
}// namespace Pupil::ddgi::spatial