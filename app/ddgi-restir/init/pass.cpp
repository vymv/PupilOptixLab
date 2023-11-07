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
extern "C" char ddgi_render_pass_embedded_ptx_code[];

namespace Pupil {
extern uint32_t g_window_w;
extern uint32_t g_window_h;
}// namespace Pupil

namespace {
int m_max_depth;
int m_num_secondary = 16;
int m_num_emission = 16;
double m_time_cnt = 1.;
bool m_direct_on = true;
bool m_indirect_on = true;
Pupil::world::World *m_world;

}// namespace
mat4x4 prev_camera_proj_view;

namespace Pupil::ddgi::render {
RenderPass::RenderPass(std::string_view name) noexcept : Pass(name) {
    auto optix_ctx = util::Singleton<optix::Context>::instance();
    auto cuda_ctx = util::Singleton<cuda::Context>::instance();
    m_stream = std::make_unique<cuda::Stream>();
    m_optix_pass =
        std::make_unique<optix::Pass<SBTTypes, OptixLaunchParams>>(optix_ctx->context, m_stream->GetStream());
    InitOptixPipeline();
    BindingEventCallback();
}

void RenderPass::OnRun() noexcept {
    if (!is_pathtracer) {
        m_timer.Start();
        {
            prev_camera_proj_view = m_optix_launch_params.camera.proj_view;
            // 由于ui线程和渲染线程分离，所以在渲染前先检查是否通过ui修改了渲染参数
            if (m_dirty) {
                auto proj = m_world_camera->GetProjectionMatrix();
                auto view = m_world_camera->GetViewMatrix();
                m_optix_launch_params.camera.proj_view = Pupil::ToCudaType(proj * view);
                m_optix_launch_params.camera.view = m_world_camera->GetViewCudaMatrix();
                m_optix_launch_params.camera.camera_to_world = m_world_camera->GetToWorldCudaMatrix();
                m_optix_launch_params.camera.sample_to_camera = m_world_camera->GetSampleToCameraCudaMatrix();

                m_optix_launch_params.num_emission = m_num_emission;
                m_optix_launch_params.directOn = m_direct_on;
                m_optix_launch_params.random_seed = 0;
                m_optix_launch_params.handle = m_world->GetIASHandle(2, true);
                m_dirty = false;
            }

            auto buf_mngr = util::Singleton<BufferManager>::instance();
            m_optix_pass->Run(m_optix_launch_params, m_optix_launch_params.config.frame.width,
                              m_optix_launch_params.config.frame.height);
            m_optix_pass->Synchronize();// 等待optix渲染结束

            ++m_optix_launch_params.random_seed;
        }

        m_timer.Stop();
        m_time_cnt = m_timer.ElapsedMilliseconds();
    }
}

void RenderPass::InitOptixPipeline() noexcept {
    auto module_mngr = util::Singleton<optix::ModuleManager>::instance();

    auto sphere_module = module_mngr->GetModule(optix::EModuleBuiltinType::SpherePrimitive);
    auto pt_module = module_mngr->GetModule(ddgi_render_pass_embedded_ptx_code);

    // 创建optix pipeline，需要填写.cu中对应的函数入口，
    // 产生光线(每一个像素都会执行，相当于一个线程)：命名前缀必须是__raygen__
    // 光线击中(当optixTrace()发出的光线与场景相交(最近的交点)时会执行)：命名前缀必须是__closesthit__
    // 光线未击中：命名前缀必须是__miss__
    // 光追算法需要追踪view ray和shadow
    // ray(判断是否在阴影中，即交点与光源中间是否有遮挡)
    // 因此对closesthit和miss需要有default和shadow两种处理，分别对应两个函数的入口
    // 但在gbuffer渲染时不需要shadow ray
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
    // mesh的求交使用built-in的三角形求交模块（默认）
    // 球的求交使用built-in的球求交模块(这里使用module_mngr->GetModule(OPTIX_PRIMITIVE_TYPE_SPHERE)生成)
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

void RenderPass::SetScene(world::World *world) noexcept {
    m_world = world;
    // 对于场景初始化参数
    m_world_camera = world->camera.get();
    m_optix_launch_params.config.frame.width = world->scene->sensor.film.w;
    m_optix_launch_params.config.frame.height = world->scene->sensor.film.h;
    // m_optix_launch_params.config.max_depth = world->scene->integrator.max_depth;
    m_output_pixel_num = m_optix_launch_params.config.frame.width * m_optix_launch_params.config.frame.height;

    auto buf_mngr = util::Singleton<BufferManager>::instance();
    BufferDesc position_buf_desc = {
        .name = "gbuffer position",
        .flag = EBufferFlag::AllowDisplay,
        .width = m_optix_launch_params.config.frame.width,
        .height = m_optix_launch_params.config.frame.height,
        .stride_in_byte = sizeof(float) * 4
    };
    auto position_buf = buf_mngr->AllocBuffer(position_buf_desc);
    m_optix_launch_params.position_buffer.SetData(position_buf->cuda_ptr, m_output_pixel_num);

    BufferDesc albedo_buf_desc = {
        .name = "gbuffer albedo",
        .flag = EBufferFlag::AllowDisplay,
        .width = m_optix_launch_params.config.frame.width,
        .height = m_optix_launch_params.config.frame.height,
        .stride_in_byte = sizeof(float) * 4
    };
    auto albedo_buf = buf_mngr->AllocBuffer(albedo_buf_desc);
    m_optix_launch_params.albedo_buffer.SetData(albedo_buf->cuda_ptr, m_output_pixel_num);

    BufferDesc normal_buf_desc = {
        .name = "gbuffer normal",
        .flag = EBufferFlag::AllowDisplay,
        .width = m_optix_launch_params.config.frame.width,
        .height = m_optix_launch_params.config.frame.height,
        .stride_in_byte = sizeof(float) * 4
    };
    auto normal_buf = buf_mngr->AllocBuffer(normal_buf_desc);
    m_optix_launch_params.normal_buffer.SetData(normal_buf->cuda_ptr, m_output_pixel_num);

    // BufferDesc emission_buf_desc = {
    //     .name = "gbuffer emission",
    //     .flag = EBufferFlag::AllowDisplay,
    //     .width = m_optix_launch_params.config.frame.width,
    //     .height = m_optix_launch_params.config.frame.height,
    //     .stride_in_byte = sizeof(float) * 3
    // };
    // auto emission_buf = buf_mngr->AllocBuffer(emission_buf_desc);
    // m_optix_launch_params.emission_buffer.SetData(emission_buf->cuda_ptr, m_output_pixel_num);

    BufferDesc reservoir_buf_desc{
        .name = "screen reservoir",
        .flag = EBufferFlag::None,
        .width = m_optix_launch_params.config.frame.width,
        .height = m_optix_launch_params.config.frame.height,
        .stride_in_byte = sizeof(Reservoir)
    };
    auto reservoir_buf = buf_mngr->AllocBuffer(reservoir_buf_desc);
    m_optix_launch_params.reservoirs.SetData(reservoir_buf->cuda_ptr, m_output_pixel_num);

    // m_max_depth = m_optix_launch_params.config.max_depth;
    m_num_secondary = 8;
    m_num_emission = 8;
    m_direct_on = true;
    m_indirect_on = true;

    m_optix_launch_params.random_seed = 0;
    // m_optix_launch_params.num_secondary = m_num_secondary;
    m_optix_launch_params.num_emission = m_num_emission;
    m_optix_launch_params.directOn = m_direct_on;
    // m_optix_launch_params.indirectOn = m_indirect_on;

    // m_optix_launch_params.frame_buffer.SetData(0, 0);
    m_optix_launch_params.handle = world->GetIASHandle(2, true);
    m_optix_launch_params.emitters = world->emitters->GetEmitterGroup();

    // m_optix_launch_params.probeStartPosition = m_probestartpos;
    // m_optix_launch_params.probeStep = m_probestep;
    // m_optix_launch_params.probeCount = make_int3(m_probecountperside, m_probecountperside, m_probecountperside);
    // m_optix_launch_params.probeirradiancesize = make_uint2(m_probeirradiancesize.w, m_probeirradiancesize.h);
    // m_optix_launch_params.probeSideLength = m_probesidelength;
    // m_optix_launch_params.probeirradiance.SetData(0, 0);

    m_dirty = true;

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
            hit_default_data.data.mat = ro->mat;
            hit_default_data.data.geo = ro->geo;
            if (ro->is_emitter) {
                hit_default_data.data.emitter_index_offset = emitter_index_offset;
                emitter_index_offset += ro->sub_emitters_num;
            }

            desc.hit_datas.push_back(hit_default_data);

            HitGroupDataRecord hit_shadow_data{};
            hit_shadow_data.program = "__closesthit__shadow";
            hit_shadow_data.data.mat.type = ro->mat.type;
            desc.hit_datas.push_back(hit_shadow_data);
        }
    }
    {
        optix::ProgDataDescPair<SBTTypes::MissDataType> miss_data = {
            .program = "__miss__default"
        };
        desc.miss_datas.push_back(miss_data);
        optix::ProgDataDescPair<SBTTypes::MissDataType> miss_shadow_data = {
            .program = "__miss__shadow"
        };
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
}

void RenderPass::BindingEventCallback() noexcept {
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

void RenderPass::Inspector() noexcept {
}
}// namespace Pupil::ddgi::render