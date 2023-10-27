#include "pass.h"
#include "imgui.h"
#include "cuda/context.h"
#include "optix/context.h"
#include "optix/module.h"

#include "util/event.h"
#include "system/gui/gui.h"
#include "system/system.h"
#include "../indirect/global.h"

// 构建期通过CMake将.cu代码编译并嵌入到.c文件中，
// 代码指令由char类型保存，只需要声明extern即可获取
extern "C" char ddgi_shading_pass_embedded_ptx_code[];

namespace Pupil {
extern uint32_t g_window_w;
extern uint32_t g_window_h;
}// namespace Pupil

namespace {
int m_max_depth;
double m_time_cnt = 1.;

int m_show_type = 0;
Pupil::world::World *m_world;
}// namespace

namespace Pupil::ddgi::shading {
ShadingPass::ShadingPass(std::string_view name) noexcept : Pass(name) {
    auto optix_ctx = util::Singleton<optix::Context>::instance();
    auto cuda_ctx = util::Singleton<cuda::Context>::instance();
    m_stream = std::make_unique<cuda::Stream>();
    m_optix_pass =
        std::make_unique<optix::Pass<SBTTypes, OptixLaunchParams>>(optix_ctx->context, m_stream->GetStream());
    InitOptixPipeline();
    BindingEventCallback();
}

void ShadingPass::OnRun() noexcept {

    if (!is_pathtracer) {
        m_timer.Start();
        {
            auto buf_mngr = util::Singleton<BufferManager>::instance();
            auto *frame_buffer = buf_mngr->GetBuffer(buf_mngr->DEFAULT_FINAL_RESULT_BUFFER_NAME);

            m_optix_launch_params.frame_buffer.SetData(frame_buffer->cuda_ptr, m_optix_launch_params.config.frame.width * m_optix_launch_params.config.frame.height);
            m_optix_pass->Run(m_optix_launch_params, m_optix_launch_params.config.frame.width, m_optix_launch_params.config.frame.height);
            m_optix_pass->Synchronize();
        }
        m_timer.Stop();
    }
}

void ShadingPass::InitOptixPipeline() noexcept {
    auto module_mngr = util::Singleton<optix::ModuleManager>::instance();

    auto sphere_module = module_mngr->GetModule(optix::EModuleBuiltinType::SpherePrimitive);
    auto pt_module = module_mngr->GetModule(ddgi_shading_pass_embedded_ptx_code);

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

void ShadingPass::SetScene(world::World *world) noexcept {
    m_world = world;
    // 对于场景初始化参数
    m_world_camera = world->camera.get();
    m_optix_launch_params.config.frame.width = world->scene->sensor.film.w;
    m_optix_launch_params.config.frame.height = world->scene->sensor.film.h;

    auto buf_mngr = util::Singleton<BufferManager>::instance();

    auto reservoir_buf = buf_mngr->GetBuffer("final screen reservoir");
    m_optix_launch_params.final_reservoirs.SetData(reservoir_buf->cuda_ptr, m_optix_launch_params.config.frame.width * m_optix_launch_params.config.frame.height);
    auto albedo_buf = buf_mngr->GetBuffer("gbuffer albedo");
    m_optix_launch_params.albedo_buffer.SetData(albedo_buf->cuda_ptr, m_optix_launch_params.config.frame.width * m_optix_launch_params.config.frame.height);
    auto position_buf = buf_mngr->GetBuffer("gbuffer position");
    m_optix_launch_params.position_buffer.SetData(position_buf->cuda_ptr, m_optix_launch_params.config.frame.width * m_optix_launch_params.config.frame.height);
    auto normal_buf = buf_mngr->GetBuffer("gbuffer normal");
    m_optix_launch_params.normal_buffer.SetData(normal_buf->cuda_ptr, m_optix_launch_params.config.frame.width * m_optix_launch_params.config.frame.height);
    auto emission_buf = buf_mngr->GetBuffer("gbuffer emission");
    m_optix_launch_params.emission_buffer.SetData(emission_buf->cuda_ptr, m_optix_launch_params.config.frame.width * m_optix_launch_params.config.frame.height);
    m_optix_launch_params.frame_buffer.SetData(0, 0);
    m_optix_launch_params.handle = world->GetIASHandle(2, true);

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

void ShadingPass::BindingEventCallback() noexcept {
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

void ShadingPass::Inspector() noexcept {
}
}// namespace Pupil::ddgi::shading