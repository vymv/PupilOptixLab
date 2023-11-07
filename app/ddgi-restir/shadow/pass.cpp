#include "pass.h"
#include "imgui.h"
#include "cuda/context.h"
#include "optix/context.h"
#include "optix/module.h"

#include "util/event.h"
#include "system/gui/gui.h"
#include "system/system.h"
#include "../indirect/global.h"

extern "C" char ddgi_shadow_pass_embedded_ptx_code[];

namespace {
double m_time_cost = 0.f;
bool m_flag = true;
}// namespace

namespace Pupil::ddgi::shadow {
ShadowRayPass::ShadowRayPass(std::string_view name) noexcept
    : Pass(name) {
    auto optix_ctx = util::Singleton<optix::Context>::instance();
    auto cuda_ctx = util::Singleton<cuda::Context>::instance();
    m_stream = std::make_unique<cuda::Stream>();
    m_optix_pass = std::make_unique<optix::Pass<SBTTypes, ShadowRayPassLaunchParams>>(optix_ctx->context, m_stream->GetStream());
    InitOptixPipeline();
    BindingEventCallback();
}

void ShadowRayPass::OnRun() noexcept {
    if (!is_pathtracer) {
        m_timer.Start();
        if (m_flag) {
            m_optix_pass->Run(m_params, m_params.config.frame.width, m_params.config.frame.height);
            m_optix_pass->Synchronize();
        }
        m_timer.Stop();
        m_time_cost = m_timer.ElapsedMilliseconds();
    }
}

void ShadowRayPass::SetScene(world::World *world) noexcept {
    m_params.config.frame.width = world->scene->sensor.film.w;
    m_params.config.frame.height = world->scene->sensor.film.h;
    m_params.handle = world->GetIASHandle(2, true);

    m_output_pixel_num = m_params.config.frame.width * m_params.config.frame.height;

    auto buf_mngr = util::Singleton<BufferManager>::instance();

    auto pos_buf = buf_mngr->GetBuffer("gbuffer position");
    auto nor_buf = buf_mngr->GetBuffer("gbuffer normal");
    auto alb_buf = buf_mngr->GetBuffer("gbuffer albedo");

    m_params.position.SetData(pos_buf->cuda_ptr, m_output_pixel_num);
    m_params.normal.SetData(nor_buf->cuda_ptr, m_output_pixel_num);
    m_params.albedo.SetData(alb_buf->cuda_ptr, m_output_pixel_num);

    auto reservoir_buf = buf_mngr->GetBuffer("final screen reservoir");
    m_params.reservoirs.SetData(reservoir_buf->cuda_ptr, m_output_pixel_num);

    {
        optix::SBTDesc<SBTTypes> desc{};
        desc.ray_gen_data = {
            .program = "__raygen__main",
        };
        {
            using HitGroupDataRecord = optix::ProgDataDescPair<SBTTypes::HitGroupDataType>;
            for (auto &&ro : world->GetRenderobjects()) {
                HitGroupDataRecord hit_default_data{};
                hit_default_data.program = "__closesthit__default";

                desc.hit_datas.push_back(hit_default_data);
                desc.hit_datas.push_back(hit_default_data);
            }
        }
        {
            optix::ProgDataDescPair<SBTTypes::MissDataType> miss_data = {
                .program = "__miss__default",
            };
            desc.miss_datas.push_back(miss_data);
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
    }
}

void ShadowRayPass::InitOptixPipeline() noexcept {
    auto module_mngr = util::Singleton<optix::ModuleManager>::instance();
    auto rt_module = module_mngr->GetModule(ddgi_shadow_pass_embedded_ptx_code);
    auto sphere_module = module_mngr->GetModule(optix::EModuleBuiltinType::SpherePrimitive);

    optix::PipelineDesc pipeline_desc;
    {
        // for mesh(triangle) geo
        optix::RayTraceProgramDesc desc{
            .module_ptr = rt_module,
            .ray_gen_entry = "__raygen__main",
            .miss_entry = "__miss__default",
            .hit_group = { .ch_entry = "__closesthit__default" },
        };
        pipeline_desc.ray_trace_programs.push_back(desc);
    }

    {
        // for sphere geo
        optix::RayTraceProgramDesc desc{
            .module_ptr = rt_module,
            .hit_group = { .ch_entry = "__closesthit__default",
                           .intersect_module = sphere_module },
        };
        pipeline_desc.ray_trace_programs.push_back(desc);
    }
    {
        auto mat_programs = Pupil::resource::GetMaterialProgramDesc();
        pipeline_desc.callable_programs.insert(
            pipeline_desc.callable_programs.end(),
            mat_programs.begin(), mat_programs.end());
    }
    m_optix_pass->InitPipeline(pipeline_desc);
}

void ShadowRayPass::BindingEventCallback() noexcept {
    EventBinder<ESystemEvent::SceneLoad>([this](void *p) {
        SetScene((world::World *)p);
    });
}

void ShadowRayPass::Inspector() noexcept {
}
}// namespace Pupil::ddgi::shadow