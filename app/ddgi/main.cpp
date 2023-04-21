#include "gbuffer/pass.h"
#include "probe/pass.h"
#include "pt/pass.h"
#include "render/pass.h"
#include "static.h"
#include "system/system.h"

int main()
{
    auto system = Pupil::util::Singleton<Pupil::System>::instance();
    system->Init(true);

    {
        // // 创建Pass
        // auto gbuffer_pass = std::make_unique<Pupil::ddgi::gbuffer::GBufferPass>();
        // auto pt_pass = std::make_unique<Pupil::ddgi::pt::PTPass>();
        // // Pass执行的顺序与AddPass的顺序一致，目前是线性执行
        // system->AddPass(gbuffer_pass.get());
        // system->AddPass(pt_pass.get());

        auto probe_pass = std::make_unique<Pupil::ddgi::probe::ProbePass>();
        system->AddPass(probe_pass.get());

        auto render_pass = std::make_unique<Pupil::ddgi::render::RenderPass>();
        system->AddPass(render_pass.get());

        std::filesystem::path scene_file_path{Pupil::DATA_DIR};
        scene_file_path /= "cornellbox.xml";
        // scene_file_path /= "mis.xml";
        system->SetScene(scene_file_path);

        system->Run();
    }

    system->Destroy();

    return 0;
}