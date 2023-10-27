#include "probe/pass.h"
#include "init/pass.h"
#include "shading/pass.h"
#include "spatial/pass.h"
#include "path_tracer/pt_pass.h"
#include "static.h"
#include "system/system.h"
#include "indirect/global.h"

int main() {
    auto system = Pupil::util::Singleton<Pupil::System>::instance();
    system->Init(true);

    {
        auto pt_pass = std::make_unique<Pupil::ddgi::pt::PTPass>("Path Tracing");
        auto probe_pass = std::make_unique<Pupil::ddgi::probe::ProbePass>();
        auto render_pass = std::make_unique<Pupil::ddgi::render::RenderPass>();
        auto spatial_pass = std::make_unique<Pupil::ddgi::spatial::SpatialPass>();
        auto shading_pass = std::make_unique<Pupil::ddgi::shading::ShadingPass>();

        system->AddPass(pt_pass.get());
        system->AddPass(probe_pass.get());
        system->AddPass(render_pass.get());
        system->AddPass(spatial_pass.get());
        system->AddPass(shading_pass.get());

        std::filesystem::path scene_file_path{ Pupil::DATA_DIR };
        scene_file_path /= "static/cornellbox.xml";
        // std::filesystem::path scene_file_path = "D:/Research/Models/MitsubaModels/living-room-white/scene_v3.xml";
        // std::filesystem::path scene_file_path = "D:/Research/Models/MitsubaModels/living-room-2/pupil_test.xml";
        // std::filesystem::path scene_file_path = "D:/Research/Models/MitsubaModels/veach-ajar/scene_v3.xml";
        // std::filesystem::path scene_file_path = "D:/Research/Models/MitsubaModels/bathroom/scene_v3.xml";
        system->SetScene(scene_file_path);

        system->Run();
    }

    system->Destroy();

    return 0;
}