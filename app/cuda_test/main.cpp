#include "pass.h"
#include "system/gui.h"
#include "system/system.h"
#include "util/event.h"

int main()
{
    auto system = Pupil::util::Singleton<Pupil::System>::instance();
    system->Init(true);
     {
         auto pass = std::make_unique<CudaPass>();
         system->AddPass(pass.get());

         struct {
             uint32_t w, h;
         } size{ 1000, 1000 };
         Pupil::EventDispatcher<Pupil::ECanvasEvent::Resize>(size);

         system->Run();
     }

    system->Destroy();

    return 0;
}