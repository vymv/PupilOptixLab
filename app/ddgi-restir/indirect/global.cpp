#include "global.h"

namespace Pupil::ddgi {
int m_probesidelength = 64;
int m_irradiancerays_perprobe = 64;
int m_probecountperside = 5;
float3 m_probestep;
float3 m_probestartpos;
float m_depthSharpness = 50.0f;

usize m_probeirradiancesize = {
    static_cast<uint32_t>(m_probecountperside * m_probecountperside * (m_probesidelength + 2) + 2),
    static_cast<uint32_t>(m_probecountperside *(m_probesidelength + 2) + 2)
};

usize m_raygbuffersize{ static_cast<uint32_t>(m_irradiancerays_perprobe),
                        static_cast<uint32_t>(std::pow(m_probecountperside, 3)) };

int m_show_type = 0;
bool show_type_changed = false;
bool is_pathtracer = false;
float m_energyconservation = 0.95f;
bool m_enable_visualize = true;
bool accumulated = false;
}// namespace Pupil::ddgi