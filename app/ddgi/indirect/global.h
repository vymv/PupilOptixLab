#pragma once
#include "render/material/optix_material.h"
//#include "optix/geometry.h"
#include "render/camera.h"
#include "render/emitter.h"

namespace Pupil::ddgi {
extern int m_probesidelength;
extern int m_irradiancerays_perprobe;
extern int m_probecountperside;
extern float3 m_probestep;
extern float3 m_probestartpos;
extern float m_depthSharpness;

struct usize {
    uint32_t w, h;
};
extern usize m_probeirradiancesize;
extern usize m_raygbuffersize;
extern int m_show_type;
extern bool show_type_changed;

}// namespace Pupil::ddgi