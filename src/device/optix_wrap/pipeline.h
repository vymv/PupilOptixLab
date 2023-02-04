#pragma once

#include <optix.h>
#include <vector>
#include <unordered_map>
#include <string>

namespace optix_wrap {
struct Module;

struct ProgramDesc {
    Module *module = nullptr;
    const char *ray_gen_entry = nullptr;
    const char *hit_miss = nullptr;
    const char *shadow_miss = nullptr;
    struct {
        const char *ch_entry = nullptr;
        const char *ah_entry = nullptr;
        Module *intersect_module = nullptr;// use for builtin type
        const char *is_entry = nullptr;
    } hit_group, shadow_grop;
};
struct PipelineDesc {
    std::vector<ProgramDesc> programs;
};

struct Pipeline {
private:
    std::unordered_map<std::string, OptixProgramGroup> m_program_map;

public:
    static OptixPipelineCompileOptions pipeline_compile_options;

    OptixPipeline pipeline;
    std::vector<OptixProgramGroup> programs;

    operator OptixPipeline() const noexcept { return pipeline; }

    Pipeline(const OptixDeviceContext device_context, const PipelineDesc &desc) noexcept;
    ~Pipeline() noexcept;

    OptixProgramGroup FindProgram(std::string) noexcept;
};
}// namespace optix_wrap