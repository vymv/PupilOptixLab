set(target_name ddgi-restir)

# cuda_compile_and_embed(ddgi_pt_pass_embedded_ptx_code pt/pt.cu)
# cuda_compile_and_embed(ddgi_gbuffer_pass_embedded_ptx_code gbuffer/gbuffer.cu)

# add_executable(${target_name}
#     ${ddgi_pt_pass_embedded_ptx_code}
#     ${ddgi_gbuffer_pass_embedded_ptx_code}
#     main.cpp
#     gbuffer/pass.h
#     gbuffer/pass.cpp
#     gbuffer/type.h
#     pt/pass.h
#     pt/pass.cpp
#     pt/type.h
# )

cuda_compile_and_embed(ddgi_probe_pass_embedded_ptx_code probe/probetrace.cu)
cuda_compile_and_embed(ddgi_render_pass_embedded_ptx_code init/init.cu)
cuda_compile_and_embed(ddgi_shading_pass_embedded_ptx_code shading/shading.cu)
cuda_compile_and_embed(ddgi_spatial_pass_embedded_ptx_code spatial/spatial_reuse.cu)
cuda_compile_and_embed(ddgi_temporal_pass_embedded_ptx_code temporal/temporal_reuse.cu)
cuda_compile_and_embed(ddgi_shadow_pass_embedded_ptx_code shadow/shadow.cu)
cuda_compile_and_embed(pathtracer_pass_embedded_ptx_code path_tracer/pathtracer.cu)
cuda_compile_and_embed(ddgi_merge_pass_embedded_ptx_code merge/merge.cu)


file(GLOB_RECURSE ALL_INCLUDE CONFIGURE_DEPENDS "*.h")
file(GLOB_RECURSE ALL_SOURCES CONFIGURE_DEPENDS "*.cpp")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${ALL_INCLUDE} ${ALL_SOURCES})

add_executable(${target_name}
    ${ddgi_probe_pass_embedded_ptx_code}
    ${ddgi_render_pass_embedded_ptx_code}
    ${ddgi_shading_pass_embedded_ptx_code}
    ${ddgi_spatial_pass_embedded_ptx_code}
    ${ddgi_temporal_pass_embedded_ptx_code}
    ${ddgi_shadow_pass_embedded_ptx_code}
    ${ddgi_merge_pass_embedded_ptx_code}
    ${pathtracer_pass_embedded_ptx_code}
    ${ALL_INCLUDE}
    ${ALL_SOURCES}
    probe/probeIntegrate.cu
    probe/probeIntegrate.h
    probe/probeCopyBorder.cu
    probe/probeCopyBorder.h
)

target_link_libraries(${target_name} PUBLIC ${pupil_framework_name})
set_target_properties(${target_name} PROPERTIES FOLDER ${pupil_application})
