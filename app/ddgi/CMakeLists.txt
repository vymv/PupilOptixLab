set(target_name ddgi)

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
cuda_compile_and_embed(ddgi_render_pass_embedded_ptx_code render/render.cu)

add_executable(${target_name}
    ${ddgi_probe_pass_embedded_ptx_code}
    ${ddgi_render_pass_embedded_ptx_code}
    main.cpp
    probe/pass.h
    probe/pass.cpp
    probe/type.h
    probe/probeIntegrate.cu
    probe/probeIntegrate.h
    render/pass.h
    render/pass.cpp
    render/type.h
    # global.h
)

target_link_libraries(${target_name} PUBLIC ${pupil_framework_name})
set_target_properties(${target_name} PROPERTIES FOLDER ${pupil_application})
