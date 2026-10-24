cmake_minimum_required(VERSION 3.10.2)

set(CMAKE_C_COMPILER "tecocc")
set(CMAKE_CXX_COMPILER "tecocc")

project(teco_ops)

set(PROJECT_DIR ${CMAKE_SOURCE_DIR})
set(OPS_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_compile_options(-Werror)

option(WITH_TORCH "Enable PyTorch support" OFF)
if(WITH_TORCH)
    find_package(Python3 REQUIRED COMPONENTS Development)
    find_package(Torch REQUIRED)
    add_definitions(-DWITH_TORCH)
    include_directories(${TORCH_INCLUDE_DIRS})
    include_directories(${Python3_INCLUDE_DIRS})
    include_directories(${TORCH_SDAA_HOME}/include)
    get_filename_component(TORCH_LIB_DIR ${TORCH_LIBRARY} DIRECTORY)
    message(STATUS "WITH_TORCH is enabled")
    message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
    message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
    message(STATUS "TORCH_LIB_DIR: ${TORCH_LIB_DIR}")
else()
    message(STATUS "WITH_TORCH is disabled")
endif()

option(WITH_INFERENCE_PLUGIN "Enable Teco-inference Plugin support" OFF)
if(WITH_INFERENCE_PLUGIN)
    add_definitions(-DWITH_INFERENCE_PLUGIN)
    file(GLOB_RECURSE SRC_PLUGIN "${OPS_DIR}/plugin/*/*.cc")
    set(FIND_SDAART "/opt/tecoai/lib64")
    set(FIND_PluginUtil "$ENV{TECO_INFER_PLUGIN_UTIL_PATH}")
    set(PLUGIN_LIBS ${FIND_SDAART}/libsdaart.so ${FIND_PluginUtil}/libTecoInferPluginUtil.so)
    set(PLUGIN_INCLUDES
        -I/opt/tecoai/include
        -I/opt/tecoai/include/dyn/include
        -I/opt/tecoai/include/dyn/src
        -I/opt/tecoai/include/teco_infer_dyn
        -I${OPS_DIR}
    )
    set(PLUGIN_COMPILE_FLAGS -O3 -std=c++17 -fPIC -DWITH_INFERENCE_PLUGIN)
    set(PLUGIN_OBJ_DIR ${CMAKE_BINARY_DIR}/plugin_objs)
    set(PLUGIN_OBJECTS "")
    add_custom_target(plugin_mkdir ALL COMMAND ${CMAKE_COMMAND} -E make_directory ${PLUGIN_OBJ_DIR})
    foreach(src ${SRC_PLUGIN})
        get_filename_component(src_name ${src} NAME)
        list(APPEND PLUGIN_OBJECTS ${PLUGIN_OBJ_DIR}/${src_name}.o)
        add_custom_command(OUTPUT ${PLUGIN_OBJ_DIR}/${src_name}.o
            COMMAND g++ -c ${src} -o ${PLUGIN_OBJ_DIR}/${src_name}.o ${PLUGIN_COMPILE_FLAGS} ${PLUGIN_INCLUDES}
            DEPENDS ${src}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endforeach()
    add_custom_target(plugin_compile ALL DEPENDS plugin_mkdir ${PLUGIN_OBJECTS})
    message(STATUS "WITH_INFERENCE_PLUGIN is enabled")
else()
    message(STATUS "WITH_INFERENCE_PLUGIN is disabled")
endif()

set(TECOOPS_OPS_PATH ${OPS_DIR}/interface/ops)

file(GLOB_RECURSE SRC_INTERFACE_COMMON "${OPS_DIR}/interface/common/*.cpp")
file(GLOB_RECURSE SRC_INTERFACE_OPS "${OPS_DIR}/interface/ops/*.cpp")
file(GLOB_RECURSE SRC_UAL_COMMON "${OPS_DIR}/ual/com/*.cpp")
file(GLOB_RECURSE SRC_UAL_OPS "${OPS_DIR}/ual/ops/*/*.cpp")
file(GLOB_RECURSE SRC_UAL_KERNEL "${OPS_DIR}/ual/kernel/*/*.scpp")

# Device-section order pinning (build-time only; no hot-path effect).
#
# Measurement method and criteria.  The order of the device-code sections inside .sdaa_fatbin equals
# the order of the object files on the official link line (tecocc $<TARGET_OBJECTS:ops_objs> -flto
# ... --sdaa-link), which equals the order of this source list; permuting only the object order
# therefore moves section VMAs.  Every number behind the three moves below was measured on one build
# root in which only libteco_ops.so was swapped between arms, with the official fixed-form benchmark
# (zoo prototxt cases, --perf_repeat=50 --warm_repeat=3), scored as the median of 3 reps, and judged
# by the ABSOLUTE device VMA address of each section rather than by a relative offset.
#
# Merge prerequisites.  The TUs used as anchors below come from other open PRs and must land first:
# weight_stream_gemv (#44), zzz_flash_attention_d128 (#41), z_reshape_and_cache_dma (#35),
# z_ms_deform_attn_forward_span and zz_ms_deform_attn_backward_ha (#40), rotary_embedding (#48),
# and rms_norm_compat (#45). Merge this layout PR after those kernel PRs and #52; then rebuild,
# remeasure absolute device addresses, and refresh the merged-tree fingerprint. Every lookup is
# existence-guarded: absent anchors leave their moves inert. A standalone-tree fingerprint is not
# evidence for the final merged layout. See doc/device_section_order.md for the rebaseline steps.
option(TECO_ENABLE_DEVICE_SECTION_ORDER "Pin the measured device-section order at link time (build-time only)" ON)
# Opt-in CI gate: with this ON a fingerprint mismatch fails the configure step instead of
# printing a WARNING.  Default OFF so a plain developer build never breaks on a stale pin,
# while CI can turn pinning drift into a hard failure.  Exemption channel: pass
# -DTECO_OPS_DEVICE_SECTION_ORDER_EXPECTED_FP= (empty) to disable the comparison entirely.
option(TECO_OPS_DEVICE_SECTION_ORDER_STRICT "Fail configure when the device-section-order fingerprint mismatches (opt-in CI gate)" OFF)
# Configure-time assertion of the relative order of anchor kernel TUs.
# Defaults to ON so refactoring or glob drift immediately fails the configure step.
# Exemption channel: pass -DTECO_OPS_DEVICE_SECTION_ORDER_ASSERT=OFF to disable the check.
option(TECO_OPS_DEVICE_SECTION_ORDER_ASSERT "Assert the relative order of anchor kernel TUs at configure time" ON)
# Backward compatibility: honour the pre-rename option name once more, then retire it silently.
if(DEFINED TECO_OPS_DEVICE_SECTION_ORDER AND NOT "x${TECO_OPS_DEVICE_SECTION_ORDER}" STREQUAL "xON")
    message(STATUS "TECO_OPS_DEVICE_SECTION_ORDER is retired; use -DTECO_ENABLE_DEVICE_SECTION_ORDER=${TECO_OPS_DEVICE_SECTION_ORDER} instead")
    set(TECO_ENABLE_DEVICE_SECTION_ORDER "${TECO_OPS_DEVICE_SECTION_ORDER}")
endif()
if(TECO_ENABLE_DEVICE_SECTION_ORDER)
    # TAX-1: pin weight_stream_gemv immediately before rotary_embedding, because that is the only
    # move that reaches weight_stream_gemv's measured fast slot while displacing just one section.
    list(FIND SRC_UAL_KERNEL "${OPS_DIR}/ual/kernel/weight_stream_gemv/weight_stream_gemv.scpp" _TAX1_WSG)
    list(FIND SRC_UAL_KERNEL "${OPS_DIR}/ual/kernel/rotary_embedding/rotary_embedding_fp16.scpp" _TAX1_ROT)
    if(_TAX1_WSG GREATER -1 AND _TAX1_ROT GREATER -1 AND _TAX1_WSG GREATER _TAX1_ROT)
        list(REMOVE_AT SRC_UAL_KERNEL ${_TAX1_WSG})
        list(INSERT SRC_UAL_KERNEL ${_TAX1_ROT} "${OPS_DIR}/ual/kernel/weight_stream_gemv/weight_stream_gemv.scpp")
    endif()

    # R10: the same FIND/REMOVE_AT/INSERT discipline as above, idempotent and existence-guarded.
    macro(_c38_r10_move_before _obj _anchor)
        list(FIND SRC_UAL_KERNEL "${_obj}" _oi)
        list(FIND SRC_UAL_KERNEL "${_anchor}" _ai)
        if(_oi GREATER -1 AND _ai GREATER -1 AND NOT _oi EQUAL _ai)
            list(REMOVE_AT SRC_UAL_KERNEL ${_oi})
            if(_oi LESS _ai)
                math(EXPR _ai "${_ai} - 1")
            endif()
            list(INSERT SRC_UAL_KERNEL ${_ai} "${_obj}")
        endif()
    endmacro()
    set(_R10K "${OPS_DIR}/ual/kernel")
    # A3: weight_stream_gemv directly before rms_norm_compat, which is the arrangement that also puts
    # rotary_embedding into its measured fast slot.
    _c38_r10_move_before("${_R10K}/weight_stream_gemv/weight_stream_gemv.scpp" "${_R10K}/rms_norm_compat/rms_norm_compat.scpp")
    # Then pack the three contributor TUs behind it: flash_d128 after the msdeform span TU, and both
    # rac-dma and the bwd-gather TU ahead of flash_d128.
    _c38_r10_move_before("${_R10K}/zzz_flash_attention_d128/zzz_flash_attention_d128.scpp" "${_R10K}/z_ms_deform_attn_forward_span/z_ms_deform_attn_forward_span.scpp")
    _c38_r10_move_before("${_R10K}/z_reshape_and_cache_dma/reshape_and_cache_dma.scpp" "${_R10K}/zzz_flash_attention_d128/zzz_flash_attention_d128.scpp")
    _c38_r10_move_before("${_R10K}/zz_ms_deform_attn_backward_ha/zz_ms_deform_attn_backward_ha.scpp" "${_R10K}/zzz_flash_attention_d128/zzz_flash_attention_d128.scpp")
    _c38_r10_move_before("${_R10K}/zz_ms_deform_attn_backward_ha/zz_ms_deform_attn_backward_ha.scpp" "${_R10K}/weight_stream_gemv/weight_stream_gemv.scpp")

    # R17: finally swap weight_stream_gemv with the bwd-gather TU.  The gather TU is its immediate
    # predecessor and weight_stream_gemv's fast addresses all lie below its unswapped slot, so this
    # one-TU swap is the cheapest way into the fast class and leaves flash_d128, rms_norm_fp16,
    # memset and the msdeform forward sections on their measured addresses.
    _c38_r10_move_before("${_R10K}/weight_stream_gemv/weight_stream_gemv.scpp" "${_R10K}/zz_ms_deform_attn_backward_ha/zz_ms_deform_attn_backward_ha.scpp")

    # Configure-time assertion of anchor TU relative order (review recommendation).
    # When enabled, asserts that the target anchor TUs are present in SRC_UAL_KERNEL and
    # maintain their calibrated relative order (weight_stream_gemv < rotary_embedding < zzz_flash_attention_d128).
    # Refactoring or glob changes that alter this relative order fail configure immediately.
    # Exemption: pass -DTECO_OPS_DEVICE_SECTION_ORDER_ASSERT=OFF (or -DTECO_ENABLE_DEVICE_SECTION_ORDER=OFF).
    if(TECO_OPS_DEVICE_SECTION_ORDER_ASSERT)
        list(LENGTH SRC_UAL_KERNEL _KERNEL_COUNT)
        # Check a copy: removing duplicates from the real list would hide a bad
        # source-list refactor and could alter the calibrated link order.
        set(_UNIQUE_KERNELS ${SRC_UAL_KERNEL})
        list(REMOVE_DUPLICATES _UNIQUE_KERNELS)
        list(LENGTH _UNIQUE_KERNELS _UNIQUE_KERNEL_COUNT)
        if(NOT _UNIQUE_KERNEL_COUNT EQUAL _KERNEL_COUNT)
            message(FATAL_ERROR
                "device-section-order assertion failed: duplicate kernel TU in SRC_UAL_KERNEL")
        endif()
        set(_ANCHOR_WSG "${_R10K}/weight_stream_gemv/weight_stream_gemv.scpp")
        set(_ANCHOR_ROT "${_R10K}/rotary_embedding/rotary_embedding_fp16.scpp")
        set(_ANCHOR_FLASH "${_R10K}/zzz_flash_attention_d128/zzz_flash_attention_d128.scpp")

        # The anchor TUs are owned by other PRs (#35/#40/#41/#44/#48); on a tree where they are
        # not merged yet the corresponding pin moves above are inert, so the assertion requires
        # only the *present* anchors to keep the calibrated relative order.  A missing anchor is
        # reported as STATUS (legitimate merge-order state), while a reordered present anchor is
        # FATAL and therefore surfaces a CMake/source-list refactor immediately.
        set(_ANCHOR_INDICES "")
        set(_ANCHOR_FOUND "")
        set(_ANCHOR_NAMES
            "weight_stream_gemv/weight_stream_gemv.scpp"
            "rotary_embedding/rotary_embedding_fp16.scpp"
            "zzz_flash_attention_d128/zzz_flash_attention_d128.scpp")
        foreach(_ANCHOR ${_ANCHOR_NAMES})
            list(FIND SRC_UAL_KERNEL "${_R10K}/${_ANCHOR}" _IDX)
            if(_IDX EQUAL -1)
                message(STATUS
                    "device-section-order assertion: anchor TU '${_ANCHOR}' not present "
                    "(expected until its owning PR is merged); ordering check skipped for it.")
            else()
                list(APPEND _ANCHOR_INDICES ${_IDX})
                if(_ANCHOR_FOUND STREQUAL "")
                    set(_ANCHOR_FOUND "${_ANCHOR}=${_IDX}")
                else()
                    string(APPEND _ANCHOR_FOUND ", ${_ANCHOR}=${_IDX}")
                endif()
            endif()
        endforeach()

        set(_PREV_INDEX -1)
        foreach(_IDX IN LISTS _ANCHOR_INDICES)
            if(_IDX LESS _PREV_INDEX)
                message(FATAL_ERROR
                    "device-section-order assertion failed: relative order mismatch among the anchors "
                    "present in this tree (list length: ${_KERNEL_COUNT}). "
                    "Expected relative order: weight_stream_gemv < rotary_embedding < zzz_flash_attention_d128. "
                    "Actual: ${_ANCHOR_FOUND}. "
                    "Exempt with -DTECO_OPS_DEVICE_SECTION_ORDER_ASSERT=OFF or -DTECO_ENABLE_DEVICE_SECTION_ORDER=OFF.")
            endif()
            set(_PREV_INDEX ${_IDX})
        endforeach()
    endif()
endif()

# Fingerprint of the final kernel TU order, so that a silent no-op (anchor TUs absent) or a silent
# misplacement (anchors present but in an unexpected order) is visible at configure time instead of
# being discovered from a device-side measurement.
#
# The hash input is the canonical RELATIVE kernel file list ("<subdir>/<file>.scpp" from
# teco/ual/kernel/, produced by get_filename_component(... NAME)), never the absolute source paths,
# so the value is identical for every checkout directory and comparable across machines.
list(LENGTH SRC_UAL_KERNEL TECO_OPS_SECTION_ORDER_COUNT)
set(TECO_OPS_SECTION_ORDER_CANONICAL "")
foreach(_fp_src ${SRC_UAL_KERNEL})
    get_filename_component(_fp_name "${_fp_src}" NAME)
    string(APPEND TECO_OPS_SECTION_ORDER_CANONICAL "${_fp_name};")
endforeach()
string(SHA1 TECO_OPS_SECTION_ORDER_FP "${TECO_OPS_SECTION_ORDER_CANONICAL}")

# Recorded expected fingerprint of the canonical kernel TU order.  It is the SHA1 of the
# relative "<file>.scpp" list above (NOT of absolute paths, so it is comparable across checkouts)
# from the accepted layout of this branch.  Value below is the configure-time fingerprint of this
# PR's own tree (kernel TU set of the base branch: the anchor TUs owned by #35/#40/#41/#44 are not
# merged yet, so every existence-guarded move is inert and the fingerprint only covers the 10 TUs
# present here).  How to refresh AFTER the dependency PRs land and the resulting device layout has
# been re-measured: run one configure, copy `fingerprint=` from the "teco-ops device-section order"
# STATUS line into the default below (or pass
# -DTECO_OPS_DEVICE_SECTION_ORDER_EXPECTED_FP=<sha> on the cmake command line).
# The comparison is enabled by default now: a configure whose TU order no longer matches the
# recorded layout emits a WARNING (build continues; nothing fails).  -DTECO_OPS_DEVICE_SECTION_ORDER_STRICT=ON
# upgrades that same mismatch to a FATAL_ERROR so CI cannot lose the pin silently; it is OFF by
# default so an unrelated developer configure is never blocked.  Pass an empty value
# (-DTECO_OPS_DEVICE_SECTION_ORDER_EXPECTED_FP=) to disable the comparison in either mode.
set(TECO_OPS_DEVICE_SECTION_ORDER_EXPECTED_FP "809c29255f8eb09af32c03072acd3d5313ea7fa4"
    CACHE STRING "Recorded canonical device-section order fingerprint of the last accepted layout")
if(TECO_OPS_DEVICE_SECTION_ORDER_EXPECTED_FP
   AND NOT TECO_OPS_DEVICE_SECTION_ORDER_EXPECTED_FP STREQUAL TECO_OPS_SECTION_ORDER_FP)
    if(TECO_OPS_DEVICE_SECTION_ORDER_STRICT)
        # CI mode: pinning must never be lost silently.  The message repeats the manual
        # refresh recipe and the explicit exemption, so the failure is actionable without
        # reading the docs.
        message(FATAL_ERROR
            "==============================================================\n"
            "teco-ops device-section order fingerprint MISMATCH (TECO_OPS_DEVICE_SECTION_ORDER_STRICT=ON)\n"
            "  expected (recorded) = ${TECO_OPS_DEVICE_SECTION_ORDER_EXPECTED_FP}\n"
            "  computed (this tree) = ${TECO_OPS_SECTION_ORDER_FP}  (count=${TECO_OPS_SECTION_ORDER_COUNT})\n"
            "The kernel TU order no longer matches the recorded layout, so the measured device-section "
            "pin would silently stop applying.  Either (a) re-measure the device section order "
            "(doc/device_section_order.md) and refresh TECO_OPS_DEVICE_SECTION_ORDER_EXPECTED_FP, "
            "(b) restore the recorded TU order, or (c) exempt this configure explicitly with "
            "-DTECO_OPS_DEVICE_SECTION_ORDER_EXPECTED_FP= (empty value disables the comparison).\n"
            "Configure stopped because TECO_OPS_DEVICE_SECTION_ORDER_STRICT=ON.\n"
            "==============================================================")
    else()
        message(WARNING
            "==============================================================\n"
            "teco-ops device-section order fingerprint MISMATCH\n"
            "  expected (recorded) = ${TECO_OPS_DEVICE_SECTION_ORDER_EXPECTED_FP}\n"
            "  computed (this tree) = ${TECO_OPS_SECTION_ORDER_FP}  (count=${TECO_OPS_SECTION_ORDER_COUNT})\n"
            "The kernel TU order has drifted from the recorded layout; if the new order is intended, "
            "re-measure the device section order (doc/device_section_order.md) and refresh "
            "TECO_OPS_DEVICE_SECTION_ORDER_EXPECTED_FP, otherwise restore the recorded order before "
            "building performance-sensitive targets.  Build continues (warning only); pass "
            "-DTECO_OPS_DEVICE_SECTION_ORDER_STRICT=ON to make this a configure failure in CI.\n"
            "==============================================================")
    endif()
endif()

message(STATUS "teco-ops device-section order: ${TECO_ENABLE_DEVICE_SECTION_ORDER} fingerprint=${TECO_OPS_SECTION_ORDER_FP} (count=${TECO_OPS_SECTION_ORDER_COUNT}) strict=${TECO_OPS_DEVICE_SECTION_ORDER_STRICT} assert=${TECO_OPS_DEVICE_SECTION_ORDER_ASSERT}")

list(APPEND SRC_ALL
    ${SRC_INTERFACE_COMMON}
    ${SRC_INTERFACE_OPS}
    ${SRC_UAL_COMMON}
    ${SRC_UAL_OPS}
    ${SRC_UAL_KERNEL})

set(complie_options -O3 -msimd -mslave -fPIC -flto -x sdaa -std=c++17 --sdaa-device-lib=${OPS_DIR}/../thirdparty/teco-hal/lib/teco_hal.bc)
add_definitions(-DUSE_TECO)

add_library(ops_objs OBJECT ${SRC_ALL})
set_source_files_properties(${SRC_UAL_KERNEL} PROPERTIES LANGUAGE CXX)
set_target_properties(ops_objs PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(ops_objs PRIVATE ${complie_options})
target_include_directories(ops_objs PRIVATE "${OPS_DIR}/../common")
target_include_directories(ops_objs PRIVATE "${OPS_DIR}/")
target_include_directories(ops_objs PRIVATE "${OPS_DIR}/../thirdparty/teco-hal/include")

if(WITH_TORCH)
    set(TORCH_LINK_FLAGS "-L${TORCH_LIB_DIR} -ltorch -ltorch_cpu -lc10")
else()
    set(TORCH_LINK_FLAGS "")
endif()

if(WITH_INFERENCE_PLUGIN)
    set(TECO_OPS_PLUGIN_LIB ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libteco_ops_plugin.so)
    set(TECO_INFER_PLUGIN_LIB ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libTecoInferPlugin.so)

    add_custom_target(ops_lib ALL
            COMMAND mkdir -p ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
            COMMAND tecocc $<TARGET_OBJECTS:ops_objs>
            -flto -ffp-contract=fast -fPIC -shared --sdaa-link -fuse-ld=lld
            -o ${TECO_OPS_PLUGIN_LIB}
            COMMAND_EXPAND_LISTS
    )
    add_dependencies(ops_lib ops_objs)

    add_custom_target(plugin_lib ALL
            COMMAND gcc ${PLUGIN_OBJECTS}
            -L${CMAKE_LIBRARY_OUTPUT_DIRECTORY} -lteco_ops_plugin
            ${PLUGIN_LIBS}
            -shared -o ${TECO_INFER_PLUGIN_LIB}
            COMMAND_EXPAND_LISTS
    )
    add_dependencies(plugin_lib ops_lib plugin_compile)
else()
    add_custom_target(ops_lib ALL
            COMMAND mkdir -p ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
            COMMAND tecocc $<TARGET_OBJECTS:ops_objs> -flto -ffp-contract=fast -fPIC -shared --sdaa-link -fuse-ld=lld
            ${TORCH_LINK_FLAGS}
            -o ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libteco_ops.so
            COMMAND_EXPAND_LISTS
    )
    add_dependencies(ops_lib ops_objs)
endif()

set(CMAKE_CXX_LINK_FLAGS "-L$ENV{SDAA_HOME}/lib64  -lsdaart")
