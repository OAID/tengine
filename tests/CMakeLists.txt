# generate tengine header file
FILE (MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tengine)
FILE (COPY ${CMAKE_SOURCE_DIR}/source/api/c_api.h DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tengine)
FILE (COPY ${CMAKE_SOURCE_DIR}/source/api/c_api_ex.h DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tengine)

function(tengine_op_test name)
    file(GLOB TENGINE_UTIL_SOURCE_FILES      ${PROJECT_SOURCE_DIR}/tests/common/util/*.c)
    add_executable(${name} "${CMAKE_CURRENT_SOURCE_DIR}/op/${name}.c" "${TENGINE_UTIL_SOURCE_FILES}")

    target_link_libraries(${name} PUBLIC "${CMAKE_PROJECT_NAME}")

    target_include_directories (${name} PRIVATE "${PROJECT_SOURCE_DIR}/source")
    target_include_directories (${name} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    target_include_directories (${name} PRIVATE "${PROJECT_BINARY_DIR}")
    target_include_directories (${name} PRIVATE "${PROJECT_BINARY_DIR}/source")
    target_include_directories (${name} PRIVATE "${PROJECT_SOURCE_DIR}/tests/common")
    target_include_directories (${name} PRIVATE "${PROJECT_SOURCE_DIR}/tests/common/util")

endfunction()
tengine_op_test(test_op_absval)
tengine_op_test(test_op_add_n)
tengine_op_test(test_op_argmax)
tengine_op_test(test_op_argmin)

if (TENGINE_ENABLE_OPENDLA)
    function (tengine_opendla_op_test name file)
        file(GLOB TENGINE_UTIL_SOURCE_FILES      ${PROJECT_SOURCE_DIR}/tests/common/util/*.c)

        add_executable (${name} ${CMAKE_CURRENT_SOURCE_DIR}/${file} "${TENGINE_UTIL_SOURCE_FILES}" "${PROJECT_SOURCE_DIR}/tests/common/tengine_operations.c")

        target_link_libraries (${name} PRIVATE ${CMAKE_PROJECT_NAME})

        target_include_directories (${name} PRIVATE "${PROJECT_SOURCE_DIR}/source")
        target_include_directories (${name} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
        target_include_directories (${name} PRIVATE "${PROJECT_BINARY_DIR}")
        target_include_directories (${name} PRIVATE "${PROJECT_BINARY_DIR}/source")
        target_include_directories (${name} PRIVATE "${PROJECT_SOURCE_DIR}/tests/common")
        target_include_directories (${name} PRIVATE "${PROJECT_SOURCE_DIR}/tests/common/util")

        if (${TENGINE_TARGET_PROCESSOR} MATCHES "ARM" AND (NOT ANDROID AND NOT OHOS) AND TENGINE_TARGET_PROCESSOR_32Bit)
            target_compile_options (${name} PRIVATE "-mfp16-format=ieee")
        endif()

        add_test (${name} ${name})

        # add to a virtual project group
        SET_PROPERTY(TARGET ${name} PROPERTY FOLDER "tests/test_opendla")
    endfunction()

    tengine_opendla_op_test(test_opendla_op_concat              op/test_opendla_op_concat.cpp)
    tengine_opendla_op_test(test_opendla_op_convolution         op/test_opendla_op_convolution.cpp)
    tengine_opendla_op_test(test_opendla_op_deconv              op/test_opendla_op_deconv.cpp)
    tengine_opendla_op_test(test_opendla_op_groupconvolution    op/test_opendla_op_groupconvolution.cpp)
    tengine_opendla_op_test(test_opendla_op_eltwise             op/test_opendla_op_eltwise.cpp)
    tengine_opendla_op_test(test_opendla_op_fc                  op/test_opendla_op_fc.cpp)
    tengine_opendla_op_test(test_opendla_op_pooling             op/test_opendla_op_pooling.cpp)
    tengine_opendla_op_test(test_opendla_op_relu                op/test_opendla_op_relu.cpp)
    tengine_opendla_op_test(test_opendla_op_split               op/test_opendla_op_split.cpp)


endif()


if (TENGINE_ENABLE_TIM_VX)
    function (tengine_timvx_op_test name file)
        file(GLOB TENGINE_UTIL_SOURCE_FILES      ${PROJECT_SOURCE_DIR}/tests/common/util/*.c)

        add_executable (${name} ${CMAKE_CURRENT_SOURCE_DIR}/${file} "${TENGINE_UTIL_SOURCE_FILES}" "${PROJECT_SOURCE_DIR}/tests/common/tengine_operations.c")

        target_link_libraries (${name} PRIVATE ${CMAKE_PROJECT_NAME})

        target_include_directories (${name} PRIVATE "${PROJECT_SOURCE_DIR}/source")
        target_include_directories (${name} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
        target_include_directories (${name} PRIVATE "${PROJECT_BINARY_DIR}")
        target_include_directories (${name} PRIVATE "${PROJECT_BINARY_DIR}/source")
        target_include_directories (${name} PRIVATE "${PROJECT_SOURCE_DIR}/tests/common")
        target_include_directories (${name} PRIVATE "${PROJECT_SOURCE_DIR}/tests/common/util")

        if (${TENGINE_TARGET_PROCESSOR} MATCHES "ARM" AND (NOT ANDROID AND NOT OHOS) AND TENGINE_TARGET_PROCESSOR_32Bit)
            target_compile_options (${name} PRIVATE "-mfp16-format=ieee")
        endif()

        add_test (${name} ${name})

        # add to a virtual project group
        SET_PROPERTY(TARGET ${name} PROPERTY FOLDER "tests/test_timvx")
    endfunction()

    tengine_timvx_op_test(test_timvx_op_clip            op/test_timvx_op_clip.cpp)
    tengine_timvx_op_test(test_timvx_op_concat          op/test_timvx_op_concat.cpp)
    tengine_timvx_op_test(test_timvx_op_convolution     op/test_timvx_op_convolution.cpp)
    tengine_timvx_op_test(test_timvx_op_convolution_depthwise     op/test_timvx_op_convolution_depthwise.cpp)
    tengine_timvx_op_test(test_timvx_op_crop            op/test_timvx_op_crop.cpp)
    tengine_timvx_op_test(test_timvx_op_deconv          op/test_timvx_op_deconv.cpp)
    tengine_timvx_op_test(test_timvx_op_deconv_depthwise          op/test_timvx_op_deconv_depthwise.cpp)
    tengine_timvx_op_test(test_timvx_op_dropout         op/test_timvx_op_dropout.cpp)
    tengine_timvx_op_test(test_timvx_op_eltwise_mul     op/test_timvx_op_eltwise_mul.cpp)
    tengine_timvx_op_test(test_timvx_op_eltwise_sum     op/test_timvx_op_eltwise_sum.cpp)
    tengine_timvx_op_test(test_timvx_op_elu             op/test_timvx_op_elu.cpp)
    tengine_timvx_op_test(test_timvx_op_fc              op/test_timvx_op_fc.cpp)
    tengine_timvx_op_test(test_timvx_op_flatten         op/test_timvx_op_flatten.cpp)
    tengine_timvx_op_test(test_timvx_op_gather          op/test_timvx_op_gather.cpp)
    tengine_timvx_op_test(test_timvx_op_hardswish       op/test_timvx_op_hardswish.cpp)
    tengine_timvx_op_test(test_timvx_op_interp          op/test_timvx_op_interp.cpp)
    tengine_timvx_op_test(test_timvx_op_leakyrelu       op/test_timvx_op_leakyrelu.cpp)
    tengine_timvx_op_test(test_timvx_op_mish            op/test_timvx_op_mish.cpp)
    tengine_timvx_op_test(test_timvx_op_pad             op/test_timvx_op_pad.cpp)
    tengine_timvx_op_test(test_timvx_op_permute         op/test_timvx_op_permute.cpp)
    tengine_timvx_op_test(test_timvx_op_pooling         op/test_timvx_op_pooling.cpp)
    tengine_timvx_op_test(test_timvx_op_prelu           op/test_timvx_op_prelu.cpp)
    tengine_timvx_op_test(test_timvx_op_reduction       op/test_timvx_op_reduction.cpp)
    tengine_timvx_op_test(test_timvx_op_relu            op/test_timvx_op_relu.cpp)
    # tengine_timvx_op_test(test_timvx_op_relu1         op/test_timvx_op_relu1.cpp)
    tengine_timvx_op_test(test_timvx_op_reshape         op/test_timvx_op_reshape.cpp)
    tengine_timvx_op_test(test_timvx_op_resize          op/test_timvx_op_resize.cpp)
    tengine_timvx_op_test(test_timvx_op_sigmoid         op/test_timvx_op_sigmoid.cpp)
    tengine_timvx_op_test(test_timvx_op_slice           op/test_timvx_op_slice.cpp)
    tengine_timvx_op_test(test_timvx_op_softmax         op/test_timvx_op_softmax.cpp)
    tengine_timvx_op_test(test_timvx_op_split           op/test_timvx_op_split.cpp)
    tengine_timvx_op_test(test_timvx_op_tanh            op/test_timvx_op_tanh.cpp)
    tengine_timvx_op_test(test_timvx_op_transpose       op/test_timvx_op_transpose.cpp)
    tengine_timvx_op_test(test_timvx_op_upsampling      op/test_timvx_op_upsampling.cpp)
    
    
    # timvx model test, need opencv
    FIND_PACKAGE(OpenCV QUIET)
    IF (OpenCV_FOUND)
        # macro for adding examples
        FUNCTION (TENGINE_TIMVX_MODEL_TEST name file)
            ADD_EXECUTABLE (${name} "${CMAKE_CURRENT_SOURCE_DIR}/${file}" "${CMAKE_CURRENT_SOURCE_DIR}/common/tengine_operations.c")

            TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
            TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
            TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${PROJECT_SOURCE_DIR}/tests/common")

            TARGET_LINK_LIBRARIES (${name} ${CMAKE_PROJECT_NAME} ${OpenCV_LIBS})
            INSTALL (TARGETS ${name} DESTINATION bin)
        ENDFUNCTION()

        TENGINE_TIMVX_MODEL_TEST (test_timvx_model_yolov5s     models/test_timvx_model_yolov5s.cpp)
    ELSE()
        MESSAGE (WARNING "OpenCV not found, some examples won't be built")
    ENDIF()    
endif()


if (TENGINE_ENABLE_TENSORRT)
    function (tengine_tensorrt_op_test name file)
        file(GLOB TENGINE_UTIL_SOURCE_FILES      ${PROJECT_SOURCE_DIR}/tests/common/util/*.c)

        add_executable (${name} ${CMAKE_CURRENT_SOURCE_DIR}/${file} "${TENGINE_UTIL_SOURCE_FILES}" "${PROJECT_SOURCE_DIR}/tests/common/tengine_operations.c")

        target_link_libraries (${name} PRIVATE ${CMAKE_PROJECT_NAME})

        target_include_directories (${name} PRIVATE "${PROJECT_SOURCE_DIR}/source")
        target_include_directories (${name} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
        target_include_directories (${name} PRIVATE "${PROJECT_BINARY_DIR}")
        target_include_directories (${name} PRIVATE "${PROJECT_BINARY_DIR}/source")
        target_include_directories (${name} PRIVATE "${PROJECT_SOURCE_DIR}/tests/common")
        target_include_directories (${name} PRIVATE "${PROJECT_SOURCE_DIR}/tests/common/util")

        if (${TENGINE_TARGET_PROCESSOR} MATCHES "ARM" AND (NOT ANDROID AND NOT OHOS) AND TENGINE_TARGET_PROCESSOR_32Bit)
            target_compile_options (${name} PRIVATE "-mfp16-format=ieee")
        endif()

        add_test (${name} ${name})

        # add to a virtual project group
        SET_PROPERTY(TARGET ${name} PROPERTY FOLDER "tests/test_tensorrt")
    endfunction()

    tengine_tensorrt_op_test(test_tensorrt_op_clip          op/test_tensorrt_op_clip.cpp)
    tengine_tensorrt_op_test(test_tensorrt_op_concat        op/test_tensorrt_op_concat.cpp)
    tengine_tensorrt_op_test(test_tensorrt_op_deconv        op/test_tensorrt_op_deconv.cpp)
    tengine_tensorrt_op_test(test_tensorrt_op_dropout       op/test_tensorrt_op_dropout.cpp)
    tengine_tensorrt_op_test(test_tensorrt_op_eltwise       op/test_tensorrt_op_eltwise.cpp)
    tengine_tensorrt_op_test(test_tensorrt_op_fc            op/test_tensorrt_op_fc.cpp)
endif()

if (TENGINE_ENABLE_TORCH)
    function (tengine_torch_op_test name file)
        file(GLOB TENGINE_UTIL_SOURCE_FILES      ${PROJECT_SOURCE_DIR}/tests/common/util/*.c)

        add_executable (${name} ${CMAKE_CURRENT_SOURCE_DIR}/${file} "${TENGINE_UTIL_SOURCE_FILES}" "${PROJECT_SOURCE_DIR}/tests/common/tengine_operations.c")

        target_link_libraries (${name} PRIVATE ${CMAKE_PROJECT_NAME})

        target_include_directories (${name} PRIVATE "${PROJECT_SOURCE_DIR}/source")
        target_include_directories (${name} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
        target_include_directories (${name} PRIVATE "${PROJECT_BINARY_DIR}")
        target_include_directories (${name} PRIVATE "${PROJECT_BINARY_DIR}/source")
        target_include_directories (${name} PRIVATE "${PROJECT_SOURCE_DIR}/tests/common")
        target_include_directories (${name} PRIVATE "${PROJECT_SOURCE_DIR}/tests/common/util")

        if (${TENGINE_TARGET_PROCESSOR} MATCHES "ARM" AND (NOT ANDROID AND NOT OHOS) AND TENGINE_TARGET_PROCESSOR_32Bit)
            target_compile_options (${name} PRIVATE "-mfp16-format=ieee")
        endif()

        add_test (${name} ${name})

        # add to a virtual project group
        SET_PROPERTY(TARGET ${name} PROPERTY FOLDER "tests/test_torch")
    endfunction()

    tengine_torch_op_test(test_torch_op_conv          op/test_torch_op_conv.cpp)
endif()

# operator level test using onnx test
find_package(Protobuf)

if(PROTOBUF_FOUND)
    protobuf_generate_cpp(ONNX_PROTO_SRCS ONNX_PROTO_HDRS onnx.proto)

    function (tengine_onnx_op_test name file)
        add_executable (${name} ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${ONNX_PROTO_SRCS} ${ONNX_PROTO_HDRS})

        target_link_libraries (${name} PRIVATE ${CMAKE_PROJECT_NAME} ${PROTOBUF_LIBRARIES})

        target_include_directories (${name} PRIVATE "${PROTOBUF_INCLUDE_DIR}")
        target_include_directories (${name} PRIVATE "${PROJECT_SOURCE_DIR}/source")
        target_include_directories (${name} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
        target_include_directories (${name} PRIVATE "${PROJECT_BINARY_DIR}")
        target_include_directories (${name} PRIVATE "${PROJECT_BINARY_DIR}/source")

        add_test (${name} ${name})

        # add to a virtual project group
        SET_PROPERTY(TARGET ${name} PROPERTY FOLDER "tests/test_onnx")
    endfunction()

    tengine_onnx_op_test(test_onnx_op_abs                           op/test_onnx_op_abs.cpp)
    tengine_onnx_op_test(test_onnx_op_acos                          op/test_onnx_op_acos.cpp)
    tengine_onnx_op_test(test_onnx_op_add                           op/test_onnx_op_add.cpp)
    tengine_onnx_op_test(test_onnx_op_and2d                         op/test_onnx_op_and2d.cpp)
    tengine_onnx_op_test(test_onnx_op_and3d                         op/test_onnx_op_and3d.cpp)
    tengine_onnx_op_test(test_onnx_op_and4d                         op/test_onnx_op_and4d.cpp)
    tengine_onnx_op_test(test_onnx_op_argmax_default_axis_example   op/test_onnx_op_argmax_default_axis_example.cpp)
    tengine_onnx_op_test(test_onnx_op_argmin_default_axis_example   op/test_onnx_op_argmin_default_axis_example.cpp)
    tengine_onnx_op_test(test_onnx_op_asin                          op/test_onnx_op_asin.cpp)
    tengine_onnx_op_test(test_onnx_op_atan                          op/test_onnx_op_atan.cpp)
    tengine_onnx_op_test(test_onnx_op_averagepool_2d_default        op/test_onnx_op_averagepool_2d_default.cpp)
    tengine_onnx_op_test(test_onnx_op_averagepool_2d_pads           op/test_onnx_op_averagepool_2d_pads.cpp)
    tengine_onnx_op_test(test_onnx_op_basic_conv_with_padding       op/test_onnx_op_basic_conv_with_padding.cpp)
    tengine_onnx_op_test(test_onnx_op_basic_conv_without_padding    op/test_onnx_op_basic_conv_without_padding.cpp)
    tengine_onnx_op_test(test_onnx_op_batchnorm_epsilon             op/test_onnx_op_batchnorm_epsilon.cpp) #test_faild
    tengine_onnx_op_test(test_onnx_op_batchnorm_example             op/test_onnx_op_batchnorm_example.cpp) #test_faild
    tengine_onnx_op_test(test_onnx_op_cast_double_to_float          op/test_onnx_op_cast_double_to_float.cpp)
    tengine_onnx_op_test(test_onnx_op_ceil                          op/test_onnx_op_ceil.cpp)
    tengine_onnx_op_test(test_onnx_op_clip_example                  op/test_onnx_op_clip_example.cpp)
    # tengine_onnx_op_test(test_onnx_op_concat_1d_axis_0              op/test_onnx_op_concat_1d_axis_0.cpp)
    tengine_onnx_op_test(test_onnx_op_concat_2d_axis_0              op/test_onnx_op_concat_2d_axis_0.cpp)
    tengine_onnx_op_test(test_onnx_op_concat_2d_axis_1              op/test_onnx_op_concat_2d_axis_1.cpp)
    tengine_onnx_op_test(test_onnx_op_concat_3d_axis_0              op/test_onnx_op_concat_3d_axis_0.cpp)
    tengine_onnx_op_test(test_onnx_op_concat_3d_axis_1              op/test_onnx_op_concat_3d_axis_1.cpp)
    tengine_onnx_op_test(test_onnx_op_concat_3d_axis_2              op/test_onnx_op_concat_3d_axis_2.cpp)
    tengine_onnx_op_test(test_onnx_op_conv_with_strides_no_padding  op/test_onnx_op_conv_with_strides_no_padding.cpp)
    tengine_onnx_op_test(test_onnx_op_conv_with_strides_padding     op/test_onnx_op_conv_with_strides_padding.cpp)
    tengine_onnx_op_test(test_onnx_op_convtranspose_pad             op/test_onnx_op_convtranspose_pad.cpp)
    tengine_onnx_op_test(test_onnx_op_convtranspose_pads            op/test_onnx_op_convtranspose_pads.cpp)
    tengine_onnx_op_test(test_onnx_op_convtranspose                 op/test_onnx_op_convtranspose.cpp)
    tengine_onnx_op_test(test_onnx_op_cos                           op/test_onnx_op_cos.cpp)
    # tengine_onnx_op_test(test_onnx_op_depthtospace_dcr_mode         op/test_onnx_op_depthtospace_dcr_mode.cpp)
    tengine_onnx_op_test(test_onnx_op_div                           op/test_onnx_op_div.cpp)
    tengine_onnx_op_test(test_onnx_op_dropout_default               op/test_onnx_op_dropout_default.cpp)
    tengine_onnx_op_test(test_onnx_op_elu                           op/test_onnx_op_elu.cpp)
    tengine_onnx_op_test(test_onnx_op_equal                         op/test_onnx_op_equal.cpp)
    tengine_onnx_op_test(test_onnx_op_exp                           op/test_onnx_op_exp.cpp)
    # tengine_onnx_op_test(test_onnx_op_expand_dim_unchanged          op/test_onnx_op_expand_dim_unchanged.cpp)
    tengine_onnx_op_test(test_onnx_op_floor                         op/test_onnx_op_floor.cpp)
    # tengine_onnx_op_test(test_onnx_op_gather_0                      op/test_onnx_op_gather_0.cpp) # onnx.tmfile 有问题
    # tengine_onnx_op_test(test_onnx_op_gather_1                      op/test_onnx_op_gather_1.cpp) # onnx.tmfile 有问题
    tengine_onnx_op_test(test_onnx_op_globalaveragepool             op/test_onnx_op_globalaveragepool.cpp)
    tengine_onnx_op_test(test_onnx_op_greater                       op/test_onnx_op_greater.cpp)
    tengine_onnx_op_test(test_onnx_op_gru_defaults                  op/test_onnx_op_gru_defaults.cpp)
    # tengine_onnx_op_test(test_onnx_op_gru_seq_length                op/test_onnx_op_gru_seq_length.cpp)
    tengine_onnx_op_test(test_onnx_op_gru_with_initial_bias         op/test_onnx_op_gru_with_initial_bias.cpp)
    tengine_onnx_op_test(test_onnx_op_hardsigmoid                   op/test_onnx_op_hardsigmoid.cpp)
    tengine_onnx_op_test(test_onnx_op_instancenorm_epsilon          op/test_onnx_op_instancenorm_epsilon.cpp)
    tengine_onnx_op_test(test_onnx_op_instancenorm_example          op/test_onnx_op_instancenorm_example.cpp)
    tengine_onnx_op_test(test_onnx_op_leakyrelu                     op/test_onnx_op_leakyrelu.cpp)
    tengine_onnx_op_test(test_onnx_op_less                          op/test_onnx_op_less.cpp)
    tengine_onnx_op_test(test_onnx_op_log                           op/test_onnx_op_log.cpp)
    tengine_onnx_op_test(test_onnx_op_pad_reflect                   op/test_onnx_op_pad_reflect.cpp)
    # tengine_onnx_op_test(test_onnx_op_logsoftmax_default_axis       op/test_onnx_op_logsoftmax_default_axis.cpp)
    tengine_onnx_op_test(test_onnx_op_lstm_defaults                 op/test_onnx_op_lstm_defaults.cpp)
    tengine_onnx_op_test(test_onnx_op_lstm_with_initial_bias        op/test_onnx_op_lstm_with_initial_bias.cpp)
    tengine_onnx_op_test(test_onnx_op_matmul_2d                     op/test_onnx_op_matmul_2d.cpp)
    tengine_onnx_op_test(test_onnx_op_matmul_3d                     op/test_onnx_op_matmul_3d.cpp)
    tengine_onnx_op_test(test_onnx_op_matmul_4d                     op/test_onnx_op_matmul_4d.cpp)
    tengine_onnx_op_test(test_onnx_op_maxpool_2d_default            op/test_onnx_op_maxpool_2d_default.cpp)
    # tengine_onnx_op_test(test_onnx_op_maxpool_2d_dilations          op/test_onnx_op_maxpool_2d_dilations.cpp)
    tengine_onnx_op_test(test_onnx_op_maxpool_2d_pads               op/test_onnx_op_maxpool_2d_pads.cpp)
    tengine_onnx_op_test(test_onnx_op_neg                           op/test_onnx_op_neg.cpp)
    tengine_onnx_op_test(test_onnx_op_pow                           op/test_onnx_op_pow.cpp)
    tengine_onnx_op_test(test_onnx_op_reciprocal                    op/test_onnx_op_reciprocal.cpp)
    tengine_onnx_op_test(test_onnx_op_reduce_log_sum_default        op/test_onnx_op_reduce_log_sum_default.cpp)
    tengine_onnx_op_test(test_onnx_op_reduce_max_default_axes_keepdim_example               op/test_onnx_op_reduce_max_default_axes_keepdim_example.cpp)
    # tengine_onnx_op_test(test_onnx_op_reduce_mean_default_axes_keepdims_example             op/test_onnx_op_reduce_mean_default_axes_keepdims_example.cpp)
    # tengine_onnx_op_test(test_onnx_op_reduce_min_default_axes_keepdims_example              op/test_onnx_op_reduce_min_default_axes_keepdims_example.cpp)
    tengine_onnx_op_test(test_onnx_op_reduce_sum_square_default_axes_keepdims_example       op/test_onnx_op_reduce_sum_square_default_axes_keepdims_example.cpp)
    tengine_onnx_op_test(test_onnx_op_relu                          op/test_onnx_op_relu.cpp)
    tengine_onnx_op_test(test_onnx_op_selu                          op/test_onnx_op_selu.cpp)
    tengine_onnx_op_test(test_onnx_op_selu_default                  op/test_onnx_op_selu_default.cpp)
    # tengine_onnx_op_test(test_onnx_op_softmax_default_axis          op/test_onnx_op_softmax_default_axis.cpp)
    tengine_onnx_op_test(test_onnx_op_softplus                      op/test_onnx_op_softplus.cpp)
    tengine_onnx_op_test(test_onnx_op_squeeze                       op/test_onnx_op_squeeze.cpp)
    tengine_onnx_op_test(test_onnx_op_sub                           op/test_onnx_op_sub.cpp)
    tengine_onnx_op_test(test_onnx_op_tanh                          op/test_onnx_op_tanh.cpp)
    tengine_onnx_op_test(test_onnx_op_unsqueeze_axis_0              op/test_onnx_op_unsqueeze_axis_0.cpp)
    tengine_onnx_op_test(test_onnx_op_unsqueeze_axis_1              op/test_onnx_op_unsqueeze_axis_1.cpp)
    tengine_onnx_op_test(test_onnx_op_unsqueeze_axis_2              op/test_onnx_op_unsqueeze_axis_2.cpp)
    tengine_onnx_op_test(test_onnx_op_unsqueeze_axis_3              op/test_onnx_op_unsqueeze_axis_3.cpp)
    tengine_onnx_op_test(test_onnx_op_unsqueeze_negative_axes       op/test_onnx_op_unsqueeze_negative_axes.cpp)
    tengine_onnx_op_test(test_onnx_op_unsqueeze_three_axes          op/test_onnx_op_unsqueeze_three_axes.cpp)
    tengine_onnx_op_test(test_onnx_op_unsqueeze_two_axes            op/test_onnx_op_unsqueeze_two_axes.cpp)
    tengine_onnx_op_test(test_onnx_op_unsqueeze_unsorted_axes       op/test_onnx_op_unsqueeze_unsorted_axes.cpp)
else()
    message(WARNING "Protobuf not found, onnx op test won't be built")
endif()


FUNCTION (TENGINE_MODEL_TEST name file)
    ADD_EXECUTABLE (${name} "${CMAKE_CURRENT_SOURCE_DIR}/${file}" "${CMAKE_CURRENT_SOURCE_DIR}/common/tengine_operations.c")

    TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${PROJECT_SOURCE_DIR}/tests/common")

    TARGET_LINK_LIBRARIES (${name} PRIVATE ${CMAKE_PROJECT_NAME})
    TARGET_LINK_LIBRARIES (${name} PRIVATE m)

    # add to a virtual project group
    SET_PROPERTY(TARGET ${name} PROPERTY FOLDER "tests/test_model")

ENDFUNCTION()

# add c api examples
TENGINE_MODEL_TEST (test_model_alphapose            models/test_model_alphapose.cpp)
TENGINE_MODEL_TEST (test_model_classification       models/test_model_classification.cpp)
TENGINE_MODEL_TEST (test_model_crnn                 models/test_model_crnn.cpp)
TENGINE_MODEL_TEST (test_model_efficientdet         models/test_model_efficientdet.c)
TENGINE_MODEL_TEST (test_model_hrnet                models/test_model_hrnet.cpp)
TENGINE_MODEL_TEST (test_model_landmark             models/test_model_landmark.cpp)
TENGINE_MODEL_TEST (test_model_mobilefacenet        models/test_model_mobilefacenet.cpp)
TENGINE_MODEL_TEST (test_model_mobilenet_ssd        models/test_model_mobilenet_ssd.c)
TENGINE_MODEL_TEST (test_model_nanodet_m            models/test_model_nanodet_m.cpp)
TENGINE_MODEL_TEST (test_model_openpose             models/test_model_openpose.cpp)
TENGINE_MODEL_TEST (test_model_retinaface           models/test_model_retinaface.cpp)
TENGINE_MODEL_TEST (test_model_ultraface            models/test_model_ultraface.cpp)
TENGINE_MODEL_TEST (test_model_unet                 models/test_model_unet.cpp)
TENGINE_MODEL_TEST (test_model_yolact               models/test_model_yolact.cpp)
TENGINE_MODEL_TEST (test_model_yolofastest          models/test_model_yolofastest.cpp)
TENGINE_MODEL_TEST (test_model_yolov3               models/test_model_yolov3.cpp)
TENGINE_MODEL_TEST (test_model_yolov3_tiny          models/test_model_yolov3_tiny.cpp)
TENGINE_MODEL_TEST (test_model_yolov4               models/test_model_yolov4.cpp)
TENGINE_MODEL_TEST (test_model_yolov4_tiny          models/test_model_yolov4_tiny.cpp)
TENGINE_MODEL_TEST (test_model_yolov5s              models/test_model_yolov5s.cpp)
TENGINE_MODEL_TEST (test_model_common               models/test_model_common.cpp)
IF(TENGINE_ENABLE_TENSORRT)
    TENGINE_MODEL_TEST (test_trt_model_classification       models/test_trt_model_classification.cpp)
ENDIF()
