# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# License); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# Copyright (c) 2021, OPEN AI LAB
# Author: lswang@openailab.com
#

# generate tengine header file
FILE (MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tengine)
FILE (COPY ${CMAKE_SOURCE_DIR}/source/api/c_api.h DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/tengine)
#FILE (COPY ${CMAKE_SOURCE_DIR}/source/api/c_api.h DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
#FILE (RENAME ${CMAKE_CURRENT_BINARY_DIR}/c_api.h ${CMAKE_CURRENT_BINARY_DIR}/tengine_c_api.h)

# macro for adding test
FUNCTION (TENGINE_EXAMPLE name file)
    ADD_EXECUTABLE (${name} "${CMAKE_CURRENT_SOURCE_DIR}/${file}" "${CMAKE_CURRENT_SOURCE_DIR}/common/tengine_operations.c")

    TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${PROJECT_SOURCE_DIR}/examples/common")

    TARGET_LINK_LIBRARIES (${name} PRIVATE ${CMAKE_PROJECT_NAME})

    IF(MSVC)
        TARGET_COMPILE_DEFINITIONS (${name} PRIVATE $<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:NOMINMAX>)
    ELSE()
        TARGET_LINK_LIBRARIES (${name} PRIVATE m)
    ENDIF()

    INSTALL (TARGETS ${name} DESTINATION bin)

    # add to a virtual project group
    SET_PROPERTY(TARGET ${name} PROPERTY FOLDER "examples")
ENDFUNCTION()

# add c++ api examples
IF (TENGINE_BUILD_CPP_API)
    TENGINE_EXAMPLE (cpp_tm_classification       cpp_tm_classification.cpp)
    TENGINE_EXAMPLE (cpp_tm_mobilenet_ssd        cpp_tm_mobilenet_ssd.cpp)
ENDIF()

# add c api examples
TENGINE_EXAMPLE (tm_classification           tm_classification.c)
TENGINE_EXAMPLE (tm_classification_int8      tm_classification_int8.c)
TENGINE_EXAMPLE (tm_classification_uint8     tm_classification_uint8.c)
TENGINE_EXAMPLE (tm_efficientdet             tm_efficientdet.c)
TENGINE_EXAMPLE (tm_efficientdet_uint8       tm_efficientdet_uint8.c)
TENGINE_EXAMPLE (tm_mobilenet_ssd            tm_mobilenet_ssd.c)
TENGINE_EXAMPLE (tm_mobilenet_ssd_uint8      tm_mobilenet_ssd_uint8.cpp)
TENGINE_EXAMPLE (tm_retinaface               tm_retinaface.cpp)
TENGINE_EXAMPLE (tm_retinaface_vulkan        tm_retinaface_vulkan.cpp)
TENGINE_EXAMPLE (tm_landmark                 tm_landmark.cpp)
TENGINE_EXAMPLE (tm_landmark_uint8           tm_landmark_uint8.cpp)
TENGINE_EXAMPLE (tm_mobilefacenet            tm_mobilefacenet.cpp)
TENGINE_EXAMPLE (tm_mobilefacenet_uint8      tm_mobilefacenet_uint8.cpp)
TENGINE_EXAMPLE (tm_yolov5                   tm_yolov5.cpp)
TENGINE_EXAMPLE (tm_yolofastest              tm_yolofastest.cpp)
TENGINE_EXAMPLE (tm_ultraface                tm_ultraface.cpp)



IF(TENGINE_ENABLE_ACL)
    TENGINE_EXAMPLE (tm_classification_acl       tm_classification_acl.c)
    TENGINE_EXAMPLE (tm_mobilenet_ssd_acl        tm_mobilenet_ssd_acl.c)
ENDIF()

IF(TENGINE_ENABLE_CUDA)
    TENGINE_EXAMPLE (tm_classification_cuda      tm_classification_cuda.cpp)
ENDIF()

IF(TENGINE_ENABLE_OPENDLA)
    TENGINE_EXAMPLE (tm_classification_opendla      tm_classification_opendla.c)
ENDIF()

IF(TENGINE_ENABLE_TENSORRT)
    TENGINE_EXAMPLE (tm_classification_trt       tm_classification_trt.cpp)
ENDIF()

IF(TENGINE_ENABLE_TIM_VX)
    TENGINE_EXAMPLE (tm_classification_timvx     tm_classification_timvx.c)
    TENGINE_EXAMPLE (tm_landmark_timvx           tm_landmark_timvx.cpp)
    TENGINE_EXAMPLE (tm_retinaface_timvx         tm_retinaface_timvx.cpp)
    TENGINE_EXAMPLE (tm_mobilenet_ssd_uint8_timvx      tm_mobilenet_ssd_uint8_timvx.cpp)
ENDIF()

IF(TENGINE_ENABLE_VULKAN)
    TENGINE_EXAMPLE (tm_classification_vulkan       tm_classification_vulkan.c)
ENDIF()

IF (TENGINE_ARCH_ARM_82)
    TENGINE_EXAMPLE (tm_classification_fp16      tm_classification_fp16.c)
ENDIF()


# add examples with opencv
# set(OpenCV_DIR /mnt/d/ubuntu/opencv_install/linux-armv7/lib/cmake/opencv4)
FIND_PACKAGE(OpenCV)

IF (OpenCV_FOUND)
    # macro for adding examples
    FUNCTION (TENGINE_EXAMPLE_CV name file)
        ADD_EXECUTABLE (${name} "${CMAKE_CURRENT_SOURCE_DIR}/${file}" "${CMAKE_CURRENT_SOURCE_DIR}/common/tengine_operations.c")

        TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
        TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
        TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${PROJECT_SOURCE_DIR}/examples/common")
        TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${PROJECT_SOURCE_DIR}/source")
        TARGET_INCLUDE_DIRECTORIES (${name} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/../source")

        TARGET_LINK_LIBRARIES (${name} ${CMAKE_PROJECT_NAME} ${OpenCV_LIBS})
        INSTALL (TARGETS ${name} DESTINATION bin)

        # add to a virtual project group
        SET_PROPERTY(TARGET ${name} PROPERTY FOLDER "examples")
    ENDFUNCTION()

    TENGINE_EXAMPLE_CV (tm_openpose              tm_openpose.cpp)
    TENGINE_EXAMPLE_CV (tm_yolact                tm_yolact.cpp)
    TENGINE_EXAMPLE_CV (tm_yolact_uint8          tm_yolact_uint8.cpp)
    TENGINE_EXAMPLE_CV (tm_crnn                  tm_crnn.cpp)
    TENGINE_EXAMPLE_CV (tm_alphapose             tm_alphapose.cpp)
    TENGINE_EXAMPLE_CV (tm_yolov3                tm_yolov3.cpp)
    TENGINE_EXAMPLE_CV (tm_yolov3_uint8          tm_yolov3_uint8.cpp)
    TENGINE_EXAMPLE_CV (tm_yolov3_opencl         tm_yolov3_opencl.cpp)
    TENGINE_EXAMPLE_CV (tm_yolov3_tiny           tm_yolov3_tiny.cpp)
    TENGINE_EXAMPLE_CV (tm_yolov3_tiny_opendla   tm_yolov3_tiny_opendla.cpp)
    TENGINE_EXAMPLE_CV (tm_yolov3_tiny_int8      tm_yolov3_tiny_int8.cpp)
    TENGINE_EXAMPLE_CV (tm_yolov3_tiny_uint8     tm_yolov3_tiny_uint8.cpp)
    TENGINE_EXAMPLE_CV (tm_yolov4                tm_yolov4.cpp)
    TENGINE_EXAMPLE_CV (tm_yolov4_uint8          tm_yolov4_uint8.cpp)
    TENGINE_EXAMPLE_CV (tm_yolov4_tiny           tm_yolov4_tiny.cpp)
    TENGINE_EXAMPLE_CV (tm_yolov4_tiny_uint8     tm_yolov4_tiny_uint8.cpp)
    TENGINE_EXAMPLE_CV (tm_yolov5s               tm_yolov5s.cpp)
    TENGINE_EXAMPLE_CV (tm_unet                  tm_unet.cpp)
    TENGINE_EXAMPLE_CV (tm_hrnet                 tm_hrnet.cpp)
    TENGINE_EXAMPLE_CV (tm_nanodet_m             tm_nanodet_m.cpp)
    TENGINE_EXAMPLE_CV (tm_yolox                 tm_yolox.cpp)
    TENGINE_EXAMPLE_CV (tm_yolox_int8            tm_yolox_int8.cpp)
    TENGINE_EXAMPLE_CV (tm_yolox_opendla         tm_yolox_opendla.cpp)
    TENGINE_EXAMPLE_CV (tm_yolox_darknet53       tm_yolox_darknet53.cpp)
    TENGINE_EXAMPLE_CV (tm_scrfd                 tm_scrfd.cpp)
    TENGINE_EXAMPLE_CV (tm_segformer             tm_segformer.cpp)
    TENGINE_EXAMPLE_CV (tm_seghuman              tm_seghuman.cpp)
    TENGINE_EXAMPLE_CV (tm_solov2                tm_solov2.cpp)
    TENGINE_EXAMPLE_CV (tm_apollo_smoke          tm_apollo_smoke.cpp)
    TENGINE_EXAMPLE_CV (tm_rvm                   tm_rvm.cpp)
    TENGINE_EXAMPLE_CV (tm_movenet               tm_movenet.cpp)
    TENGINE_EXAMPLE_CV (tm_picodet               tm_picodet.cpp)
    TENGINE_EXAMPLE_CV (tm_handpose              tm_handpose.cpp)
    TENGINE_EXAMPLE_CV (tm_nanodet_plus          tm_nanodet_plus.cpp)
    IF(TENGINE_ENABLE_TIM_VX)
        TENGINE_EXAMPLE_CV (tm_yolov3_timvx          tm_yolov3_timvx.cpp)
        TENGINE_EXAMPLE_CV (tm_yolov4_tiny_timvx     tm_yolov4_tiny_timvx.cpp)
        TENGINE_EXAMPLE_CV (tm_yolov5s_timvx         tm_yolov5s_timvx.cpp)
        TENGINE_EXAMPLE_CV (tm_hrnet_timvx           tm_hrnet_timvx.cpp)
        TENGINE_EXAMPLE_CV (tm_nanodet_m_timvx       tm_nanodet_m_timvx.cpp)
        TENGINE_EXAMPLE_CV (tm_yolox_timvx           tm_yolox_timvx.cpp)
    ENDIF()
    IF(TENGINE_ENABLE_TENSORRT)
        TENGINE_EXAMPLE_CV (tm_scrfd_trt_fp16        tm_scrfd_trt_fp16.cpp)
    ENDIF()

    # pthread
    IF (NOT OHOS)
        INCLUDE (${PROJECT_SOURCE_DIR}/cmake/libraries/pthread.cmake)
        TENGINE_CHECK_LIB_PTHREAD (_has_pthread)
        IF (_has_pthread)
            TENGINE_USE_LIB_PTHREAD (${TENGINE_LITE_NAME}-static ON)
            TENGINE_USE_LIB_PTHREAD (${TENGINE_LITE_NAME}        ON)

            TENGINE_EXAMPLE_CV(tm_pipeline_camera_preview   pipeline/test/test_camera_preview.cpp)
            TENGINE_EXAMPLE_CV(tm_pipeline_estimate_ped_distance  pipeline/test/test_pedestrian_distance_estimation.cpp)
            TENGINE_EXAMPLE_CV(tm_pipeline_enroll_face  pipeline/test/test_face_enroll.cpp)
        ENDIF()
        UNSET (_has_pthread)
    ENDIF()

ELSE()
    MESSAGE (WARNING "OpenCV not found, some examples won't be built")
ENDIF()
