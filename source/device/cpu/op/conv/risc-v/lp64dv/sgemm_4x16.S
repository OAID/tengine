/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2020, OPEN AI LAB
 * Author: ddzhao@openailab.com
*/
//
// 4*16 single precise floating point matric multiplication
//
//    --              --      --               --     --               --         --                 --
//    | i0 - - - - - - |      |  k0  k1  ..  kf |     |  b0  b1  ..  bf |         | i0k0 i0k1 .. i0kf |
//    |                |      |  .   .   .   .  |     |                 |         |                   |
//    | i1 - - - - - - |      |  .   .   .   .  |     |  b0  b1  .   bf |         | i1k0 i1k1 .. i1kf |
//    |                |  x   |  .   .   .   .  |  +  |                 |     =   |                   |
//    | i2 - - - - - - |      |  .   .   .   .  |     |  b0  b1  .   bf |         | i2k0 i2k1 .. i2kf |
//    |                |      |  .   .   .   .  |     |                 |         |                   |
//    | i3 - - - - - - |      |  .   .   .   .  |     |  b0  b1  .   bf |         | i3k0 i3k1 .. i3kf |
//    --              --      --               --     --               --         --                 --
//      input 4 x p             kernel p x 16            biases 4 x 16                 output 4 x 16           p = kernel size
//
//
// load 4 more input and 8 more kernel to improve loop performance
//
// input: 
//         x0 arg0  biases address {b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15}  nullptr means no biases 
//         x1 arg1  input  address {i[0-3][0],i1[0-3][1],i[0-3][2],i[0-3][3],i[0-3][4],...}
//         x2 arg2  kernel address {k[0-15][0],k[0-15][1],k[0-15][2],k[0-15][3],...}
//         x3 arg3  kernel size
//         x4 arg4  output address 
//                  indirect save: output {i[0-3]k[0],i[0-3]k[1],i[0-3]k[2],i[0-3]k[3],i[0-3]k[4]..}
//                    direct save: output                 : {i0k0  i1k0  i2k0  i3k0}
//                                 output + ouput_xy      : {i0k1  i1k1  i2k1  i3k1}
//                                 output + ouput_xy * 2  : {i0k2  i1k2  i2k2  i3k2}
//                                 ...
//                                 output + ouput_xy * 15 : {i0k15 i1k15 i2k15 i3k15}
//         x5 arg5  output xy 
//         x6 arg6  activation flag     activation layers is integrated after convolution
//
// output: no
//
// register definition
// x0        biases start address
// x1        input start address
// x2        kernel start address
// x3        kernal size 
// x4        output start address
// x5        output_x * output_y
// x6        activation flag
// x9 ~ x10  temp loop counter
// x11~ x13  temp output save address
// x14       output_xy * 4
// x7~8 x15  not used
// x9       t1
// x10      t2
// x11      t3
// x12      t4
// x13      t5
// x14      t6
//
// v0~1 4S data of input0   {i3   i2   i1   i0}
// v2~3 not used
// v4   4S kernal data      {k3 | k2 | k1 | k0}
// v5   4S kernal data      {k7 | k6 | k5 | k4}
// v6   4S kernal data      {kb | ka | k9 | k8}
// v7   4S kernal data      {kf | ke | kd | kc}
// v8~15 not used
// v16 dot product for {i3k0, i2k0, i1k0, i0k0}
// v17 dot product for {i3k1, i2k1, i1k1, i0k1}
// v18 dot product for {i3k2, i2k2, i1k2, i0k2}
// v19 dot product for {i3k3, i2k3, i1k3, i0k3}
// v20 dot product for {i3k4, i2k4, i1k4, i0k4}
// v21 dot product for {i3k5, i2k5, i1k5, i0k5}
// v22 dot product for {i3k6, i2k6, i1k6, i0k6}
// v23 dot product for {i3k7, i2k7, i1k7, i0k7}
// v24 dot product for {i3k8, i2k8, i1k8, i0k8}
// v25 dot product for {i3k9, i2k9, i1k9, i0k9}
// v26 dot product for {i3ka, i2ka, i1ka, i0ka}
// v27 dot product for {i3kb, i2kb, i1kb, i0kb}
// v28 dot product for {i3kc, i2kc, i1kc, i0kc}
// v29 dot product for {i3kd, i2kd, i1kd, i0kd}
// v30 dot product for {i3ke, i2ke, i1ke, i0ke}
// v31 dot product for {i3kf, i2kf, i1kf, i0kf}

    .section .text,"ax"
    .align 5

    .type sgemm_4x16_rv64 STT_FUNC
    .global sgemm_4x16_rv64
    .hidden sgemm_4x16_rv64
sgemm_4x16_rv64:
    addi            sp, sp, -64
    sd              t0, 0(sp)
    sd              t1, 8(sp)
    sd              t2, 16(sp)
    sd              t3, 24(sp)
    sd              t4, 32(sp)
    sd              t5, 40(sp)
    sd              t6, 48(sp)
    sd              ra, 56(sp)

    call            vsetvl_e32_m1
    ld              ra, 56(sp)

// biases_initial
    beqz            a0, none_biases
    vle32.v           v0, (a0)
    vrgather.vi     v16, v0, 0
    vrgather.vi     v17, v0, 1
    vrgather.vi     v18, v0, 2
    vrgather.vi     v19, v0, 3
    addi            a0, a0, 0x10
    vle32.v           v0, (a0)
    vrgather.vi     v20, v0, 0
    vrgather.vi     v21, v0, 1
    vrgather.vi     v22, v0, 2
    vrgather.vi     v23, v0, 3
    addi            a0, a0, 0x10
    vle32.v           v0, (a0)
    vrgather.vi     v24, v0, 0
    vrgather.vi     v25, v0, 1
    vrgather.vi     v26, v0, 2
    vrgather.vi     v27, v0, 3
    addi            a0, a0, 0x10
    vle32.v           v0, (a0)
    vrgather.vi     v28, v0, 0
    vrgather.vi     v29, v0, 1
    vrgather.vi     v30, v0, 2
    vrgather.vi     v31, v0, 3

    j               convolution_start

none_biases:
    vmv.v.x         v16, x0
    vmv.v.x         v17, x0
    vmv.v.x         v18, x0
    vmv.v.x         v19, x0
    vmv.v.x         v20, x0
    vmv.v.x         v21, x0
    vmv.v.x         v22, x0
    vmv.v.x         v23, x0
    vmv.v.x         v24, x0
    vmv.v.x         v25, x0
    vmv.v.x         v26, x0
    vmv.v.x         v27, x0
    vmv.v.x         v28, x0
    vmv.v.x         v29, x0
    vmv.v.x         v30, x0
    vmv.v.x         v31, x0

convolution_start:
    vle32.v           v0, (a1)
    addi            t0, a2, 0
    vle32.v           v4, (t0)
    addi            t0, a2, 0x10
    vle32.v           v5, (t0)

    andi             t2, a3, 0x3
    slli            a5, a5, 0x2
    bltz            t2, loop4_end
    srli            t1, a3, 0x2

// main loop     each loop generate dot prodcut for 4x16x4SP
loop4:  
    addi            t1, t1, -1
    addi            t0, a2, 0x20
    vle32.v           v6, (t0)
    addi            t0, a2, 0x30
    vle32.v           v7, (t0)

    vrgather.vi     v8, v4, 0
    vrgather.vi     v9, v4, 1
    vrgather.vi     v10, v4, 2
    vrgather.vi     v11, v4, 3
    vfmacc.vv       v16, v0, v8
    vfmacc.vv       v17, v0, v9
    vfmacc.vv       v18, v0, v10
    vfmacc.vv       v19, v0, v11
    
    addi            t0, a1, 0x10
    vle32.v           v1, (t0)
    
    vrgather.vi     v8,  v5, 0
    vrgather.vi     v9,  v5, 1
    vrgather.vi     v10, v5, 2
    vrgather.vi     v11, v5, 3
    vfmacc.vv       v20, v0, v8
    vfmacc.vv       v21, v0, v9
    vfmacc.vv       v22, v0, v10
    vfmacc.vv       v23, v0, v11
    
    addi            t0, a2, 0x40
    vle32.v           v4, (t0)
    addi            t0, a2, 0x50
    vle32.v           v5, (t0)
    
    vrgather.vi     v8,  v6, 0
    vrgather.vi     v9,  v6, 1
    vrgather.vi     v10, v6, 2
    vrgather.vi     v11, v6, 3
    vfmacc.vv       v24, v0, v8
    vfmacc.vv       v25, v0, v9
    vfmacc.vv       v26, v0, v10
    vfmacc.vv       v27, v0, v11
    
    vrgather.vi     v8,  v7, 0
    vrgather.vi     v9,  v7, 1
    vrgather.vi     v10, v7, 2
    vrgather.vi     v11, v7, 3
    vfmacc.vv       v28, v0, v8
    vfmacc.vv       v29, v0, v9
    vfmacc.vv       v30, v0, v10
    vfmacc.vv       v31, v0, v11

    addi            t0, a2, 0x60
    vle32.v           v6, (t0)
    addi            t0, a2, 0x70
    vle32.v           v7, (t0)
    
    vrgather.vi     v8, v4, 0
    vrgather.vi     v9, v4, 1
    vrgather.vi     v10, v4, 2
    vrgather.vi     v11, v4, 3
    vfmacc.vv       v16, v1, v8
    vfmacc.vv       v17, v1, v9
    vfmacc.vv       v18, v1, v10
    vfmacc.vv       v19, v1, v11
    
    addi            t0, a1, 0x20
    vle32.v           v0, (t0)
    
    vrgather.vi     v8,  v5, 0
    vrgather.vi     v9,  v5, 1
    vrgather.vi     v10, v5, 2
    vrgather.vi     v11, v5, 3
    vfmacc.vv       v20, v1, v8
    vfmacc.vv       v21, v1, v9
    vfmacc.vv       v22, v1, v10
    vfmacc.vv       v23, v1, v11
    
    addi            t0, a2, 0x80
    vle32.v           v4, (t0)
    addi            t0, a2, 0x90
    vle32.v           v5, (t0)
    
    vrgather.vi     v8,  v6, 0
    vrgather.vi     v9,  v6, 1
    vrgather.vi     v10, v6, 2
    vrgather.vi     v11, v6, 3
    vfmacc.vv       v24, v1, v8
    vfmacc.vv       v25, v1, v9
    vfmacc.vv       v26, v1, v10
    vfmacc.vv       v27, v1, v11
    
    vrgather.vi     v8,  v7, 0
    vrgather.vi     v9,  v7, 1
    vrgather.vi     v10, v7, 2
    vrgather.vi     v11, v7, 3
    vfmacc.vv       v28, v1, v8
    vfmacc.vv       v29, v1, v9
    vfmacc.vv       v30, v1, v10
    vfmacc.vv       v31, v1, v11
    
    addi            t0, a2, 0xa0
    vle32.v           v6, (t0)
    addi            t0, a2, 0xb0
    vle32.v           v7, (t0)
    
    vrgather.vi     v8, v4, 0
    vrgather.vi     v9, v4, 1
    vrgather.vi     v10, v4, 2
    vrgather.vi     v11, v4, 3
    vfmacc.vv       v16, v0, v8
    vfmacc.vv       v17, v0, v9
    vfmacc.vv       v18, v0, v10
    vfmacc.vv       v19, v0, v11
    
    addi            t0, a1, 0x30
    vle32.v           v1, (t0)
    addi             a1, a1, 0x40
    
    vrgather.vi     v8,  v5, 0
    vrgather.vi     v9,  v5, 1
    vrgather.vi     v10, v5, 2
    vrgather.vi     v11, v5, 3
    vfmacc.vv       v20, v0, v8
    vfmacc.vv       v21, v0, v9
    vfmacc.vv       v22, v0, v10
    vfmacc.vv       v23, v0, v11
    
    addi            t0, a2, 0xc0
    vle32.v           v4, (t0)
    addi            t0, a2, 0xd0
    vle32.v           v5, (t0)
    
    vrgather.vi     v8,  v6, 0
    vrgather.vi     v9,  v6, 1
    vrgather.vi     v10, v6, 2
    vrgather.vi     v11, v6, 3
    vfmacc.vv       v24, v0, v8
    vfmacc.vv       v25, v0, v9
    vfmacc.vv       v26, v0, v10
    vfmacc.vv       v27, v0, v11
    
    vrgather.vi     v8,  v7, 0
    vrgather.vi     v9,  v7, 1
    vrgather.vi     v10, v7, 2
    vrgather.vi     v11, v7, 3
    vfmacc.vv       v28, v0, v8
    vfmacc.vv       v29, v0, v9
    vfmacc.vv       v30, v0, v10
    vfmacc.vv       v31, v0, v11
    
    addi            t0, a2, 0xe0
    vle32.v           v6, (t0)
    addi            t0, a2, 0xf0
    vle32.v           v7, (t0)
    addi            a2, a2, 0x100
    vrgather.vi     v8, v4, 0
    vrgather.vi     v9, v4, 1
    vrgather.vi     v10, v4, 2
    vrgather.vi     v11, v4, 3
    vfmacc.vv       v16, v1, v8
    vfmacc.vv       v17, v1, v9
    vfmacc.vv       v18, v1, v10
    vfmacc.vv       v19, v1, v11

    vle32.v           v0, (a1)
    
    vrgather.vi     v8,  v5, 0
    vrgather.vi     v9,  v5, 1
    vrgather.vi     v10, v5, 2
    vrgather.vi     v11, v5, 3
    vfmacc.vv       v20, v1, v8
    vfmacc.vv       v21, v1, v9
    vfmacc.vv       v22, v1, v10
    vfmacc.vv       v23, v1, v11
    
    addi            t0, a2, 0x0
    vle32.v           v4, (t0)
    addi            t0, a2, 0x10
    vle32.v           v5, (t0)
    
    vrgather.vi     v8,  v6, 0
    vrgather.vi     v9,  v6, 1
    vrgather.vi     v10, v6, 2
    vrgather.vi     v11, v6, 3
    vfmacc.vv       v24, v1, v8
    vfmacc.vv       v25, v1, v9
    vfmacc.vv       v26, v1, v10
    vfmacc.vv       v27, v1, v11

    vrgather.vi     v8,  v7, 0
    vrgather.vi     v9,  v7, 1
    vrgather.vi     v10, v7, 2
    vrgather.vi     v11, v7, 3
    vfmacc.vv       v28, v1, v8
    vfmacc.vv       v29, v1, v9
    vfmacc.vv       v30, v1, v10
    vfmacc.vv       v31, v1, v11
    bnez            t1, loop4

loop4_end:
    slli            t6, a5, 2
    beqz            t2, activation

loop1:
    addi            t0, a2, 0x20
    vle32.v           v6, (t0)
    addi            t0, a2, 0x30
    vle32.v           v7, (t0)
    addi            a2, a2, 0x40
    vrgather.vi     v8, v4, 0
    vrgather.vi     v9, v4, 1
    vrgather.vi     v10, v4, 2
    vrgather.vi     v11, v4, 3
    vfmacc.vv       v16, v0, v8
    vfmacc.vv       v17, v0, v9
    vfmacc.vv       v18, v0, v10
    vfmacc.vv       v19, v0, v11
    addi            a1, a1, 0x10
    addi            t2, t2, -1
    vrgather.vi     v8,  v5, 0
    vrgather.vi     v9,  v5, 1
    vrgather.vi     v10, v5, 2
    vrgather.vi     v11, v5, 3
    vfmacc.vv       v20, v0, v8
    vfmacc.vv       v21, v0, v9
    vfmacc.vv       v22, v0, v10
    vfmacc.vv       v23, v0, v11
    addi            t0, a2, 0x0
    vle32.v           v4, (t0)
    addi            t0, a2, 0x10
    vle32.v           v5, (t0)
    vrgather.vi     v8,  v6, 0
    vrgather.vi     v9,  v6, 1
    vrgather.vi     v10, v6, 2
    vrgather.vi     v11, v6, 3
    vfmacc.vv       v24, v0, v8
    vfmacc.vv       v25, v0, v9
    vfmacc.vv       v26, v0, v10
    vfmacc.vv       v27, v0, v11
    vrgather.vi     v8,  v7, 0
    vrgather.vi     v9,  v7, 1
    vrgather.vi     v10, v7, 2
    vrgather.vi     v11, v7, 3
    vfmacc.vv       v28, v0, v8
    vfmacc.vv       v29, v0, v9
    vfmacc.vv       v30, v0, v10
    vfmacc.vv       v31, v0, v11
    
    vle32.v           v0, (a1)
    bnez            t2, loop1

activation:
    add             t3, a4, a5
    bltz            a6, save_result
    vmv.v.x         v0, x0
    vmv.v.x         v0, a6          // FIXME: change DataType
    vfmax.vv        v16, v16, v0
    vfmax.vv        v17, v17, v0
    vfmax.vv        v18, v18, v0
    vfmax.vv        v19, v19, v0
    vfmax.vv        v20, v20, v0
    vfmax.vv        v21, v21, v0
    vfmax.vv        v22, v22, v0
    vfmax.vv        v23, v23, v0
    vfmax.vv        v24, v24, v0
    vfmax.vv        v25, v25, v0
    vfmax.vv        v26, v26, v0
    vfmax.vv        v27, v27, v0
    vfmax.vv        v28, v28, v0
    vfmax.vv        v29, v29, v0
    vfmax.vv        v30, v30, v0
    vfmax.vv        v31, v31, v0

    beqz            a6, save_result
    vfmin.vv        v16, v16, v1
    vfmin.vv        v17, v17, v1
    vfmin.vv        v18, v18, v1
    vfmin.vv        v19, v19, v1
    vfmin.vv        v20, v20, v1
    vfmin.vv        v21, v21, v1
    vfmin.vv        v22, v22, v1
    vfmin.vv        v23, v23, v1
    vfmin.vv        v24, v24, v1
    vfmin.vv        v25, v25, v1
    vfmin.vv        v26, v26, v1
    vfmin.vv        v27, v27, v1
    vfmin.vv        v28, v28, v1
    vfmin.vv        v29, v29, v1
    vfmin.vv        v30, v30, v1
    vfmin.vv        v31, v31, v1

save_result:
    slli            t0, a5, 1
    add             t4, a4, t0
    add             t5, t3, t0
#     // store result
    beqz            a7, save_result_nchw

    vsse32.v        v16, (a4), a5
    addi            a4, a4, 4
    vsse32.v        v17, (a4), a5
    addi            a4, a4, 4
    vsse32.v        v18, (a4), a5
    addi            a4, a4, 4
    vsse32.v        v19, (a4), a5
    addi            a4, a4, 4
    vsse32.v        v20, (a4), a5
    addi            a4, a4, 4
    vsse32.v        v21, (a4), a5
    addi            a4, a4, 4
    vsse32.v        v22, (a4), a5
    addi            a4, a4, 4
    vsse32.v        v23, (a4), a5
    addi            a4, a4, 4
    vsse32.v        v24, (a4), a5
    addi            a4, a4, 4
    vsse32.v        v25, (a4), a5
    addi            a4, a4, 4
    vsse32.v        v26, (a4), a5
    addi            a4, a4, 4
    vsse32.v        v27, (a4), a5
    addi            a4, a4, 4
    vsse32.v        v28, (a4), a5
    addi            a4, a4, 4
    vsse32.v        v29, (a4), a5
    addi            a4, a4, 4
    vsse32.v        v30, (a4), a5
    addi            a4, a4, 4
    vsse32.v        v31, (a4), a5

    j               end

save_result_nchw:
    vse32.v           v16, (a4)
    add             a4, a4, t6
    vse32.v           v17, (t3)
    add             t3, t3, t6
    vse32.v           v18, (t4)
    add             t4, t4, t6
    vse32.v           v19, (t5)
    add             t5, t5, t6
    
    vse32.v           v20, (a4)
    add             a4, a4, t6
    vse32.v           v21, (t3)
    add             t3, t3, t6
    vse32.v           v22, (t4)
    add             t4, t4, t6
    vse32.v           v23, (t5)
    add             t5, t5, t6
    
    vse32.v           v24, (a4)
    add             a4, a4, t6
    vse32.v           v25, (t3)
    add             t3, t3, t6
    vse32.v           v26, (t4)
    add             t4, t4, t6
    vse32.v           v27, (t5)
    add             t5, t5, t6
    
    vse32.v           v28, (a4)
    vse32.v           v29, (t3)
    vse32.v           v30, (t4)
    vse32.v           v31, (t5)

end:
    ld            t0, 0(sp)
    ld            t1, 8(sp)
    ld            t2, 16(sp)
    ld            t3, 24(sp)
    ld            t4, 32(sp)
    ld            t5, 40(sp)
    ld            t6, 48(sp)
    addi          sp, sp, 64 
    ret
    .end
