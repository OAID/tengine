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
// 4*4 single precise floating point matric multiplication
//
//    --              --      --               --     --               --         --                   --
//    | i0 - - - - - - |      |  k0  k1  k2  k3 |     |  b0  b1  b2  b3 |         | i0k0 i0k1 i0k2 i0k3 |
//    |                |      |  .   .   .   .  |     |                 |         |                     |
//    | i1 - - - - - - |      |  .   .   .   .  |     |  b0  b1  b2  b3 |         | i1k0 i1k1 i1k2 i1k3 |
//    |                |  x   |  .   .   .   .  |  +  |                 |     =   |                     |
//    | i2 - - - - - - |      |  .   .   .   .  |     |  b0  b1  b2  b3 |         | i2k0 i2k1 i2k2 i2k3 |
//    |                |      |  .   .   .   .  |     |                 |         |                     |
//    | i3 - - - - - - |      |  .   .   .   .  |     |  b0  b1  b2  b3 |         | i3k0 i3k1 i3k2 i3k3 |
//    --              --      --               --     --               --         --                   --
//      input 4 x p             kernel p x 4             biases 4 x 4                 output 4 x 4         p = kernel size
//
//
//
// input:  
//         x0 arg0  biases address {b0,b1,b2,b3}  nullptr means no biases 
//         x1 arg1  input  address {i[0-3][0],i1[0-3][1],i[0-3][2],i[0-3][3],i[0-3][4],...}
//         x2 arg2  kernel address {k[0-3][0],k[0-3][1],k[0-3][2],k[0-3][3],...}
//         x3 arg3  kernel size
//         x4 arg4  output address 
//                  indirect save: output {i[0-3]k[0],i[0-3]k[1],i[0-3]k[2],i[0-3]k[3]}
//                    direct save: output                 : {i0k0  i1k0  i2k0  i3k0}
//                                 output + ouput_xy      : {i0k1  i1k1  i2k1  i3k1}
//                                 output + ouput_xy * 2  : {i0k2  i1k2  i2k2  i3k2}
//                                 output + ouput_xy * 3  : {i0k3  i1k3  i2k3  i3k3}
//         x5 arg5  output xy
//         x6 arg6  activation flag     relu layers is integrated after convolution
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
// x6        fused relu flag
// x9 ~ x10  temp loop counter
// x11~ x13  temp output save address
// x7~8 14~15 not used

//
// v0-3 4S data of input0   {i3   i2   i1   i0}
// v4-7 4S kernal data      {k3   k2   k1   k0}
// v8~v15 not used
// v16 dot product for {i3k0, i2k0, i1k0, i0k0}
// v17 dot product for {i3k1, i2k1, i1k1, i0k1}
// v18 dot product for {i3k2, i2k2, i1k2, i0k2}
// v19 dot product for {i3k3, i2k3, i1k3, i0k3}
// v20~V31 not used

        .section .text,"ax"
        .align 5

        .type sgemm_4x4_rv64 STT_FUNC
        .global sgemm_4x4_rv64
        .hidden sgemm_4x4_rv64
sgemm_4x4_rv64:
        addi    sp, sp, -8
        sd      ra, (sp)
        call    vsetvl_e32_m1
        ld      ra, (sp)

        slli            a5, a5, 0x2
#       // initial biases
        beqz            a0, non_biases

        vle32.v           v0, (a0)
        vrgather.vi     v16, v0, 0
        vrgather.vi     v17, v0, 1
        vrgather.vi     v18, v0, 2
        vrgather.vi     v19, v0, 3

        j               convoluation_start
	
non_biases:
        vmv.v.x         v16, x0
        vmv.v.x         v17, x0
        vmv.v.x         v18, x0
        vmv.v.x         v19, x0

convoluation_start:
        add             t4, a4, a5
        
        andi	        t3, a3, 0x3

        li              t0, 4
        blt             a3, t0, loop4_end
        srli            t2, a3, 0x2

// main loop: each loop generate dot prodcut for 4x4SFP
loop4:  
        addi            t2, t2, -1
        
        vle32.v           v0, (a1)
        addi            a1, a1, 16
        vle32.v           v1, (a1)
        addi            a1, a1, 16
        vle32.v           v2, (a1)
        addi            a1, a1, 16
        vle32.v           v3, (a1)
        addi            a1, a1, 16
        
        vle32.v           v4, (a2)
        addi            a2, a2, 16
        vle32.v           v5, (a2)
        addi            a2, a2, 16
        vle32.v           v6, (a2)
        addi            a2, a2, 16
        vle32.v           v7, (a2)
        addi            a2, a2, 16
        
        vrgather.vi     v20, v4, 0
        vrgather.vi     v21, v4, 1
        vrgather.vi     v22, v4, 2
        vrgather.vi     v23, v4, 3
        vfmacc.vv       v16, v20, v0
        vfmacc.vv       v17, v21, v0
        vfmacc.vv       v18, v22, v0
        vfmacc.vv       v19, v23, v0
        
        vrgather.vi     v20, v5, 0
        vrgather.vi     v21, v5, 1
        vrgather.vi     v22, v5, 2
        vrgather.vi     v23, v5, 3
        vfmacc.vv       v16, v20, v1
        vfmacc.vv       v17, v21, v1
        vfmacc.vv       v18, v22, v1
        vfmacc.vv       v19, v23, v1

        vrgather.vi     v20, v6, 0
        vrgather.vi     v21, v6, 1
        vrgather.vi     v22, v6, 2
        vrgather.vi     v23, v6, 3
        vfmacc.vv       v16, v20, v2
        vfmacc.vv       v17, v21, v2
        vfmacc.vv       v18, v22, v2
        vfmacc.vv       v19, v23, v2

        vrgather.vi     v20, v7, 0
        vrgather.vi     v21, v7, 1
        vrgather.vi     v22, v7, 2
        vrgather.vi     v23, v7, 3
        vfmacc.vv       v16, v20, v3
        vfmacc.vv       v17, v21, v3
        vfmacc.vv       v18, v22, v3
        vfmacc.vv       v19, v23, v3

        bnez            t2, loop4

loop4_end:
        slli            t0, a5, 1
        add             t5, a4, t0
        beqz            t3, activation

loop1:
        addi            t3, t3, -1
        
        vle32.v           v0, (a1)
        addi            a1, a1, 16
        
        vle32.v           v4, (a2)
        addi            a2, a2, 16

        vrgather.vi     v20, v4, 0
        vrgather.vi     v21, v4, 1
        vrgather.vi     v22, v4, 2
        vrgather.vi     v23, v4, 3
        vfmacc.vv       v16, v20, v0
        vfmacc.vv       v17, v21, v0
        vfmacc.vv       v18, v22, v0
        vfmacc.vv       v19, v23, v0

        bnez            t3, loop1


activation:
        slli            t0, a5, 1
        add             t6, t4, t0
        
        bltz            a6, save_result
        
        vmv.v.i         v0, 0
        vmv.v.x         v1, a6

        vfmax.vv        v16, v16, v0
        vfmax.vv        v17, v17, v0
        vfmax.vv        v18, v18, v0
        vfmax.vv        v19, v19, v0    

        beqz            a6, save_result
        vfmin.vv        v16, v16, v1
        vfmin.vv        v17, v17, v1
        vfmin.vv        v18, v18, v1
        vfmin.vv        v19, v19, v1 

save_result:
# 	// store result
        beqz            a7, save_result_nchw
        
        vsse32.v        v16, (a4), a5
        addi            a4, a4, 4
        vsse32.v        v17, (a4), a5
        addi            a4, a4, 4
        vsse32.v        v18, (a4), a5
        addi            a4, a4, 4
        vsse32.v        v19, (a4), a5

        j               end

save_result_nchw:
        vse32.v           v16, (a4)
        vse32.v           v17, (t4)
        vse32.v           v18, (t5)
        vse32.v           v19, (t6)

end:
    addi    sp, sp, 8
	ret
    .end

