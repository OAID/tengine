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
 * Copyright (c) 2021, OPEN AI LAB
 * Author: ddzhao@openailab.com
 */
//
// im2col for kernel 1x1 s1p0d1
//
// input:
//         x0 arg0  input address 
//         x1 arg1  input_xy
//         x2 arg2  col address
//         x3 arg3  col_cnt must be multiply of 4
//         x4 arg4  input channel
//
// register definition
//    x0 input address 
//    x1 input_xy x 4
//    x2 col address
//    x3 col_cnt
//    x4 input channel
//    x6 input start pointer		t6
//    x7 input pointer
//    x9 channel cnt
//    x11
//    x12 = input_xy size * 2		// x12 -> t5

        .section .text,"ax"
        .align 5

        .type   im2col_fp32_1x1 STT_FUNC
        .global im2col_fp32_1x1
        .hidden im2col_fp32_1x1
im2col_fp32_1x1:
	addi    sp, sp, -64
	sd      t0, 0(sp)
	sd      t1, 8(sp)
	sd      t2, 16(sp)
	sd      t3, 24(sp)
	sd      t4, 32(sp)
	sd      t5, 40(sp)
	sd      t6, 48(sp)
    sd      ra, 56(sp)

    call    vsetvl_e32_m1
    ld      ra, 56(sp)

	li 		t0, 4
	blt 	a3, t0, col_end
	
	srli	a3, a3, 2
	
	slli	a1, a1, 2
	
	mv 		t6, a0
	
	slli	t5, a1, 1
	
	add 	t4, a4, 1								// x10 -> t4

	// col loop
col_loop:
	mv 		t3, t6
	srli	t2, a4, 1
	beqz	t2, channel_last
	add 	t1, t3, a1						
	// kernel size loop
channel_loop2:
	vle32.v 	v0,(t3)
	vle32.v 	v1,(t1)
	addi 	t2, t2, -1
	add 	t3, t3, t5
	add 	t1, t1, t5
	vse32.v 	v0, (a2)
	addi 	a2, a2, 16
	vse32.v 	v1, (a2)
	addi 	a2, a2, 16
	bnez	t2, channel_loop2

channel_last:
	beqz 	t4, channel_loop_end
	vle32.v 	v0,(t3)
	vse32.v 	v0, (a2)
	addi 	a2, a2, 16

channel_loop_end:
	addi 	t6, t6, 16
	addi 	a3, a3, -1
	bnez	a3, col_loop

col_end:
	ld      t0, 0(sp)
	ld      t1, 8(sp)
	ld      t2, 16(sp)
	ld      t3, 24(sp)
	ld      t4, 32(sp)
	ld      t5, 40(sp)
	ld      t6, 48(sp)
	addi    sp, sp, 64 
	ret
	.end
