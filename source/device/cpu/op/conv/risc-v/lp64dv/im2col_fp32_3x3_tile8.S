// input:
//         x0 arg0  input address 
//         x1 arg1  input_x
//         x2 arg2  input_y
//         x3 arg3  input channel cnt
//         x4 arg4  col address
//         x5 arg5  stride_x

.section .text, "ax"
.align 5

.type im2col_fp32_3x3_tile8 STT_FUNC
.global im2col_fp32_3x3_tile8
.hidden im2col_fp32_3x3_tile8

im2col_fp32_3x3_tile8:
    li  t0, 8
    vsetvli t1, t0, e32, m2

    slli    a1, a1, 2
    // a2 = out_xy
    mul     a2, a2, a1

    //t0 = input[1, :]
    //t1 = input[2, :]
    add     t0, a0, a1
    add     t1, t0, a1
    
    li      t2, 2 
    beq     a5, t2, stride2_channel_loop

stride1_channel_loop:
    vle32.v v0, (a0)
    vle32.v v2, (t0)
    vle32.v v4, (t1)

    addi    a3, a3, -1

    addi    t2, a0, 4
    vle32.v v6, (t2)
    addi    t2, a0, 8
    vle32.v v8, (t2)

    add     a0, a0, a2

    addi    t2, t0, 4
    vle32.v v10, (t2)
    addi    t2, t0, 8
    vle32.v v12, (t2)
    
    add     t0, t0, a2

    addi    t2, t1, 4
    vle32.v v14, (t2)
    addi    t2, t1, 8
    vle32.v v16, (t2)

    add     t1, t1, a2

    vse32.v v0, (a4)
    addi    a4, a4, 32
    vse32.v v6, (a4)
    addi    a4, a4, 32
    vse32.v v8, (a4)

    addi    a4, a4, 32
    vse32.v v2, (a4)
    addi    a4, a4, 32
    vse32.v v10, (a4)
    addi    a4, a4, 32
    vse32.v v12, (a4)

    addi    a4, a4, 32
    vse32.v v4, (a4)
    addi    a4, a4, 32
    vse32.v v14, (a4)
    addi    a4, a4, 32
    vse32.v v16, (a4)
    addi    a4, a4, 32

    bnez    a3, stride1_channel_loop
    j finish

stride2_channel_loop:
    li  t2, 8
    mv  t3, a0

    vlse32.v    v0, (t3), t2
    addi        t3, a0, 0x4
    vlse32.v    v2, (t3), t2
    addi        t3, a0, 0x8
    vlse32.v    v4, (t3), t2

    addi        a3, a3, -1

    mv  t3, t0
    vlse32.v    v6, (t3), t2 
    addi        t3, t3, 0x4
    vlse32.v    v8, (t3), t2
    addi        t3, t3, 0x4
    vlse32.v    v10, (t3), t2

    add         a0, a0, a2

    mv  t3, t1
    vlse32.v    v12, (t3), t2
    addi        t3, t3, 0x4
    vlse32.v    v14, (t3), t2
    addi        t3, t3, 0x4
    vlse32.v    v16, (t3), t2

    add         t0, t0, a2

    vse32.v     v0, (a4)
    addi        a4, a4, 32
    vse32.v     v2, (a4)
    addi        a4, a4, 32
    vse32.v     v4, (a4)
    addi        a4, a4, 32

    add         t1, t1, a2

    vse32.v     v6, (a4)
    addi        a4, a4, 32
    vse32.v     v8, (a4)
    addi        a4, a4, 32
    vse32.v     v10, (a4)
    addi        a4, a4, 32

    vse32.v     v12, (a4)
    addi        a4, a4, 32
    vse32.v     v14, (a4)
    addi        a4, a4, 32
    vse32.v     v16, (a4)
    addi        a4, a4, 32

    bnez        a3, stride2_channel_loop
    
finish:
    ret
    .end
