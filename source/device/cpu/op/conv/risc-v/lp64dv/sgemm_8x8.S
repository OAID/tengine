.section .text
.align 5
.type sgemm_8x8_rv64 STT_FUNC
.global sgemm_8x8_rv64

//a0 cur_col
//a1 cur_kernel
//a2 bias
//a3 act
//a4 cur_output
//a5 output_xy
//a6 kernel_size

sgemm_8x8_rv64:
    addi    sp, sp, -8
    sd      ra, (sp)
    call    vsetvl_e32_m2
    ld      ra, (sp)

    srli    t0, a6, 0x2
    andi    t1, a6, 0x7
    slli    a5, a5, 0x2

    beqz    a2, none_biases
    // bias init
    vle32.v v0, (a2)
    vrgather.vi v16, v0, 0
    vrgather.vi v18, v0, 1
    vrgather.vi v20, v0, 2
    vrgather.vi v22, v0, 3
    vrgather.vi v24, v0, 4
    vrgather.vi v26, v0, 5
    vrgather.vi v28, v0, 6
    vrgather.vi v30, v0, 7
    j loop4

none_biases:
    vmv.v.x v16, x0
    vmv.v.x v18, x0
    vmv.v.x v20, x0
    vmv.v.x v22, x0
    vmv.v.x v24, x0
    vmv.v.x v26, x0
    vmv.v.x v28, x0
    vmv.v.x v30, x0

loop4:
    vle32.v v0, (a0)
    addi    a0, a0, 32
    vle32.v v2, (a1)
    addi    a1, a1, 32
    vle32.v v4, (a0)
    addi    a0, a0, 32
    vle32.v v6, (a1)
    addi    a1, a1, 32

    vrgather.vi  v8, v2, 0
    vrgather.vi  v10, v2, 1
    vrgather.vi  v12, v2, 2
    vrgather.vi  v14,v2, 3

    vfmacc.vv   v16, v0, v8
    vfmacc.vv   v18, v0, v10
    vfmacc.vv   v20, v0, v12
    vfmacc.vv   v22, v0, v14

    vrgather.vi  v8, v2, 4
    vrgather.vi  v10, v2, 5
    vrgather.vi  v12, v2, 6
    vrgather.vi  v14,v2, 7

    vfmacc.vv   v24, v0, v8
    vfmacc.vv   v26, v0, v10
    vfmacc.vv   v28, v0, v12
    vfmacc.vv   v30, v0, v14

    vle32.v v0, (a0)
    addi    a0, a0, 32

    vrgather.vi  v8,  v6, 0
    vrgather.vi  v10, v6, 1
    vrgather.vi  v12, v6, 2
    vrgather.vi  v14, v6, 3

    vfmacc.vv   v16, v4, v8
    vfmacc.vv   v18, v4, v10
    vfmacc.vv   v20, v4, v12
    vfmacc.vv   v22, v4, v14

    vle32.v v2, (a1)
    addi    a1, a1, 32

    vrgather.vi  v8,  v6, 4
    vrgather.vi  v10, v6, 5
    vrgather.vi  v12, v6, 6
    vrgather.vi  v14, v6, 7

    vfmacc.vv   v24, v4, v8
    vfmacc.vv   v26, v4, v10
    vfmacc.vv   v28, v4, v12
    vfmacc.vv   v30, v4, v14

    vle32.v v4, (a0)
    addi    a0, a0, 32

    vrgather.vi  v8, v2, 0
    vrgather.vi  v10, v2, 1
    vrgather.vi  v12, v2, 2
    vrgather.vi  v14,v2, 3

    vfmacc.vv   v16, v0, v8
    vfmacc.vv   v18, v0, v10
    vfmacc.vv   v20, v0, v12
    vfmacc.vv   v22, v0, v14

    vle32.v v6, (a1)
    addi    a1, a1, 32

    vrgather.vi  v8, v2, 4
    vrgather.vi  v10, v2, 5
    vrgather.vi  v12, v2, 6
    vrgather.vi  v14,v2, 7

    vfmacc.vv   v24, v0, v8
    vfmacc.vv   v26, v0, v10
    vfmacc.vv   v28, v0, v12
    vfmacc.vv   v30, v0, v14

    addi        t0, t0, -1

    vrgather.vi  v8,  v6, 0
    vrgather.vi  v10, v6, 1
    vrgather.vi  v12, v6, 2
    vrgather.vi  v14, v6, 3

    vfmacc.vv   v16, v4, v8
    vfmacc.vv   v18, v4, v10
    vfmacc.vv   v20, v4, v12
    vfmacc.vv   v22, v4, v14

    vrgather.vi  v8,  v6, 4
    vrgather.vi  v10, v6, 5
    vrgather.vi  v12, v6, 6
    vrgather.vi  v14, v6, 7

    vfmacc.vv   v24, v4, v8
    vfmacc.vv   v26, v4, v10
    vfmacc.vv   v28, v4, v12
    vfmacc.vv   v30, v4, v14

    bnez    t0, loop4

loop1:
    beqz    t1, activation 
    vle32.v v0, (a0)
    addi    a0, a0, 32
    vle32.v v2, (a1)
    addi    a1, a1, 32

    vrgather.vi  v8, v2, 0
    vrgather.vi  v10, v2, 1
    vrgather.vi  v12, v2, 2
    vrgather.vi  v14,v2, 3

    vfmacc.vv   v16, v0, v8
    vfmacc.vv   v18, v0, v10
    vfmacc.vv   v20, v0, v12
    vfmacc.vv   v22, v0, v14

    vrgather.vi  v8, v2, 4
    vrgather.vi  v10, v2, 5
    vrgather.vi  v12, v2, 6
    vrgather.vi  v14,v2, 7

    vfmacc.vv   v24, v0, v8
    vfmacc.vv   v26, v0, v10
    vfmacc.vv   v28, v0, v12
    vfmacc.vv   v30, v0, v14

    addi        t1, t1, -1
    bnez        t1, loop1

activation:
    bltz    a3, save_result
    vmv.v.x v0, x0
    vmv.v.x v2, a3

    vfmax.vv    v16, v16, v0
    vfmax.vv    v18, v18, v0
    vfmax.vv    v20, v20, v0
    vfmax.vv    v22, v22, v0
    vfmax.vv    v24, v24, v0
    vfmax.vv    v26, v26, v0
    vfmax.vv    v28, v28, v0
    vfmax.vv    v30, v30, v0

    beqz        a3, save_result
    vfmin.vv    v16, v16, v2
    vfmin.vv    v18, v18, v2
    vfmin.vv    v20, v20, v2
    vfmin.vv    v22, v22, v2
    vfmin.vv    v24, v24, v2
    vfmin.vv    v26, v26, v2
    vfmin.vv    v28, v28, v2
    vfmin.vv    v30, v30, v2

save_result:
    vse32.v     v16, (a4)
    add         a4, a4, a5
    vse32.v     v18, (a4)
    add         a4, a4, a5
    vse32.v     v20, (a4)
    add         a4, a4, a5
    vse32.v     v22, (a4)
    add         a4, a4, a5
    vse32.v     v24, (a4)
    add         a4, a4, a5
    vse32.v     v26, (a4)
    add         a4, a4, a5
    vse32.v     v28, (a4)
    add         a4, a4, a5
    vse32.v     v30, (a4)
finish:
    addi        sp, sp, 8
    ret
    .end
