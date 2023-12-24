// input:
//         x0 arg0  input address 
//         x1 arg1  input_xy
//         x2 arg2  col address
//         x3 arg3  input channel
//         x4 arg4  tile_size 

.section .text, "ax"
.align 5

.type im2col_fp32_1x1_tile8 STT_FUNC
.global im2col_fp32_1x1_tile8
.hidden im2col_fp32_1x1_tile8

im2col_fp32_1x1_tile8:
    li  t0, 8
    vsetvli t1, t0, e32, m2

    slli    a1, a1, 2
    slli    t0, a1, 1
    
    srli    t1, a3, 1
    andi    t4, a3, 1

    mv      t2, a0
    add t3, t2, a1

chan_loop:
    vle32.v v0, (t2)
    vle32.v v2, (t3)

    vse32.v v0, (a2)
    addi    a2, a2, 32
    vse32.v v2, (a2)
    addi    a2, a2, 32

//TODO: move update ops up
    add     t2, t2, t0
    add     t3, t3, t0
    addi    t1, t1, -1

    bnez    t1, chan_loop

channel_last:
    beqz    t4, end 
    vle32.v v0, (t2)
    vse32.v v0, (a2)

end:
    ret
    .end
