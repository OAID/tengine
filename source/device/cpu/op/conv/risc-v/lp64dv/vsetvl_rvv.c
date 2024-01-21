#include "vsetvl_rvv.h"

void vsetvl_e32_m1(void)
{
#ifdef __FIX_RVV_C906
    __asm__("li     t0, 8;\n"
            "li     t1, 4;\n"
            "vsetvl t0, t1, t0;\n"
            :
            :
            : "t0", "t1");
#else
    __asm__("vsetvli %0, %1, e32, m1;\n"
            : "=r"(n)
            : "r"(packn));
#endif
}

void vsetvl_e32_m2(void)
{
#ifdef __FIX_RVV_C906
    __asm__("li t0, 9;\n"
            "li t1, 8;\n"
            "vsetvl t0, t1, t0;\n"
            :
            :
            : "t0", "t1");
#else
    __asm__("vsetvli %0, %1, e32, m2;\n"
            : "=r"(n)
            : "r"(packn));
#endif
}
