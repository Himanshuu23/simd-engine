#include <stdio.h>
#include <stdlib.h>
#include "simd_abstraction.h"

int main() {
    simd_dispatch_t* dispatch = simd_init_dispatch();
    simd_print_backend(dispatch->backend);
    printf("\n");
    
    float a[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    float b[8] = {8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    float c[8] = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    float result[8];
    
    simd_vec_t va = simd_load(a);
    simd_vec_t vb = simd_load(b);
    simd_vec_t vc = simd_load(c);
    simd_vec_t vr;
    
    printf("Addition\n");
    printf("A = ");
    for(int i = 0; i < 8; i++) printf("%.1f ", a[i]);
    printf("\nB = ");
    for(int i = 0; i < 8; i++) printf("%.1f ", b[i]);
    
    vr = dispatch->add(va, vb);
    simd_store(result, vr);
    
    printf("\nA + B = ");
    for(int i = 0; i < 8; i++) printf("%.1f ", result[i]);
    printf("\n\n");
    
    printf("Multiplication\n");
    vr = dispatch->mul(va, vb);
    simd_store(result, vr);
    
    printf("A * B = ");
    for(int i = 0; i < 8; i++) printf("%.1f ", result[i]);
    printf("\n\n");
    
    printf("Fused Multiply-Add\n");
    printf("C = ");
    for(int i = 0; i < 8; i++) printf("%.1f ", c[i]);
    
    vr = dispatch->fmadd(va, vb, vc);
    simd_store(result, vr);
    
    printf("\n(A * B) + C = ");
    for(int i = 0; i < 8; i++) printf("%.1f ", result[i]);
    printf("\n\n");
    
    simd_free_dispatch(dispatch);
    return 0;
}
