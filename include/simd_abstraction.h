#ifndef SIMD_ABSTRACTION_H
#define SIMD_ABSTRACTION_H

#include <stddef.h>

typedef enum {
	BACKEND_SCALAR,
	BACKEND_SSE,
	BACKEND_AVX2,
	BACKEND_AVX512,
	BACKEND_NEON
} simd_backend_t;

typedef struct {
	float data[8];
} simd_vec_t;

// detecting which SIMD the CPU supports
simd_backend_t simd_detect(void);

// printing the detected SIMD
void simd_print_backend(simd_backend_t backend);

// loading 8 floats from memory into SIMD register
simd_vec_t simd_load(const float* ptr);

// storing SIMD register back to memory
void simd_store(float* ptr, simd_vec_t vec);

typedef simd_vec_t (*simd_add_func)(simd_vec_t, simd_vec_t);
typedef simd_vec_t (*simd_mul_func)(simd_vec_t, simd_vec_t);
typedef simd_vec_t (*simd_fmadd_func)(simd_vec_t, simd_vec_t, simd_vec_t);

typedef struct {
	simd_backend_t backend;
	simd_add_func add;
	simd_mul_func mul;
	simd_fmadd_func fmadd;
} simd_dispatch_t;

simd_dispatch_t* simd_init_dispatch(void);

void simd_free_dispatch(simd_dispatch_t* dispatch);

#endif
