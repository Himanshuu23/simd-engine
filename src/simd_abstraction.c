#include "simd_abstraction.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __SSE2__
#include <emmintrin.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

simd_backend_t simd_detect(void) {
    #if defined(__AVX512F__)
        return BACKEND_AVX512;
    #elif defined(__AVX2__)
        return BACKEND_AVX2;
    #elif defined(__SSE2__)
        return BACKEND_SSE;
    #elif defined(__ARM_NEON)
        return BACKEND_NEON;
    #else
        return BACKEND_SCALAR;
    #endif
}

void simd_print_backend(simd_backend_t backend) {
    const char* names[] = {
        "Scalar (no SIMD)",
        "SSE (4 floats)",
        "AVX2 (8 floats)",
        "AVX-512 (16 floats)",
        "NEON (4 floats)"
    };
    printf("SIMD Backend: %s\n", names[backend]);
}

simd_vec_t simd_load(const float* ptr) {
    simd_vec_t vec;
    
    #ifdef __AVX2__
        __m256 native = _mm256_loadu_ps(ptr);
        _mm256_storeu_ps(vec.data, native);
    #elif defined(__SSE2__)
        __m128 low = _mm_loadu_ps(ptr);
        __m128 high = _mm_loadu_ps(ptr + 4);
        _mm_storeu_ps(vec.data, low);
        _mm_storeu_ps(vec.data + 4, high);
    #elif defined(__ARM_NEON)
        float32x4_t low = vld1q_f32(ptr);
        float32x4_t high = vld1q_f32(ptr + 4);
        vst1q_f32(vec.data, low);
        vst1q_f32(vec.data + 4, high);
    #else
        for(int i = 0; i < 8; i++) {
            vec.data[i] = ptr[i];
        }
    #endif
    
    return vec;
}

void simd_store(float* ptr, simd_vec_t vec) {
    #ifdef __AVX2__
        __m256 native = _mm256_loadu_ps(vec.data);
        _mm256_storeu_ps(ptr, native);
    #elif defined(__SSE2__)
        __m128 low = _mm_loadu_ps(vec.data);
        __m128 high = _mm_loadu_ps(vec.data + 4);
        _mm_storeu_ps(ptr, low);
        _mm_storeu_ps(ptr + 4, high);
    #elif defined(__ARM_NEON)
        float32x4_t low = vld1q_f32(vec.data);
        float32x4_t high = vld1q_f32(vec.data + 4);
        vst1q_f32(ptr, low);
        vst1q_f32(ptr + 4, high);
    #else
        for(int i = 0; i < 8; i++) {
            ptr[i] = vec.data[i];
        }
    #endif
}
static simd_vec_t simd_add_scalar(simd_vec_t a, simd_vec_t b) {
    simd_vec_t result;
    for(int i = 0; i < 8; i++) 
        result.data[i] = a.data[i] + b.data[i];
    
    return result;
}

static simd_vec_t simd_mul_scalar(simd_vec_t a, simd_vec_t b) {
    simd_vec_t result;
    for(int i = 0; i < 8; i++) 
        result.data[i] = a.data[i] * b.data[i];
    
    return result;
}

static simd_vec_t simd_fmadd_scalar(simd_vec_t a, simd_vec_t b, simd_vec_t c) {
    simd_vec_t result;
    for(int i = 0; i < 8; i++) 
        result.data[i] = a.data[i] * b.data[i] + c.data[i];
    
    return result;
}

#ifdef __SSE2__
static simd_vec_t simd_add_sse(simd_vec_t a, simd_vec_t b) {
    simd_vec_t result;
    
    __m128 va_low = _mm_loadu_ps(a.data);
    __m128 vb_low = _mm_loadu_ps(b.data);
    __m128 vr_low = _mm_add_ps(va_low, vb_low);
    _mm_storeu_ps(result.data, vr_low);
    
    __m128 va_high = _mm_loadu_ps(a.data + 4);
    __m128 vb_high = _mm_loadu_ps(b.data + 4);
    __m128 vr_high = _mm_add_ps(va_high, vb_high);
    _mm_storeu_ps(result.data + 4, vr_high);
    
    return result;
}

static simd_vec_t simd_mul_sse(simd_vec_t a, simd_vec_t b) {
    simd_vec_t result;
    
    __m128 va_low = _mm_loadu_ps(a.data);
    __m128 vb_low = _mm_loadu_ps(b.data);
    __m128 vr_low = _mm_mul_ps(va_low, vb_low);
    _mm_storeu_ps(result.data, vr_low);
    
    __m128 va_high = _mm_loadu_ps(a.data + 4);
    __m128 vb_high = _mm_loadu_ps(b.data + 4);
    __m128 vr_high = _mm_mul_ps(va_high, vb_high);
    _mm_storeu_ps(result.data + 4, vr_high);
    
    return result;
}

static simd_vec_t simd_fmadd_sse(simd_vec_t a, simd_vec_t b, simd_vec_t c) {
    simd_vec_t temp = simd_mul_sse(a, b);
    return simd_add_sse(temp, c);
}
#endif

#ifdef __AVX2__
static simd_vec_t simd_add_avx2(simd_vec_t a, simd_vec_t b) {
    simd_vec_t result;
    __m256 va = _mm256_loadu_ps(a.data);
    __m256 vb = _mm256_loadu_ps(b.data);
    __m256 vr = _mm256_add_ps(va, vb);
    _mm256_storeu_ps(result.data, vr);
    return result;
}

static simd_vec_t simd_mul_avx2(simd_vec_t a, simd_vec_t b) {
    simd_vec_t result;
    __m256 va = _mm256_loadu_ps(a.data);
    __m256 vb = _mm256_loadu_ps(b.data);
    __m256 vr = _mm256_mul_ps(va, vb);
    _mm256_storeu_ps(result.data, vr);
    return result;
}

static simd_vec_t simd_fmadd_avx2(simd_vec_t a, simd_vec_t b, simd_vec_t c) {
    simd_vec_t result;
    __m256 va = _mm256_loadu_ps(a.data);
    __m256 vb = _mm256_loadu_ps(b.data);
    __m256 vc = _mm256_loadu_ps(c.data);
    __m256 vr = _mm256_fmadd_ps(va, vb, vc);
    _mm256_storeu_ps(result.data, vr);
    return result;
}
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <cpuid.h>

static int cpu_has_sse2(void) {
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        return (edx & bit_SSE2) != 0;
    }
    return 0;
}

static int cpu_has_avx2(void) {
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_max(0, NULL) >= 7) {
        __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);
        return (ebx & bit_AVX2) != 0;
    }
    return 0;
}
#else
static int cpu_has_sse2(void) { return 0; }
static int cpu_has_avx2(void) { return 0; }
#endif

simd_dispatch_t* simd_init_dispatch(void) {
    simd_dispatch_t* dispatch = malloc(sizeof(simd_dispatch_t));
    
    if (cpu_has_avx2()) {
        dispatch->backend = BACKEND_AVX2;
        #ifdef __AVX2__
        dispatch->add = simd_add_avx2;
        dispatch->mul = simd_mul_avx2;
        dispatch->fmadd = simd_fmadd_avx2;
        printf("Using AVX2 implementation\n");
        #else
        dispatch->add = simd_add_scalar;
        dispatch->mul = simd_mul_scalar;
        dispatch->fmadd = simd_fmadd_scalar;
        printf("AVX2 detected but not compiled in, using scalar\n");
        #endif
    } else if (cpu_has_sse2()) {
        dispatch->backend = BACKEND_SSE;
        #ifdef __SSE2__
        dispatch->add = simd_add_sse;
        dispatch->mul = simd_mul_sse;
        dispatch->fmadd = simd_fmadd_sse;
        printf("Using SSE2 implementation\n");
        #else
        dispatch->add = simd_add_scalar;
        dispatch->mul = simd_mul_scalar;
        dispatch->fmadd = simd_fmadd_scalar;
        printf("SSE2 detected but not compiled in, using scalar\n");
        #endif
    } else {
        dispatch->backend = BACKEND_SCALAR;
        dispatch->add = simd_add_scalar;
        dispatch->mul = simd_mul_scalar;
        dispatch->fmadd = simd_fmadd_scalar;
        printf("Using scalar fallback\n");
    }
    
    return dispatch;
}

void simd_free_dispatch(simd_dispatch_t* dispatch) {
    free(dispatch);
}
