#include <stdio.h>
#include <stdlib.h>
#include "array.h"
#include "simd_abstraction.h"

void test_basic_creation() {
    printf("Basic Array Creation \n");
    
    size_t shape1d[1] = {5};
    array_t* arr1d = array_create(shape1d, 1);
    
    for (size_t i = 0; i < 5; i++) {
        size_t idx[1] = {i};
        array_set(arr1d, idx, (float)(i + 1));
    }
    
    printf("1D Array: ");
    array_print(arr1d);
    
    size_t shape2d[2] = {3, 4};
    array_t* arr2d = array_create(shape2d, 2);
    
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 4; j++) {
            size_t idx[2] = {i, j};
            array_set(arr2d, idx, (float)(i * 4 + j + 1));
        }
    }
    
    printf("2D Array:\n");
    array_print(arr2d);
    
    array_free(arr1d);
    array_free(arr2d);
    printf("\n");
}

void test_slicing() {
    printf("Zero-Copy Slicing \n");
    
    size_t shape[2] = {4, 5};
    array_t* arr = array_create(shape, 2);
    
    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 5; j++) {
            size_t idx[2] = {i, j};
            array_set(arr, idx, (float)(i * 5 + j));
        }
    }
    
    printf("Original 4x5 array:\n");
    array_print(arr);
    
    size_t start[2] = {1, 1};
    size_t end[2] = {3, 4};
    array_t* view = array_view(arr, start, end);
    
    printf("View [1:3, 1:4]:\n");
    array_print(view);
    
    printf("View owns data: %s\n", view->owns_data ? "yes" : "no");
    
    array_free(view);
    array_free(arr);
    printf("\n");
}

void test_broadcasting() {
    printf("Broadcasting \n");
    
    size_t shape_a[2] = {3, 4};
    size_t shape_b[1] = {4};
    
    array_t* a = array_create(shape_a, 2);
    array_t* b = array_create(shape_b, 1);
    
    array_fill(a, 1.0f);
    for (size_t i = 0; i < 4; i++) {
        size_t idx[1] = {i};
        array_set(b, idx, (float)(i + 1));
    }
    
    printf("Array A (3x4):\n");
    array_print(a);
    printf("Array B (4,):\n");
    array_print(b);
    
    printf("Broadcastable: %s\n", array_broadcastable(a, b) ? "YES" : "NO");
    
    size_t out_ndim;
    size_t* out_shape = array_broadcast_shape(a, b, &out_ndim);
    printf("Broadcast shape: (");
    for (size_t i = 0; i < out_ndim; i++) {
        printf("%zu", out_shape[i]);
        if (i < out_ndim - 1) printf(", ");
    }
    printf(")\n");
    
    free(out_shape);
    array_free(a);
    array_free(b);
    printf("\n");
}

void test_eager_operations() {
    printf("Eager SIMD Operations\n");
    
    simd_dispatch_t* dispatch = simd_init_dispatch();
    
    size_t shape[1] = {8};
    array_t* a = array_create(shape, 1);
    array_t* b = array_create(shape, 1);
    array_t* result = array_create(shape, 1);
    
    for (size_t i = 0; i < 8; i++) {
        size_t idx[1] = {i};
        array_set(a, idx, (float)(i + 1));
        array_set(b, idx, (float)(8 - i));
    }
    
    printf("A: ");
    array_print(a);
    printf("B: ");
    array_print(b);
    
    array_add_eager(result, a, b, dispatch);
    printf("A + B: ");
    array_print(result);
    
    array_mul_eager(result, a, b, dispatch);
    printf("A * B: ");
    array_print(result);
    
    simd_free_dispatch(dispatch);
    array_free(a);
    array_free(b);
    array_free(result);
    printf("\n");
}

void test_lazy_evaluation() {
    printf("Lazy Evaluation (Expression Templates) \n");
    
    simd_dispatch_t* dispatch = simd_init_dispatch();
    
    size_t shape[1] = {5};
    array_t* a = array_create(shape, 1);
    array_t* b = array_create(shape, 1);
    array_t* c = array_create(shape, 1);
    
    for (size_t i = 0; i < 5; i++) {
        size_t idx[1] = {i};
        array_set(a, idx, (float)(i + 1));
        array_set(b, idx, 2.0f);
        array_set(c, idx, 3.0f);
    }
    
    printf("A: ");
    array_print(a);
    printf("B: ");
    array_print(b);
    printf("C: ");
    array_print(c);
    
    expr_t* expr_a = expr_from_array(a);
    expr_t* expr_b = expr_from_array(b);
    expr_t* expr_c = expr_from_array(c);
    
    expr_t* sum = expr_add(expr_a, expr_b);  // A + B
    expr_t* product = expr_mul(sum, expr_c); // (A + B) * C
    
    printf("Expression: (A + B) * C\n");
    
    array_t* result = array_create(shape, 1);
    expr_eval(product, result, dispatch);
    
    printf("Result: ");
    array_print(result);
    
    expr_free(product);
    array_free(a);
    array_free(b);
    array_free(c);
    array_free(result);
    simd_free_dispatch(dispatch);
    printf("\n");
}

int main() {
    test_basic_creation();
    test_slicing();
    test_broadcasting();
    test_eager_operations();
    test_lazy_evaluation();
    
    return 0;
}
