#include "array.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

static void compute_strides(size_t* strides, size_t* shape, size_t ndim) {
    size_t stride = 1;
    for (int i = (int)ndim - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= shape[i];
    }
}

array_t* array_create(size_t* shape, size_t ndim) {
    array_t* arr = malloc(sizeof(array_t));
    
    arr->ndim = ndim;
    arr->shape = malloc(ndim * sizeof(size_t));
    arr->strides = malloc(ndim * sizeof(size_t));
    memcpy(arr->shape, shape, ndim * sizeof(size_t));
    
    compute_strides(arr->strides, shape, ndim);
    
    arr->size = 1;
    for (size_t i = 0; i < ndim; i++) {
        arr->size *= shape[i];
    }
    
    arr->data = calloc(arr->size, sizeof(float));
    arr->owns_data = true;
    arr->base = NULL;
    
    return arr;
}

array_t* array_from_data(float* data, size_t* shape, size_t ndim) {
    array_t* arr = malloc(sizeof(array_t));
    
    arr->ndim = ndim;
    arr->shape = malloc(ndim * sizeof(size_t));
    arr->strides = malloc(ndim * sizeof(size_t));
    memcpy(arr->shape, shape, ndim * sizeof(size_t));
    
    compute_strides(arr->strides, shape, ndim);
    
    arr->size = 1;
    for (size_t i = 0; i < ndim; i++) {
        arr->size *= shape[i];
    }
    
    arr->data = data;  
    arr->owns_data = false;
    arr->base = NULL;
    
    return arr;
}

array_t* array_view(array_t* arr, size_t* start, size_t* end) {
    array_t* view = malloc(sizeof(array_t));
    
    view->ndim = arr->ndim;
    view->shape = malloc(arr->ndim * sizeof(size_t));
    view->strides = malloc(arr->ndim * sizeof(size_t));
    
    size_t offset = 0;
    for (size_t i = 0; i < arr->ndim; i++) {
        view->shape[i] = end[i] - start[i];
        view->strides[i] = arr->strides[i];
        offset += start[i] * arr->strides[i];
    }
    
    view->size = 1;
    for (size_t i = 0; i < view->ndim; i++) {
        view->size *= view->shape[i];
    }
    
    view->data = arr->data + offset;  
    view->owns_data = false;
    view->base = arr;
    
    return view;
}

void array_free(array_t* arr) {
    if (arr->owns_data && arr->data) {
        free(arr->data);
    }
    free(arr->shape);
    free(arr->strides);
    free(arr);
}

size_t array_offset(array_t* arr, size_t* indices) {
    size_t offset = 0;
    for (size_t i = 0; i < arr->ndim; i++) {
        offset += indices[i] * arr->strides[i];
    }
    return offset;
}

float array_get(array_t* arr, size_t* indices) {
    return arr->data[array_offset(arr, indices)];
}

void array_set(array_t* arr, size_t* indices, float value) {
    arr->data[array_offset(arr, indices)] = value;
}

bool array_broadcastable(array_t* a, array_t* b) {
    size_t max_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
    
    for (int i = 0; i < (int)max_ndim; i++) {
        int a_idx = (int)a->ndim - 1 - i;
        int b_idx = (int)b->ndim - 1 - i;
        
        size_t a_dim = (a_idx >= 0) ? a->shape[a_idx] : 1;
        size_t b_dim = (b_idx >= 0) ? b->shape[b_idx] : 1;
        
        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
            return false;
        }
    }
    return true;
}

size_t* array_broadcast_shape(array_t* a, array_t* b, size_t* out_ndim) {
    *out_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
    size_t* shape = malloc(*out_ndim * sizeof(size_t));
    
    for (int i = 0; i < (int)*out_ndim; i++) {
        int a_idx = (int)a->ndim - 1 - i;
        int b_idx = (int)b->ndim - 1 - i;
        
        size_t a_dim = (a_idx >= 0) ? a->shape[a_idx] : 1;
        size_t b_dim = (b_idx >= 0) ? b->shape[b_idx] : 1;
        
        shape[*out_ndim - 1 - i] = (a_dim > b_dim) ? a_dim : b_dim;
    }
    
    return shape;
}

void array_broadcast_prepare(array_t* arr, size_t* target_shape, size_t target_ndim) {
    size_t* new_strides = malloc(target_ndim * sizeof(size_t));
    
    for (int i = 0; i < (int)target_ndim; i++) {
        int arr_idx = (int)arr->ndim - (int)target_ndim + i;
        
        if (arr_idx < 0) {
            new_strides[i] = 0;
        } else if (arr->shape[arr_idx] == 1 && target_shape[i] > 1) {
            new_strides[i] = 0;
        } else {
            new_strides[i] = arr->strides[arr_idx];
        }
    }
    
    free(arr->strides);
    arr->strides = new_strides;
}

void array_add_eager(array_t* result, array_t* a, array_t* b, simd_dispatch_t* dispatch) {
    assert(array_broadcastable(a, b));
    
    if (a->ndim == 1 && b->ndim == 1 && a->size == b->size && result->size == a->size) {
        size_t i = 0;
        for (; i + 8 <= a->size; i += 8) {
            simd_vec_t va = simd_load(&a->data[i]);
            simd_vec_t vb = simd_load(&b->data[i]);
            simd_vec_t vr = dispatch->add(va, vb);
            simd_store(&result->data[i], vr);
        }
        
        for (; i < a->size; i++) {
            result->data[i] = a->data[i] + b->data[i];
        }
    } else {
        for (size_t i = 0; i < result->size; i++) {
            result->data[i] = a->data[i] + b->data[i];
        }
    }
}

void array_mul_eager(array_t* result, array_t* a, array_t* b, simd_dispatch_t* dispatch) {
    assert(array_broadcastable(a, b));
    
    if (a->ndim == 1 && b->ndim == 1 && a->size == b->size && result->size == a->size) {
        size_t i = 0;
        for (; i + 8 <= a->size; i += 8) {
            simd_vec_t va = simd_load(&a->data[i]);
            simd_vec_t vb = simd_load(&b->data[i]);
            simd_vec_t vr = dispatch->mul(va, vb);
            simd_store(&result->data[i], vr);
        }
        
        for (; i < a->size; i++) {
            result->data[i] = a->data[i] * b->data[i];
        }
    } else {
        for (size_t i = 0; i < result->size; i++) {
            result->data[i] = a->data[i] * b->data[i];
        }
    }
}

expr_t* expr_from_array(array_t* arr) {
    expr_t* expr = malloc(sizeof(expr_t));
    expr->type = EXPR_ARRAY;
    expr->data.leaf.array = arr;
    
    expr->ndim = arr->ndim;
    expr->shape = malloc(arr->ndim * sizeof(size_t));
    memcpy(expr->shape, arr->shape, arr->ndim * sizeof(size_t));
    
    return expr;
}

expr_t* expr_add(expr_t* left, expr_t* right) {
    expr_t* expr = malloc(sizeof(expr_t));
    expr->type = EXPR_ADD;
    expr->data.binary.left = left;
    expr->data.binary.right = right;
    
    expr->ndim = left->ndim;
    expr->shape = malloc(left->ndim * sizeof(size_t));
    memcpy(expr->shape, left->shape, left->ndim * sizeof(size_t));
    
    return expr;
}

expr_t* expr_mul(expr_t* left, expr_t* right) {
    expr_t* expr = malloc(sizeof(expr_t));
    expr->type = EXPR_MUL;
    expr->data.binary.left = left;
    expr->data.binary.right = right;
    
    expr->ndim = left->ndim;
    expr->shape = malloc(left->ndim * sizeof(size_t));
    memcpy(expr->shape, left->shape, left->ndim * sizeof(size_t));
    
    return expr;
}

expr_t* expr_scalar_mul(float scalar, expr_t* operand) {
    expr_t* expr = malloc(sizeof(expr_t));
    expr->type = EXPR_SCALAR_MUL;
    expr->data.scalar_op.scalar = scalar;
    expr->data.scalar_op.operand = operand;
    
    expr->ndim = operand->ndim;
    expr->shape = malloc(operand->ndim * sizeof(size_t));
    memcpy(expr->shape, operand->shape, operand->ndim * sizeof(size_t));
    
    return expr;
}

static float expr_eval_at(expr_t* expr, size_t* indices) {
    switch (expr->type) {
        case EXPR_ARRAY:
            return array_get(expr->data.leaf.array, indices);
        
        case EXPR_ADD: {
            float left = expr_eval_at(expr->data.binary.left, indices);
            float right = expr_eval_at(expr->data.binary.right, indices);
            return left + right;
        }
        
        case EXPR_MUL: {
            float left = expr_eval_at(expr->data.binary.left, indices);
            float right = expr_eval_at(expr->data.binary.right, indices);
            return left * right;
        }
        
        case EXPR_SCALAR_MUL: {
            float val = expr_eval_at(expr->data.scalar_op.operand, indices);
            return expr->data.scalar_op.scalar * val;
        }
    }
    return 0.0f;
}

void expr_eval(expr_t* expr, array_t* result, simd_dispatch_t* dispatch) {
    size_t* indices = calloc(result->ndim, sizeof(size_t));
    
    for (size_t flat = 0; flat < result->size; flat++) {
        size_t temp = flat;
        for (int i = (int)result->ndim - 1; i >= 0; i--) {
            indices[i] = temp % result->shape[i];
            temp /= result->shape[i];
        }
        
        result->data[flat] = expr_eval_at(expr, indices);
    }
    
    free(indices);
}

void expr_free(expr_t* expr) {
    if (!expr) return;
    
    switch (expr->type) {
        case EXPR_ADD:
        case EXPR_MUL:
            expr_free(expr->data.binary.left);
            expr_free(expr->data.binary.right);
            break;
        
        case EXPR_SCALAR_MUL:
            expr_free(expr->data.scalar_op.operand);
            break;
        
        case EXPR_ARRAY:
            break;
    }
    
    free(expr->shape);
    free(expr);
}

void array_fill(array_t* arr, float value) {
    for (size_t i = 0; i < arr->size; i++) {
        arr->data[i] = value;
    }
}

array_t* array_copy(array_t* src) {
    array_t* dst = array_create(src->shape, src->ndim);
    memcpy(dst->data, src->data, src->size * sizeof(float));
    return dst;
}

void array_print(array_t* arr) {
    if (arr->ndim == 1) {
        printf("[");
        for (size_t i = 0; i < arr->shape[0]; i++) {
            size_t idx[1] = {i};
            printf("%.2f", array_get(arr, idx));
            if (i < arr->shape[0] - 1) printf(", ");
        }
        printf("]\n");
    } else if (arr->ndim == 2) {
        printf("[\n");
        for (size_t i = 0; i < arr->shape[0]; i++) {
            printf("  [");
            for (size_t j = 0; j < arr->shape[1]; j++) {
                size_t idx[2] = {i, j};
                printf("%.2f", array_get(arr, idx));
                if (j < arr->shape[1] - 1) printf(", ");
            }
            printf("]");
            if (i < arr->shape[0] - 1) printf(",");
            printf("\n");
        }
        printf("]\n");
    } else {
        printf("Array with %zu dimensions, size=%zu\n", arr->ndim, arr->size);
    }
}
