#ifndef ARRAY_H
#define ARRAY_H

#include <stddef.h>
#include <stdbool.h>
#include "simd_abstraction.h"

typedef struct {
    float* data;           
    size_t* shape;         
    size_t* strides;      
    size_t ndim;           
    size_t size;           
    bool owns_data;       
    void* base;           
} array_t;

array_t* array_create(size_t* shape, size_t ndim);

array_t* array_from_data(float* data, size_t* shape, size_t ndim);

array_t* array_view(array_t* arr, size_t* start, size_t* end);

void array_free(array_t* arr);

float array_get(array_t* arr, size_t* indices);

void array_set(array_t* arr, size_t* indices, float value);

size_t array_offset(array_t* arr, size_t* indices);

bool array_broadcastable(array_t* a, array_t* b);

size_t* array_broadcast_shape(array_t* a, array_t* b, size_t* out_ndim);

void array_broadcast_prepare(array_t* arr, size_t* target_shape, size_t target_ndim);

void array_add_eager(array_t* result, array_t* a, array_t* b, simd_dispatch_t* dispatch);
void array_mul_eager(array_t* result, array_t* a, array_t* b, simd_dispatch_t* dispatch);

typedef enum {
    EXPR_ARRAY,      
    EXPR_ADD,      
    EXPR_MUL,     
    EXPR_SCALAR_MUL 
} expr_type_t;

typedef struct expr_t expr_t;

struct expr_t {
    expr_type_t type;
    
    union {
        struct {
            array_t* array;
        } leaf;
        
        struct {
            expr_t* left;
            expr_t* right;
        } binary;
        
        struct {
            float scalar;
            expr_t* operand;
        } scalar_op;
    } data;
    
    size_t* shape;      
    size_t ndim;       
};

expr_t* expr_from_array(array_t* arr);

expr_t* expr_add(expr_t* left, expr_t* right);
expr_t* expr_mul(expr_t* left, expr_t* right);

expr_t* expr_scalar_mul(float scalar, expr_t* operand);

void expr_eval(expr_t* expr, array_t* result, simd_dispatch_t* dispatch);

void expr_free(expr_t* expr);

void array_print(array_t* arr);
void array_fill(array_t* arr, float value);
array_t* array_copy(array_t* src);

#endif 
