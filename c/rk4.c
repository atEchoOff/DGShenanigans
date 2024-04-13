#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "flux.c"


// Helper function to multiply matrix and vector
// inputs are A matrix, x vector to be multipled with
// y store the resulting vector Ax
// A = R^{m x n}

void matrix_k_vector_multiply(double** u, double*** k_vector, double coef[4]) {
    for (int k = 0; k < 4; k++) {
        for (int i = 0; i < NUM_SPATIAL; i++) {
            for (int j = 0; j < NUM_ELEMENTS; j++) {
                u[i][j] = u[i][j] + h/6.0 * coef[k] * (k_vector[k][i][j]); 
            }
        }
    }
}

void k_vector_vector_addition(double** u, double** u_eval, double** u_next, double** MASS_MATRIX, double* weight, \
                double** uM_eval, double** uP_eval, double** uM, double** uP, \
                double** uavg_uM, double** L, double** rhs, double** normal_direc, \
                double*** k_vector, double factor, int k, double delta) {

    for (int i = 0; i < NUM_SPATIAL; i++) {
        for (int j = 0; j < NUM_ELEMENTS; j++) {
            u_next[i][j] = u[i][j] + factor * k_vector[k - 1][i][j];
        }
    }
    RHS(u_next, u_eval, MASS_MATRIX, weight, \
        uM_eval, uP_eval, uM, uP, \
        uavg_uM, L, rhs, normal_direc, delta);
    for (int i = 0; i < NUM_SPATIAL; i++) {
        memcpy(k_vector[k][i], rhs[i], 8*NUM_ELEMENTS);
    }
}

void rk4_rhs(double** u, double** u_eval, double** MASS_MATRIX, double* weight, \
                double** uM_eval, double** uP_eval, double** uM, double** uP, \
                double** uavg_uM, double** L, double** rhs, double** normal_direc, \
                double*** k_vector, double coef[4], double** u_next, double delta) {
    
    RHS(u, u_eval, MASS_MATRIX, weight, \
        uM_eval, uP_eval, uM, uP, \
        uavg_uM, L, rhs, normal_direc, delta);

    for (int i = 0; i < NUM_SPATIAL; i++) {
        memcpy(k_vector[0][i], rhs[i], 8*NUM_ELEMENTS);
    }
    k_vector_vector_addition(u, u_eval, u_next, MASS_MATRIX, weight, uM_eval, uP_eval, \
        uM, uP, uavg_uM, L, rhs, normal_direc, k_vector, h/2, 1, delta);
    k_vector_vector_addition(u, u_eval, u_next, MASS_MATRIX, weight, uM_eval, uP_eval, \
        uM, uP, uavg_uM, L, rhs, normal_direc, k_vector, h/2, 2, delta);
    k_vector_vector_addition(u, u_eval, u_next, MASS_MATRIX, weight, uM_eval, uP_eval, \
        uM, uP, uavg_uM, L, rhs, normal_direc, k_vector, h, 3, delta);

    matrix_k_vector_multiply(u, k_vector, coef);
 
};

void euler_multiply(double** u, double** rhs) {
    for (int i = 0; i < NUM_SPATIAL; i++) {
        for (int j = 0; j < NUM_ELEMENTS; j++) {
            //printf("%f\t", u[i][j]);
            u[i][j] = u[i][j] + h * rhs[i][j];
            
        }
        //printf("\n");
    }
    //printf("\n");
}


void euler_rhs(double** u, double** u_eval, double** MASS_MATRIX, double* weight, \
                double** uM_eval, double** uP_eval, double** uM, double** uP, \
                double** uavg_uM, double** L, double** rhs, double** normal_direc, \
                double delta) {
    
    RHS(u, u_eval, MASS_MATRIX, weight, \
        uM_eval, uP_eval, uM, uP, \
        uavg_uM, L, rhs, normal_direc, delta);
    euler_multiply(u, rhs);
 
};
