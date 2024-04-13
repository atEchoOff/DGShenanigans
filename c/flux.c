#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "quadrature.c"
#include "lagrange.c"

void eval_function(double** u, double** ueval, int n, int m) {
    // Burgers' equation
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            ueval[i][j] = u[i][j] * u[i][j]/2;
        }
    }
}

void normal(double** normal_direc) {
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        normal_direc[0][i] = -1;
        normal_direc[1][i] = 1;
    }
    
}

void dot_product (double** a, double** b, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j <n;j++) {
            a[i][j] = a[i][j] * b[i][j];
        }
    }
}

void RHS(double** u, double** u_eval, double** MASS_MATRIX, double* weight,\
    double** uM_eval, double** uP_eval, double** uM, double** uP, \
    double** uavg_uM, double** L, double** rhs, double** normal_direc, double delta) {

    for (int i = 0; i < NUM_ELEMENTS;i++) {
        uM[0][i] = u[0][i];
        uM[1][i] = u[NUM_SPATIAL - 1][i]; 
    }
    for (int i = 1;i < NUM_ELEMENTS;i++) {
        uP[0][i] = uM[1][i - 1];
        uP[1][i - 1] = uM[0][i];
    }
    //uP[0][0] = uM[0][NUM_ELEMENTS-1];
    //uP[1][NUM_ELEMENTS - 1] = uM[0][0];

    uP[0][0] = uM[0][0];
    uP[1][NUM_ELEMENTS - 1] = uM[1][NUM_ELEMENTS-1];

    eval_function(u, u_eval, NUM_SPATIAL, NUM_ELEMENTS);
    eval_function(uM, uM_eval, 2, NUM_ELEMENTS);
    eval_function(uP, uP_eval, 2, NUM_ELEMENTS);
    
    for (int i = 0;i < 2; i++) {
        for (int j = 0; j < NUM_ELEMENTS;j++) {
            uavg_uM[i][j] = ((uM_eval[i][j] + uP_eval[i][j])/2 - uM_eval[i][j]) \
             * normal_direc[i][j]; //- 0.5 * (uP[i][j] - uM[i][j]);
        }
    }

    double sum;
    for (int i = 0; i < NUM_SPATIAL; i++) {
        for (int j = 0; j < NUM_ELEMENTS; j++) {
            sum = 0;
            for (int k = 0; k < 2; k++) {
                sum += L[i][k] * uavg_uM[k][j];
            }
            rhs[i][j] = sum;
        }
    }


    for (int i = 0; i < NUM_SPATIAL; i++) {
        for (int j = 0; j < NUM_ELEMENTS;j++) {
            rhs[i][j] = 1/ weight[i] * rhs[i][j];
        }
    }
    
    for (int i = 0; i < NUM_SPATIAL; i++) {
        for (int j = 0; j < NUM_ELEMENTS;j++) {
            sum = 0;
            for (int k = 0; k < NUM_SPATIAL; k++) {
                sum += MASS_MATRIX[i][k] * u_eval[k][j];
            }
            rhs[i][j] += sum;
            rhs[i][j] = (-2/delta) * rhs[i][j];
            //printf("%1.10e\t", rhs[i][j]);
        }
        //printf("\n");
    }
    //printf("\n");
}

