#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define NUM_TIME 20480
#define NUM_SPATIAL 4
#define NUM_ELEMENTS 64
#define h 0.7/20480

void mass_matrix(double* roots, double** D, double** MASS_MATRIX, int N) {

    double* phi = malloc(sizeof(double)*(N+1));
    double* d_phi = malloc(sizeof(double)*(N+1));
    double* dd_phi = malloc(sizeof(double)*(N+1));


    double r2;
    double PEval;
    double PdEval;
    double Pd2Eval;

    for (int i = 0; i <N+1;i++) {
        r2 = roots[i] * roots[i];
        PEval = poly_eval(roots[i], D[1], N);
        PdEval = poly_eval(roots[i], D[2], N);
        Pd2Eval = poly_eval(roots[i], D[3], N);

        phi[i] = (r2 - 1) * PEval;
        d_phi[i] = (r2 - 1) * PdEval + 2 * roots[i] * PEval;
        dd_phi[i] = (r2 - 1) * Pd2Eval + 4 * roots[i] * PdEval + 2 * PEval;

    }

    for (int i = 0; i < N+1;i++) {
        for (int j = 0; j < N+1;j++) {
            if (i != j) {
                MASS_MATRIX[j][i] = 1.0/(pow(roots[j] - roots[i], 2))/d_phi[i] * \
                (d_phi[j] * (roots[j] - roots[i]) - phi[j]);
            }
            else {
                MASS_MATRIX[j][i] = dd_phi[i]/ d_phi[i] / 2.0;
            }
        
        }
    }
    /*
    for (int i = 0; i <N+1;i++) {
        for (int j = 0; j < N+1; j++) {
            printf("%f \t", MASS_MATRIX[i][j]);
        }
        printf("\n");
    }
    */
    
}

