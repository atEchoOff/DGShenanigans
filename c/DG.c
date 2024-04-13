#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "rk4.c"

void init_cond(double** u, double** nodes) {
    // u0(x) = e^(-10x^2);
    for (int i = 0; i < NUM_SPATIAL;i++) {
        for (int j = 0; j < NUM_ELEMENTS; j++) {
            u[i][j] = exp(-10 * nodes[i][j] * nodes[i][j]);
            //u[i][j] = 1.0/(1 + exp(-3*nodes[i][j])) + 1;
            //u[i][j] = sin(M_PI*nodes[i][j]);
        }
    }
}

int main() {
    // interval (a, b)
    clock_t start, end;
    
    start = clock();
    double a = -1;
    double b = 1;

    double delta = (b - a)/NUM_ELEMENTS;

    // Set number of derivative - 1 to take for P'_{n-1}
    int d = 4;
    
    int N = NUM_SPATIAL - 1;
    // C contains coefficient of 1st to Nth Legendre polynomial
    double** C;

    // D contains coefficient of 0th to 2nd derivative
    // of P'(n-1) legendre polynomial
    double** D;

    // This should be differential matrix
    // MASS_MATRIX contains evalulation MASS_MATRIX_ij = l'j(x_i) 
    double** MASS_MATRIX;

    // Matrix of nodes, each column represent each element
    double** nodes;

    // number of elements

    double* coef = malloc(sizeof(double)*(N+1));
    double* roots = malloc(sizeof(double) * (N + 1));
    double* weight = malloc(sizeof(double)*(N+1));

    C = malloc(sizeof(double)*(N+1));
    for (int i = 0; i < N + 1;i++) {
        C[i] = (double*) malloc(sizeof(double) * (N+1));
    }

    D = malloc(sizeof(double)*(d));
    for (int i = 0; i < d;i++) {
        D[i] = (double*) malloc(sizeof(double) * (N+1));
    }

    MASS_MATRIX = malloc(sizeof(double)*(N+1));
    for (int i = 0; i < N + 1;i++) {
        MASS_MATRIX[i] = (double*) malloc(sizeof(double) * (N+1));
    }

    nodes = malloc(sizeof(double)*(N+1));
    for (int i = 0; i < N + 1;i++) {
        nodes[i] = (double*) malloc(sizeof(double) * (NUM_ELEMENTS));
    }

    printf("Legendre Coefficients:\n");
    legendre(N, C);
    printf("\n");
    newton_raphson_legendre(N, C, D, roots);
    printf("\n");
    memcpy(coef, C[N], 8* (N+1));
    weights(N, roots, coef, weight);
    printf("\n");
    printf("Mass Matrix: \n");
    mass_matrix(roots, D, MASS_MATRIX, N);
    printf("\n");
    printf("Nodal Points: \n");
    generate_nodal(NUM_ELEMENTS, N, a, b, roots, nodes);
    printf("\n");

    double** u; 
    double** u_next;
    double** uM;
    double** uP;
    double** u_eval;
    double** uM_eval;
    double** uP_eval;
    double** uavg_uM;
    double** uavg_uM_L;
    double** rhs;
    double** L;
    double** normal_direc;

    u = malloc(sizeof(double)*(NUM_SPATIAL));
    L = malloc(sizeof(double)*(NUM_SPATIAL));
    for (int i = 0; i < (NUM_SPATIAL);i++) {
        u[i] = malloc(sizeof(double)*(NUM_ELEMENTS));
        L[i] = malloc(sizeof(double)* 2);
    }
    L[0][0] = 1;
    L[N][1] = 1;

    uM = malloc(sizeof(double)*(2));
    uP = malloc(sizeof(double)*(2));
    uM_eval = malloc(sizeof(double)*2);
    uP_eval = malloc(sizeof(double)*2);
    uavg_uM = malloc(sizeof(double)*2);
    uavg_uM_L = malloc(sizeof(double)*NUM_ELEMENTS);
    normal_direc = malloc(sizeof(double)*2);

    for (int i = 0; i < 2;i++) {
        uM[i] = malloc(sizeof(double)*(NUM_ELEMENTS));
        uP[i] = malloc(sizeof(double)*(NUM_ELEMENTS));
        uM_eval[i] = malloc(sizeof(double)*(NUM_ELEMENTS));
        uP_eval[i] = malloc(sizeof(double)*(NUM_ELEMENTS));
        uavg_uM[i] = malloc(sizeof(double)*(NUM_ELEMENTS));
        normal_direc[i] = malloc(sizeof(double)*NUM_ELEMENTS);
    }
    normal(normal_direc);

    for (int i = 0; i < NUM_ELEMENTS;i++) {
        uavg_uM_L[i] = malloc(sizeof(double)*(NUM_ELEMENTS));
    }

    double*** U;
    double*** k_vector;
    double rk_coef[4] = {1, 2, 2, 1};

    k_vector = malloc(sizeof(double)*4);
    for (int i = 0; i <4; i++) {
        k_vector[i] = malloc(sizeof(double)*NUM_SPATIAL);
    }
    for (int i = 0; i < 4;i++) {
        for (int j = 0; j < NUM_SPATIAL;j++) {
            k_vector[i][j] = malloc(sizeof(double)*NUM_ELEMENTS);
        }
    }

    U = malloc(sizeof(double)*NUM_TIME);
    for (int i = 0; i <NUM_TIME; i++) {
        U[i] = malloc(sizeof(double)*NUM_SPATIAL);
    }
    for (int i = 0; i < NUM_TIME;i++) {
        for (int j = 0; j < NUM_SPATIAL;j++) {
            U[i][j] = malloc(sizeof(double)*NUM_ELEMENTS);
        }
    }

    // Initial condition
    u = malloc(sizeof(double) *NUM_SPATIAL);
    u_next = malloc(sizeof(double)* NUM_SPATIAL);
    u_eval = malloc(sizeof(double)* NUM_SPATIAL);
    rhs = malloc(sizeof(double)* NUM_SPATIAL);

    for (int i = 0; i < NUM_SPATIAL;i++) {
        u[i] = malloc(sizeof(double)* NUM_ELEMENTS);
        u_next[i] = malloc(sizeof(double)* NUM_ELEMENTS);
        u_eval[i] = malloc(sizeof(double)* NUM_ELEMENTS);
        rhs[i] = malloc(sizeof(double)* NUM_ELEMENTS);
    }

    init_cond(u, nodes);
    end = clock();

    double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Time For Setting Up: %f\n", cpu_time_used);
    
    FILE* file;
    FILE* x_nodes;
    FILE* param;
    start = clock();
    
    for (int t = 0; t < NUM_TIME;t++) {
        
        //memcpy(U[t], u, 8*NUM_ELEMENTS*NUM_SPATIAL);
        for (int i = 0; i < NUM_SPATIAL;i++) {
            memcpy(U[t][i], u[i], 8* NUM_ELEMENTS);
        }
        
        
        
        /*rk4_rhs(u, u_eval, MASS_MATRIX, weight, \
                uM_eval, uP_eval, uM, uP, \
                uavg_uM, L, rhs, normal_direc, \
                k_vector, rk_coef, u_next, delta); 
        */
        euler_rhs(u, u_eval, MASS_MATRIX, weight, \
                uM_eval, uP_eval, uM, uP, \
                uavg_uM, L, rhs, normal_direc, \
                delta);
                
    }
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Time for ODE: %f\n", cpu_time_used);
    printf("Num Elements: %d\n", NUM_ELEMENTS);
    printf("Degree: %d\n", NUM_SPATIAL);

    file = fopen("meh.bin", "w");
    x_nodes = fopen("x.bin", "w");

    for (int i = 0; i < NUM_TIME; i++) {
        for (int j = 0; j < NUM_SPATIAL; j++) {
            fwrite(U[i][j], sizeof(double), NUM_ELEMENTS, file);
        }
    }
    fclose(file);
    for (int i = 0; i < NUM_SPATIAL; i++) {
        fwrite(nodes[i], sizeof(double), NUM_ELEMENTS, x_nodes);
    }
    fclose(x_nodes);
    param = fopen("param.bin", "w");
    int para[3] = {NUM_TIME, NUM_SPATIAL, NUM_ELEMENTS};
    fwrite(para, sizeof(int), 3, param);

}