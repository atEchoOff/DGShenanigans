#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define TOL 1e-10


// Gauss-Lobatto Quadrature weights
// Here N is the number of nodal points (including endpoints) + 1

double** W;


void legendre(int N, double** C) {

    C[0][0] = 1;
    C[1][0] = 0;
    C[1][1] = 1;

    for (int i = 2; i < N + 1; i++) {
        for (int j = 0; j < i;j++) {

            C[i][j] += -(double) (i - 1)/i * C[i - 2][j];
            C[i][j + 1] += (double) (2*i - 1)/i * C[i - 1][j];         
        }
    }
    // Print out Legendre coefficient matrix
    /*
    for (int i = 0; i < N + 1;i++) {
        for (int j = 0; j < N + 1;j++) {
            printf("%0.4f \t", C[i][j]);
        }
        printf("\n");
    }
    */
    
}

double factorial(int n) {
    double f = 1;
    for (int i = 2; i < n + 1; i++) {
        f = f * i;
    }
    return f;
}

void legendre_deriv_matrix(double** D, int N, int d) {
    
    for (int i = 1; i < d;i++){
        for (int j = 0; j < N;j++) {
            D[i][j] = D[i-1][j+1] * (j + 1);
        }
    }
}

double poly_eval(double x, double* coef, int N) {
    double* vand;
    double sum = 0;
    vand = malloc(sizeof(double)* (N + 1));
    vand[0] = 1;

    for (int i = 1; i < N + 1; i++) {
        vand[i] = vand[i - 1] * x;
    }

    for (int i = 0; i < N+1;i++) {
        sum += coef[i] * vand[i];
    }
    return sum;

}

void legendre_deriv_poly_eval(double** D, double x, int N, double* feval, int d) {

    double sum;
    double* vand;
    vand = malloc(sizeof(double)* (N + 1));
    vand[0] = 1;

    for (int i = 1; i < N + 1; i++) {
        vand[i] = vand[i - 1] * x;
    }

    for (int i = 0; i < d; i++) {
        sum = 0;
        for (int j = 0; j < N + 1; j++) {
            sum += D[i][j] * vand[j];
        }
        feval[i] = sum;
    }    
}

void weights(int N, double* roots, double* coef, double* weight) {
    double eval;
    for (int i = 1; i < N;i++) {
        eval = poly_eval(roots[i], coef, N);
        weight[i] = 2.0/(N+1)/(N)/(eval * eval);
    }
    weight[0] = 2.0/(N + 1)/(N);
    weight[N] = 2.0/(N + 1)/(N);

    
    for (int i = 0; i < N+1;i++) {
        printf("Weights %d: %f\n", i, weight[i]);
    }


   
}

// Accepts N, C- Coefficient matrix of legendre polynomials
// D- coefficient of derivatives of P'_(N). including 0th derivative

void newton_raphson_legendre(int N, double** C, double** D, double* roots) {

    int d = 4;
    double* fevals = malloc(sizeof(double)*d);
    double x0;

    for (int i = 0; i < N+1;i++) {
        D[0][i] = C[N][i];
    }

    legendre_deriv_matrix(D, N, d);

    for (int n = 1; n < N; n++) {
        // Intial x0 should be chebyshev node

        x0 = cos(M_PI * ((double) n)/ N);
        // Set high first derivative value

        fevals[1] = 1000;

        legendre_deriv_poly_eval(D, x0, N, fevals, d);

        // Newton's method
        int iter = 0;
        while (fabs(fevals[1]) > TOL) {
            
            x0 = x0 - fevals[1]/fevals[2];
            iter = iter + 1;
            legendre_deriv_poly_eval(D, x0, N, fevals, d);
        
            //printf("Root %d: Function Value = %f, Iter =  %d \n", n + 1, fevals[1], iter);

        }
        roots[N-n] = x0;
        
        printf("###ROOT: %1.12f\n", x0);
        
    }
    roots[0] = -1;
    roots[N] = 1;
    
}

void generate_nodal(int m, int N, double a, double b, double* roots, double** nodes) {
    double L = (b - a)/m;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < N+1;j++) {
            nodes[j][i] = (roots[j] + 1)/2 * L + a + i*L;
        }
    }
    /*
    for (int i = 0; i < N+1; i++) {
        for (int j = 0; j <m;j++) {
            printf("%f\t", nodes[i][j]);
        }
        printf("\n");
    }
    */
}


