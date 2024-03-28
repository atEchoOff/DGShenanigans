#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define TOL pow(10, -8)

// Gauss-Lobatto Quadrature weights
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
}

void legendre_deriv_poly_eval(double* coef, double** C, double x, int N, double* f) {
    double sum = 0;
    double feval = 0;
    double* vand = malloc(sizeof(double)* (N+1));
    vand[0] = 1;

    for (int i = 1; i < N + 1; i++) {
        vand[i] = vand[i] * x;
    }

    for (int i = 0; i < N + 1; i++) {
        sum += vand[i] * coef[i];
        feval += vand[i] * C[N][i];
    }
    sum += pow(x, N+1) * coef[N + 1];

    sum = N* sum / (pow(x, 2) - 1);
    f[0] = feval;
    f[1] = sum;
}


void newton_raphson_legendre(int N, double** C) {

    double* deriv = malloc(sizeof(double)*(N+1));
    deriv[0] = -C[N-1][0];
    for (int i = 1; i < N + 2; i++) {
        deriv[i] = C[N][i - 1] - C[N-1][i];

    }
    double* fevals = malloc(sizeof(double)*2);
    legendre_deriv_poly_eval(deriv, C, 0.5, N, fevals);
    printf("%f \n %f", fevals[0], fevals[1]);

    /*for (int n = 0; n < N; n++) {

        x = cos(M_PI * (n - 1.0/4/(N + 1.0/2)));
        while (grad > TOL) {
            
        
        }
    }
    */
    
}

int main() {
    int N = 5;
    double** C;
    for (int i = 0; i < N + 1;i++) {
        C[i] = malloc(sizeof(double) * (N+1));
    }
    legendre(N, C);
    for (int i = 0; i < N + 1;i++) {
        for (int j = 0; j < N + 1;j++) {
            printf("%f \t", C[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    newton_raphson_legendre(N, C);

}