#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>

/* Problem size. */
// #define NX 8192
// #define NY 8192

#ifndef M_PI
#define M_PI 3.14159
#endif

void init_array(double *x, double *A, int NX, int NY)
{
	int i, j;

	for (i = 0; i < NX; i++) {
		for (j = 0; j < NY; j++) {
			A[i*NY + j] = ((double) i*(j)) / NX;
		}
	}
	for (j = 0; j < NY; j++) {
		x[j] = j * M_PI;
	}
	// printf("this is A \n");
	// for (int i = 0; i < NX; i++) {
	// 	for (int j = 0; j < NY; j++) {
	// 		printf("%f ",A[i*NY+j]);
	// 	}
	// 	printf("\n");
	// }
	// printf("this is X \n");
	// for (int j = 0; j < NY; j++) {
	// 	printf("%f ",x[j]);
	// 	printf("\n");
	// }
	// printf("\n");
}

void trans_norm_vector(double* A, double* x, double* y, double* tmp, int NX, int NY)
{
	int i,j;

	for (i= 0; i < NY; i++) {
    	y[i] = 0;
	}

	for (i = 0; i < NX; i++) {
  	tmp[i] = 0;
		//Α*Χ
	  for (j = 0; j < NY; j++) {
			tmp[i] = tmp[i] + A[i*NY + j] * x[j];
		}
		// Ατ*x
	  for (j = 0; j < NY; j++) {
			y[j] = y[j] + A[i*NY + j] * tmp[i];
		}
	}
	// print y
		// for (int j = 0; j < NY; j++) {
		// 	printf("%f ",y[j]);
		// 	printf("\n");
		// }

	}

int main(int argc, char *argv[])
{
	double		*A;
	double		*x;
	double		*y;
	double		*tmp;
	struct timeval	cpu_start, cpu_end;
	FILE		*output1;
	int NX = atoi(argv[1]), NY = atoi(argv[2]);
	A = (double*)malloc(NX*NY*sizeof(double));
	x = (double*)malloc(NY*sizeof(double));
	y = (double*)malloc(NY*sizeof(double));
	tmp = (double*)malloc(NX*sizeof(double));
	output1 = fopen("transcpu.out", "w");

	init_array(x, A, NX, NY);

	gettimeofday(&cpu_start, NULL);
	trans_norm_vector(A, x, y, tmp, NX, NY);
	gettimeofday(&cpu_end, NULL);
	fprintf(stdout, "CPU Runtime :%0.6lfs\n", ((cpu_end.tv_sec - cpu_start.tv_sec) * 1000000.0 + (cpu_end.tv_usec - cpu_start.tv_usec)) / 1000000.0);

	// // print y to cmd
	// for (int j = 0; j < NY; j++) {
	// 	printf("%f ",y[j]);
	// 	printf("\n");
	// }
	//print y to file
	for (int j = 0; j < NY; j++) {
		fprintf(output1, "%f ", y[j]);
		fprintf(output1, "\n");
	}

	free(A);
	free(x);
	free(y);
	free(tmp);
	fclose(output1);

  return 0;
}
