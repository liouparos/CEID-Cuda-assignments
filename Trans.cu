#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda_runtime.h>

/* Problem size. */

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

}

__global__ void trans_norm_vector(double* A, double* x, double* y, double* tmp, int NX, int NY)
{
	int j;
	int i = blockDim.x * blockIdx.x + threadIdx.x;

  	tmp[i] = 0;
		//Α*Χ
	  for (j = 0; j < NY; j++) {
			tmp[i] = tmp[i] + A[i*NY + j] * x[j];
		}

}

__global__ void atemp(double* A, double* y, double* tmp, int NX, int NY)
{
	int j;
	int i = blockDim.x * blockIdx.x + threadIdx.x;
		// Α(T)*temp
		if (i <= NY){
	  	for (j = 0; j < NX; j++) {
				y[i] = y[i] + A[i + j*NY] * tmp[j];
			}
		}
}



int main(int argc, char *argv[])
{
	FILE		*output1;
	double		*A;
	double		*x;
	double		*y;
	double		*tmp;
	cudaEvent_t start, stop;
 	float elapsedTime;
	int NX = atoi(argv[1]), NY = atoi(argv[2]);

	int sizeY = NY*sizeof(double);
	int sizeX = NY*sizeof(double);
	int sizeA = NX*NY*sizeof(double);
	int sizeTMP = NX*sizeof(double);


	A = (double*)malloc(sizeA);
	x = (double*)malloc(sizeX);
	y = (double*)malloc(sizeY);
	tmp = (double*)malloc(sizeTMP);
	output1 = fopen("transgpu.out", "w");
	cudaError_t err = cudaSuccess;

	int threadsPerBlock=64;
	int threadsPerBlock2=64;
	int blocksPerGrid;
	int blocksPerGrid2;

	if ((NX%512==0) && (NY%512==0)) {
		threadsPerBlock=128;
		threadsPerBlock2=128;
	}


	if ((NX%threadsPerBlock != 0) && (NY%threadsPerBlock2 != 0)) {
		blocksPerGrid = NX/threadsPerBlock+1;
		blocksPerGrid2 = NY/threadsPerBlock2+1;
		}
	else if ((NX%threadsPerBlock == 0) && (NY%threadsPerBlock2 != 0)) {
		blocksPerGrid = NX/threadsPerBlock;
		blocksPerGrid2 = NY/threadsPerBlock2+1;
	}
	else if ((NX%threadsPerBlock != 0) && (NY%threadsPerBlock2 == 0)) {
		blocksPerGrid = NX/threadsPerBlock+1;
		blocksPerGrid2 = NY/threadsPerBlock2;
	}
	else {
		blocksPerGrid = NX/threadsPerBlock;
		blocksPerGrid2 = NY/threadsPerBlock2;
	}

	printf("blocksPerGrid %d\n", blocksPerGrid);
	printf("blocksPerGrid2 %d\n", blocksPerGrid2);
	printf("threadsPerBlock %d\n", threadsPerBlock);
	printf("threadsPerBlock2 %d\n", threadsPerBlock2);



	//Matrix A
	double *d_A = NULL;
	err = cudaMalloc((void **)&d_A, sizeA);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device matrix A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	//Vector x
	double *d_x = NULL;
	err = cudaMalloc((void **)&d_x, sizeX);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector x (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	//Vector y
	double *d_y = NULL;
	err = cudaMalloc((void **)&d_y, sizeY);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector y (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	// tmp
	double *d_tmp = NULL;
	err = cudaMalloc((void **)&d_tmp, sizeTMP);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector tmp (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	init_array(x, A, NX, NY);

	err = cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to copy matrix A from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_x, x, sizeX, cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to copy vector x from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	trans_norm_vector<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_x, d_y, d_tmp, NX, NY);
	atemp<<<blocksPerGrid2, threadsPerBlock2>>>(d_A, d_y, d_tmp, NX, NY);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start,stop);

	printf("Elapsed time : %f s\n" ,elapsedTime/1000);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to fail to run (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(y, d_y, sizeY, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess){
		fprintf(stderr, "Failed to copy vector Y from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(tmp, d_tmp, sizeTMP, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess){
		fprintf(stderr, "Failed to copy vector tmp from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	for (int j = 0; j < NY; j++) {
		fprintf(output1, "%f ", y[j]);
		fprintf(output1, "\n");
	}

	//free gpu pointers
	cudaFree(d_A);
	cudaFree(d_y);
	cudaFree(d_x);
	cudaFree(d_tmp);

	//free cpu pointers
	free(A);
	free(x);
	free(y);
	free(tmp);
	fclose(output1);

  return 0;
}
