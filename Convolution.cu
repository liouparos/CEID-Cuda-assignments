#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda_runtime.h>

__global__ void Convolution(double* A, double* B, int I, int J)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	double c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
	c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
	c13 = +0.4;  c23 = +0.7;  c33 = +0.1;

	if (i>J && i<I*J-J && (i%J!=0) && ((i+1)%J!=0)) {
		B[i] = c11 * A[i-J-1]  +  c12 * A[i-1]  +  c13 * A[i+J-1]
				 + c21 * A[i-J]  +  c22 * A[i]  +  c23 * A[i+J]
				 + c31 * A[i-J+1]  +  c32 * A[i+1]  +  c33 * A[i+J+1];
	}

}

void init(double* A, int I, int J)
{
	int i, j;

	for (i = 0; i < I; ++i) {
		for (j = 0; j < J; ++j) {
			A[i*J + j] = (double)rand()/RAND_MAX;
        	}
    	}
}

int main(int argc, char *argv[])
{
	FILE		*output1;
	double		*A;
	double		*B;
	cudaEvent_t start, stop;
 	float elapsedTime;

	output1 = fopen("convgpu.out", "w");

	int I = atoi(argv[1]), J = atoi(argv[2]);
	int size = I*J*sizeof(double);

	A = (double*)malloc(size);
	B = (double*)malloc(size);

	cudaError_t err = cudaSuccess;
	double *d_A = NULL;
	err = cudaMalloc((void **)&d_A, size);
	if (err != cudaSuccess)
	{
			fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
	}
	double *d_B = NULL;
	err = cudaMalloc((void **)&d_B, size);

	if (err != cudaSuccess)
	{
			fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
	}


	//initialize the arrays
	init(A, I, J);

	//host to Device
	err = cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

		if (err != cudaSuccess)
		{
				fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
				exit(EXIT_FAILURE);
		}
		int threadsPerBlock=128;
		int blocksPerGrid;
		if (I*J%threadsPerBlock != 0){
			blocksPerGrid = I*J/threadsPerBlock+1;
		}else {
			blocksPerGrid=I*J/threadsPerBlock;
		}

		printf("blocksPerGrid: %d\n", blocksPerGrid);
		printf("threadsPerBlock: %d\n", threadsPerBlock);

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		Convolution<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, I, J);
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTime, start,stop);

		err = cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost);

		if (err != cudaSuccess)
		{
				fprintf(stderr, "Failed to copy vector B from device to host (error code %s)!\n", cudaGetErrorString(err));
				exit(EXIT_FAILURE);
		}

		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
				fprintf(stderr, "error code %s\n", cudaGetErrorString(err));
				exit(EXIT_FAILURE);
		}

		for (int i = 0; i < I; i++) {
			for (int j = 0; j < J; j++) {
				fprintf(output1, "%f ", B[i*J+j]);
			}
			fprintf(output1, "\n");
		}

	printf("Elapsed time : %f s\n" ,elapsedTime/1000);

	free(A);
	free(B);
	cudaFree(d_A);
	cudaFree(d_B);
	fclose(output1);
	return 0;
}
