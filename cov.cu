#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define FLOAT_N 3214212.01

void init_arrays(double* data, int M, int N)
{
	int i, j;

	for (i = 1; i < (M+1); i++) {
		for (j = 1; j < (N+1); j++) {
			data[i*(N+1) + j] = ((double) i*j) / M;
		}
	}
}


/* Determine mean of column vectors of input data matrix */
__global__ void calcmean(double* d_data, double* d_mean, int M, int N)
{
	int	i;
  int j = blockDim.x * blockIdx.x + threadIdx.x+1;
	if (j<=(M+1)) {
		d_mean[j] = 0.0;
		for (i = 1; i < (N+1); i++) {
        		d_mean[j] += d_data[i*(M+1) + j];
		}
		d_mean[j] /= FLOAT_N;
	}
}


/* Center the column vectors. */
__global__ void calcdata(double* d_data, double* d_mean, int M, int N)
{
  int j;
  int i = blockDim.x * blockIdx.x + threadIdx.x+1;
	if (i<=(N+1)) {
		for (j = 1; j < (M+1); j++) {
			d_data[i*(M+1) + j] -= d_mean[j];
		}
	}
}


/* Calculate the m * m covariance matrix. */
__global__ void calcsymmat(double* d_data, double* d_symmat, int M, int N)
{
  int	i, j2;
  int j1 = blockDim.x * blockIdx.x + threadIdx.x+1;
	if (j1<=(M+1)) {
		for (j2 = j1; j2 < (M+1); j2++) {
	       		d_symmat[j1*(M+1) + j2] = 0.0;
			for (i = 1; i < N+1; i++) {
				d_symmat[j1*(M+1) + j2] += d_data[i*(M+1) + j1] * d_data[i*(M+1) + j2];
			}
    d_symmat[j2*(M+1) + j1] = d_symmat[j1*(M+1) + j2];
    }
	}
}

int main(int argc, char *argv[])
{
	double		*data;
	double		*symmat;
	double		*mean;
  cudaEvent_t start, stop;
  float elapsedTime;
	int i, j;
	FILE		*output1;
	output1 = fopen("covgpu.out", "w");
  cudaError_t err = cudaSuccess;
  int M = atoi(argv[1]), N = atoi(argv[2]);

  int sizedata = (M+1)*(N+1)*sizeof(double);
  int sizesymmat = (M+1)*(M+1)*sizeof(double);
  int sizemean = (M+1)*sizeof(double);


	data = (double*)malloc(sizedata);
	symmat = (double*)malloc(sizesymmat);
	mean = (double*)malloc(sizemean);

  int threadsPerBlock = 128;
  int threadsPerBlock2 = 128;
  int threadsPerBlock3 = 128;
  int blocksPerGrid;
  int blocksPerGrid2;
  int blocksPerGrid3;

  if (M%threadsPerBlock != 0){
		blocksPerGrid = M/128+1;
	}else {
    blocksPerGrid=M/128+1;
  }

  if (N%threadsPerBlock2 != 0){
    blocksPerGrid2 = N/128+1;
  }else {
    blocksPerGrid2=N/128+1;
  }

  if (M%threadsPerBlock3 != 0){
		blocksPerGrid3 = M/128+1;
		}
  else {
    blocksPerGrid3=M/128+1;
  }

  printf("blocksPerGrid %d\n", blocksPerGrid);
  printf("blocksPerGrid2 %d\n", blocksPerGrid2);
  printf("blocksPerGrid3 %d\n", blocksPerGrid3);

  //Matrix data
  double *d_data = NULL;
  err = cudaMalloc((void **)&d_data, sizedata);
  if (err != cudaSuccess){
    fprintf(stderr, "Failed to allocate device matrix DATA (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  //Matrix symmat
  double *d_symmat = NULL;
  err = cudaMalloc((void **)&d_symmat, sizesymmat);
  if (err != cudaSuccess){
    fprintf(stderr, "Failed to allocate device vector SYMMAT (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }


  //Vector mean
  double *d_mean = NULL;
  err = cudaMalloc((void **)&d_mean, sizemean);
  if (err != cudaSuccess){
    fprintf(stderr, "Failed to allocate device vector MEAN (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

	init_arrays(data, M, N);

  err = cudaMemcpy(d_data, data, sizedata, cudaMemcpyHostToDevice);
  if (err != cudaSuccess){
    fprintf(stderr, "Failed to copy matrix DATA from host to device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	calcmean<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_mean, M, N);
  calcdata<<<blocksPerGrid2, threadsPerBlock2>>>(d_data, d_mean, M, N);
	calcsymmat<<<blocksPerGrid3, threadsPerBlock3>>>(d_data, d_symmat, M, N);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start,stop);

	printf("Elapsed time : %f s\n" ,elapsedTime/1000);

  /*Return results*/
  err = cudaMemcpy(symmat, d_symmat, sizesymmat, cudaMemcpyDeviceToHost);

  if (err != cudaSuccess){
    fprintf(stderr, "Failed to return results from device to host (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  /*Print results to file*/
	for (i = 1; i < (M+1); i++) {
		for (j = 1; j < (N+1); j++) {
			fprintf(output1, "%f ", symmat[i*(M+1)+j]);
		}
		fprintf(output1, "\n");
	}

	free(data);
	free(symmat);
	free(mean);
	fclose(output1);

  	return 0;
}
