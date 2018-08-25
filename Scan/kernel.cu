
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define BlockSize	1
#define ThreadSize	19
#define ArraySize	(BlockSize*ThreadSize)

__device__ void __syncthreads();
__global__ void scanHillisSteele(int *b, int *a)
{
	__shared__ int x[ThreadSize];

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	x[threadIdx.x] = a[id];
	__syncthreads(); //wait copy compelete

	for (int d = 1; d<blockDim.x; d <<= 1)
	{
		if (threadIdx.x >= d) {
			x[threadIdx.x] += x[threadIdx.x - d];
		} //keep 
		__syncthreads();
	}

	b[threadIdx.x] = x[threadIdx.x]; 
}

int main()
{
	int host_a[ArraySize];
	int host_b[ArraySize];
	int *dev_a = 0;
	int *dev_b = 0;
	int sum = 0;
	float elapsedTime;

	// setup performance meter from CUDA ----------
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaSetDevice(0);

	cudaMalloc((void**)&dev_a, ArraySize * sizeof(int));
	for (int i = 0; i < ArraySize; i++)
		host_a[i] = i + 1;
	cudaMemcpy(dev_a, host_a, ArraySize * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dev_b, ArraySize * sizeof(int));
	//cudaMemset(dev_b, 0, ArraySize * sizeof(int));

	// Run scanHillisSteele

	cudaEventRecord(start, 0); //keep start time
	scanHillisSteele << <BlockSize, ThreadSize >> > (dev_b, dev_a);	//calculate
	cudaEventRecord(stop, 0); //keep stop time
	cudaEventSynchronize(stop); //wait stop event		
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaMemcpy(host_b, dev_b, ArraySize * sizeof(int), cudaMemcpyDeviceToHost);

	//Print result
	printf("pdf:\n");
	for (int i = 0; i < ArraySize; i++) {
		printf("%4d ", host_a[i]);
	}
	printf("\n");

	printf("cdf:\n");
	for (int i = 0; i < ArraySize; i++) {
		printf("%4d ", host_b[i]);
	}
	printf("\nt=%f\n\n", elapsedTime);




	//cudaDeviceSynchronize();
	getchar();

	cudaFree(dev_a);
	cudaFree(dev_b);
	return 0;
}
