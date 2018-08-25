
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define BlockSize	8
#define ThreadSize	1024
#define ArraySize	(BlockSize*ThreadSize)

__device__ void __syncthreads();
__global__ void globalReduceBlockSum(int *b, int *a)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	for (int d = blockDim.x / 2; d > 0; d >>= 1)
	{
		if (threadIdx.x < d)
		{
			a[id] += a[id + d];
		}		
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		b[blockIdx.x] = a[id]; 
	}
}

__global__ void sharedReduceBlockSum(int *b, int *a)
{
	__shared__ int x[ThreadSize];

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	x[threadIdx.x] = a[id]; //copy to shared memory of block
	__syncthreads(); //wait all threads copy complete

	for (int d = blockDim.x / 2; d > 0; d >>= 1)
	{
		if (threadIdx.x < d)
		{
			x[threadIdx.x] += x[threadIdx.x + d];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		b[blockIdx.x] = x[0];
	}
}

int main()
{
	int host_a[ArraySize];
	int host_b[BlockSize];
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
	for (int i = 0; i <  ArraySize; i++)
		host_a[i] = i+1;
	cudaMemcpy(dev_a, host_a, ArraySize * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dev_b, BlockSize * sizeof(int));
	cudaMemset(dev_b, 0, BlockSize * sizeof(int));

	// Run sharedReduce first, because b[] is modified in globalReduce

	cudaEventRecord(start, 0); //keep start time
	sharedReduceBlockSum << <BlockSize, ThreadSize >> > (dev_b, dev_a);	//calculate
	cudaEventRecord(stop, 0); //keep stop time
	cudaEventSynchronize(stop); //wait stop event		
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaMemcpy(host_b, dev_b, BlockSize * sizeof(int), cudaMemcpyDeviceToHost);

	//Print result
	int answer = (ArraySize + 1)*ArraySize / 2;

	printf("shared:\n");
	sum = 0;
	for (int i = 0; i < BlockSize; i++) {
		sum += host_b[i];
		printf("%d ", host_b[i]);
	}
	printf("sum=%d answer=%d t=%f\n\n", sum, answer, elapsedTime);

	// run globalReduce
	cudaEventRecord(start, 0); //keep start time
	globalReduceBlockSum << <BlockSize, ThreadSize >> > (dev_b, dev_a);	//calculate
	cudaEventRecord(stop, 0); //keep stop time
	cudaEventSynchronize(stop); //wait stop event		
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaMemcpy(host_b, dev_b, BlockSize * sizeof(int), cudaMemcpyDeviceToHost);

	//Print result
	printf("global:\n");
	sum = 0;
	for (int i = 0; i < BlockSize; i++) {
		sum += host_b[i];
		printf("%d ", host_b[i]);
	}
	printf("sum=%d answer=%d t=%f\n\n", sum, answer, elapsedTime);
	//cudaDeviceSynchronize();
	getchar();

	cudaFree(dev_a);
	cudaFree(dev_b);
	return 0;
}
