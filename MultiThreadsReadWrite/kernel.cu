
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>

#define ThreadSize 1000
#define BlockSize  10000
#define ArraySize 10


//int atomicAdd(int* address, int val);
__global__ void incKernel(int *a)
{
	int i = (blockIdx.x*blockDim.x + threadIdx.x) % ArraySize;
	a[i] = a[i] + 1;
	//atomicAdd(&a[i], 1);
}

int main()
{
	int host_a[ArraySize];
	int *dev_a = 0;
	float elapsedTime;

	// setup performance meter from CUDA ----------
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaSetDevice(0);
	cudaMalloc((void**)&dev_a, ArraySize * sizeof(int));

	for (int run = 0; run < 10; run++) {

		cudaMemset(dev_a, 0, ArraySize * sizeof(int));		//clear 

		cudaEventRecord(start, 0); //keep start time
		incKernel << <BlockSize, ThreadSize >> > (dev_a);	//calculate
		cudaEventRecord(stop, 0); //keep stop time
		cudaEventSynchronize(stop); //wait stop event		
		cudaEventElapsedTime(&elapsedTime, start, stop);	

		cudaMemcpy(host_a, dev_a, ArraySize * sizeof(int), cudaMemcpyDeviceToHost);
		//Print result
		printf("run {%d}: ",run);
		for (int i = 0; i < ArraySize; i++) {
			printf("%d ", host_a[i]);
		}
		printf(" t=%f\n",elapsedTime);
	}
	//cudaDeviceSynchronize();
	getchar();

	cudaFree(dev_a);
	return 0;

}
