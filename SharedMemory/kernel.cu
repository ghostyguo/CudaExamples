
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define BlockSize  2
#define ThreadSize 10
#define ArraySize (BlockSize*ThreadSize)

__global__ void shared(int* a) //running on device
{
	__shared__ int x[ThreadSize]; //shared in the same block

	int i = blockIdx.x*ThreadSize+threadIdx.x;
	x[threadIdx.x] = a[i]; //copy to shared memory

	if (threadIdx.x < ThreadSize/2) {
		x[threadIdx.x] = x[threadIdx.x + 1];
	}

	a[i] = x[threadIdx.x];
}

int main() //running on host
{
	int host_a[ArraySize], host_b[ArraySize]; //memory in host
	int *dev_a = 0; //global memory on device

	cudaSetDevice(0); //select a device

	// init array values
	for (int i = 0; i < ArraySize; i++) {
		host_a[i] = i;
		printf("a[%d]=%d ", i, host_a[i]);
		if (i%ThreadSize == ThreadSize-1) printf("\n");
	}
	printf("\n");
	
	// init device memory array values
	cudaMalloc((void**)&dev_a, ArraySize * sizeof(int));
	cudaMemcpy(dev_a, host_a, ArraySize * sizeof(int), cudaMemcpyHostToDevice);

	// running kernel in parallel
	shared << < BlockSize, ThreadSize >> > (dev_a);

	//copy result back

	cudaMemcpy(host_b, dev_a, ArraySize * sizeof(int), cudaMemcpyDeviceToHost);

	//	waits for the kernel to finish, 
	cudaDeviceSynchronize();

	//output
	for (int i = 0; i < ArraySize; i++) {
		printf("b[%d]=%d ", i, host_b[i]);
		if (i%ThreadSize == ThreadSize-1) printf("\n");
	}
	printf("\n");
	getchar(); //wait keypressed

	return 0;
}
