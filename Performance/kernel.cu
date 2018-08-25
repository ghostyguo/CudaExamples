
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "cooperative_groups_helpers.h"
#include <iostream>
#include <memory>
#include <string>
#include <Windows.h>

const int repeat = 10000; //重複計算次數;
const int arraySize = 1024 * 1024; //受限於host與gpu記憶體大小
BOOL WINAPI QueryPerformanceCounter(_Out_ LARGE_INTEGER *lpPerformanceCount);
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
void addWithCpu(int *c, const int *a, const int *b, unsigned int size);
__global__ void addKernel(int *c, const int *a, const int *b, int size);


// data array for test, global or local static in heap to prevent stck overflow
/*
int a[arraySize];
int b[arraySize];
int c[arraySize];
int d[arraySize];
*/
int main()
{
	// data array for test,  global or local static in heap to prevent stck overflow
	static int a[arraySize];
	static int b[arraySize];
	static int c[arraySize];
	static int d[arraySize];

	// setup performance measure from windows ----------
	LARGE_INTEGER frequency;        // ticks per second
	LARGE_INTEGER t1, t2;           // ticks
	float elapsedTime;

	// setup performance measure from windows ---------
	QueryPerformanceFrequency(&frequency);

	// setup performance meter from CUDA ----------
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// init array ----------
	for (int i = 0; i < arraySize; i++) {
		a[i] = i;
		b[i] = arraySize + i;
	}

	/// Add by CPU ----------
	QueryPerformanceCounter(&t1); //keep start time
	for (int i = 0; i < repeat; i++) {
		addWithCpu(c, a, b, arraySize);
	}
	QueryPerformanceCounter(&t2); //keep stop time
	elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
	printf("c[%d]=%d, cpu t=%f\n",
		arraySize - 1, c[arraySize - 1], elapsedTime);
	

	// Add by CUDA ----------
	cudaEventRecord(start, 0); //keep start time
	cudaError_t cudaStatus = addWithCuda(d, a, b, arraySize);
	cudaEventRecord(stop, 0); //keep stop time
	cudaEventSynchronize(stop); //wait stop event
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}
	cudaEventElapsedTime(&elapsedTime, start, stop);	
	printf("d[%d]=%d, gpu t=%f\n",
		arraySize - 1, c[arraySize - 1], elapsedTime);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	getchar();
	return 0;
}

// Add with CPU ---------
void addWithCpu(int *c, const int *a, const int *b, unsigned int size)
{
	for (unsigned int i = 0; i < size; i++) {
		c[i] = a[i] + b[i];
	}
}

// Add with GPU ---------
__global__ void addKernel(int *c, const int *a, const int *b, int size)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < size) {
		c[i] = a[i] + b[i];
	}
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	int dev = 0;
	cudaStatus = cudaSetDevice(dev);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaSetDevice(dev);

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	/*
	int block = 1;
	unsigned int thread_x = (size-1) / blockDim.x+1;
	unsigned int thread_y = (size-1) % blockDim.x+1;
	dim3 thread = {thread_x, thread_y, 1 };
	*/
	int block = (size - 1) / 1024 + 1;
	int thread = (size>1024) ? 1024 : (size - 1);

	for (int i = 0; i < repeat; i++) {
		addKernel << <block, thread >> > (dev_c, dev_a, dev_b, size);
	}

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}