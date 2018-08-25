
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void hello()
{
	printf("This is thread %d in block %d\n", threadIdx.x, blockIdx.x);
}

int main()
{
	int blockSize =4, threadSize = 3;

    // running in parallel
	hello << <blockSize, threadSize >> > ();

	//force the printf to flush
	cudaDeviceSynchronize();
	getchar(); //wait keypressed

    return 0;
}
