#include <chrono>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <iomanip>
// CUDA header file
#include <cuda_runtime.h>

// other sorting
#include "quicksort.h"
#include "bitonicCPU.h"

//thruse
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
//

using namespace std;
using namespace std::chrono;

const int ntpb = 512;

__global__ void BitonicSort(int* d_a, int n, int stage)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int gap = 1 << (stage - 1);
	int idx = 0;
	int temp = 0;
	int group = (i / gap) % 2; // 0 asc, 1 desc


	if (i < n)
	{
		for (int round = 1; round <= stage; round++)
		{
			gap = 1 << (stage - round); // 1,2,4,8,16,32....
			idx = (i / gap)*gap * 2 + i%gap; 
			
			if (group == 0)
			{
				if (d_a[idx] > d_a[idx + gap]) // swap
				{
					temp = d_a[idx];
					d_a[idx] = d_a[idx + gap];
					d_a[idx + gap] = temp;
				}
			}
			if (group == 1)
			{
				if (d_a[idx] < d_a[idx + gap]) // swap
				{
					temp = d_a[idx];
					d_a[idx] = d_a[idx + gap];
					d_a[idx + gap] = temp;
				}
			}
			__syncthreads();
		}
	}
}
// 16777216 tested
__global__ void BitonicSortLarge(int* d_a, int n, int stage, int round)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int gap = 1 << (stage - round);
	int group = (i / (1 << (stage - 1))) % 2; // 0 asc, 1 desc
	int idx = (i / gap)*gap * 2 + i%gap;
	int temp = 0;

	//printf("%d stage = %d, round = %d, compare %d %d\n",group, stage, round, idx, idx + gap);

	if (group == 0)
	{
		if (d_a[idx] > d_a[idx + gap]) // swap
		{
			temp = d_a[idx];
			d_a[idx] = d_a[idx + gap];
			d_a[idx + gap] = temp;
		}
	}
	if (group == 1)
	{
		if (d_a[idx] < d_a[idx + gap]) // swap
		{
			temp = d_a[idx];
			d_a[idx] = d_a[idx + gap];
			d_a[idx + gap] = temp;
		}
	}
}

__global__ void BitonicSortLargeOptimized(int* d_a, int n, int stage, int round)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int gap = 1 << (stage - round);
	int group = (i / (1 << (stage - 1))) % 2; // 0 asc, 1 desc
	int idx = (i / gap)*gap * 2 + i%gap;
	int temp = 0;
	int tidx = threadIdx.x;
	__shared__ int s_a[ntpb];
	__shared__ int s_gap[ntpb];

	s_a[tidx] = d_a[idx];
	s_gap[tidx] = d_a[idx + gap];
	__syncthreads();

	//printf("%d stage = %d, round = %d, compare %d %d\n",group, stage, round, idx, idx + gap);

	if (group == 0)
	{
		if (s_a[tidx] > s_gap[tidx]) // swap
		{
			temp = s_gap[tidx];
			s_gap[tidx] = s_a[tidx];
			s_a[tidx] = temp;
		}
	}
	if (group == 1)
	{
		if (s_a[tidx] < s_gap[tidx]) // swap
		{
			temp = s_gap[tidx];
			s_gap[tidx] = s_a[tidx];
			s_a[tidx] = temp;
		}
	}

	d_a[idx] = s_a[tidx];
	d_a[idx+gap] = s_gap[tidx];
}

void reportTime(const char* msg, steady_clock::duration span) {
	auto ms = duration_cast<microseconds>(span);
	std::cout << msg << " - took - " <<
		ms.count() << " microseconds\n" << std::endl;
}

void generateRandom(int array[], int size)
{
	for (int i = 0; i < size; i++)
	{
		array[i] = rand();
	}
}

void generateRandom(int array[], int array2[], int array3[], int size)
{
	for (int i = 0; i < size; i++)
	{
		array[i] = rand();
		array2[i] = array[i];
		array3[i] = array[i];
	}
}

void generateRandom(int array[], int array2[], int array3[], thrust::host_vector<int>& h_vec, int size)
{
	for (int i = 0; i < size; i++)
	{
		array[i] = rand();
		array2[i] = array[i];
		array3[i] = array[i];
		h_vec[i] = array[i];
	}
}

void bitonicGPU(int* h_a, int n ,int nblks)
{
	steady_clock::time_point ts, te;
	int* d_a = nullptr;

	std::cout << "Memcpy Host to Device" << std::endl;
	ts = steady_clock::now();
	cudaMalloc((void**)&d_a, n * sizeof(int));
	cudaMemcpy(d_a, h_a, n * sizeof(int), cudaMemcpyHostToDevice);
	te = steady_clock::now();
	reportTime("Memcpy Host to Device", te - ts);

	// thread is required n/2, so block is n/2
	dim3 blocks(nblks / 2, 1);
	dim3 threads(ntpb, 1);

	// bitonic sort using cuda
	std::cout << "bitonic sort ";
	if (n <= 4096)
	{
		std::cout << "short kernel using GPU" << std::endl;
		ts = steady_clock::now();
		for (int i = 1; i <= std::log2(n); i++) // log2(n) times loop.
			BitonicSort << < blocks, threads >> > (d_a, n, i);
		te = steady_clock::now();
	}
	else
	{
		std::cout << "long kernel using GPU" << std::endl;
		ts = steady_clock::now();
		for (int i = 1; i <= std::log2(n); i++) // if large number, loops are outside of kernel
			for (int j = 1; j <= i; j++)
				BitonicSortLarge << < blocks, threads >> > (d_a, n, i, j);
		te = steady_clock::now();
	}
	reportTime("bitonic sort", te - ts);
	// end bitonic


	std::cout << "Memcpy Device to Host" << std::endl;
	ts = steady_clock::now();
	cudaMemcpy(h_a, d_a, n * sizeof(int), cudaMemcpyDeviceToHost);
	te = steady_clock::now();
	reportTime("Memcpy Device to Host", te - ts);
	cudaFree(d_a);
	cudaDeviceReset();
}

int main(int argc, char* argv[])
{
	if (argc != 2) {
		std::cerr << "Enter length of data ";
		return 1;
	}
	steady_clock::time_point ts, te;
	int n = std::atoi(argv[1]);
	int nblks = n/ntpb;

	// array with generated random number.
	std::cout << "array n = " << n << std::endl;

	std::cout << "generate number" << std::endl;
	ts = steady_clock::now();
	int* bGPU = new int[n];
	int* qCPU = new int[n];
	int* bCPU = new int[n];
	thrust::host_vector<int> h_vec(n);
	generateRandom(bGPU, qCPU, bCPU, h_vec, n);
	te = steady_clock::now();
	reportTime("generate number", te - ts);

	bitonicGPU(bGPU, n, nblks);

	// quick sort using CPU
	std::cout << "quick sort using CPU" << std::endl;
	ts = steady_clock::now();
	quickSort(qCPU, 0, n - 1);
	te = steady_clock::now();
	reportTime("quick sort", te - ts);
	// end quick sort

	// bitonic sort using CPU
	std::cout << "bitonic sort using CPU" << std::endl;
	ts = steady_clock::now();
	bitonicSort(bCPU, n, 1);
	te = steady_clock::now();
	reportTime("bitonic sort", te - ts);
	// end bitonic sort

	// sort using thrust
	std::cout << "sort using thrust" << std::endl;
	thrust::device_vector<int> d_vec = h_vec;
	ts = steady_clock::now();
	thrust::sort(d_vec.begin(), d_vec.end());
	te = steady_clock::now();
	reportTime("thrust sort", te - ts);
	thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
	// end thrust sort

	// check correctness
	std::cout << "correctness test" << std::endl;
	ts = steady_clock::now();
	for (int i = 0; i < n; i++)
	{
		if (bGPU[i] != qCPU[i] || bGPU[i] != bCPU[i] || bGPU[i] != h_vec[i])
		{
			//std::cout << "Wrong" << std::endl;
			std::cout << bGPU[i] << " != " << qCPU[i] <<" != "<< bCPU[i] <<" != "<< h_vec[i] << std::endl;
			//break;
		}
	}
	te = steady_clock::now();
	reportTime("correctness test", te - ts);

	//cudaDeviceReset();
	// end
	delete[] bGPU;
	delete[] qCPU;
	delete[] bCPU;

	getchar();
	return 0;
}