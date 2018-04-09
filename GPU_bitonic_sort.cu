#include <chrono>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <iomanip>
// CUDA header file
#include <cuda_runtime.h>

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
	int group = (i / (1<<(stage-1))) % 2; // 0 asc, 1 desc
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

void reportTime(const char* msg, steady_clock::duration span) {
	auto ms = duration_cast<milliseconds>(span);
	std::cout << msg << " - took - " <<
		ms.count() << " millisecs" << std::endl;
}

void generateRandom(int array[], int size)
{
	for (int i = 0; i < size; i++)
	{
		array[i] = rand();
	}
}

/////////////////// quick sort
// copied from https://www.geeksforgeeks.org/quick-sort/
void swap(int* a, int* b)
{
	int t = *a;
	*a = *b;
	*b = t;
}

/* This function takes last element as pivot, places
the pivot element at its correct position in sorted
array, and places all smaller (smaller than pivot)
to left of pivot and all greater elements to right
of pivot */
int partition(int arr[], int low, int high)
{
	int pivot = arr[high];    // pivot
	int i = (low - 1);  // Index of smaller element

	for (int j = low; j <= high - 1; j++)
	{
		// If current element is smaller than or
		// equal to pivot
		if (arr[j] <= pivot)
		{
			i++;    // increment index of smaller element
			swap(&arr[i], &arr[j]);
		}
	}
	swap(&arr[i + 1], &arr[high]);
	return (i + 1);
}

/* The main function that implements QuickSort
arr[] --> Array to be sorted,
low  --> Starting index,
high  --> Ending index */
void quickSort(int arr[], int low, int high)
{
	if (low < high)
	{
		/* pi is partitioning index, arr[p] is now
		at right place */
		int pi = partition(arr, low, high);

		// Separately sort elements before
		// partition and after partition
		quickSort(arr, low, pi - 1);
		quickSort(arr, pi + 1, high);
	}
}
/////////////////// quick sort

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
	ts = steady_clock::now();
	int* array = new int[n];
	generateRandom(array, n);
	te = steady_clock::now();
	reportTime("generate number", te - ts);

	int* d_a = nullptr;

	cudaMalloc((void**)&d_a, n * sizeof(int));
	cudaMemcpy(d_a, array, n * sizeof(int), cudaMemcpyHostToDevice);

	// thread is required n/2, so block is n/2
	dim3 blocks(nblks/2, 1);
	dim3 threads(ntpb, 1);

	// bitonic sort using cuda
	if (n <= 4096)
	{
		std::cout << "short kernel"<< std::endl;
		ts = steady_clock::now();
		for (int i = 1; i <= std::log2(n); i++) // log2(n) times loop.
			BitonicSort << < blocks, threads >> > (d_a, n, i);
		te = steady_clock::now();
	}
	else
	{
		std::cout << "long kernel"<< std::endl;
		ts = steady_clock::now();
		for (int i = 1; i <= std::log2(n); i++) // if large number, loops are outside of kernel
			for (int j = 1; j <= i; j++)
				BitonicSortLarge << < blocks, threads >> > (d_a, n, i, j);
		te = steady_clock::now();
	}
	reportTime("bitonic sort", te - ts);
	// end bitonic

	int* h_b = new int[n];
	cudaMemcpy(h_b, d_a, n * sizeof(int), cudaMemcpyDeviceToHost);

	// quick sort using CPU
	ts = steady_clock::now();
	quickSort(array, 0, n - 1);
	te = steady_clock::now();
	reportTime("bubble sort", te - ts);
	// end quick sort

	// check correctness
	std::cout << "correctness test" << std::endl;

	ts = steady_clock::now();
	for (int i = 0; i < n; i++)
	{
		if (h_b[i] != array[i])
		{
			std::cout << h_b[i] << " = " << array[i] << std::endl;
			break;
		}
	}
	te = steady_clock::now();
	reportTime("correctness test", te - ts);

	// end
	cudaFree(d_a);
	delete[] array;
	delete[] h_b;

	return 0;
}