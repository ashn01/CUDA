#include "bitonicCPU.h"
/////////////////// bitonic sort CPU https://www.geeksforgeeks.org/bitonic-sort/

void compAndSwap(int a[], int i, int j, int dir)
{
	if (dir == (a[i]>a[j]))
		std::swap(a[i], a[j]);
}

void bitonicMerge(int a[], int low, int cnt, int dir)
{
	if (cnt>1)
	{
		int k = cnt / 2;
		for (int i = low; i<low + k; i++)
			compAndSwap(a, i, i + k, dir);
		bitonicMerge(a, low, k, dir);
		bitonicMerge(a, low + k, k, dir);
	}
}

/* This function first produces a bitonic sequence by recursively
sorting its two halves in opposite sorting orders, and then
calls bitonicMerge to make them in the same order */
void bitonicSortMain(int a[], int low, int cnt, int dir)
{
	if (cnt>1)
	{
		int k = cnt / 2;

		// sort in ascending order since dir here is 1
		bitonicSortMain(a, low, k, 1);

		// sort in descending order since dir here is 0
		bitonicSortMain(a, low + k, k, 0);

		// Will merge wole sequence in ascending order
		// since dir=1.
		bitonicMerge(a, low, cnt, dir);
	}
}

/////////////////// bitonic sort CPU

/* Caller of bitonicSort for sorting the entire array of
length N in ASCENDING order */
void bitonicSort(int a[], int N, int up)
{
	bitonicSortMain(a, 0, N, up);
}
