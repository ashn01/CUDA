#pragma once
#include <iostream>
/////////////////// bitonic sort CPU https://www.geeksforgeeks.org/bitonic-sort/

void compAndSwap(int a[], int i, int j, int dir);

void bitonicMerge(int a[], int low, int cnt, int dir);
/* This function first produces a bitonic sequence by recursively
sorting its two halves in opposite sorting orders, and then
calls bitonicMerge to make them in the same order */
void bitonicSortMain(int a[], int low, int cnt, int dir);

/////////////////// bitonic sort CPU

/* Caller of bitonicSort for sorting the entire array of
length N in ASCENDING order */
void bitonicSort(int a[], int N, int up);
