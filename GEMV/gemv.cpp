/*
 * Matrix vector multiplication with multiple tasklet
 *
 */
#include <stdint.h>
#include <stdio.h>
#include <getopt.h>
#include <unistd.h>
#include <omp.h>
#include <ctime>
#include <iomanip>
#include "../../../misc/hooks/zsim_hooks.h"
#include "../common.h"
#include "../timer.h"
#include "params.h"
using namespace std;

static T* A;
static T* B;
static T* C;
static T* C_dpu;

// Create input arrays
static void init_data(T* A, T* B, unsigned int m_size, unsigned long n_size) {
	srand(0);

	for (unsigned long i = 0; i < m_size * n_size; i++)
	{
		A[i] = (unsigned int) (rand()%50);
	}

	for (unsigned long i = 0; i < n_size; i++)
	{
		B[i] = (unsigned int) (rand()%50);
	}
}

static void gemv(T *bufferC, T *bufferA, T *bufferB, int pos, unsigned long n_size) {
	for (unsigned long i = 0; i < n_size; i++) {
		bufferC[pos] += bufferA[i] * bufferB[i];
	}
	return;
}

int main(int argc, char **argv) {
	struct Params p = input_params(argc, argv);
	unsigned int i;
	unsigned int n_threads = p.n_threads;
	omp_set_num_threads(n_threads);	
	printf("Number of threads %d \n", omp_get_max_threads());
	unsigned long m_size = p.m_size;
	unsigned long n_size = p.n_size;
	uint32_t *prev_rows_dpu = new uint32_t[n_threads];
	uint32_t *rows_per_dpu = new uint32_t[n_threads];
	uint32_t chunks = m_size / n_threads;
	uint32_t rest_rows = m_size % n_threads;
	prev_rows_dpu[0] = 0;
	for(int i = 0; i < n_threads; i++) {
		if (i < rest_rows) {
			rows_per_dpu[i] = chunks + 1;
		} else {
			rows_per_dpu[i] = chunks;
		}
		if (rest_rows > 0) {
			if (i >= rest_rows)
				prev_rows_dpu[i] = rest_rows * (chunks + 1) + (i - rest_rows) * chunks;
			else
				prev_rows_dpu[i] = i * (chunks + 1);
		} else {
			prev_rows_dpu[i] = i * chunks;
		}
	}
	A = new T[m_size * n_size];
	B = new T[n_size];
	C = new T[m_size];

	init_data(A, B, m_size, n_size);

	// Print Matrices A, B and C before calculation
	// printf("\nMatrix A:\n");
	// for(unsigned long i = 0; i < m_size * n_size; i++) {
	// 	printf("%d ", (unsigned int)A[i]);
	// 	if((i+1) % n_size == 0) {
	// 		printf("\n");
	// 	}
	// }
	// printf("\nMatrix B:\n");
	// for(unsigned long i = 0; i < n_size; i++) {
	// 	printf("%d\n", (unsigned int)B[i]);
	// }
	// printf("\nMatrix C:\n");
	// for(unsigned long i = 0; i < m_size; i++) {
	// 	printf("%d ", (unsigned int)C[i]);
	// }
	// printf("\n");

	#pragma omp target data map(to: n_threads, n_size, rows_per_dpu[0:n_threads], prev_rows_dpu[0:n_threads], A[0:m_size*n_size], B[0:n_size], C[0:m_size])
	{
		for (unsigned int rep = 0; rep < p.n_warmup; rep++) {
			#pragma omp target
            #pragma omp parallel for
			for(int tid = 0; tid < n_threads; tid++) {
				zsim_PIM_function_begin();
				uint32_t rows_per_tasklet = rows_per_dpu[tid];
				uint32_t start_row = prev_rows_dpu[tid];
				T* bufferA = &(A[start_row * n_size]);
				for(int pos = start_row; pos < start_row + rows_per_tasklet; pos++) {
					C[pos] = 0;
					gemv(C, bufferA, B, pos, n_size);
					bufferA = bufferA + n_size;
				}
				zsim_PIM_function_end();
			}
		}

		zsim_roi_begin();
		for (unsigned int rep = 0; rep < p.n_reps; rep++) {
			#pragma omp target
            #pragma omp parallel for
			for(int tid = 0; tid < n_threads; tid++) {
				zsim_PIM_function_begin();
				uint32_t rows_per_tasklet = rows_per_dpu[tid];
				uint32_t start_row = prev_rows_dpu[tid];
				T* bufferA = &(A[start_row * n_size]);
				for(int pos = start_row; pos < start_row + rows_per_tasklet; pos++) {
					C[pos] = 0;
					gemv(C, bufferA, B, pos, n_size);
					bufferA = bufferA + n_size;
				}
				zsim_PIM_function_end();
			}
		}
		zsim_roi_end(); 
	}

	// Print Matrix C after calculation
	// printf("\nMatrix C:\n");
	// for(unsigned long i = 0; i < m_size; i++) {
	// 	printf("%d ", (unsigned int)C[i]);
	// }
	// printf("\n");

	delete[] A;
	delete[] B;
	delete[] C;
	delete[] prev_rows_dpu;
	delete[] rows_per_dpu;
	return 0;
}