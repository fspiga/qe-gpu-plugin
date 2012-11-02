/*
 * Copyright (C) 2001-2013 Quantum ESPRESSO Foundation
 *
 * This file is distributed under the terms of the
 * GNU General Public License. See the file `License'
 * in the root directory of the present distribution,
 * or http://www.gnu.org/copyleft/gpl.txt .
 *
 */

#include <stdio.h>

#include "cuda_env.h"

extern "C" void qvan2_(int * ptr_ngm, int * iih, int * jjh, int * ptr_nt, double * qmod, double * qgm, double * ylmk0);

extern "C" void start_clock_(char * label, unsigned int length_arg );
extern "C" void stop_clock_(char * label, unsigned int length_arg );


__global__ void kernel_compute_aux( const double * __restrict eigts1, const double * __restrict eigts2, const double * __restrict eigts3,
		const int * __restrict ig1, const int * __restrict ig2, const int * __restrict ig3,  const int nr1, const int nr2, const int nr3,
		const double * __restrict qgm, const double * __restrict becsum, double * aux, const int na, const int nspin_mag, const int ngm,
		const int first_becsum, const int ijh, const int nat, const int * __restrict ityp, const int nt )
{
	register int is = blockIdx.x * blockDim.x + threadIdx.x;
	register int ig = blockIdx.y * blockDim.y + threadIdx.y;
	register int threads_id = threadIdx.y * blockDim.x + threadIdx.x;
	
	double eigts1_local[2], eigts2_local[2], eigts3_local[2], skk[2], element[2], sup, s_qgm[2], aux_sup[2];
	int s_ig1, s_ig2, s_ig3;
	
	__shared__ int s_ityp[1024];
	
	if( threads_id < nat )
		s_ityp[ threads_id ] = ityp[threads_id];
	
	if( ig < ngm && is < nspin_mag ) {
	
		s_qgm[ 0 ] = qgm[ ig * 2 ];
		s_qgm[ 1 ] = qgm[ ( ig * 2 ) + 1];
		aux_sup[0] = aux[ ( ig + (is * ngm) ) * 2 ];
		aux_sup[1] = aux[ ( ( ig + (is * ngm) ) * 2 ) + 1 ];
		s_ig1 = ig1[ig];
		s_ig2 = ig2[ig];
		s_ig3 = ig3[ig];

		for ( int ina = 0; ina < nat; ina++) {

			if ( s_ityp[ina] == nt ) {

			eigts1_local[0] = eigts1[ ( ( (nr1 + s_ig1) + ( ina * ( nr1 * 2 + 1 ) ) ) * 2 ) ];
			eigts1_local[1] = eigts1[ ( ( (nr1 + s_ig1) + ( ina * ( nr1 * 2 + 1 ) ) ) * 2 ) + 1 ];

			eigts2_local[0] = eigts2[ ( ( (nr2 + s_ig2) + ( ina * ( nr2 * 2 + 1 ) ) ) * 2 ) ];
			eigts2_local[1] = eigts2[ ( ( (nr2 + s_ig2) + ( ina * ( nr2 * 2 + 1 ) ) ) * 2 ) + 1 ];

			element[0] = eigts1_local[0] * eigts2_local[0] - eigts1_local[1] * eigts2_local[1];
			element[1] = eigts1_local[0] * eigts2_local[1] + eigts1_local[1] * eigts2_local[0];

			eigts3_local[0] = eigts3[ ( ( (nr3 + s_ig3) + ( ina * ( nr3 * 2 + 1 ) ) ) * 2 ) ];
			eigts3_local[1] = eigts3[ ( ( (nr3 + s_ig3) + ( ina * ( nr3 * 2 + 1 ) ) ) * 2 ) + 1 ];

			skk[0] = element[0] * eigts3_local[0] - element[1] * eigts3_local[1];
			skk[1] = element[1] * eigts3_local[0] + element[0] * eigts3_local[1];

			sup = becsum[ ijh + ( ina * first_becsum ) + ( is * first_becsum * nat ) ];

			skk[0] = skk[0] * sup;
			skk[1] = skk[1] * sup;

			element[0] = skk[0] * s_qgm[ 0 ] - skk[1] * s_qgm[ 1 ];
			aux_sup[0] = aux_sup[0] + element[0];

			element[1] = skk[1] * s_qgm[ 0 ] + skk[0] * s_qgm[ 1 ];
			aux_sup[1] = aux_sup[1] + element[1];

			}

		}

		aux[ ( ig + (is * ngm) ) * 2 ] = aux_sup[0];
		aux[ ( ( ig + (is * ngm) ) * 2 ) + 1 ] = aux_sup[1];

	}
}

extern "C" int addusdens_cuda_(int * ptr_nr1, int * ptr_nr2, int * ptr_nr3, int * ptr_first_becsum,
		int * ptr_nat, int * nh, int * ptr_nt, int * ptr_ngm, double * qmod, double * qgm,
		double * ylmk0, double * eigts1, double * eigts2, double * eigts3, int * ig1, int * ig2,
		int * ig3, double * aux, double * becsum, int * ityp, int * ptr_nspin_mag, int * ptr_nspin)
{
	int ijh, ih, jh, na, iih, jjh;

	void * ityp_D, * qgm_D, * becsum_D, * aux_D;
    void * eigts1_D, * eigts2_D, * eigts3_D;
    void * ig1_D, * ig2_D, * ig3_D;

	int nat = (* ptr_nat);
	int nt = (* ptr_nt);
	int ngm = (* ptr_ngm);
	int nspin_mag = (* ptr_nspin_mag);
	int nr1 = (* ptr_nr1);
	int nr2 = (* ptr_nr2);
	int nr3 = (* ptr_nr3);
	int nspin = (* ptr_nspin);
	int first_becsum = (* ptr_first_becsum );

	ijh = 0;

#if defined(__CUDA_DEBUG)
	printf("\n[ADDUSDENS] Enter \n");fflush(stdout);
#endif

	int number_of_block = (ngm + __CUDA_TxB_ADDUSDENS_COMPUTE_AUX__ - 1) / __CUDA_TxB_ADDUSDENS_COMPUTE_AUX__;
	if ( number_of_block > 65535) {
		fprintf( stderr, "\n[NEWD] kernel_compute_aux cannot run, blocks requested ( %d ) > blocks allowed!!!", number_of_block );
		return 1;
	}

	cudaSetDevice(qe_gpu_bonded[0]);

#if defined(__CUDA_NOALLOC)
	/* Do real allocation */
	int ierr = cudaMalloc ( (void**) &(qe_dev_scratch[0]), (size_t) qe_gpu_mem_unused[0] );
	if ( ierr != cudaSuccess) {
		fprintf( stderr, "\nError in memory allocation, program will be terminated (%d)!!! Bye...\n\n", ierr );
		exit(EXIT_FAILURE);
	}
#endif

	size_t shift = 0;
	aux_D = (char*) qe_dev_scratch[0] + shift;
	shift += ( ngm * nspin_mag * 2 )*sizeof(double);
	becsum_D = (char*) qe_dev_scratch[0] + shift;
	shift += ( first_becsum * nat * nspin )*sizeof(double);
	qgm_D = (char*) qe_dev_scratch[0] + shift;
	shift += ( ngm * 2 )*sizeof(double);
	ityp_D = (char*) qe_dev_scratch[0] + shift;
	shift += ( (nat%2==0 ? nat : nat+1) )*sizeof( int );
	eigts1_D = (char*) qe_dev_scratch[0] + shift;
	shift += ( ( ( nr1 * 2 + 1 ) * nat ) * 2 )*sizeof(double);
	eigts2_D = (char*) qe_dev_scratch[0] + shift;
	shift += ( ( ( nr2 * 2 + 1 ) * nat ) * 2 )*sizeof(double);
	eigts3_D = (char*) qe_dev_scratch[0] + shift;
	shift += ( ( ( nr3 * 2 + 1 ) * nat ) * 2 )*sizeof(double);
	ig1_D = (char*) qe_dev_scratch[0] + shift;
	shift += ( (ngm%2==0) ? ngm : ngm+1 )*sizeof(int);
	ig2_D = (char*) qe_dev_scratch[0] + shift;
	shift += ( (ngm%2==0) ? ngm : ngm+1 )*sizeof(int);
	ig3_D = (char*) qe_dev_scratch[0] + shift;
	shift += ( (ngm%2==0) ? ngm : ngm+1 )*sizeof(int);

	// 	shift contains the amount of byte required on the GPU to compute
	if ( shift > qe_gpu_mem_unused[0] ) {
		fprintf( stderr, "\n[ADDUSDENS] Problem don't fit in GPU memory, requested ( %lu ) > memory allocated  (%lu )!!!", shift, qe_gpu_mem_unused[0] );
#if defined(__CUDA_NOALLOC)
		/* Deallocating... */
		ierr = cudaFree ( qe_dev_scratch[0] );
		if(ierr != cudaSuccess) {
			fprintf( stderr, "\nError in memory release, program will be terminated!!! Bye...\n\n" );
			exit(EXIT_FAILURE);
		}
#endif
		return 1;
	}

	qecudaSafeCall( cudaMemcpy( (double *) aux_D, aux,  sizeof( double ) * ( ngm * nspin_mag * 2 ), cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( (double *) becsum_D, becsum,  sizeof( double ) * ( first_becsum * nat * nspin ), cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( (int * ) ityp_D, ityp,  sizeof( int ) * nat, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( ig1_D, ig1,  sizeof( int ) * ngm, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( ig2_D, ig2,  sizeof( int ) * ngm, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( ig3_D, ig3,  sizeof( int ) * ngm, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( eigts1_D, eigts1,  sizeof( double ) * ( ( ( nr1 * 2 + 1 ) * nat ) * 2 ), cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( eigts2_D, eigts2,  sizeof( double ) * ( ( ( nr2 * 2 + 1 ) * nat ) * 2 ), cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( eigts3_D, eigts3,  sizeof( double ) * ( ( ( nr3 * 2 + 1 ) * nat ) * 2 ), cudaMemcpyHostToDevice ) );

	dim3 threads2(1, __CUDA_TxB_ADDUSDENS_COMPUTE_AUX__);
	dim3 grid2( nspin_mag / 1 ? nspin_mag / 1 : 1,
			number_of_block ? number_of_block : 1);

	qecudaSafeCall( cudaFuncSetCacheConfig(kernel_compute_aux, cudaFuncCachePreferShared) );

	for( ih = 0, iih = 1; ih < nh[nt - 1]; ih++, iih++) {
		for( jh = ih, jjh = iih; jh < nh[nt - 1]; jh++, jjh++, ijh++ ) {

			qvan2_(ptr_ngm, &iih, &jjh, ptr_nt, qmod, qgm, ylmk0);
			// Protective guard....
			cudaDeviceSynchronize();

			qecudaSafeCall( cudaMemcpy( (double *) qgm_D, qgm,  sizeof( double ) * ngm * 2, cudaMemcpyHostToDevice ) );

			kernel_compute_aux<<<grid2, threads2>>>( (double *) eigts1_D,
					(double *) eigts2_D, (double *) eigts3_D,
					(int *)  ig1_D, (int *)  ig2_D,
					(int *)  ig3_D, nr1, nr2, nr3, (double *) qgm_D,
					(double *) becsum_D, (double *) aux_D, na, nspin_mag, ngm,
					first_becsum, ijh, nat, (int *) ityp_D, nt );
			qecudaGetLastError("kernel launch failure");
//			cudaDeviceSynchronize();

		}
	}

	qecudaSafeCall( cudaMemcpy( aux, (double *) aux_D, sizeof( double ) * ( ngm * nspin_mag * 2 ), cudaMemcpyDeviceToHost ) );
//	qecudaSafeCall( cudaFuncSetCacheConfig(kernel_compute_aux, cudaFuncCachePreferNone) );

#if defined(__CUDA_NOALLOC)
	/* Deallocating... */
	ierr = cudaFree ( qe_dev_scratch[0] );
	if(ierr != cudaSuccess) {
		fprintf( stderr, "\nError in memory release, program will be terminated!!! Bye...\n\n" );
		exit(EXIT_FAILURE);
	}

#else

#if defined(__CUDA_KERNEL_MEMSET)
	qecudaSafeCall( cudaMemset( qe_dev_scratch[0], 0, (size_t) qe_gpu_mem_unused[0] ) );
#endif

#endif
	return 0;
}
