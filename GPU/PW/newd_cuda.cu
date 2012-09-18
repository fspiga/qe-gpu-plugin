/*
 * Copyright (C) 2001-2012 Quantum ESPRESSO group
 * Copyright (C) 2010-2011 Irish Centre for High-End Computing (ICHEC)
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

__device__ inline void complex_by_complex_device( const  double * __restrict A, const  double * __restrict B, double * C)
{
	double re_a = A[0], re_b = B[0];
	double img_a = A[1], img_b = B[1];

	C[0] = (re_a * re_b) - (img_a * img_b);
	C[1] = (re_a * img_b) + (re_b * img_a);
}

__global__ void kernel_compute_qgm_na( const  double * __restrict eigts1, const  double * __restrict eigts2, const  double * __restrict eigts3,
		const  int * __restrict ig1, const  int * __restrict ig2, const  int * __restrict ig3, const  double * __restrict qgm, const int nr1,
		const int nr2, const int nr3, const int na, const int ngm, double * qgm_na )
{
	int global_index = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

	int ind_eigts1, ind_eigts2, ind_eigts3;
	double sup_prod_1[2], sup_prod_2[2];

	if( global_index < ngm ){

		ind_eigts1 = ( ( ( nr1 + ig1[global_index] ) + ( na * ( nr1 * 2 + 1 ) ) ) * 2 );
		ind_eigts2 = ( ( ( nr2 + ig2[global_index] ) + ( na * ( nr2 * 2 + 1 ) ) ) * 2 );
		ind_eigts3 = ( ( ( nr3 + ig3[global_index] ) + ( na * ( nr3 * 2 + 1 ) ) ) * 2 );

		complex_by_complex_device( &eigts1[ ind_eigts1 ], &eigts2[ ind_eigts2 ], sup_prod_1 );
		complex_by_complex_device( sup_prod_1, &eigts3[ ind_eigts3 ], sup_prod_2 );
		complex_by_complex_device( sup_prod_2,  &qgm[ global_index * 2 ],  &qgm_na[ global_index * 2 ] );

	}
}

__global__ void kernel_compute_deeq( const double * qgm, double * deeq, const double * aux,
		const int na, const int nspin_mag, const int ngm, const int nat, const int flag,
		const int ih, const int jh, const int nhm, const double omega, const double fact,
		const  double * __restrict qgm_na, double * dtmp )
{
	int global_index = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	double sup_prod_1[2];

	if( global_index < nspin_mag ){

		int index = ih + ( jh * nhm ) + ( na * nhm * nhm ) + ( global_index * nhm * nhm * nat );
		int rev_index = jh + ( ih * nhm ) + ( na * nhm * nhm ) + ( global_index * nhm * nhm * nat );
		double temp = fact * omega * dtmp[global_index];

		if ( flag ) {
			complex_by_complex_device( &aux[ global_index * ngm * 2 ], qgm_na, sup_prod_1 );
			temp -= omega * sup_prod_1[0];
		}

		deeq[ rev_index ] = temp;
		deeq[ index ] = temp;
	}
}

extern "C" int newd_cuda_( int * ptr_nr1, int * ptr_nr2, int * ptr_nr3, int * ptr_na, int * nh,
		double * ptr_fact, int * ptr_nt, int * ptr_nat, int * ptr_ngm, int * ptr_nhm, int * ptr_nspin,
		double * qmod, double * ylmk0, double * eigts1, double * eigts2, double * eigts3, int * ig1,
		int * ig2, int * ig3, double * deeq, int * ityp, double * ptr_omega, int * ptr_flag,
		double * aux, int * ptr_nspin_mag)
{
	int ih, jh, jjh, iih, is;
	double fact = (* ptr_fact);
	int nt = (* ptr_nt);
	int na = (* ptr_na);
	int ngm = (* ptr_ngm);
	int nhm = (* ptr_nhm);
	int nspin = (* ptr_nspin);
	int nat = (* ptr_nat);
	int nr1 = (* ptr_nr1);
	int nr2 = (* ptr_nr2);
	int nr3 = (* ptr_nr3);
	int flag = (* ptr_flag);
	double omega = (* ptr_omega);
	int nspin_mag = (* ptr_nspin_mag);

	double * qgm_na;

	cudaStream_t  vlocStreams[ MAX_QE_GPUS ];
	cublasHandle_t vlocHandles[ MAX_QE_GPUS ];

//	size_t buffer_size = 0L;

	int blocksPerGrid;

	double * qgm;

	void * qgm_D, * deeq_D, * aux_D, * dtmp_D, * qgm_na_D;
    void * local_eigts1_D, * local_eigts2_D, * local_eigts3_D;
    void * local_ig1_D, * local_ig2_D, * local_ig3_D;

//#if defined(__CUDA_PRELOADING_DATA)
//	buffer_size =  sizeof(double) + ngm * 5 +
//			sizeof( double ) * ( ngm * nspin_mag * 2 ) +
//			sizeof( double ) * ( nhm * nhm * nat * nspin );
//#else
//	buffer_size =  sizeof(double) + ngm * 5 +
//			sizeof( double ) * ( ( ( nr1 * 2 + 1 ) * nat ) * 2 ) +
//			sizeof( double ) * ( ( ( nr2 * 2 + 1 ) * nat ) * 2 ) +
//			sizeof( double ) * ( ( ( nr3 * 2 + 1 ) * nat ) * 2 ) +
//			sizeof( double ) * ( ngm * nspin_mag * 2 ) +
//			sizeof( double ) * ( nhm * nhm * nat * nspin );
//#endif
//
//	if ( buffer_size > cuda_memory_unused[0] ) {
//		fprintf( stderr, "\n[NEWD] Problem don't fit in GPU memory, memory requested ( %lu ) > memory allocated  (%lu )!!!", buffer_size, cuda_memory_allocated[0] );
//		return 1;
//	}

	if ( ((nspin_mag * ngm) / __CUDA_TxB_NEWD_QGM__) > 65535) {
		fprintf( stderr, "\n[NEWD] kernel_compute_qgm_na cannot run, blocks requested ( %d ) > blocks allowed!!!", (nspin_mag * ngm * 2 / __CUDA_TxB_NEWD_QGM__) );
		return 1;
	}

	qgm_na = (double *) malloc( ngm * 2 * sizeof(double) );

	cudaSetDevice(qe_gpu_bonded[0]);

	if ( cublasCreate( &vlocHandles[ 0 ] ) != CUBLAS_STATUS_SUCCESS ) {
		printf("\n*** CUDA NEWD *** ERROR *** cublasInit() for device %d failed!",0);
		fflush(stdout);
		exit(EXIT_FAILURE);
	}

	if( cudaStreamCreate( &vlocStreams[ 0 ] ) != cudaSuccess ) {
		printf("\n*** CUDA NEWD *** ERROR *** creating stream for device %d failed!", 0);
		fflush(stdout);
		exit(EXIT_FAILURE);
	}

	size_t shift = 0;
	dtmp_D = (char*) dev_scratch_QE[0] + shift;
	shift += ( nspin_mag )*sizeof(double);
	aux_D = (char*) dev_scratch_QE[0] + shift;
	shift += ( ( ngm * nspin_mag ) * 2  )*sizeof(double);
	qgm_na_D = (char*) dev_scratch_QE[0] + shift;
	shift += ( ngm * 2 )*sizeof(double);
	qgm_D = (char*) dev_scratch_QE[0] + shift;
	shift += ( ngm * 2 )*sizeof(double);
	deeq_D = (char*) dev_scratch_QE[0] + shift;
	shift += ( nhm * nhm * nat * nspin )*sizeof( double );
#if defined(__CUDA_PRELOADING_DATA)
	// now	shift contains the amount of byte required on the GPU to compute
	local_eigts1_D = (void *) preloaded_eigts1_D;
	local_eigts2_D = (void *) preloaded_eigts2_D;
	local_eigts3_D = (void *) preloaded_eigts3_D;
	local_ig1_D = (void *) preloaded_ig1_D;
	local_ig2_D = (void *) preloaded_ig2_D;
	local_ig3_D = (void *) preloaded_ig3_D;
#else
	shift += ( nhm * nhm * nat * nspin )*sizeof( double );
	local_eigts1_D = (char*) dev_scratch_QE[0] + shift;
	shift += ( ( ( nr1 * 2 + 1 ) * nat ) * 2 )*sizeof(double);
	local_eigts2_D = (char*) dev_scratch_QE[0] + shift;
	shift += ( ( ( nr2 * 2 + 1 ) * nat ) * 2 )*sizeof(double);
	local_eigts3_D = (char*) dev_scratch_QE[0] + shift;
	shift += ( ( ( nr3 * 2 + 1 ) * nat ) * 2 )*sizeof(double);
	local_ig1_D = (char*) dev_scratch_QE[0] + shift;
	shift += ( (ngm%2==0) ? ngm : ngm+1 )*sizeof(int);
	local_ig2_D = (char*) dev_scratch_QE[0] + shift;
	shift += ( (ngm%2==0) ? ngm : ngm+1 )*sizeof(int);
	local_ig3_D = (char*) dev_scratch_QE[0] + shift;
	shift += ( (ngm%2==0) ? ngm : ngm+1 )*sizeof(int);
#endif
	// now	shift contains the amount of byte required on the GPU to compute

	if ( shift > cuda_memory_unused[0] ) {
		fprintf( stderr, "\n[NEWD] Problem don't fit in GPU memory, memory requested ( %lu ) > memory allocated  (%lu )!!!", shift, cuda_memory_allocated[0] );
		return 1;
	}

	qecudaSafeCall( cudaHostAlloc( (void**) &qgm,  ngm * 2 * sizeof(double), cudaHostAllocDefault ) );

	// Before do anything force sync to terminate async data transfer
#if defined(__CUDA_PRELOADING_DATA) && defined(__CUDA_PRELOAD_PINNED)
	cudaDeviceSynchronize();
#endif

	qecudaSafeCall( cudaMemcpy( (double *) aux_D, aux,  sizeof( double ) * ( ngm * nspin_mag * 2 ), cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( (double *) deeq_D, deeq,  sizeof( double ) * ( nhm * nhm * nat * nspin ), cudaMemcpyHostToDevice ) );
#if !defined(__CUDA_PRELOADING_DATA)
	qecudaSafeCall( cudaMemcpy( local_ig1_D, ig1,  sizeof( int ) * ngm, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( local_ig2_D, ig2,  sizeof( int ) * ngm, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( local_ig3_D, ig3,  sizeof( int ) * ngm, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( local_eigts1_D, eigts1,  sizeof( double ) * ( ( ( nr1 * 2 + 1 ) * nat ) * 2 ), cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( local_eigts2_D, eigts2,  sizeof( double ) * ( ( ( nr2 * 2 + 1 ) * nat ) * 2 ), cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( local_eigts3_D, eigts3,  sizeof( double ) * ( ( ( nr3 * 2 + 1 ) * nat ) * 2 ), cudaMemcpyHostToDevice ) );
#endif

	qecudaSafeCall( cudaMemset( (double *) qgm_na_D, 0, sizeof( double ) * ngm * 2  ) );
	qecudaSafeCall( cudaMemset( (double *) dtmp_D, 0, sizeof( double ) * nspin_mag ) );
	
	cublasSetPointerMode(vlocHandles[ 0 ] , CUBLAS_POINTER_MODE_DEVICE);

	for( ih = 0, iih = 1; ih < nh[nt - 1]; ih++, iih++ )
	{
		for( jh = ih, jjh = iih; jh < nh[nt - 1]; jh++, jjh++ )
		{

			qvan2_(ptr_ngm, &iih, &jjh, ptr_nt, qmod, qgm, ylmk0);

			qecudaSafeCall( cudaMemcpy( (double *) qgm_D, qgm,  sizeof( double ) * ngm * 2, cudaMemcpyHostToDevice ) );

			for( na = 0;  na < nat; na++ ){

				if( ityp[na] == nt ) {

#if defined(__CUDA_DEBUG)
					printf("\n[DEBUG] kernel_compute_qgm_na= ih:%d, jh:%d, na:%d, ngm:%d\n", ih, jh, na, ngm);fflush(stdout);
#endif
					blocksPerGrid = ( (nspin_mag * ngm ) + __CUDA_TxB_NEWD_QGM__ - 1) / __CUDA_TxB_NEWD_QGM__;
					kernel_compute_qgm_na<<<blocksPerGrid, __CUDA_TxB_NEWD_QGM__>>>( (double *) local_eigts1_D, (double *) local_eigts2_D, (double *) local_eigts3_D, (int *) local_ig1_D, (int *) local_ig2_D, (int *) local_ig3_D, (double *) qgm_D, nr1, nr2, nr3, na, ngm, (double *) qgm_na_D );
					qecudaGetLastError("kernel kernel_compute_qgm_na launch failure");

					for( is = 0; is < nspin_mag; is++ ){
						cublasDdot(vlocHandles[ 0 ] , ngm * 2, (double *) aux_D + (is * ngm * 2), 1, (double *) qgm_na_D, 1, (double *) dtmp_D + is );
					}

#if defined(__CUDA_DEBUG)
					printf("\n[DEBUG] kernel_compute_deeq= ih:%d, jh:%d, na:%d, ngm:%d\n", ih, jh, na, ngm);fflush(stdout);
#endif

					blocksPerGrid = ( (nspin_mag) + __CUDA_TxB_NEWD_DEEPQ__ - 1) / __CUDA_TxB_NEWD_DEEPQ__;
					kernel_compute_deeq<<<blocksPerGrid, __CUDA_TxB_NEWD_DEEPQ__>>>( (double *) qgm_D, (double *) deeq_D, (double *) aux_D, na, nspin_mag, ngm, nat, flag, ih, jh, nhm, omega, fact, (double *) qgm_na_D, (double *) dtmp_D );
					qecudaGetLastError("kernel kernel_compute_deeq launch failure");
				}
			}
		}
	}
	
	cublasSetPointerMode(vlocHandles[ 0 ] , CUBLAS_POINTER_MODE_HOST);

	qecudaSafeCall( cudaMemcpy( deeq, (double *) deeq_D, sizeof( double ) * ( nhm * nhm * nat * nspin ), cudaMemcpyDeviceToHost ) );

	free( qgm_na );
	cudaFreeHost(qgm);

	cudaStreamDestroy( vlocStreams[ 0 ] );
	cublasDestroy( vlocHandles[ 0 ]);

#if defined(__CUDA_KERNEL_MEMSET)
	qecudaSafeCall( cudaMemset( dev_scratch_QE[0], 0, (size_t) cuda_memory_unused[0] ) );
#endif

	return 0;
}
