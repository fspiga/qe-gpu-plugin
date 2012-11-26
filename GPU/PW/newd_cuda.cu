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

__device__ inline void complex_by_complex_device( const  double * __restrict A, const  double * __restrict B, double * C)
{
	double re_a = A[0], re_b = B[0];
	double img_a = A[1], img_b = B[1];

	C[0] = (re_a * re_b) - (img_a * img_b);
	C[1] = (re_a * img_b) + (re_b * img_a);

	return;
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

	return;
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

	return;
}

extern "C" int newd_cuda_( int * ptr_nr1, int * ptr_nr2, int * ptr_nr3, int * ptr_na, int * nh,
		double * ptr_fact, int * ptr_nt, int * ptr_nat, int * ptr_ngm, int * ptr_nhm, int * ptr_nspin,
		double * qmod, double * ylmk0, double * eigts1, double * eigts2, double * eigts3, int * ig1,
		int * ig2, int * ig3, double * deeq, int * ityp, double * ptr_omega, int * ptr_flag,
		double * aux, int * ptr_nspin_mag)
{
	void * qgm_D, * deeq_D, * aux_D, * dtmp_D, * qgm_na_D;
    void * eigts1_D, * eigts2_D, * eigts3_D;
    void * ig1_D, * ig2_D, * ig3_D;

	double * qgm_na, * qgm;

    int ih, jh, jjh, iih, is, number_of_block;
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

	dim3 threads2_qgm( __CUDA_TxB_NEWD_QGM__ );
	dim3 grid2_qgm( qe_compute_num_blocks( (nspin_mag * ngm ), threads2_qgm.x) );

	dim3 threads2_deepq( __CUDA_TxB_NEWD_DEEPQ__ );
	dim3 grid2_deepq( qe_compute_num_blocks( nspin_mag, threads2_deepq.x));

#if defined(__CUDA_DEBUG)
	printf("\n[NEWD] Enter \n");fflush(stdout);
#endif

	if ( grid2_qgm.x > __CUDA_MAXNUMBLOCKS__) {
		fprintf( stderr, "\n[NEWD] kernel_compute_qgm_na cannot run, blocks requested ( %d ) > blocks allowed!!!", (nspin_mag * ngm * 2 / __CUDA_TxB_NEWD_QGM__) );
		return 1;
	}

	qgm_na = (double *) malloc( ngm * 2 * sizeof(double) );

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
	dtmp_D = (char*) qe_dev_scratch[0] + shift;
	shift += ( nspin_mag )*sizeof(double);
	aux_D = (char*) qe_dev_scratch[0] + shift;
	shift += ( ( ngm * nspin_mag ) * 2  )*sizeof(double);
	qgm_na_D = (char*) qe_dev_scratch[0] + shift;
	shift += ( ngm * 2 )*sizeof(double);
	qgm_D = (char*) qe_dev_scratch[0] + shift;
	shift += ( ngm * 2 )*sizeof(double);
	deeq_D = (char*) qe_dev_scratch[0] + shift;
	shift += ( nhm * nhm * nat * nspin )*sizeof( double );
	shift += ( nhm * nhm * nat * nspin )*sizeof( double );
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
	// now	shift contains the amount of byte required on the GPU to compute

	if ( shift > qe_gpu_mem_unused[0] ) {
		fprintf( stderr, "\n[NEWD] Problem don't fit in GPU memory, memory requested ( %lu ) > memory allocated  (%lu )!!!", shift, qe_gpu_mem_unused[0] );
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

	qecudaSafeCall( cudaHostAlloc( (void**) &qgm,  ngm * 2 * sizeof(double), cudaHostAllocDefault ) );

	qecudaSafeCall( cudaMemcpy( (double *) aux_D, aux,  sizeof( double ) * ( ngm * nspin_mag * 2 ), cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( (double *) deeq_D, deeq,  sizeof( double ) * ( nhm * nhm * nat * nspin ), cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( ig1_D, ig1,  sizeof( int ) * ngm, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( ig2_D, ig2,  sizeof( int ) * ngm, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( ig3_D, ig3,  sizeof( int ) * ngm, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( eigts1_D, eigts1,  sizeof( double ) * ( ( ( nr1 * 2 + 1 ) * nat ) * 2 ), cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( eigts2_D, eigts2,  sizeof( double ) * ( ( ( nr2 * 2 + 1 ) * nat ) * 2 ), cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( eigts3_D, eigts3,  sizeof( double ) * ( ( ( nr3 * 2 + 1 ) * nat ) * 2 ), cudaMemcpyHostToDevice ) );

	qecudaSafeCall( cudaMemset( (double *) qgm_na_D, 0, sizeof( double ) * ngm * 2  ) );
	qecudaSafeCall( cudaMemset( (double *) dtmp_D, 0, sizeof( double ) * nspin_mag ) );
	
	cublasSetPointerMode(qecudaHandles[ 0 ] , CUBLAS_POINTER_MODE_DEVICE);

	for( ih = 0, iih = 1; ih < nh[nt - 1]; ih++, iih++ )
	{
		for( jh = ih, jjh = iih; jh < nh[nt - 1]; jh++, jjh++ )
		{

			qvan2_(ptr_ngm, &iih, &jjh, ptr_nt, qmod, qgm, ylmk0);

			qecudaSafeCall( cudaMemcpy( (double *) qgm_D, qgm,  sizeof( double ) * ngm * 2, cudaMemcpyHostToDevice ) );

			for( na = 0;  na < nat; na++ ){

				if( ityp[na] == nt ) {

					kernel_compute_qgm_na<<< grid2_qgm, threads2_qgm, 0, qecudaStreams[ 0 ] >>>(
							(double *) eigts1_D, (double *) eigts2_D, (double *) eigts3_D,
							(int *) ig1_D, (int *) ig2_D, (int *) ig3_D, (double *) qgm_D,
							nr1, nr2, nr3, na, ngm, (double *) qgm_na_D );
					qecudaGetLastError("kernel kernel_compute_qgm_na launch failure");

					for( is = 0; is < nspin_mag; is++ ){
						cublasDdot(qecudaHandles[ 0 ] , ngm * 2, (double *) aux_D + (is * ngm * 2), 1, (double *) qgm_na_D, 1, (double *) dtmp_D + is );
					}

					kernel_compute_deeq<<< grid2_deepq, threads2_deepq, 0, qecudaStreams[ 0 ] >>>(
							(double *) qgm_D, (double *) deeq_D, (double *) aux_D,
							na, nspin_mag, ngm, nat, flag, ih, jh, nhm, omega, fact,
							(double *) qgm_na_D, (double *) dtmp_D );
					qecudaGetLastError("kernel kernel_compute_deeq launch failure");
				}
			}
		}
	}
	
	cublasSetPointerMode(qecudaHandles[ 0 ] , CUBLAS_POINTER_MODE_HOST);

	qecudaSafeCall( cudaMemcpy( deeq, (double *) deeq_D, sizeof( double ) * ( nhm * nhm * nat * nspin ), cudaMemcpyDeviceToHost ) );

	free( qgm_na );
	cudaFreeHost(qgm);

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
