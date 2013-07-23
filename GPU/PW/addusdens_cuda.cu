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

#define _CUDA_QVAN_ 1
extern "C" void qvan2_(int * ptr_ngm, int * iih, int * jjh, int * ptr_nt, double * qmod, double * qgm, double * ylmk0);
int qvan2_cuda( int ngy, int ih, int jh, 
                            int np, double *qmod_D, double *qg_D, double *ylmk0_D, 
                            int ylmk0_s1, int nlx,  
                            double dq, double *qrad_D, int qrad_s1, int qrad_s2,
                            int qrad_s3, int *indv, int indv_s1,
                            int *nhtolm, int nhtolm_s1,
                            int nbetam, int *lpx, int lpx_s1,
                            int *lpl, int lpl_s1, int lpl_s2,
                            double *ap, int ap_s1, int ap_s2,
                            cudaStream_t st) ;



extern "C" void start_clock_(char * label, unsigned int length_arg );
extern "C" void stop_clock_(char * label, unsigned int length_arg );


__global__ void kernel_compute_aux( const double * __restrict eigts1, const double * __restrict eigts2, const double * __restrict eigts3,
		const int * __restrict ig1, const int * __restrict ig2, const int * __restrict ig3,  const int nr1, const int nr2, const int nr3,
		const double * __restrict qgm, const double * __restrict becsum, double * aux, const int na, const int nspin_mag, const int ngm,
		const int first_becsum, const int ijh, const int nat, const int it )
{
	register int ig = blockDim.x * blockIdx.x + threadIdx.x;
	// register int igg = ig * 2;
	register int is;
	register int s_ig1, s_ig2, s_ig3;
	
	double eigts1_local[2], eigts2_local[2], eigts3_local[2], skk[2], element[2], sup, s_qgm[2], aux_sup[2];

	if( ig < ngm ) {
	
		s_ig1 = ig1[ig];
		s_ig2 = ig2[ig];
		s_ig3 = ig3[ig];

		s_qgm[ 0 ] = qgm[ ig * 2 ];
		s_qgm[ 1 ] = qgm[ ( ig * 2 ) + 1];

		eigts1_local[0] = eigts1[ ( ( (nr1 + s_ig1) + ( it * ( nr1 * 2 + 1 ) ) ) * 2 ) ];
		eigts1_local[1] = eigts1[ ( ( (nr1 + s_ig1) + ( it * ( nr1 * 2 + 1 ) ) ) * 2 ) + 1 ];

		eigts2_local[0] = eigts2[ ( ( (nr2 + s_ig2) + ( it * ( nr2 * 2 + 1 ) ) ) * 2 ) ];
		eigts2_local[1] = eigts2[ ( ( (nr2 + s_ig2) + ( it * ( nr2 * 2 + 1 ) ) ) * 2 ) + 1 ];

		eigts3_local[0] = eigts3[ ( ( (nr3 + s_ig3) + ( it * ( nr3 * 2 + 1 ) ) ) * 2 ) ];
		eigts3_local[1] = eigts3[ ( ( (nr3 + s_ig3) + ( it * ( nr3 * 2 + 1 ) ) ) * 2 ) + 1 ];

		element[0] = eigts1_local[0] * eigts2_local[0] - eigts1_local[1] * eigts2_local[1];
		element[1] = eigts1_local[0] * eigts2_local[1] + eigts1_local[1] * eigts2_local[0];

		skk[0] = element[0] * eigts3_local[0] - element[1] * eigts3_local[1];
		skk[1] = element[1] * eigts3_local[0] + element[0] * eigts3_local[1];

		for (is = 0 ; is < nspin_mag; is++) {

			aux_sup[0] = aux[ ( ig + (is * ngm) ) * 2 ];
			aux_sup[1] = aux[ ( ( ig + (is * ngm) ) * 2 ) + 1 ];

			sup = becsum[ ijh + ( it * first_becsum ) + ( is * first_becsum * nat ) ];

			skk[0] = skk[0] * sup;
			skk[1] = skk[1] * sup;

			element[0] = skk[0] * s_qgm[ 0 ] - skk[1] * s_qgm[ 1 ];
			aux_sup[0] = aux_sup[0] + element[0];

			element[1] = skk[1] * s_qgm[ 0 ] + skk[0] * s_qgm[ 1 ];
			aux_sup[1] = aux_sup[1] + element[1];
	
			aux[ ( ig + (is * ngm) ) * 2 ] = aux_sup[0];
			aux[ ( ( ig + (is * ngm) ) * 2 ) + 1 ] = aux_sup[1];
		}
	}

	return;
}
template <unsigned int N>
__global__ void debugMark() {
   //This is only for putting marks into the profile.
}

extern "C" int addusdens_cuda_(int * ptr_nr1, int * ptr_nr2, int * ptr_nr3, int * ptr_first_becsum,
		int * ptr_nat, int * nh, int * ptr_nt, int * ptr_ngm, double * qmod, double * qgm,
		double * ylmk0, double * eigts1, double * eigts2, double * eigts3, int * ig1, int * ig2,
		int * ig3, double * aux, double * becsum, int * ityp, int * ptr_nspin_mag, int * ptr_nspin,
      double * qrad, int * ptr_qrad_s1, int * ptr_qrad_s2, int * ptr_qrad_s3, int * ptr_qrad_s4,
      int * ptr_lmaxq, int * ptr_nlx, double * ptr_dq, int * indv, int * nhtolm, int * ptr_nbetam,
      int * lpx, int * lpl, double * ap, int * ptr_ap_s1, int * ptr_nhm)
{
	int ijh, ih, jh, na, iih, jjh, iit;

	void * qgm_D, * becsum_D, * aux_D;
    void * eigts1_D, * eigts2_D, * eigts3_D;
    void * ig1_D, * ig2_D, * ig3_D;

   void *qrad_D, *qmod_D, *ylmk0_D;

   double *qgm_H;
	int nat = (* ptr_nat);
	int nt = (* ptr_nt);
	int ngm = (* ptr_ngm);
	int nspin_mag = (* ptr_nspin_mag);
	int nr1 = (* ptr_nr1);
	int nr2 = (* ptr_nr2);
	int nr3 = (* ptr_nr3);
	int nspin = (* ptr_nspin);
	int first_becsum = (* ptr_first_becsum );
   int qrad_s1 = (* ptr_qrad_s1);
   int qrad_s2 = (* ptr_qrad_s2);
   int qrad_s3 = (* ptr_qrad_s3);
   int qrad_s4 = (* ptr_qrad_s4);
   int lmaxq = (* ptr_lmaxq);
   int nlx = (* ptr_nlx);
   double dq = (* ptr_dq);
   int nbetam = (* ptr_nbetam);
   int ap_s1 = (* ptr_ap_s1);
   int nhm = (* ptr_nhm);

	dim3 threads2_aux(__CUDA_TxB_ADDUSDENS_COMPUTE_AUX__);
	dim3 grid2_aux( qe_compute_num_blocks(ngm, threads2_aux.x) );

#define MAX_STREAMS 4
   cudaStream_t usdensStreams[MAX_STREAMS];

   for (int q = 0; q < MAX_STREAMS ; q++ )
      cudaStreamCreate( &usdensStreams[q]);


#if defined(__CUDA_DEBUG)
	printf("\n[ADDUSDENS] Enter \n");fflush(stdout);
#endif
	printf("\n[ADDUSDENS] Enter \n");fflush(stdout);

	if ( grid2_aux.x > __CUDA_MAXNUMBLOCKS__) {
		fprintf( stderr, "\n[ADDUSDENS] kernel_compute_aux cannot run, blocks requested ( %d ) > blocks allowed!!!", grid2_aux.x );
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

	int n_streams = MAX_STREAMS+1;
	size_t shift;
	do {
		shift = 0;
		n_streams--;
		aux_D = (char*) qe_dev_scratch[0] + shift;
		shift += ( ngm * nspin_mag * 2 )*sizeof(double);
		becsum_D = (char*) qe_dev_scratch[0] + shift;
		shift += ( first_becsum * nat * nspin )*sizeof(double);
		qgm_D = (char*) qe_dev_scratch[0] + shift;
		shift += ( ngm * 2 )*sizeof(double)* n_streams;
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
#if _CUDA_QVAN_
		qrad_D = (char*) qe_dev_scratch[0] + shift;
   		shift += ( qrad_s1 * qrad_s2 * qrad_s3 * qrad_s4 )*sizeof(double);
		qmod_D = (char*) qe_dev_scratch[0] + shift;
	   	shift += ( ngm ) *sizeof(double);
		ylmk0_D = (char*) qe_dev_scratch[0] + shift;
	   	shift += ( ngm * lmaxq * lmaxq )*sizeof(double);
#endif
	} while ( shift > qe_gpu_mem_unused[0] );
   
	// 	shift contains the amount of byte required on the GPU to compute
	if ( shift > qe_gpu_mem_unused[0] ) {
		fprintf( stderr, "\n[ADDUSDENS] Problem doesn't fit in GPU memory, requested ( %lu ) > memory allocated  (%lu )!!!", shift, qe_gpu_mem_unused[0] );
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

	qecudaSafeCall( cudaHostAlloc( (void**) &qgm_H, ngm * 2 * sizeof(double) * n_streams, cudaHostAllocDefault ) );
	qecudaSafeCall( cudaMemcpy( (double *) aux_D, aux,  sizeof( double ) * ( ngm * nspin_mag * 2 ), cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( (double *) becsum_D, becsum,  sizeof( double ) * ( first_becsum * nat * nspin ), cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( ig1_D, ig1,  sizeof( int ) * ngm, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( ig2_D, ig2,  sizeof( int ) * ngm, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( ig3_D, ig3,  sizeof( int ) * ngm, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( eigts1_D, eigts1,  sizeof( double ) * ( ( ( nr1 * 2 + 1 ) * nat ) * 2 ), cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( eigts2_D, eigts2,  sizeof( double ) * ( ( ( nr2 * 2 + 1 ) * nat ) * 2 ), cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( eigts3_D, eigts3,  sizeof( double ) * ( ( ( nr3 * 2 + 1 ) * nat ) * 2 ), cudaMemcpyHostToDevice ) );
#if _CUDA_QVAN_
	qecudaSafeCall( cudaMemcpy( qrad_D, qrad,  sizeof( double ) * ( qrad_s1 * qrad_s2 * qrad_s3 * qrad_s4 ), cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( qmod_D, qmod,  sizeof( double ) * ( ngm ), cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( ylmk0_D, ylmk0,  sizeof( double ) * ( ngm * lmaxq * lmaxq ), cudaMemcpyHostToDevice ) );
#endif


	ijh = 0;

	int this_stream = -1;
	for( ih = 0, iih = 1; ih < nh[nt - 1]; ih++, iih++) {
		for( jh = ih, jjh = iih; jh < nh[nt - 1]; jh++, jjh++, ijh++ ) {
			this_stream = (this_stream+1)%n_streams;
 

			qecudaSafeCall( cudaStreamSynchronize( usdensStreams[ this_stream ] ) );
#if _CUDA_QVAN_
			qvan2_cuda(ngm, iih, jjh, nt, (double *)qmod_D, (double*)qgm_D + ngm * 2 * this_stream, 
			           (double *)ylmk0_D, ngm, nlx-1, dq, (double *)qrad_D, qrad_s1, qrad_s2, qrad_s3, indv,
			           nhm, nhtolm, nhm, nbetam, lpx, nlx, lpl, nlx, nlx, ap, ap_s1, nlx, 
			           usdensStreams[this_stream] );

#else
			debugMark<1><<<1,1,0,usdensStreams[ this_stream ]>>>();
			qvan2_(ptr_ngm, &iih, &jjh, ptr_nt, qmod, qgm_H + ngm * 2 * this_stream, ylmk0);
			debugMark<1><<<1,1,0,usdensStreams[ this_stream ]>>>();
			// Protective guard....
			//cudaDeviceSynchronize();

			qecudaSafeCall( cudaMemcpyAsync( (double *) qgm_D + ngm * 2 * this_stream, 
                                          (double *) qgm_H + ngm * 2 * this_stream,  
                                          sizeof( double ) * ngm * 2, cudaMemcpyHostToDevice, 
                                          usdensStreams[ this_stream ] ) );
#endif

			for( iit = 0; iit < nat ; iit++ ) {

				if ( ityp[iit] == nt ) {

					kernel_compute_aux<<< grid2_aux, threads2_aux, 0, usdensStreams[ this_stream ] >>>(
							(double *) eigts1_D, (double *) eigts2_D, (double *) eigts3_D,
							(int *) ig1_D, (int *)  ig2_D, (int *) ig3_D, nr1, nr2, nr3,
							(double *) qgm_D + ngm * 2 * this_stream, (double *) becsum_D, (double *) aux_D,
							na, nspin_mag, ngm, first_becsum, ijh, nat, iit );
					qecudaGetLastError("kernel launch failure");

				}

			}

		}
	}

	for (this_stream = 0; this_stream < n_streams; this_stream++) {
		qecudaSafeCall( cudaStreamSynchronize(usdensStreams[ this_stream ]) );
		qecudaSafeCall( cudaStreamDestroy(usdensStreams[ this_stream ]) );
	}
	qecudaSafeCall( cudaMemcpy( aux, (double *) aux_D, sizeof( double ) * ( ngm * nspin_mag * 2 ), cudaMemcpyDeviceToHost ) );

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

        cudaFreeHost(qgm_H);
#endif
	return 0;
}
