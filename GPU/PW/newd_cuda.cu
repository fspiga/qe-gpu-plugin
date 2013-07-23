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

__device__ inline void complex_by_complex_device( const  double * __restrict A, const  double * __restrict B, double * C)
{
	double re_a = A[0], re_b = B[0];
	double img_a = A[1], img_b = B[1];

	C[0] = (re_a * re_b) - (img_a * img_b);
	C[1] = (re_a * img_b) + (re_b * img_a);

	return;
}

__device__ inline void complex_by_complex_device_new( const  cuDoubleComplex * __restrict A, const  cuDoubleComplex * __restrict B, cuDoubleComplex * C)
{

	(*C).x = ((*A).x * (*B).x) - ((*A).y * (*B).y);
	(*C).y = ((*A).x * (*B).y) + ((*B).x * (*A).y);

	return;
}

template <unsigned int N>
__global__ void debugMark() {
   //This is only for putting marks into the profile.
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

__device__ double atomicAdd(double* address, double val)

{

double old = *address, assumed;

do {

assumed = old;

old =__longlong_as_double(atomicCAS((unsigned long long int*)address,__double_as_longlong(assumed),__double_as_longlong(val + assumed)));

} while (assumed != old);

return old;

}

#define _N_BINS 2
__global__ void kernel_compute_qgm_na_new( const  cuDoubleComplex * __restrict eigts1, 
              const  cuDoubleComplex * __restrict eigts2, const  cuDoubleComplex * __restrict eigts3,
		        const  int * __restrict ig1, const  int * __restrict ig2, const  int * __restrict ig3, 
              const  cuDoubleComplex * __restrict qgm, const int nr1,
		        const int nr2, const int nr3, const int na, const int ngm, cuDoubleComplex * /*__restrict*/ aux, 
              const int nspin_mag, double * dtmp, cuDoubleComplex * qgm_na )
{
	int global_index = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

	int ind_eigts1, ind_eigts2, ind_eigts3;
   cuDoubleComplex sup_prod_1, sup_prod_2;
   cuDoubleComplex out;

   cuDoubleComplex tmp[_N_BINS];
   double __shared__ sdata[_N_BINS][__CUDA_TxB_NEWD_QGM__];
   int is, parity;

	if( global_index < ngm ){

		ind_eigts1 = ( ( ( nr1 + ig1[global_index] ) + ( na * ( nr1 * 2 + 1 ) ) ) * 1 );
		ind_eigts2 = ( ( ( nr2 + ig2[global_index] ) + ( na * ( nr2 * 2 + 1 ) ) ) * 1 );
		ind_eigts3 = ( ( ( nr3 + ig3[global_index] ) + ( na * ( nr3 * 2 + 1 ) ) ) * 1 );

      sup_prod_1 = cuCmul( __ldg(&eigts1[ ind_eigts1 ] ), __ldg(&eigts2[ ind_eigts2 ] ) );
      sup_prod_2 = cuCmul( sup_prod_1, __ldg(&eigts3[ ind_eigts3 ] ) );
      out = cuCmul( sup_prod_2, qgm[ global_index ] );
      qgm_na[ global_index ] = out;
   }

   //dot product with aux, for each spin mag
   parity = 0;
   int tid = threadIdx.x + blockDim.x * threadIdx.y;
   #pragma unroll _N_BINS
   for (is = 0; is < nspin_mag; is++) {

      if( global_index < ngm ){
         tmp[parity] = aux[ is * ngm + global_index ];
         sdata[parity][tid] = out.x * tmp[parity].x + out.y * tmp[parity].y;
	   } else {
         sdata[parity][tid] = 0.0;
      }
      __syncthreads();
      for (unsigned int s = blockDim.x/2; s>0; s >>= 1) {
         if (tid<s)
            sdata[parity][tid] += sdata[parity][tid+s];
         __syncthreads();
      }
      if (tid == 0) {
         atomicAdd(&dtmp[is], sdata[parity][tid]);
      }
      parity = (parity+1)%_N_BINS;
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
      //LDB zero this for use in the next iteration
      dtmp[global_index]=0.0;

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
		double * aux, int * ptr_nspin_mag, double * qrad, int *ptr_qrad_s1, int *ptr_qrad_s2, 
      int *ptr_qrad_s3, int *ptr_qrad_s4, int *ptr_lmaxq, int *ptr_nlx, double *ptr_dq, 
      int *indv, int *nhtolm, int *ptr_nbetam, int *lpx, int *lpl, double *ap, int *ptr_ap_s1)
{
	void * qgm_D, * deeq_D, * aux_D, * dtmp_D, * qgm_na_D;
    void * eigts1_D, * eigts2_D, * eigts3_D;
    void * ig1_D, * ig2_D, * ig3_D;

   void *qrad_D, *qmod_D, *ylmk0_D;

	double * qgm;

   int ih, jh, jjh, iih;
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
   int qrad_s1 = (* ptr_qrad_s1);
   int qrad_s2 = (* ptr_qrad_s2);
   int qrad_s3 = (* ptr_qrad_s3);
   int qrad_s4 = (* ptr_qrad_s4);
   int lmaxq = (* ptr_lmaxq);
   int nlx = (* ptr_nlx);
   double dq = (* ptr_dq);
   int nbetam = (* ptr_nbetam);
   int ap_s1 = (* ptr_ap_s1);

	dim3 threads2_qgm( __CUDA_TxB_NEWD_QGM__ );
	dim3 grid2_qgm( qe_compute_num_blocks( (nspin_mag * ngm ), threads2_qgm.x) );

	dim3 threads2_deepq( __CUDA_TxB_NEWD_DEEPQ__ );
	dim3 grid2_deepq( qe_compute_num_blocks( nspin_mag, threads2_deepq.x));

   cudaEvent_t estart, eend;
   cudaEventCreate(&estart);
   cudaEventCreate(&eend);

   cudaEventRecord(estart);

#define MAX_STREAMS 2
   cudaStream_t newdcudaStreams[MAX_STREAMS];

   for (int q = 0; q < MAX_STREAMS ; q++ )
      cudaStreamCreate( &newdcudaStreams[q]);

#if defined(__CUDA_DEBUG)
	printf("\n[NEWD] Enter \n");fflush(stdout);
#endif
	printf("\n[NEWD] Enter \n");fflush(stdout);

	if ( grid2_qgm.x > __CUDA_MAXNUMBLOCKS__) {
		fprintf( stderr, "\n[NEWD] kernel_compute_qgm_na cannot run, blocks requested ( %d ) > blocks allowed!!!", (nspin_mag * ngm * 2 / __CUDA_TxB_NEWD_QGM__) );
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
      n_streams -= 1;
 	   shift = 0;
   	dtmp_D = (char*) qe_dev_scratch[0] + shift;
   	shift += ( nspin_mag )*sizeof(double)*n_streams;
      //LDB adding this so that aux_D is aligned on a cuDoubleComplex boundary
      shift = shift + sizeof(double)*2/sizeof(char) - 1;
      shift -= shift % (sizeof(double)*2/sizeof(char));
   	aux_D = (char*) qe_dev_scratch[0] + shift;
   	shift += ( ( ngm * nspin_mag ) * 2  )*sizeof(double);
   	qgm_na_D = (char*) qe_dev_scratch[0] + shift;
   	shift += ( ngm * 2 )*sizeof(double) * n_streams;
   	qgm_D = (char*) qe_dev_scratch[0] + shift;
   	shift += ( ngm * 2 )*sizeof(double) * n_streams;
   	deeq_D = (char*) qe_dev_scratch[0] + shift;
   	shift += ( nhm * nhm * nat * nspin )*sizeof( double );
      //LDB adding this so that eigts1_D is aligned on a cuDoubleComplex boundary
      shift = shift + sizeof(double)*2/sizeof(char) - 1;
      shift -= shift % (sizeof(double)*2/sizeof(char));
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
   	// now	shift contains the amount of byte required on the GPU to compute
   } while (n_streams > 0 && shift > qe_gpu_mem_unused[0]);
   
	if ( n_streams < 1) {
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

	qecudaSafeCall( cudaHostAlloc( (void**) &qgm,  ngm * 2 * sizeof(double) * n_streams, cudaHostAllocDefault ) );

	qecudaSafeCall( cudaMemcpy( (double *) aux_D, aux,  sizeof( double ) * ( ngm * nspin_mag * 2 ), cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( (double *) deeq_D, deeq,  sizeof( double ) * ( nhm * nhm * nat * nspin ), cudaMemcpyHostToDevice ) );
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
	qecudaSafeCall( cudaMemset( (double *) qgm_na_D, 0, sizeof( double ) * ngm * 2 * n_streams  ) );
	qecudaSafeCall( cudaMemset( (double *) dtmp_D, 0, sizeof( double ) * nspin_mag * n_streams ) );
	
	cublasSetPointerMode(qecudaHandles[ 0 ] , CUBLAS_POINTER_MODE_DEVICE);

   int this_stream = -1;
	for( ih = 0, iih = 1; ih < nh[nt - 1]; ih++, iih++ )
	{
		for( jh = ih, jjh = iih; jh < nh[nt - 1]; jh++, jjh++ )
		{
         this_stream = (this_stream+1)%n_streams;

         qecudaSafeCall( cudaStreamSynchronize( newdcudaStreams[ this_stream ] ) );
#if _CUDA_QVAN_ 
			qvan2_cuda(ngm, iih, jjh, nt, (double *)qmod_D, (double*)qgm_D + ngm * 2 * this_stream, 
                    (double *)ylmk0_D, ngm, nlx-1, dq, (double *)qrad_D, qrad_s1, qrad_s2, qrad_s3, indv,
                    nhm, nhtolm, nhm, nbetam, lpx, nlx, lpl, nlx, nlx, ap, ap_s1, nlx, 
                    newdcudaStreams[this_stream] );
#else
         //debugMark<1><<<1,1,0,newdcudaStreams[ this_stream ]>>>();
			qvan2_(ptr_ngm, &iih, &jjh, ptr_nt, qmod, (double*)qgm + ngm * 2 * this_stream, ylmk0);
         //debugMark<2><<<1,1,0,newdcudaStreams[ this_stream ]>>>();
  		   qecudaSafeCall( cudaMemcpyAsync( (double *) qgm_D + ngm * 2 * this_stream,  
                                             (double *) qgm + ngm * 2 * this_stream,  
                                             sizeof( double ) * ngm * 2, 
                                             cudaMemcpyHostToDevice, newdcudaStreams[ this_stream] ) );
#endif

			for( na = 0;  na < nat; na++ ){
            //No need for this since kernel_compute_deeq is zeroing dtmp_D
            //qecudaSafeCall ( cudaMemsetAsync((double*)dtmp_D + nspin_mag * this_stream, 0, sizeof(double)*nspin_mag, 
            //                newdcudaStreams[ this_stream ] ) );

				if( ityp[na] == nt ) {

					kernel_compute_qgm_na_new<<< grid2_qgm, threads2_qgm, 0, newdcudaStreams[ this_stream ] >>>(
							(cuDoubleComplex *) eigts1_D, (cuDoubleComplex *) eigts2_D, (cuDoubleComplex *) eigts3_D,
							(int *) ig1_D, (int *) ig2_D, (int *) ig3_D, (cuDoubleComplex *) qgm_D + ngm * this_stream,
							nr1, nr2, nr3, na, ngm, (cuDoubleComplex *) aux_D, nspin_mag, 
                     (double*) dtmp_D + nspin_mag * this_stream, 
                     (cuDoubleComplex *) qgm_na_D + ngm * this_stream );

					qecudaGetLastError("kernel kernel_compute_qgm_na launch failure");

					kernel_compute_deeq<<< grid2_deepq, threads2_deepq, 0, newdcudaStreams[ this_stream ] >>>(
							(double *) qgm_D + ngm * 2 * this_stream, (double *) deeq_D, (double *) aux_D,
							na, nspin_mag, ngm, nat, flag, ih, jh, nhm, omega, fact,
							(double *) qgm_na_D + ngm * 2 * this_stream, 
                     (double *) dtmp_D + nspin_mag * this_stream );
					qecudaGetLastError("kernel kernel_compute_deeq launch failure");
				}
			}
		}
	}
	
	cublasSetPointerMode(qecudaHandles[ 0 ] , CUBLAS_POINTER_MODE_HOST);

   for (this_stream = 0; this_stream < n_streams; this_stream++) {
      qecudaSafeCall( cudaStreamSynchronize(newdcudaStreams[ this_stream ]) );
      qecudaSafeCall( cudaStreamDestroy(newdcudaStreams[ this_stream ]) );
   }
	qecudaSafeCall( cudaMemcpy( deeq, (double *) deeq_D, sizeof( double ) * ( nhm * nhm * nat * nspin ), cudaMemcpyDeviceToHost ) );

#if 0
   for (int q = 0; q < nhm * nhm * nat * nspin; q++) {
      printf("deeq [%d] = %25.8e\n", q, deeq[q]);
   }
#endif
   

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

#if 1
   float elapsed;
   cudaEventRecord(eend);
   cudaEventSynchronize(eend);
   cudaEventElapsedTime(&elapsed, estart, eend);
   printf("Newd elapsed time: %f\n", elapsed);
#endif
   cudaEventDestroy(estart);
   cudaEventDestroy(eend);

#if defined(__CUDA_DEBUG)
	printf("\n[NEWD] Exit \n");fflush(stdout);
#endif

	return 0;
}
