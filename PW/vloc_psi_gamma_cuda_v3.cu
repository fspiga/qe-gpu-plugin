/*****************************************************************************\
 * Copyright (C) 2011-2014 Quantum ESPRESSO Foundation
 *
 * This file is distributed under the terms of the
 * GNU General Public License. See the file `License'
 * in the root directory of the present distribution,
 * or http://www.gnu.org/copyleft/gpl.txt .
 *
 * Filippo Spiga (filippo.spiga@quantum-espresso.org)
 *
 * NOTES: it works only if __CUDA_NOALLOC
 *
\*****************************************************************************/

#include <stdlib.h>
#include <stdio.h>

#include "cuda_env.h"

extern "C" void start_clock_(char * label, unsigned int length_arg );
extern "C" void stop_clock_(char * label, unsigned int length_arg );

extern "C" int nls_precompute_gamma_( int * ptr_n, int * igk, int * nls,  int * nlsm, int * ptr_ngms, int * ptr_ngm);
extern "C" int nls_precompute_gamma_cleanup_();


__global__ void kernel_vec_prod( double *a, const  double * __restrict b, int dimx )
{	   
	register int ix = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	register double sup;
	register int ii = ix / 2;

	if( ix < ( dimx * 2 ) ){
		sup = a[ix] * b[ii];
		a[ix] = sup;
	}
}

__global__ void kernel_init_psic( const  int * __restrict psic_index_nls, const  int * __restrict psic_index_nlsm, const  double * __restrict psi, double * psic, const int n, const int m, const int lda, const int ibnd )
{	   
	register int ix = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

	register int pos = ibnd * lda;
	register int pos_plus = (ibnd+1) * lda;

	register int psi_index = (ix + pos) * 2;
	register int psi_index_plus = (ix + pos_plus) * 2;

	if ( ix < n ) {

		// psic_index_nls[ix] = ( nls[ igk[ ix ] - 1 ] - 1 ) * 2;
		// psic_index_nlsm[ix] = ( nlsm[ igk[ ix ] - 1 ] - 1 ) * 2;

		if ( ibnd  < ( m - 1 ) ) {

			psic[ psic_index_nls[ix] ] = psi[ psi_index ] - psi[ psi_index_plus + 1 ];
			psic[ psic_index_nls[ix] + 1 ] = psi[ psi_index + 1 ] + psi[ psi_index_plus ];

			psic[ psic_index_nlsm[ix] ] = psi[ psi_index ] + psi[ psi_index_plus + 1 ];
			psic[ psic_index_nlsm[ix] + 1 ] = -1.0 * ( psi[ psi_index + 1 ] - psi[ psi_index_plus ] );

		} else {

			psic[ psic_index_nls[ix] ] = psi[ psi_index ];
			psic[ psic_index_nls[ix] + 1 ] = psi[ psi_index + 1 ];

			psic[ psic_index_nlsm[ix] ] = psi[ psi_index ];
			psic[ psic_index_nlsm[ix] + 1 ] = - 1.0 * psi[ psi_index + 1 ];

		}
	}
}

__global__ void kernel_save_hpsi(  const  int * __restrict psic_index_nls, const  int * __restrict psic_index_nlsm, double * hpsi, const  double * __restrict psic, const int n, const int m, const int lda, const int ibnd )
{	   
	register int ix = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

	register int pos = ibnd * lda;
	register int pos_plus = (ibnd+1) * lda;

	register int psi_index = (ix + pos) * 2;
	register int psi_index_plus = (ix + pos_plus) * 2;

	register double real_fp, cmplx_fp, real_fm, cmplx_fm;

	if ( ix < n ) {

		// psic_index_nls[ix] = (nls[ igk[ ix ] - 1 ] - 1) * 2;
		// psic_index_nlsm[ix] = (nlsm[ igk[ ix ] - 1 ] - 1) * 2;

		if( ibnd  < ( m - 1 ) ) {

			real_fp = ( psic[ psic_index_nls[ix] ] + psic[ psic_index_nlsm[ix] ] ) * 0.5;
			cmplx_fp = ( psic[ psic_index_nls[ix] + 1 ] + psic[ psic_index_nlsm[ix] + 1 ] ) * 0.5;

			real_fm = ( psic[ psic_index_nls[ix] ] - psic[ psic_index_nlsm[ix] ] ) * 0.5;
			cmplx_fm = ( psic[ psic_index_nls[ix] + 1 ] - psic[ psic_index_nlsm[ix] + 1 ] ) * 0.5;

			hpsi[ psi_index ] = hpsi[ psi_index ] + real_fp;
			hpsi[ psi_index + 1 ] = hpsi[ psi_index + 1 ] + cmplx_fm;

			hpsi[ psi_index_plus ] = hpsi[ psi_index_plus ] + cmplx_fp;
			hpsi[ psi_index_plus + 1 ] = hpsi[ psi_index_plus + 1 ] - real_fm;

		} else {

			hpsi[ psi_index ] = hpsi[ psi_index ] + psic[ psic_index_nls[ix] ];
			hpsi[ psi_index + 1 ] = hpsi[ psi_index + 1 ] + psic[ psic_index_nls[ix] + 1 ];

		}
	}
}


extern "C"  int vloc_psi_cuda_(int * ptr_lda, int * ptr_nrxxs, int * ptr_nr1s, int * ptr_nr2s, int * ptr_nr3s, int * ptr_n, int * ptr_m, void * psi, double * v, void * hpsi, int * igk, int * nls, int * nlsm, int * ptr_ngms, int * ptr_ngm)
{
	cufftHandle p_global;

    void * psic_D, * psi_D, * hpsi_D; // cufftDoubleComplex*
	void * v_D; // double*
	int blocksPerGrid, ibnd;

	double tscale;

	int n = (* ptr_n);
	int m = (* ptr_m);
	int m_fake;
	int nr1s = (* ptr_nr1s);
	int nr2s = (* ptr_nr2s);
	int nr3s = (* ptr_nr3s);
	int nrxxs = (* ptr_nrxxs);
	int ngms = (* ptr_ngms);
	int ngm = (* ptr_ngm);
	int lda = (* ptr_lda);

	int size_psic = nr1s * nr2s * nr3s;

	bool precomp_psic_indexes = false;
	int ierr;

#if defined(__CUDA_DEBUG)
	printf("[VLOC_PSI_GAMMA_OPT2] Enter (n=%d, m=%d, ngms=%d, ngm=%d)\n", n, m, ngms, ngm); fflush(stdout);
#endif

	/* Padding -- really necessary?*/
	if (m%2 == 0)
		m_fake = m ;
	else
		m_fake = m + 1;

	blocksPerGrid = ( n + __CUDA_TxB_VLOCPSI_PSIC__ - 1) / __CUDA_TxB_VLOCPSI_PSIC__ ;
	if ( blocksPerGrid > __CUDA_MAXNUMBLOCKS__) {
		fprintf( stderr, "\n[VLOC_PSI_GAMMA] kernel_init_psic cannot run, blocks requested ( %d ) > blocks allowed!!!", blocksPerGrid );
		return 1;
	}

	blocksPerGrid = ( (nrxxs * 2) + __CUDA_TxB_VLOCPSI_PROD__  - 1) / __CUDA_TxB_VLOCPSI_PROD__ ;
	if ( blocksPerGrid > __CUDA_MAXNUMBLOCKS__) {
		fprintf( stderr, "\n[VLOC_PSI_GAMMA] kernel_vec_prod cannot run, blocks requested ( %d ) > blocks allowed!!!", blocksPerGrid );
		return 1;
	}

	blocksPerGrid = ( n + __CUDA_TxB_VLOCPSI_HPSI__ - 1) / __CUDA_TxB_VLOCPSI_HPSI__ ;
	if ( blocksPerGrid > __CUDA_MAXNUMBLOCKS__) {
		fprintf( stderr, "\n[VLOC_PSI_GAMMA] kernel_save_hpsi cannot run, blocks requested ( %d ) > blocks allowed!!!", blocksPerGrid );
		return 1;
	}

	cudaSetDevice(qe_gpu_bonded[0]);

#if defined(__CUDA_NOALLOC)
	// If preloaded_nls_D is NULL, calculate preloaded_nls_D/preloaded_nlsm_D on-the-fly
	if (preloaded_nls_D == NULL){
#if defined(__CUDA_DEBUG)
		printf("[VLOC_PSI_GAMMA_OPT2] Perform nls_precompute_gamma_ on the fly...\n");
#endif
		precomp_psic_indexes = true;
		ierr = nls_precompute_gamma_ ( ptr_n, igk, nls, nlsm, ptr_ngms, ptr_ngm);
	}

	/* Do real allocation */
	ierr = cudaMalloc ( (void**) &(qe_dev_scratch[0]), (size_t) qe_gpu_mem_unused[0] );
	qecudaGenericErr((cudaError_t) ierr, "VLOC_PSI_GAMMA_OPT2", "error in memory allocation");

#if defined(__CUDA_KERNEL_MEMSET)
	qecudaSafeCall( cudaMemset( qe_dev_scratch[0], 0, (size_t) qe_gpu_mem_unused[0] ) );
#endif
#endif

	size_t shift = 0;
	psic_D = (char*) qe_dev_scratch[0] + shift;
	shift += ( size_psic )*sizeof( cufftDoubleComplex );
	psi_D = (char*) qe_dev_scratch[0] + shift;
	shift += ( lda * m_fake )*sizeof( cufftDoubleComplex );
	hpsi_D = (char*) qe_dev_scratch[0] + shift;
	shift += ( lda * m_fake )*sizeof( cufftDoubleComplex );
	v_D = (char*) qe_dev_scratch[0] + shift;
	shift += ( nrxxs )*sizeof( double );
	// now	shift contains the amount of byte required on the GPU to compute

	if ( shift > qe_gpu_mem_unused[0] ) {
		fprintf( stderr, "[VLOC_PSI_GAMMA_OPT2] Problem don't fit in GPU memory --- memory requested ( %lu ) > memory allocated  (%lu )!!!", shift, qe_gpu_mem_unused[0] );
#if defined(__CUDA_NOALLOC)
		if (precomp_psic_indexes) {
			ierr = nls_precompute_gamma_cleanup_();
			precomp_psic_indexes = false;
		}

		ierr = cudaFree ( qe_dev_scratch[0] );
		qecudaGenericErr((cudaError_t) ierr, "VLOC_PSI_GAMMA_OPT2", "error in memory release");
#endif
		return 1;
	}

#if defined(__CUDA_KERNEL_MEMSET)
	qecudaSafeCall( cudaMemset( psi_D, 0, sizeof( cufftDoubleComplex ) * lda * m_fake ) );
	qecudaSafeCall( cudaMemset( psic_index_nls_D , 0, sizeof( int ) * n ) );
	qecudaSafeCall( cudaMemset( psic_index_nlsm_D , 0, sizeof( int ) * n ) );
#endif
	qecudaSafeCall( cudaMemcpy( psi_D, psi,  sizeof( cufftDoubleComplex ) * lda * m, cudaMemcpyHostToDevice ) );
#if defined(__CUDA_KERNEL_MEMSET)
	if (m_fake > m) {
		qecudaSafeCall( cudaMemset( (psi_D + ( lda * m )) , 0, sizeof( cufftDoubleComplex ) * size_psic ) ); // Post-set of (m_fake) zeros
	}
#endif
	qecudaSafeCall( cudaMemcpy( hpsi_D, hpsi,  sizeof( cufftDoubleComplex ) * lda * m, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( v_D, v,  sizeof( double ) * nrxxs, cudaMemcpyHostToDevice ) );

	qecheck_cufft_call( cufftPlan3d( &p_global, nr3s, nr2s,  nr1s, CUFFT_Z2Z ) );
    qecheck_cufft_call( cufftSetStream(p_global,qecudaStreams[ 0 ]) );

	for( ibnd =  0; ibnd < m_fake; ibnd = ibnd + 2 )
	{
		qecudaSafeCall( cudaMemset( psic_D , 0, size_psic * sizeof( cufftDoubleComplex ) ) );

		blocksPerGrid = ( n + __CUDA_TxB_VLOCPSI_PSIC__ - 1) / __CUDA_TxB_VLOCPSI_PSIC__ ;
		kernel_init_psic<<<blocksPerGrid, __CUDA_TxB_VLOCPSI_PSIC__ >>>( (int *) preloaded_nls_D, (int *) preloaded_nlsm_D, (double *) psi_D, (double *) psic_D, n, m, lda, ibnd );
		qecudaGetLastError("kernel launch failure");

		qecheck_cufft_call( cufftExecZ2Z( p_global, (cufftDoubleComplex *) psic_D, (cufftDoubleComplex *) psic_D, CUFFT_INVERSE ) );

		blocksPerGrid = ( (nrxxs * 2) + __CUDA_TxB_VLOCPSI_PROD__  - 1) / __CUDA_TxB_VLOCPSI_PROD__ ;
		kernel_vec_prod<<<blocksPerGrid, __CUDA_TxB_VLOCPSI_PROD__ >>>( (double *) psic_D, (double *) v_D , nrxxs );
		qecudaGetLastError("kernel launch failure");

		qecheck_cufft_call( cufftExecZ2Z( p_global, (cufftDoubleComplex *) psic_D, (cufftDoubleComplex *) psic_D, CUFFT_FORWARD ) );

		tscale = 1.0 / (double) ( size_psic );
		cublasZdscal(qecudaHandles[ 0 ] , size_psic, &tscale, (cufftDoubleComplex *) psic_D, 1);

		blocksPerGrid = ( n + __CUDA_TxB_VLOCPSI_HPSI__ - 1) / __CUDA_TxB_VLOCPSI_HPSI__ ;
		kernel_save_hpsi<<<blocksPerGrid, __CUDA_TxB_VLOCPSI_HPSI__ >>>( (int *) preloaded_nls_D, (int *) preloaded_nlsm_D, (double *) hpsi_D, (double *) psic_D, n, m, lda, ibnd );
		qecudaGetLastError("kernel launch failure");

	}

	qecudaSafeCall( cudaMemcpy( hpsi, (cufftDoubleComplex *) hpsi_D, sizeof( cufftDoubleComplex ) * lda * m, cudaMemcpyDeviceToHost ) );

	qecheck_cufft_call( cufftDestroy(p_global) );

#if defined(__CUDA_NOALLOC)
	if (precomp_psic_indexes) {
		ierr = nls_precompute_gamma_cleanup_();
		precomp_psic_indexes = false;
	}

	ierr = cudaFree ( qe_dev_scratch[0] );
	qecudaGenericErr((cudaError_t) ierr, "VLOC_PSI_GAMMA_OPT2", "error in memory release");
#else

#if defined(__CUDA_KERNEL_MEMSET)
	qecudaSafeCall( cudaMemset( qe_dev_scratch[0], 0, (size_t) qe_gpu_mem_unused[0] ) );
#endif

#endif

#if defined(__CUDA_DEBUG)
	printf("[VLOC_PSI_GAMMA_OPT2] Exit\n"); fflush(stdout);
#endif

	return 0;
}

#if defined(__CUDA_MULTIPLAN_FFT)
extern "C" void vloc_psi_multiplan_cuda_(int * ptr_lda, int * ptr_nrxxs, int * ptr_nr1s, int * ptr_nr2s, int * ptr_nr3s, int * ptr_n, int * ptr_m, void * psi, double * v, void * hpsi, int * igk, int * nls, int * nlsm, int * ptr_ngms, int * ptr_ngm)
{
	cufftHandle p_global;

	void * psic_D, * psi_D, * hpsi_D; // cufftDoubleComplex *
	void * v_D; // double *
	void * igk_D, * nls_D, * nlsm_D; // int *

	double tscale;

	int n = (* ptr_n);
	int m = (* ptr_m);
	int nr1s = (* ptr_nr1s);
	int nr2s = (* ptr_nr2s);
	int nr3s = (* ptr_nr3s);
	int nrxxs = (* ptr_nrxxs);
	int ngms = (* ptr_ngms);
	int ngm = (* ptr_ngm);
	int lda = (* ptr_lda);
	int size_psic = nr1s * nr2s * nr3s;

	int array[3];
	int dim_multiplepsic, n_singlepsic, n_multiplepsic, size_multiplepsic, v_size;
	int m_fake, m_buf, blocksPerGrid, i, j;

	psic_D = (cufftDoubleComplex * ) qe_dev_scratch[0];

	/* Padding */
	if (m%2 == 0) {
		m_buf = m;
		m_fake = m_buf/2 ;
	} else {
		m_buf = m+1;
		m_fake = m_buf/2;
	}

	dim_multiplepsic = __NUM_FFT_MULTIPLAN__;
	n_multiplepsic = m_fake/dim_multiplepsic;
	n_singlepsic = m_fake%dim_multiplepsic;

	size_multiplepsic = size_psic * (dim_multiplepsic);

	cudaSetDevice(qe_gpu_bonded[0]);

	size_t shift = 0;
	psic_D = (char*) qe_dev_scratch[0] + shift;
	shift += ( size_multiplepsic )*sizeof( cufftDoubleComplex );
	psi_D = (char*) qe_dev_scratch[0] + shift;
	shift += ( n * m_buf )*sizeof( cufftDoubleComplex );
	hpsi_D = (char*) qe_dev_scratch[0] + shift;
	shift += ( n * m_buf )*sizeof( cufftDoubleComplex );
	v_D = (char*) qe_dev_scratch[0] + shift;
	shift += ( nrxxs )*sizeof( double );
	nls_D = (char*) qe_dev_scratch[0] + shift;
	shift += ( (ngms % 2 == 0)? ngms : ngms + 1 )*sizeof(int);
	nlsm_D = (char*) qe_dev_scratch[0] + shift;
	shift += ( (ngm % 2 == 0)? ngm : ngm + 1 )*sizeof(int);
	igk_D = (char*) qe_dev_scratch[0] + shift;
	shift += ( (n % 2 == 0)? n : n + 1 )*sizeof(int);

	// now	shift contains the amount of byte required on the GPU to compute

	if ( shift > qe_gpu_mem_tot[0] ) {
		fprintf( stderr, "\n[VLOC_PSI_GAMMA] Problem don't fit in GPU memory --- memory requested ( %lu ) > memory allocated  (%lu )!!!", shift, qe_gpu_mem_tot[0] );
		exit(EXIT_FAILURE);
	}

	qecudaSafeCall( cudaMemset( qe_dev_scratch[0], 0, (size_t) qe_gpu_mem_tot[0] ) );

	qecudaSafeCall( cudaMemcpy( psi_D, psi,  sizeof( cufftDoubleComplex ) * n * m_buf, cudaMemcpyHostToDevice ) );
	shift = ( n * m )*sizeof( cufftDoubleComplex );
	qecudaSafeCall( cudaMemset( (cufftDoubleComplex*)( (char*)psi_D + shift) , 0, sizeof( cufftDoubleComplex ) * m_buf ) ); // Post-set of (m_fake) zeros

	qecudaSafeCall( cudaMemcpy( hpsi_D, hpsi,  sizeof( cufftDoubleComplex ) * n * m, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( nls_D, nls,  sizeof( int ) * ngms, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( nlsm_D, nlsm,  sizeof( int ) * ngm, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( igk_D, igk,  sizeof( int ) * n, cudaMemcpyHostToDevice ) );
	qecudaSafeCall( cudaMemcpy( v_D, v,  sizeof( double ) * nrxxs, cudaMemcpyHostToDevice ) );

	array[0] = nr3s;
	array[1] = nr2s;
	array[2] = nr1s;

	v_size = nr1s * nr2s * nr3s;

	if ( n_multiplepsic > 0 ) {

		qecheck_cufft_call( cufftPlanMany( &p_global, 3, array, NULL, 1, 0, NULL,1,0,CUFFT_Z2Z,dim_multiplepsic ) );

		if( cufftSetStream(p_global,qecudaStreams[ 0 ]) != CUFFT_SUCCESS ) {
			printf("\n*** CUDA VLOC_PSI_GAMMA *** ERROR *** cufftSetStream for device %d failed!",qe_gpu_bonded[0]);
			fflush(stdout);
			exit( EXIT_FAILURE );
		}

		for(j = 0; j< (m_fake-n_singlepsic); j+=dim_multiplepsic ) {

			qecudaSafeCall( cudaMemset( psic_D , 0, dim_multiplepsic * size_psic * sizeof( cufftDoubleComplex ) ) );

			blocksPerGrid = ( ( n * 2) + __CUDA_THREADPERBLOCK__ - 1) / __CUDA_THREADPERBLOCK__ ;
			for (i = 0; i < dim_multiplepsic; i++  )
			{
				shift = 2*i*size_psic*sizeof(double);
				kernel_init_psic<<<blocksPerGrid, __CUDA_THREADPERBLOCK__ >>>( (int*) nls_D, (int*) nlsm_D, (int*) igk_D, (double*) psi_D, (double*) ( (char*) psic_D + shift), n, m, lda, ((j+i)*2) );
				qecudaGetLastError("kernel launch failure");
			}

			qecheck_cufft_call( cufftExecZ2Z( p_global,  (cufftDoubleComplex*) psic_D, (cufftDoubleComplex*) psic_D , CUFFT_INVERSE ) );

			blocksPerGrid = ( (v_size * 2) + __CUDA_THREADPERBLOCK__  - 1) / __CUDA_THREADPERBLOCK__ ;
			for( i = 0; i < dim_multiplepsic; i++ ) {
				shift = 2*i*size_psic*sizeof(double);
				kernel_vec_prod<<<blocksPerGrid, __CUDA_THREADPERBLOCK__ >>>( (double*) ( (char*)psic_D + shift), (double*) v_D , v_size );
				qecudaGetLastError("kernel launch failure");
			}

			qecheck_cufft_call( cufftExecZ2Z( p_global, (cufftDoubleComplex*) psic_D, (cufftDoubleComplex*) psic_D, CUFFT_FORWARD ) );

			tscale = 1.0 / (double) ( size_psic );
			cublasZdscal(qecudaHandles[ 0 ] , size_psic*dim_multiplepsic, &tscale, (cuDoubleComplex *) psic_D, 1);

			blocksPerGrid = ( ( n * 2) + __CUDA_THREADPERBLOCK__ - 1) / __CUDA_THREADPERBLOCK__ ;
			for (i = 0; i < dim_multiplepsic; i++  )
			{
				shift = 2*i*size_psic*sizeof(double);
				kernel_save_hpsi<<<blocksPerGrid, __CUDA_THREADPERBLOCK__ >>>( (int*) nls_D, (int*) nlsm_D, (int*) igk_D, (double*) hpsi_D, (double*) ( (char*) psic_D + shift), n, m, lda, ((j+i)*2) );
				qecudaGetLastError("kernel launch failure");
			}
		}

		qecheck_cufft_call( cufftDestroy(p_global) );
	}

	if (n_singlepsic > 0 ) {

		qecudaSafeCall( cudaMemset( psic_D , 0, n_singlepsic * size_psic * sizeof( cufftDoubleComplex ) ) );

		qecheck_cufft_call( cufftPlanMany( &p_global, 3, array, NULL, 1, 0, NULL,1,0,CUFFT_Z2Z,n_singlepsic ) );

		if( cufftSetStream(p_global,qecudaStreams[ 0 ]) != CUFFT_SUCCESS ) {
			printf("\n*** CUDA VLOC_PSI_GAMMA *** ERROR *** cufftSetStream for device %d failed!",qe_gpu_bonded[0]);
			fflush(stdout);
			exit( EXIT_FAILURE );
		}

		blocksPerGrid = ( ( n * 2) + __CUDA_THREADPERBLOCK__ - 1) / __CUDA_THREADPERBLOCK__ ;
		for (i = 0; i < n_singlepsic; i++  )
		{
			shift = 2*i*size_psic*sizeof(double);
			kernel_init_psic<<<blocksPerGrid, __CUDA_THREADPERBLOCK__ >>>( (int*) nls_D, (int*) nlsm_D, (int*) igk_D, (double*) psi_D, (double*) ( (char*) psic_D + shift), n, m, lda, (dim_multiplepsic*n_multiplepsic + i)*2 );
			qecudaGetLastError("kernel launch failure");
		}

		qecheck_cufft_call( cufftExecZ2Z( p_global,  (cufftDoubleComplex*) psic_D, (cufftDoubleComplex*) psic_D , CUFFT_INVERSE ) );

		blocksPerGrid = ( (v_size * 2) + __CUDA_THREADPERBLOCK__  - 1) / __CUDA_THREADPERBLOCK__ ;
		for( i = 0; i < n_singlepsic; i++ ){
			shift = 2*i*size_psic*sizeof(double);
			kernel_vec_prod<<<blocksPerGrid, __CUDA_THREADPERBLOCK__ >>>( (double*) ((char*) psic_D + shift), (double*) v_D , v_size );
			qecudaGetLastError("kernel launch failure");
		}

		tscale = 1.0 / (double) ( size_psic );
		cublasZdscal(qecudaHandles[ 0 ] , size_psic * n_singlepsic, &tscale, (cuDoubleComplex *) psic_D, 1);

		qecheck_cufft_call( cufftExecZ2Z( p_global, (cufftDoubleComplex*) psic_D, (cufftDoubleComplex*) psic_D, CUFFT_FORWARD ) );

		blocksPerGrid = ( ( n * 2) + __CUDA_THREADPERBLOCK__ - 1) / __CUDA_THREADPERBLOCK__ ;
		for (i = 0; i < n_singlepsic; i++  )
		{
			shift = 2*i*size_psic*sizeof(double);
			kernel_save_hpsi<<<blocksPerGrid, __CUDA_THREADPERBLOCK__ >>>( (int*) nls_D, (int*) nlsm_D, (int*) igk_D, (double*) hpsi_D, (double*) ( (char*) psic_D + shift), n, m, lda, (dim_multiplepsic*n_multiplepsic + i)*2 );
			qecudaGetLastError("kernel launch failure");
		}

		qecheck_cufft_call( cufftDestroy(p_global) );
	}

	qecudaSafeCall( cudaMemcpy( hpsi, hpsi_D, sizeof( cufftDoubleComplex ) * n * m, cudaMemcpyDeviceToHost ) );
	qecudaSafeCall( cudaMemset( qe_dev_scratch[0], 0, (size_t) qe_gpu_mem_tot[0] ) );

}
#endif
