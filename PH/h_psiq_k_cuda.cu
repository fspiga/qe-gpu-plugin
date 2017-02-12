/*!
 * \author Kwanmgin Yu <kyu@bnl.gov>
 *
 */


#include <iostream>

#include <stdlib.h>
#include <stdio.h>
#include "cuda_env.h"
#include <omp.h>

using namespace std;

#define MAX_STREAMS 1

typedef double fftw_complex[2];


int cufft_nr1s;
int cufft_nr2s;
int cufft_nr3s;


cufftHandle p_global[MAX_STREAMS];

#if defined(__CUDA_DEBUG)
double h_psiq_time_init_1;
double h_psiq_time_init_2;
double h_psiq_time_init_3;
double h_psiq_time_init_4;
double h_psiq_time_init_5;
double h_psiq_time_core;
double h_psiq_time_down;

extern "C" void h_psiq_cuda_k_time_init_() {
  h_psiq_time_init_1 = 0.0;
  h_psiq_time_init_2 = 0.0;
  h_psiq_time_init_3 = 0.0;
  h_psiq_time_init_4 = 0.0;
  h_psiq_time_init_5 = 0.0;
  h_psiq_time_core = 0.0;
  h_psiq_time_down = 0.0;
}

extern "C" double get_h_psiq_time_init_1_() {return h_psiq_time_init_1;}
extern "C" double get_h_psiq_time_init_2_() {return h_psiq_time_init_2;}
extern "C" double get_h_psiq_time_init_3_() {return h_psiq_time_init_3;}
extern "C" double get_h_psiq_time_init_4_() {return h_psiq_time_init_4;}
extern "C" double get_h_psiq_time_init_5_() {return h_psiq_time_init_5;}
extern "C" double get_h_psiq_time_core_() {return h_psiq_time_core;}
extern "C" double get_h_psiq_time_down_() {return h_psiq_time_down;}
#endif


__global__ void h_psiq_kernel_vec_prod_k( double *a, const  double * __restrict b, int dimx )
{
  register int ix = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
  register double sup;
  register int ii = ix / 2;

  if ( ix < ( dimx * 2 ) ) {
    sup = a[ix] * b[ii];
    a[ix] = sup;
  }

  return;
}


__global__ void h_psiq_kernel_init_hpsic(double * hpsi, double *psi, const int row_size, const int n, const int m, double* g2kin )
{
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
        
  int psi_index;

  for(int ibnd =  0; ibnd < m; ibnd = ibnd + 1) {
    if ( ix < n ) {
        
      psi_index = (ix + ( ibnd * row_size )) * 2;

      hpsi[ psi_index ] = g2kin[ix] * psi[ psi_index ];
      hpsi[ psi_index + 1 ] = g2kin[ix] * psi[ psi_index + 1 ];
  
    }
  }
      
  return;

}

__global__ void h_psiq_kernel_init_psic_k( const  int * __restrict nls, const  int * __restrict igkq, const  double * __restrict psi, double *psic, const int n, const int lda, const int ibnd, const int inx_max )
{
        int ix = blockIdx.x * blockDim.x + threadIdx.x;
        int psic_index_nls, psi_index = ( ix + ( ibnd * lda ) ) * 2;

        if ( ix < n ) {
          psic_index_nls = ( nls[ igkq[ ix ] - 1 ] - 1 ) * 2;
          if (psic_index_nls <= inx_max) {
             psic[ psic_index_nls ] = psi[ psi_index ];
             psic[ psic_index_nls + 1 ] = psi[ psi_index + 1 ];
  
          }
        }
        return;

}

__global__ void h_psiq_kernel_save_hpsi_k( const  int * __restrict nls, const  int * __restrict igkq, double * hpsi, const  double * __restrict psic, const int n, const int ibnd, const int lda)
{
        int ix = blockIdx.x * blockDim.x + threadIdx.x;
        int pos = ibnd * lda;
        int psic_index_nls, psi_index = (ix + pos) * 2;

        if ( ix < (n) ) {
                psic_index_nls = (nls[ igkq[ ix ] - 1 ] - 1) * 2;
                hpsi[ psi_index ] += psic[ psic_index_nls ];
                hpsi[ psi_index + 1 ] += psic[ psic_index_nls + 1 ];
        }

  return;

}

extern "C" void h_psiq_cuda_k_cufftplan_init_(int * ptr_nr1s, int * ptr_nr2s, int * ptr_nr3s) {
  cufft_nr1s = *ptr_nr1s;
  cufft_nr2s = *ptr_nr2s;
  cufft_nr3s = *ptr_nr3s;
 
  qecheck_cufft_call( cufftPlan3d( &p_global[0], *ptr_nr3s, *ptr_nr2s,  *ptr_nr1s, CUFFT_Z2Z ) );
}

extern "C" void h_psiq_cuda_k_cufftplan_destroy_() {
  qecheck_cufft_call( cufftDestroy(p_global[0]) );
}

extern "C" int h_psiq_cuda_k_( int * ptr_lda, int * ptr_npol, int * ptr_nrxxs, int * ptr_nr1s, int * ptr_nr2s, int * ptr_nr3s, int * ptr_n, int * ptr_m, cufftDoubleComplex * psi, double * vrs, fftw_complex * hpsi, int * igkq, int * nls, int * ptr_ngms, double* g2kin)
{
#if defined(__CUDA_DEBUG)
        std::cout << "[CUDA DEBUG] H_PSI_K" << endl;
#endif
        



        void * psic_D, * psi_D, * hpsi_D; // cufftDoubleComplex *
        void * vrs_D; // double *
        void * igkq_D, * nls_D; // int*
        void * g2kin_D; // double*

        double tscale;

        int n = (* ptr_n);
        int m = (* ptr_m);
        int nr1s = (* ptr_nr1s);
        int nr2s = (* ptr_nr2s);
        int nr3s = (* ptr_nr3s);
        int nrxxs = (* ptr_nrxxs);
        int ngms = (* ptr_ngms);
        int lda = (* ptr_lda);
	int npol = (* ptr_npol);
        
#if defined(__CUDA_NOALLOC)
        int ierr;
#endif
        int size_psic = nr1s * nr2s * nr3s;
        int ibnd;

        dim3 threads2_psic(qe_gpu_kernel_launch[0].__CUDA_TxB_VLOCPSI_PSIC);
        dim3 grid2_psic( qe_compute_num_blocks(n, threads2_psic.x) );

        dim3 threads2_prod(qe_gpu_kernel_launch[0].__CUDA_TxB_VLOCPSI_PROD);
        dim3 grid2_prod( qe_compute_num_blocks((nrxxs * 2), threads2_prod.x) );

#if defined(__CUDA_DEBUG)        
        double start_time, end_time;
#endif 

#if defined(__CUDA_DEBUG)
        //*
        std::cout << "m : " << m << std::endl;
        std::cout << "n : " << n << std::endl;
        std::cout << "nrxxs : " << nrxxs << std::endl;
        std::cout << "lda : " << lda << std::endl;
        std::cout << "ngms : " << ngms << std::endl;
        std::cout << "size_psic : " << size_psic << std::endl;
        std::cout << "nr1s : " << nr1s << std::endl;
        std::cout << "nr2s : " << nr2s << std::endl;
        std::cout << "nr3s : " << nr3s << std::endl;
        //*/
#endif 
        
        

#if defined(__CUDA_DEBUG)
        start_time=omp_get_wtime();
#endif
      	
        cudaStream_t hpsiStreams[MAX_STREAMS];
        
        
        for ( int q = 0; q < MAX_STREAMS; q++ ) cudaStreamCreate( &hpsiStreams[q] );
                
        if ( grid2_psic.x > qe_gpu_kernel_launch[0].__MAXNUMBLOCKS) {
                fprintf( stderr, "\n[H_PSI_K] h_psiq_kernel_init_psic_k cannot run, blocks requested ( %d ) > blocks allowed!!!", grid2_psic.x );
                return 1;
        }

        if ( grid2_prod.x > qe_gpu_kernel_launch[0].__MAXNUMBLOCKS) {
                fprintf( stderr, "\n[H_PSI_K] h_psiq_kernel_vec_prod cannot run, blocks requested ( %d ) > blocks allowed!!!", grid2_prod.x );
                return 1;
        }
        

        cudaSetDevice(qe_gpu_bonded[0]);
        
#if defined(__CUDA_NOALLOC)
        ierr = cudaMalloc ( (void**) &(qe_dev_scratch[0]), (size_t) qe_gpu_mem_unused[0] );
        qecudaGenericErr((cudaError_t) ierr, "H_PSI_K", "error in memory allocation (qe_dev_scratch)");
#endif



        int n_streams = MAX_STREAMS+1;
        size_t shift = 0;
        do {
                shift = 0;
                n_streams--;
                psic_D = (char*) qe_dev_scratch[0] + shift;
                shift += ( size_psic * n_streams)*sizeof( cufftDoubleComplex );
                hpsi_D = (char*) qe_dev_scratch[0] + shift;
                shift += ( (lda*npol) * m )*sizeof( cufftDoubleComplex );
                psi_D = (char*) qe_dev_scratch[0] + shift;
                shift += ( (lda*npol) * m )*sizeof( cufftDoubleComplex );
                vrs_D = (char*) qe_dev_scratch[0] + shift;
                shift += ( nrxxs )*sizeof( double );
                nls_D = (char*) qe_dev_scratch[0] + shift;
                shift += ( (ngms % 2 == 0)? ngms : ngms + 1 )*sizeof(int);
                igkq_D = (char*) qe_dev_scratch[0] + shift;
                shift += ( (n % 2 == 0)? n : n + 1 )*sizeof(int);
                g2kin_D = (char*) qe_dev_scratch[0] + shift;
                shift += ( lda * m )*sizeof( double );
                
        } while (n_streams > 0 && shift > qe_gpu_mem_unused[0]);
        

        if ( n_streams < 1 ) {
                fprintf( stderr, "\n[H_PSI_K] Problem don't fit in GPU memory --- memory requested ( %lu ) > memory allocated  (%lu )!!!", shift, qe_gpu_mem_unused[0] );
#if defined(__CUDA_NOALLOC)
                ierr = cudaFree ( qe_dev_scratch[0] );
                qecudaGenericErr((cudaError_t) ierr, "H_PSI_K", "error memory release (qe_dev_scratch)");
#endif
                return 1;
        }
#if defined(__CUDA_DEBUG)
        end_time=omp_get_wtime();
        h_psiq_time_init_1 += (end_time - start_time);
#endif
        

#if defined(__CUDA_DEBUG)
        start_time=omp_get_wtime();
#endif
        qecudaSafeCall( cudaMemcpy( psi_D, psi,  sizeof( cufftDoubleComplex ) * (lda*npol) * m, cudaMemcpyHostToDevice ) );
        qecudaSafeCall( cudaMemcpy( vrs_D, vrs,  sizeof( double ) * nrxxs, cudaMemcpyHostToDevice ) );
        qecudaSafeCall( cudaMemcpy( nls_D, nls,  sizeof( int ) * ngms, cudaMemcpyHostToDevice ) );
        qecudaSafeCall( cudaMemcpy( igkq_D, igkq,  sizeof( int ) * n, cudaMemcpyHostToDevice ) );
      	
      	qecudaSafeCall( cudaMemcpy( g2kin_D, g2kin,  sizeof( double ) * n, cudaMemcpyHostToDevice ) );

#if defined(__CUDA_DEBUG)
      	end_time=omp_get_wtime();
        h_psiq_time_init_2 += (end_time - start_time);
#endif
     
      	
#if defined(__CUDA_DEBUG)
      	start_time=omp_get_wtime();
#endif

      	h_psiq_kernel_init_hpsic<<< grid2_psic, threads2_psic >>>( (double *) hpsi_D, (double *) psi_D, lda*npol, n, m, (double *) g2kin_D );

#if defined(__CUDA_DEBUG)
      	end_time=omp_get_wtime();
        h_psiq_time_init_3 += (end_time - start_time);
#endif


#if defined(__CUDA_DEBUG)
      	start_time=omp_get_wtime();
#endif
        if( (cufft_nr1s != *ptr_nr1s) || (cufft_nr2s != *ptr_nr2s) || (cufft_nr3s != *ptr_nr3s) ) {
          qecheck_cufft_call( cufftDestroy(p_global[0]) );
          
          cufft_nr1s = *ptr_nr1s;
          cufft_nr2s = *ptr_nr2s;
          cufft_nr3s = *ptr_nr3s;
          
          qecheck_cufft_call( cufftPlan3d( &p_global[0], *ptr_nr3s, *ptr_nr2s,  *ptr_nr1s, CUFFT_Z2Z ) );
        }
#if defined(__CUDA_DEBUG)
      	end_time=omp_get_wtime();
        h_psiq_time_init_4 += (end_time - start_time);
#endif



#if defined(__CUDA_DEBUG)
        start_time=omp_get_wtime();
#endif
        int this_stream = -1;
        double * this_psic_D;
        for ( ibnd =  0; ibnd < m; ibnd = ibnd + 1) {

                this_stream = (this_stream+1)%n_streams;
                cudaStreamSynchronize(hpsiStreams[this_stream]);
                this_psic_D = ((double*)psic_D) + 2 * size_psic * this_stream;
                qecudaSafeCall( cudaMemsetAsync( this_psic_D, 0, size_psic * sizeof( cufftDoubleComplex ), hpsiStreams[this_stream]) );

		
		h_psiq_kernel_init_psic_k<<< grid2_psic, threads2_psic, 0, hpsiStreams[this_stream] >>>( (int *) nls_D, (int *) igkq_D, (double *) psi_D, (double *) this_psic_D, n, lda, ibnd, size_psic*2 );

                qecudaGetLastError("kernel launch failure");

                qecheck_cufft_call( cufftSetStream( p_global[this_stream], hpsiStreams[this_stream] ) );
                qecheck_cufft_call( cufftExecZ2Z( p_global[this_stream], (cufftDoubleComplex *) this_psic_D, (cufftDoubleComplex *) this_psic_D, CUFFT_INVERSE ) );

                h_psiq_kernel_vec_prod_k<<< grid2_prod, threads2_prod, 0, hpsiStreams[this_stream] >>>( (double *) this_psic_D, (double *) vrs_D , nrxxs );
                qecudaGetLastError("kernel launch failure");

                qecheck_cufft_call( cufftExecZ2Z( p_global[this_stream], (cufftDoubleComplex *) this_psic_D, (cufftDoubleComplex *)this_psic_D, CUFFT_FORWARD ) );

                tscale = 1.0 / (double) ( size_psic );

                cublasSetStream( qecudaHandles[ 0 ], hpsiStreams[this_stream] );
                cublasZdscal(qecudaHandles[ 0 ], size_psic, &tscale, (cuDoubleComplex *) this_psic_D, 1);


                h_psiq_kernel_save_hpsi_k<<<grid2_psic, threads2_psic, 0, hpsiStreams[this_stream]>>>( (int*)nls_D, (int*)igkq_D, (double*)hpsi_D, (double*)this_psic_D, n, ibnd, lda );

        }
#if defined(__CUDA_DEBUG)
        end_time=omp_get_wtime();
        h_psiq_time_core += (end_time - start_time);
#endif
        
        
#if defined(__CUDA_DEBUG)
        start_time=omp_get_wtime();
#endif
        qecudaSafeCall( cudaMemcpy ( (cufftDoubleComplex*)hpsi, hpsi_D, sizeof(cufftDoubleComplex) * (lda*npol) * m, cudaMemcpyDeviceToHost) );
#if defined(__CUDA_DEBUG)
        end_time=omp_get_wtime();
        h_psiq_time_down += (end_time - start_time);
#endif

	

        for ( int q = 0; q < MAX_STREAMS; q++ ) cudaStreamDestroy( hpsiStreams[q] );
        

#if defined(__CUDA_NOALLOC)
        ierr = cudaFree ( qe_dev_scratch[0] );
        qecudaGenericErr((cudaError_t) ierr, "H_PSI_K", "error memory release (qe_dev_scratch)");
#else

#if defined(__CUDA_KERNEL_MEMSET)
        qecudaSafeCall( cudaMemset( qe_dev_scratch[0], 0, (size_t) qe_gpu_mem_unused[0] ) );
#endif

#endif


        return 0;

}

