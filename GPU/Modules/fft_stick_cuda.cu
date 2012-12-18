/*
 * Copyright (C) 2001-2013 Quantum ESPRESSO group
 *
 * This file is distributed under the terms of the
 * GNU General Public License. See the file `License'
 * in the root directory of the present distribution,
 * or http://www.gnu.org/copyleft/gpl.txt .
 *
 */

#include "../../include/c_defs.h"
#include "fftw.h"

#include "cuda_env.h"

extern "C" void check_cufftplan_call(  cufftResult cufft_err, char *origin ){
  switch( cufft_err ){
  case CUFFT_ALLOC_FAILED :
    fprintf( stderr, "\n[%s] CUFFT_ALLOC_FAILED! Program exits... \n", origin );
    exit(1);
    break;

  case CUFFT_INVALID_TYPE :
    fprintf( stderr, "\n[%s] CUFFT_INVALID_TYPE! Program exits... \n", origin  );
    exit(1);
    break;

  case CUFFT_INTERNAL_ERROR :
    fprintf( stderr, "\n[%s] CUFFT_INTERNAL_ERROR! Program exits... \n", origin  );
    exit(1);
    break;

  case CUFFT_SETUP_FAILED :
    fprintf( stderr, "\n[%s] CUFFT_SETUP_FAILED! Program exits... \n", origin  );
    exit(1);
    break;

  case CUFFT_INVALID_SIZE :
    fprintf( stderr, "\n[%s] CUFFT_INVALID_SIZE! Program exits... \n", origin  );
    exit(1);
    break;

  case CUFFT_SUCCESS:
    break;

  default:
    //    fprintf( stderr, "\nCUFFT returned not recognized value! Program exits... \n" );
    fprintf( stderr, "\n[%s] CUFFT returned not recognized value! %d\n", origin , cufft_err );
    break;
  }
}

extern "C" void check_cufft_call(  cufftResult cufft_err, char *origin ){
  switch( cufft_err ){
  case CUFFT_INVALID_PLAN :
    fprintf( stderr, "\n[%s] The plan parameter is not a valid handle! Program exits... \n", origin );
    exit(1);
    break;

  case CUFFT_INVALID_VALUE :
    fprintf( stderr, "\n[%s] The idata, odata, and/or direction parameter is not valid! Program exits... \n", origin  );
    exit(1);
    break;

  case CUFFT_EXEC_FAILED :
    fprintf( stderr, "\n[%s] CUFFT failed to execute the transform on GPU! Program exits... \n", origin  );
    exit(1);
    break;

  case CUFFT_SETUP_FAILED :
    fprintf( stderr, "\n[%s] CUFFT library failed to initialize! Program exits... \n", origin  );
    exit(1);
    break;

  case CUFFT_INVALID_SIZE :
    fprintf( stderr, "\n[%s] The nx parameter is not a supported size! Program exits... \n", origin  );
    exit(1);
    break;

  case CUFFT_INVALID_TYPE :
    fprintf( stderr, "\n[%s] The type parameter is not supported! Program exits... \n", origin  );
    exit(1);
    break;

  case CUFFT_ALLOC_FAILED :
    fprintf( stderr, "\n[%s] Allocation of GPU resources for the plan failed! Program exits... \n", origin  );
    exit(1);
    break;

  case CUFFT_SUCCESS:
    break;

  default:
    //    fprintf( stderr, "\nCUFFT returned not recognized value! Program exits... \n" );
    fprintf( stderr, "\n[%s] CUFFT returned not recognized value! %d\n", origin , cufft_err );
    break;
  }
}

#define __FUNCT__ "fft_stick::create_plan_1d_cuda"
extern "C" int F77_FUNC_ (create_plan_1d_cuda, CREATE_PLAN_1D_CUDA)(cufftHandle *plan, int *nx, int *batch)
{
	check_cufftplan_call( cufftPlan1d( plan, (* nx) , CUFFT_Z2Z, (* batch) ), __FUNCT__ );
  
  return 0;
}
#undef __FUNCT__

#define __FUNCT__ "fft_stick::create_plan_1d_many_cuda"
int F77_FUNC_ (create_plan_1d_many_cuda, CREATE_PLAN_1D_MANY_CUDA)(cufftHandle *plan, int *nx, int *batch)
{
  int dims[1];
  dims[0] = *nx;

  check_cufftplan_call( cufftPlanMany(plan, 1, dims, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, (*batch)), __FUNCT__ );

  return 0;
}
#undef __FUNCT__

#define __FUNCT__ "fft_stick::fft_z_stick_cuda"
extern "C" int F77_FUNC_ (fft_z_stick_cuda, FFT_Z_STICK_CUDA)
   (cufftHandle * plan, FFTW_COMPLEX *zstick, int *ldz, int *nstick_l, int idir)
{
   int howmany, idist;
   cufftDoubleComplex *devPtr;

   howmany = (*nstick_l) ;
   idist   = (*ldz);

   // Bottleneck - experimental code only
   cudaMalloc((void**)&devPtr, sizeof(cufftDoubleComplex)*howmany*idist);
   cudaMemcpy(devPtr, zstick, sizeof(cufftDoubleComplex)*howmany*idist, cudaMemcpyHostToDevice);

   // fftw(*p, howmany, zstick, 1, idist, 0, 0, 0);
   check_cufft_call( cufftExecZ2Z((* plan), (cufftDoubleComplex *)devPtr, (cufftDoubleComplex *)devPtr, idir ), __FUNCT__);

   // Bottleneck - experimental code only
   cudaMemcpy(zstick,(cufftDoubleComplex *)devPtr, sizeof(cufftDoubleComplex)*howmany*idist, cudaMemcpyDeviceToHost);
   cudaFree(devPtr);

   return 0;
}
#undef __FUNCT__

#define __FUNCT__ "fft_stick::destroy_plan_1d_cuda"
extern "C" int F77_FUNC_ (destroy_plan_1d_cuda, DESTROY_PLAN_1D_CUDA)(cufftHandle * plan)
{
  if ( *plan != NULL )
	  check_cufft_call( cufftDestroy( (* plan) ), __FUNCT__ );
  else
	  fprintf(stderr," *** DESTROY_PLAN: warning empty plan ***\n");
  return 0;
}
#undef __FUNCT__

#define __FUNCT__ "fft_stick::destroy_plan_3d_cuda"
extern "C" int F77_FUNC_ (destroy_plan_3d_cuda, DESTROY_PLAN_3D_CUDA)(cufftHandle * plan)
{
  if ( *plan != NULL )
	  check_cufft_call( cufftDestroy( (* plan) ), __FUNCT__ );
  else
	  fprintf(stderr," *** DESTROY_PLAN: warning empty plan ***\n");

  return 0;
}
#undef __FUNCT__

#define __FUNCT__ "fft_stick::create_plan_3d_cuda"
extern "C" int F77_FUNC_ (create_plan_3d_cuda, CREATE_PLAN_3D_CUDA)(cufftHandle *plan, int *nx, int *ny, int *nz )
{
  check_cufft_call(cufftPlan3d( plan, (* nz) , (* ny) , (* nx) , CUFFT_Z2Z ), __FUNCT__ );

  return 0;
}
#undef __FUNCT__

#define __FUNCT__ "fft_stick::fftw_inplace_drv_1d_cuda"
extern "C" int F77_FUNC_ ( fftw_inplace_drv_1d_cuda, FFTW_INPLACE_DRV_1D_CUDA )
     (cufftHandle *plan, long int *a, int * direction)
{
	check_cufft_call( cufftExecZ2Z( (* plan), (cufftDoubleComplex *) (* a), (cufftDoubleComplex *) (* a), (* direction) ), __FUNCT__ );

  return 0;
}
#undef __FUNCT__

#define __FUNCT__ "fft_stick::fftw_inplace_drv_3d_cuda"
extern "C" int F77_FUNC_ ( fftw_inplace_drv_3d_cuda, FFTW_INPLACE_DRV_3D_CUDA )
     (cufftHandle *plan, long int *a, int * direction)
{
  check_cufft_call( cufftExecZ2Z( (* plan), (cufftDoubleComplex *) (* a), (cufftDoubleComplex *) (* a), (* direction) ), __FUNCT__);

  return 0;
}
#undef __FUNCT__
