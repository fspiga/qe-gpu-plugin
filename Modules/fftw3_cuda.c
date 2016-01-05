/*****************************************************************************\
 * Copyright (C) 2001-2013 Quantum ESPRESSO Foundation
 *
 * This file is distributed under the terms of the
 * GNU General Public License. See the file `License'
 * in the root directory of the present distribution,
 * or http://www.gnu.org/copyleft/gpl.txt .
\*****************************************************************************/

#include "c_defs.h"

// Getting CUDA_VERSION ...
#include "cuda.h"

#if defined(__FFTW3) && defined(__CUDA) && defined(__CUFFTW) && (CUDA_VERSION >= 5050)

#include "cufftw.h"

int F77_FUNC_( dfftw_plan_dft_3d, DFFTW_PLAN_DFT_3D )
               (fftw_plan* plan, int* nx, int* ny, int* nz, fftw_complex *in,
			    fftw_complex *out, int* sign, unsigned* flags)
{
    *plan = fftw_plan_dft_3d(*nz, *ny, *nx, in, out, *sign, *flags);
    return 0;
}


int F77_FUNC_( dfftw_execute_dft, DFFTW_EXECUTE_DFT )
              (const fftw_plan* plan, fftw_complex* in, fftw_complex* out)
{
    fftw_execute_dft(*plan, in, out);
    return 0;
}

int F77_FUNC_(dfftw_plan_many_dft, DFFTW_PLAN_MANY_DFT)
              (fftw_plan* plan, int* rank, const int* n, int* howmany,
	           fftw_complex *in, const int* inembed, int* istride, int* idist,
			   fftw_complex *out, const int* onembed, int* ostride, int* odist,
		       int* sign, unsigned* flags)
{
	int nn[7];
	int inem[7];
	int onem[7];
	int i;

    for (i=0; i < *rank; i++) {
        nn[i] = n[*rank-i-1];
        inem[i] = inembed[*rank-i-1];
        onem[i] = onembed[*rank-i-1];
    }

    *plan = fftw_plan_many_dft(*rank, nn, *howmany, in, inem, *istride, *idist,
						       out, onem, *ostride, *odist, *sign, *flags);
    return 0;
}


int F77_FUNC_(dfftw_destroy_plan, DFFTW_DESTROY_PLAN)
              (fftw_plan* plan)
{
      fftw_destroy_plan(*plan);
      return 0;
}

#else

/* This dummy subroutine is there for compilers that dislike empty files */

int dumcufftwdrv() {
  return 0;
}

#endif
