/*****************************************************************************\
 * Copyright (C) 2001-2013 Quantum ESPRESSO Foundation
 *
 * This file is distributed under the terms of the
 * GNU General Public License. See the file `License'
 * in the root directory of the present distribution,
 * or http://www.gnu.org/copyleft/gpl.txt .
\*****************************************************************************/

// Active by default...
#define __CUDA_QVAN2

#define sixth 1.0/6.0

#include <stdio.h>

__global__ void qvan_kernel( double dqi, double *qmod, double *qrad, double *ylmk0,
		double sig, double *qg, int qg_stride, int ngy)
{
	double qm, work;
	double px, ux, vx, wx, uvx, pwx;
	int i0, i1, i2, i3;
	int ig = threadIdx.x + blockIdx.x * blockDim.x;
	if (ig < ngy) {
		qm = qmod[ig] * dqi;
		px = qm - int(qm);
		ux = 1.0 - px;
		vx = 2.0 - px;
		wx = 3.0 - px;
		// Not adding 1 here since it's an array index
		i0 = int(qm);
		i1 = i0 + 1;
		i2 = i0 + 2;
		i3 = i0 + 3;
		uvx = ux * vx * sixth;
		pwx = px * wx * 0.5;
		work = qrad [i0] * uvx * wx +
				qrad [i1] * pwx * vx -
				qrad [i2] * pwx * ux +
				qrad [i3] * px * uvx;

		qg[qg_stride*ig] = qg[qg_stride*ig ] + sig * ylmk0[ig] * work;
	}

} 

extern "C" int qvan2_cuda( int ngy, int ih, int jh, 
		int np, double *qmod, double *qg, double *ylmk0,
		int ylmk0_s1, int nlx,
		double dq, double *qrad, int qrad_s1, int qrad_s2,
		int qrad_s3, int *indv, int indv_s1,
		int *nhtolm, int nhtolm_s1,
		int nbetam, int *lpx, int lpx_s1,
		int *lpl, int lpl_s1, int lpl_s2,
		double *ap, int ap_s1, int ap_s2, cudaStream_t st)
{
	// adjust array indeces to begin with 0
	np = np - 1;
	ih = ih - 1;
	jh = jh - 1;
	// double *qmod, *ylmk0;

	// the nonzero real or imaginary part of (-i)^L
	double sig;

#if defined(__CUDA_QVAN2)
        cudaError_t err;
#else
	int qg_s1 = 2;
	int i0, i1, i2, i3, ig;
	double qm, px, ux, vx, wx, uvx, pwx, work;
	double qm1 = -1.0; // any number smaller than qmod[1]
#endif
        int nb, mb, ijv, ivl, jvl, lp, l, lm, ind;
	double dqi;

	//
	//     compute the indices which correspond to ih,jh
	//
	dqi = 1.0/dq;
	nb = indv[ih + indv_s1*np ];
	mb = indv[jh + indv_s1*np ];

	if (nb > mb)
		ijv = nb * (nb - 1) / 2 + mb - 1;
	else
		ijv = mb * (mb - 1) / 2 + nb - 1;

	ivl = nhtolm[ih + nhtolm_s1*np ] - 1;
	jvl = nhtolm[jh + nhtolm_s1*np ] - 1;

	if (nb > nbetam || mb > nbetam) {
		fprintf(stderr, "  In qvan2, wrong dimensions (1) %d\n", max(nb,mb));
		exit(EXIT_FAILURE);
	}

	if (ivl > nlx || jvl > nlx) {
		fprintf(stderr, "  In qvan2, wrong dimensions (2) %d\n", max(ivl, jvl));
		exit(EXIT_FAILURE);
	}

#if defined(__CUDA_QVAN2)
	cudaMemsetAsync(qg, 0, sizeof(double)*ngy*2, st);
	err = cudaGetLastError();
	if (err) printf("qvan memset error number %d: %s\n", err, cudaGetErrorString(err));
#else
	for (int q=0;q<2*ngy;q++) qg[q] = 0.0;
#endif

	for (lm=0;lm<lpx[ivl + lpx_s1*jvl];lm++) {
		lp = lpl[ivl + lpl_s1*(jvl + lpl_s2*lm) ];
		if (lp == 1) {
			l = 0;
			sig = 1.0;
			ind = 0;
		} else if ( lp <= 4) {
			l = 1;
			sig =-1.0;
			ind = 1;
		} else if ( lp <= 9 ) {
			l = 2;
			sig =-1.0;
			ind = 0;
		} else if ( lp <= 16 ) {
			l = 3;
			sig = 1.0;
			ind = 1;
		} else if ( lp <= 25 ) {
			l = 4;
			sig = 1.0;
			ind = 0;
		} else if ( lp <= 36 ) {
			l = 5;
			sig =-1.0;
			ind = 1;
		} else {
			l = 6;
			sig =-1.0;
			ind = 0;
		}

		//Note: To avoid major changes to the comparisons above, we're leaving lp alone
		//    and subtracting 1 here
		sig = sig * ap[lp-1 + ap_s1*(ivl + ap_s2*jvl) ];

#if defined(__CUDA_QVAN2)

		qvan_kernel<<<(ngy+127)/128,128,0,st>>>(dqi, qmod, qrad + (qrad_s1*(ijv + qrad_s2*(l + qrad_s3*np))),
				ylmk0+(lp-1)*ylmk0_s1, sig, qg + ind, 2, ngy);
		err = cudaGetLastError();
		if (err) printf("qvan_kernel error number %d: %s\n", err, cudaGetErrorString(err));

#else

#pragma omp parallel for default(shared), private(qm,px,ux,vx,wx,i0,i1,i2,i3,uvx,pwx,work)
		for (ig = 0; ig < ngy; ig++ ) {

#if ! defined(__OPENMP)
			if ( abs(qmod[ig] - qm1) > 1.0) {
#endif
				qm = qmod[ig] * dqi;
				px = qm - int(qm);
				ux = 1.0 - px;
				vx = 2.0 - px;
				wx = 3.0 - px;
				// Not adding 1 here since it's an array index
				i0 = int(qm);
				i1 = i0 + 1;
				i2 = i0 + 2;
				i3 = i0 + 3;
				uvx = ux * vx * sixth;
				pwx = px * wx * 0.5;
				work = qrad [i0+ qrad_s1*(ijv + qrad_s2*(l + qrad_s3*np))] * uvx * wx +
						qrad [i1+ qrad_s1*(ijv + qrad_s2*(l + qrad_s3*np))] * pwx * vx -
						qrad [i2+ qrad_s1*(ijv + qrad_s2*(l + qrad_s3*np))] * pwx * ux +
						qrad [i3+ qrad_s1*(ijv + qrad_s2*(l + qrad_s3*np))] * px * uvx;

#if ! defined(__OPENMP)
				qm1 = qmod[ig];
			}
#endif
			qg[ind + qg_s1*ig] = qg[ind + qg_s1*ig ] + sig * ylmk0[ig + ylmk0_s1*(lp-1) ] * work;
		}
#endif
	}

	return 0;
}
