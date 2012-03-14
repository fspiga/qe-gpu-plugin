!
! Copyright (C) 2010-2011 Irish Centre for High-End Computing (ICHEC)
! Copyright (C) 2001-2011 Quantum ESPRESSO group
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!

#if defined(__CUDA)
! Interface to cudaMallocHost and cudaFree
MODULE cuda_mem_alloc

    ! equivalent type
    INTEGER, PARAMETER :: fp_kind = selected_real_kind(15,307)
    INTEGER, PARAMETER :: int_kind = kind(1)

   INTERFACE

	! CUDA cudaMallocHost
    FUNCTION cudaHostAlloc(buffer, size, flag)  bind(C,name='cudaHostAlloc')
		USE iso_c_binding
		IMPLICIT NONE
		INTEGER (C_INT) :: cudaHostAlloc
		TYPE (C_PTR) :: buffer
		INTEGER (C_SIZE_T), value :: size
		INTEGER (C_SIZE_T), value :: flag
	END FUNCTION cudaHostAlloc

	! CUDA cudaFreeHost
	FUNCTION cudaFreeHost(buffer)  bind(C,name='cudaFreeHost')
		USE iso_c_binding
		IMPLICIT NONE
		INTEGER (C_INT) :: cudaFreeHost
		TYPE (C_PTR), value :: buffer
	END FUNCTION cudaFreeHost

	END INTERFACE

END MODULE cuda_mem_alloc
#else

module cuda_mem_alloc
	INTERFACE
	SUBROUTINE fake_cudaHostAlloc()
	END SUBROUTINE fake_cudaHostAlloc
	END INTERFACE
END MODULE cuda_mem_alloc

#endif
