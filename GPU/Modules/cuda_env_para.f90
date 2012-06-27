!
! Copyright (C) 2010-2011 Irish Centre for High-End Computing (ICHEC)
! Copyright (C) 2001-2011 Quantum ESPRESSO group
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!

#if defined(__PARA)
subroutine paralleldetect(lRankThisNode, lSizeThisNode, lRank)

	USE parallel_include

	interface

#if defined(__PGI)
		subroutine quicksort(base, nmemb, elemsize, compar)  bind(C, name='qsort')
			use iso_c_binding
			implicit none
             !pgi$ ignore_tkr base, nmemb, elemsize
			type(C_PTR), value :: base
			integer(C_SIZE_T), value :: nmemb , elemsize
			type(C_FUNPTR), value :: compar
		end subroutine quicksort

		integer function strcmp(a,b) bind(C, name='strcmp')
			use iso_c_binding
			implicit none
             !pgi$ ignore_tkr a, b
			type(C_PTR), value :: a, b
		end function strcmp
#endif

#if defined(__INTEL)
        subroutine quicksort(base, nmemb, elemsize, compar)  bind(C, name='qsort')
            use iso_c_binding
            implicit none
!DEC$ ATTRIBUTES NO_ARG_CHECK :: base, nmemb, elemsize, compar
            type(C_PTR), value :: base
            integer(C_SIZE_T), value :: nmemb , elemsize
            type(C_FUNPTR), value :: compar
        end subroutine quicksort

        integer function strcmp(a,b) bind(C, name='strcmp')
            use iso_c_binding
            implicit none
!DEC$ ATTRIBUTES NO_ARG_CHECK :: a, b
            type(C_PTR), value :: a, b
        end function strcmp
#endif
	end interface

    INTEGER, INTENT(INOUT)  :: lRank, lRankThisNode, lSizeThisNode
	integer :: dev
	character (len=MPI_MAX_PROCESSOR_NAME), allocatable :: hosts(:)
	character (len=MPI_MAX_PROCESSOR_NAME) :: hostname
	integer :: namelength , color , i
	integer :: nProcs , newComm , newRank , ierr, myrank

	call MPI_COMM_SIZE(MPI_COMM_WORLD , nProcs ,  ierr)
	call MPI_COMM_RANK(MPI_COMM_WORLD , lRank , ierr)

	! allocate array of hostnames
	allocate( hosts(0:nProcs -1) )

	! Every process collects the hostname of all the nodes
	call MPI_GET_PROCESSOR_NAME(hostname , namelength , ierr)

    hosts(myrank)=hostname(1:namelength)

    do i=0, nProcs -1
        call MPI_BCAST(hosts(i),MPI_MAX_PROCESSOR_NAME ,MPI_CHARACTER ,i, MPI_COMM_WORLD ,ierr)
    enddo

    call quicksort(hosts, nProcs, MPI_MAX_PROCESSOR_NAME, strcmp)

    color=0
	do i=0, nProcs -1
		if (i > 0) then
		   if (.not. lle( hosts(i-1) ,hosts(i) ) .and. lge( hosts(i-1) ,hosts(i) ) ) color=color+1
		end if

		if (lle( hostname ,hosts(i) ) .and. lge( hostname ,hosts(i) ) )  exit
	end do

	call MPI_COMM_SPLIT(MPI_COMM_WORLD ,color ,0,newComm ,ierr)

	call MPI_COMM_SIZE(newComm , lSizeThisNode ,  ierr)
	call MPI_COMM_RANK(newComm , lRankThisNode , ierr)

	deallocate(hosts)
	! destroy MPI communicator

end subroutine

subroutine myBarrier()
	USE mpi
	integer :: ierr
	CALL  MPI_BARRIER(MPI_COMM_WORLD , ierr)
end subroutine
#endif
