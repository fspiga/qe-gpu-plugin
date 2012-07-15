!
! Copyright (C) 2001-2012 Quantum ESPRESSO group
! Copyright (C) 2010-2011 Irish Centre for High-End Computing (ICHEC)
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!

#if defined(__MPI)
subroutine paralleldetect(lRankThisNode, lSizeThisNode, lRank)

	USE parallel_include

#if defined(__PGI) || defined(__INTEL)
	interface
        subroutine quicksort(base, nmemb, elemsize, compar)  bind(C, name='qsort')
            use iso_c_binding
            implicit none
!pgi$ ignore_tkr base, nmemb, elemsize, compar
!DEC$ ATTRIBUTES NO_ARG_CHECK :: base, nmemb, elemsize, compar
            type(C_PTR), value :: base
            integer(C_SIZE_T), value :: nmemb , elemsize
            type(C_FUNPTR), value :: compar
        end subroutine quicksort

        integer function strcmp(a,b) bind(C, name='strcmp')
            use iso_c_binding
            implicit none
!DEC$ ATTRIBUTES NO_ARG_CHECK :: a, b
!pgi$ ignore_tkr a, b
            type(C_PTR), value :: a, b
        end function strcmp
	end interface
#endif

    INTEGER, INTENT(INOUT)  :: lRank, lRankThisNode, lSizeThisNode
	integer :: dev
	character (len=MPI_MAX_PROCESSOR_NAME), allocatable :: hosts(:)
	character (len=MPI_MAX_PROCESSOR_NAME) :: hostname
	integer :: namelength , color , i
	integer :: nProcs , newComm , newRank , ierr
#if !defined(__PGI) && !defined(__INTEL)
    integer, allocatable :: marked_hosts(:)
    integer :: lWorldGroup, lThisNodeGroup
#endif

    call mpi_comm_size(mpi_comm_world , nProcs ,  ierr)
    call mpi_comm_rank(mpi_comm_world , lRank , ierr)

	! allocate array of hostnames
	allocate( hosts(0:nProcs -1) )

	! Every process collects the hostname of all the nodes
	call mpi_get_processor_name(hostname , namelength , ierr)

    hosts(lRank)=hostname(1:namelength)

    do i=0, nProcs -1
        call mpi_bcast(hosts(i),MPI_MAX_PROCESSOR_NAME ,mpi_character ,i, mpi_comm_world ,ierr)
    enddo

#if defined(__PGI) || defined(__INTEL)

    call quicksort(hosts, nProcs, mpi_max_processor_name, strcmp)

    color=0
	do i=0, nProcs -1
		if (i > 0) then
		   if (.not. lle( hosts(i-1) ,hosts(i) ) .and. lge( hosts(i-1) ,hosts(i) ) ) color=color+1
		end if

		if (lle( hostname ,hosts(i) ) .and. lge( hostname ,hosts(i) ) )  exit
	end do

	call mpi_comm_split(mpi_comm_world ,color ,0, newComm ,ierr)

#else

    ! For GNU (or other compilers) we do in a old-fashion way because
    ! they might not support "ignoring TKR" features through compiler
    ! directives (NdFilippo)

    allocate( marked_hosts(0:nProcs -1) )

    dev = 0
    do i=0, nProcs -1
        if ( lle(hosts(lRank) ,hosts(i)) .and. lge(hosts(lRank) ,hosts(i)) ) then
            marked_hosts(dev)=i
            dev = dev + 1
        end if
    end do

    ! Create a communicator consisting of the ranks running on this node.
    call mpi_comm_group(mpi_comm_world, lWorldGroup, ierr)
    call mpi_group_incl(lWorldGroup, dev, marked_hosts, lThisNodeGroup, ierr)
    call mpi_comm_create(mpi_comm_world, lThisNodeGroup, newComm, ierr)

    deallocate(marked_hosts)
#endif

	call mpi_comm_size(newComm , lSizeThisNode ,  ierr)
	call mpi_comm_rank(newComm , lRankThisNode , ierr)

	deallocate(hosts)
	! destroy MPI communicator

end subroutine

subroutine myBarrier()

	USE parallel_include

	integer :: ierr
	CALL  mpi_barrier(mpi_comm_world , ierr)
end subroutine
#endif
