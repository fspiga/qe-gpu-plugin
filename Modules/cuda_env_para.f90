! Copyright (C) 2001-2014 Quantum ESPRESSO Foundation
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!
SUBROUTINE paralleldetect(lRankThisNode, lSizeThisNode, lRank)

#if defined(__MPI)
	USE parallel_include

    INTEGER, INTENT(INOUT)  :: lRank, lRankThisNode, lSizeThisNode
	integer :: dev
	character (len=MPI_MAX_PROCESSOR_NAME), allocatable :: hosts(:)
	character (len=MPI_MAX_PROCESSOR_NAME) :: hostname
	integer :: namelength , color , i
	integer :: nProcs , newComm , newRank , ierr
    integer, allocatable :: marked_hosts(:)
    integer :: lWorldGroup, lThisNodeGroup

    call mpi_comm_size(mpi_comm_world , nProcs ,  ierr)
    call mpi_comm_rank(mpi_comm_world , lRank , ierr)

	! allocate array of hostnames
	allocate( hosts(0:nProcs -1) )

	! Every process collects the hostname of all the nodes
	call mpi_get_processor_name(hostname , namelength , ierr)

    hosts(lRank)=hostname(1:namelength)

    ! TODO: Add a note here
#if defined (__CUDA_WORKAROUND1)
    do i=0, nProcs -1
     call mpi_bcast(hosts(i),MPI_MAX_PROCESSOR_NAME ,&
                       mpi_character ,i, mpi_comm_world ,ierr)
    enddo
#else
    CALL mpi_allgather(hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHARACTER, &
            hosts, MPI_MAX_PROCESSOR_NAME, MPI_CHARACTER, mpi_comm_world, ierr)
#endif

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

	call mpi_comm_size(newComm , lSizeThisNode ,  ierr)
	call mpi_comm_rank(newComm , lRankThisNode , ierr)

	deallocate(hosts)
	! destroy MPI communicator
#endif

END SUBROUTINE

#if defined(__MPI)
subroutine myBarrier()

	USE parallel_include

	integer :: ierr
	CALL  mpi_barrier(mpi_comm_world , ierr)
END SUBROUTINE
#endif
