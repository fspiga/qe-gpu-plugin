/*
 * Copyright (C) 2010-2011 Irish Centre for High-End Computing (ICHEC)
 * Copyright (C) 2001-2011 Quantum ESPRESSO group
 *
 * This file is distributed under the terms of the
 * GNU General Public License. See the file `License'
 * in the root directory of the present distribution,
 * or http://www.gnu.org/copyleft/gpl.txt .
 *
 */

#include <stdlib.h>
#include <stdio.h>

#if defined(__PARA)
#include <mpi.h>
#include <string.h>
#endif

#if defined(__PARA)
int lRank;
char lNodeName[MPI_MAX_PROCESSOR_NAME];
#else
const char lNodeName[] = "localhost";
#endif


#if defined(__PARA)
void paralleldetect_(int * lRankThisNode, int * lSizeThisNode) {

	int lSize, lNodeNameLength, i;
	int lNumRanksThisNode = 0;
	size_t free, total;
	char *lNodeNameRbuf;
	int *lRanksThisNode;

	MPI_Group lWorldGroup;
	MPI_Group lThisNodeGroup;
	MPI_Comm  lThisNodeComm;

	MPI_Comm_rank(MPI_COMM_WORLD, &lRank);
	MPI_Comm_size(MPI_COMM_WORLD, &lSize);

	MPI_Get_processor_name(lNodeName, &lNodeNameLength);

	lNodeNameRbuf = (char*) malloc(lSize * MPI_MAX_PROCESSOR_NAME * sizeof(char));

	MPI_Allgather(lNodeName, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, lNodeNameRbuf, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	// lRanksThisNode is a list of the global ranks running on this node
	lRanksThisNode = (int*) malloc(lSize * sizeof(int));

	for(i=0; i<lSize; i++)
	{
		if(strncmp(lNodeName, (lNodeNameRbuf + i * MPI_MAX_PROCESSOR_NAME), MPI_MAX_PROCESSOR_NAME) == 0)
		{
			lRanksThisNode[lNumRanksThisNode] = i;
			lNumRanksThisNode++;
		}
	}

	/* Create a communicator consisting of the ranks running on this node. */
	MPI_Comm_group(MPI_COMM_WORLD, &lWorldGroup);
	MPI_Group_incl(lWorldGroup, lNumRanksThisNode, lRanksThisNode, &lThisNodeGroup);
	MPI_Comm_create(MPI_COMM_WORLD, lThisNodeGroup, &lThisNodeComm);
	MPI_Comm_rank(lThisNodeComm, lRankThisNode);
	MPI_Comm_size(lThisNodeComm, lSizeThisNode);

}
#endif
