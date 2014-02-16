/*@ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 **
 **    Copyright (c) 2012, 
 **        Dave A. May [dave.may@erdw.ethz.ch]
 **        Geophysical Fluid Dynamics, 
 **        Department of Earth Sciences,
 **        ETH Zürich,
 **        Sonneggstrasse 5,
 **        CH-8092 Zurich,
 **        Switzerland
 **
 **    Project:       pTatin3d
 **    Filename:      data_exchanger.c
 **
 **
 **    pTatin3d is free software: you can redistribute it and/or modify
 **    it under the terms of the GNU General Public License as published by
 **    the Free Software Foundation, either version 3 of the License, or
 **    (at your option) any later version.
 **
 **    pTatin3d is distributed in the hope that it will be useful,
 **    but WITHOUT ANY WARRANTY; without even the implied warranty of
 **    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 **    GNU General Public License for more details.
 **
 **    You should have received a copy of the GNU General Public License
 **    along with pTatin3d.  If not, see <http://www.gnu.org/licenses/>.
 **
 **
 **    $Id$
 **
 ** ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~@*/
/*
Build a few basic tools to help with partitioned domains.

1)
On each processor, have a DomainExchangerTopology.
This is a doubly-connected edge list which enumerates the 
communication paths between connected processors. By numbering
these paths we can always uniquely assign message identifers.

        edge
         10
proc  --------->  proc
 0    <--------    1
         11
        twin

Eg: Proc 0 send to proc 1 with message id is 10. To recieve the correct
message, proc 1 looks for the edge connected to proc 0, and then the
messgae id comes from the twin of that edge

2)
A DomainExchangerArrayPacker.
A little function which given a piece of data, will memcpy the data into
an array (which will be sent to procs) into the correct place.

On Proc 1 we sent data to procs 0,2,3. The data is on different lengths.
All data gets jammed into single array. Need to "jam" data into correct locations
The Packer knows how much is to going to each processor and keeps track of the inserts
so as to avoid ever packing TOO much into one slot, and inevatbly corrupting some memory

data to 0    data to 2       data to 3

|--------|-----------------|--|


User has to unpack message themselves. I can get you the pointer for each i 
entry, but you'll have to cast it to the appropriate data type.




Phase A: Build topology

Phase B: Define message lengths

Phase C: Pack data

Phase D: Send data

//
DataExCreate()
// A
DataExTopologyInitialize()
DataExTopologyAddNeighbour()
DataExTopologyAddNeighbour()
DataExTopologyFinalize()
// B
DataExZeroAllSendCount()
DataExAddToSendCount()
DataExAddToSendCount()
DataExAddToSendCount()
// C
DataExPackInitialize()
DataExPackData()
DataExPackData()
DataExPackFinalize()
// D
DataExBegin()
// ... perform any calculations ... ///
DataExEnd()

// Call any getters //


*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include <petsc.h>
#include <petscvec.h>
#include <petscmat.h>

#include "data_exchanger.h"

const char *status_names[] = { "initialized", "finalized", "unknown" };

#undef __FUNCT__  
#define __FUNCT__ "DataExCreate"
DataEx DataExCreate(MPI_Comm comm,const PetscInt count)
{
	DataEx         d;
	PetscErrorCode ierr;
	
	d = (DataEx)malloc( sizeof(struct _p_DataEx) );
	memset( d, 0, sizeof(struct _p_DataEx) );
	
	ierr = MPI_Comm_dup(comm,&d->comm);
	ierr = MPI_Comm_rank(d->comm,&d->rank);
	
	d->instance = count;
	
	d->topology_status        = DEOBJECT_STATE_UNKNOWN;
	d->message_lengths_status = DEOBJECT_STATE_UNKNOWN;
	d->packer_status          = DEOBJECT_STATE_UNKNOWN;
	d->communication_status   = DEOBJECT_STATE_UNKNOWN;
	
	d->n_neighbour_procs = -1;
	d->neighbour_procs   = NULL;
	
	d->messages_to_be_sent      = NULL;
	d->message_offsets          = NULL;
	d->messages_to_be_recvieved = NULL;
	
	d->unit_message_size   = -1;
	d->send_message        = NULL;
	d->send_message_length = -1;
	d->recv_message        = NULL;
	d->recv_message_length = -1;
	d->total_pack_cnt      = -1;
	d->pack_cnt            = NULL;
	
	d->send_tags = NULL;
	d->recv_tags = NULL;
	
	d->_stats    = NULL;
	d->_requests = NULL;
	
	return d;
}

#undef __FUNCT__  
#define __FUNCT__ "DataExView"
PetscErrorCode DataExView(DataEx d)
{
	PetscMPIInt p;
	
	
	PetscFunctionBegin;
	PetscPrintf( PETSC_COMM_WORLD, "DataEx: instance=%d\n",d->instance);
	
	PetscPrintf( PETSC_COMM_WORLD, "  topology status:        %s \n", status_names[d->topology_status]);
	PetscPrintf( PETSC_COMM_WORLD, "  message lengths status: %s \n", status_names[d->message_lengths_status] );
	PetscPrintf( PETSC_COMM_WORLD, "  packer status status:   %s \n", status_names[d->packer_status] );
	PetscPrintf( PETSC_COMM_WORLD, "  communication status:   %s \n", status_names[d->communication_status] );
	
	if (d->topology_status == DEOBJECT_FINALIZED) {
		PetscPrintf( PETSC_COMM_WORLD, "  Topology:\n");
		PetscPrintf( PETSC_COMM_SELF, "    [%d] neighbours: %d \n", d->rank, d->n_neighbour_procs );
		for (p=0; p<d->n_neighbour_procs; p++) {
			PetscPrintf( PETSC_COMM_SELF, "    [%d]   neighbour[%d] = %d \n", d->rank, p, d->neighbour_procs[p]);
		}
	}
	
	if (d->message_lengths_status == DEOBJECT_FINALIZED) {
		PetscPrintf( PETSC_COMM_WORLD, "  Message lengths:\n");
		PetscPrintf( PETSC_COMM_SELF, "    [%d] atomic size: %d \n", d->rank, d->unit_message_size );
		for (p=0; p<d->n_neighbour_procs; p++) {
			PetscPrintf( PETSC_COMM_SELF, "    [%d] >>>>> ( %d units :: tag = %d ) >>>>> [%d] \n", d->rank, d->messages_to_be_sent[p], d->send_tags[p], d->neighbour_procs[p] );
		}
		for (p=0; p<d->n_neighbour_procs; p++) {
			PetscPrintf( PETSC_COMM_SELF, "    [%d] <<<<< ( %d units :: tag = %d ) <<<<< [%d] \n", d->rank, d->messages_to_be_recvieved[p], d->recv_tags[p], d->neighbour_procs[p] );
		}
	}
	
	if (d->packer_status == DEOBJECT_FINALIZED) {
	
	}
	
	if (d->communication_status == DEOBJECT_FINALIZED) {
	
	}
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DataExDestroy"
PetscErrorCode DataExDestroy(DataEx d)
{
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	ierr = MPI_Comm_free(&d->comm);CHKERRQ(ierr);
	
	if (d->neighbour_procs != NULL) {
		free(d->neighbour_procs);
	}
	
	if (d->messages_to_be_sent != NULL) {
		free(d->messages_to_be_sent);
	}
	
	if (d->message_offsets != NULL) {
		free(d->message_offsets);
	}
	
	if (d->messages_to_be_recvieved != NULL) {
		free(d->messages_to_be_recvieved);
	}
	
	if (d->send_message != NULL) {
		free(d->send_message);
	}
	
	if (d->recv_message != NULL) {
		free(d->recv_message);
	}
	
	if (d->pack_cnt != NULL) {
		free(d->pack_cnt);
	}
	
	if (d->send_tags != NULL) {
		free(d->send_tags);
	}
	if (d->recv_tags != NULL) {
		free(d->recv_tags);
	}
	
	if (d->_stats != NULL) {
		free(d->_stats);
	}
	if (d->_requests != NULL) {
		free(d->_requests);
	}
	
	free(d);
	
	PetscFunctionReturn(0);
}

/* === Phase A === */

#undef __FUNCT__  
#define __FUNCT__ "DataExTopologyInitialize"
PetscErrorCode DataExTopologyInitialize(DataEx d)
{
	PetscFunctionBegin;
	d->topology_status = DEOBJECT_INITIALIZED;
	
	d->n_neighbour_procs = 0;
	if (d->neighbour_procs          != NULL)  {  free(d->neighbour_procs);            d->neighbour_procs          = NULL;  }
	if (d->messages_to_be_sent      != NULL)  {  free(d->messages_to_be_sent);        d->messages_to_be_sent      = NULL;  }
	if (d->message_offsets          != NULL)  {  free(d->message_offsets);            d->message_offsets          = NULL;  }
	if (d->messages_to_be_recvieved != NULL)  {  free(d->messages_to_be_recvieved);   d->messages_to_be_recvieved = NULL;  }
	if (d->pack_cnt                 != NULL)  {  free(d->pack_cnt);                   d->pack_cnt                 = NULL;  }
	
	if (d->send_tags != NULL)                 {  free(d->send_tags);                  d->send_tags = NULL;  }
	if (d->recv_tags != NULL)                 {  free(d->recv_tags);                  d->recv_tags = NULL;  }
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DataExTopologyAddNeighbour"
PetscErrorCode DataExTopologyAddNeighbour(DataEx d,const PetscMPIInt proc_id)
{
	PetscMPIInt    n,found;
	PetscMPIInt    nproc;
	PetscErrorCode ierr;
	
	
	PetscFunctionBegin;
	if (d->topology_status == DEOBJECT_FINALIZED) {
		SETERRQ( d->comm, PETSC_ERR_ARG_WRONGSTATE, "Topology has been finalized. To modify or update call DataExTopologyInitialize() first" );
	}
	else if (d->topology_status != DEOBJECT_INITIALIZED) {
		SETERRQ( d->comm, PETSC_ERR_ARG_WRONGSTATE, "Topology must be intialised. Call DataExTopologyInitialize() first" );
	}
	
	/* error on negative entries */
	if (proc_id < 0) {
		SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Trying to set proc neighbour with a rank < 0");
	}
	/* error on ranks larger than number of procs in communicator */
	ierr = MPI_Comm_size(d->comm,&nproc);CHKERRQ(ierr);
	if (proc_id >= nproc) {
		SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Trying to set proc neighbour with a rank >= nproc");
	}
	
	if (d->n_neighbour_procs == 0) {
		d->neighbour_procs = (PetscMPIInt*)malloc( sizeof(PetscMPIInt) );
	}
	
	/* check for proc_id */
	found = 0;
	for (n=0; n<d->n_neighbour_procs; n++) {
		if (d->neighbour_procs[n] == proc_id) {
			found  = 1;
		}
	}
	if (found == 0) { /* add it to list */
		PetscMPIInt *tmp;
		
		tmp = (PetscMPIInt*)realloc( d->neighbour_procs, sizeof(PetscMPIInt)*(d->n_neighbour_procs+1) );
		d->neighbour_procs = tmp;
		
		d->neighbour_procs[ d->n_neighbour_procs ] = proc_id;
		d->n_neighbour_procs++;
	}
	
	PetscFunctionReturn(0);
}

/*
counter: the index of the communication object
N: the number of processors
r0: rank of sender
r1: rank of receiver

procs = { 0, 1, 2, 3 }

0 ==> 0		e=0
0 ==> 1		e=1
0 ==> 2		e=2
0 ==> 3		e=3

1 ==> 0		e=4
1 ==> 1		e=5
1 ==> 2		e=6
1 ==> 3		e=7

2 ==> 0		e=8
2 ==> 1		e=9
2 ==> 2		e=10
2 ==> 3		e=11

3 ==> 0		e=12
3 ==> 1		e=13
3 ==> 2		e=14
3 ==> 3		e=15

If we require that proc A sends to proc B, then the SEND tag index will be given by
  N * rank(A) + rank(B) + offset
If we require that proc A will receive from proc B, then the RECV tag index will be given by
  N * rank(B) + rank(A) + offset

*/
void _get_tags( PetscInt counter, PetscMPIInt N, PetscMPIInt r0,PetscMPIInt r1, PetscMPIInt *_st, PetscMPIInt *_rt )
{
	PetscMPIInt st,rt;
	
	
	st = N*r0 + r1   +   N*N*counter;
	rt = N*r1 + r0   +   N*N*counter;
	
	*_st = st;
	*_rt = rt;
}

/*
Makes the communication map symmetric
*/
#undef __FUNCT__  
#define __FUNCT__ "_DataExCompleteCommunicationMap"
PetscErrorCode _DataExCompleteCommunicationMap(MPI_Comm comm,PetscMPIInt n,PetscMPIInt proc_neighbours[],PetscMPIInt *n_new,PetscMPIInt **proc_neighbours_new)
{
	Mat               A,redA;
	PetscInt          offset,index,i,j,nc;
	PetscInt          n_, *proc_neighbours_;
  PetscInt          size_, rank_i_,_rank_j_;
	PetscMPIInt       size,  rank_i,  rank_j;
	PetscInt          max_nnz;
	PetscScalar       *vals, inserter;
	const PetscInt    *cols;
	const PetscScalar *red_vals;
	PetscMPIInt       _n_new, *_proc_neighbours_new;
	PetscBool         is_seqaij;
	PetscLogDouble    t0,t1;
	PetscErrorCode    ierr;

	
	PetscFunctionBegin;
	PetscPrintf(PETSC_COMM_WORLD,"************************** Starting _DataExCompleteCommunicationMap ************************** \n");
	PetscGetTime(&t0);

	n_ = n;
	ierr = PetscMalloc( sizeof(PetscInt) * n_, &proc_neighbours_ );CHKERRQ(ierr);
	for (i=0; i<n_; i++) {
		proc_neighbours_[i] = proc_neighbours[i];
	}

	ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
	size_ = size;
	ierr = MPI_Comm_rank(comm,&rank_i);CHKERRQ(ierr);
	rank_i_ = rank_i;

	ierr = MatCreate(comm,&A);CHKERRQ(ierr);
	ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,size,size);CHKERRQ(ierr);
	ierr = MatSetType(A,MATAIJ);CHKERRQ(ierr);
	
	ierr = MPI_Allreduce(&n_,&max_nnz,1,MPIU_INT,MPI_MAX,comm);CHKERRQ(ierr);
	ierr = MatSeqAIJSetPreallocation(A,n_,PETSC_NULL);CHKERRQ(ierr);
	PetscPrintf(PETSC_COMM_WORLD,"max_nnz = %D \n", max_nnz );
	//printf("[%d]: nnz = %d \n", rank_i,n_ );
	{
		ierr = MatMPIAIJSetPreallocation(A,1,PETSC_NULL,n_,PETSC_NULL);CHKERRQ(ierr);
	}
		
		
	/* Build original map */
	ierr = PetscMalloc( sizeof(PetscScalar)*n_, &vals );CHKERRQ(ierr);
	for (i=0; i<n_; i++) {
		vals[i] = 1.0;
	}
	ierr = MatSetValues( A, 1,&rank_i_, n_,proc_neighbours_, vals, INSERT_VALUES );CHKERRQ(ierr);
	
	ierr = MatAssemblyBegin(A,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyEnd(A,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
	
	/* Now force all other connections if they are not already there */
	/* It's more efficient to do them all at once */
	for (i=0; i<n_; i++) {
		vals[i] = 2.0;
	}
	ierr = MatSetValues( A, n_,proc_neighbours_, 1,&rank_i_, vals, INSERT_VALUES );CHKERRQ(ierr);
	
	ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

	ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
	ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

	/* 
	What the fuck is this? Is this necessary at all?
	Why cannot I just use MatGetRow on the single row living on the current rank??
	Seems like a fine thing to do provided the operation MatGetRow() is supported for MATMPIAIJ
	~ DAM, Feb 26, 2013 (using petsc 3.2)		
	*/
	/* Duplicate the entire matrix on ALL cpu's */
	/* 
	 MatGetRedundantMatrix is not supported for SEQAIJ, thus we
	 fake a redundant matrix by setting equal to A. This enables the
	 code to run on one cpu (even if this seems slightly odd).
	 */
	is_seqaij = PETSC_FALSE;
	ierr = PetscTypeCompare((PetscObject)A,MATSEQAIJ,&is_seqaij);CHKERRQ(ierr);
//	if (is_seqaij==PETSC_FALSE) {
//		ierr = MatGetRedundantMatrix( A, size_, PETSC_COMM_SELF, size_, MAT_INITIAL_MATRIX, &redA );CHKERRQ(ierr);
//	} else {
		redA = A;
		ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
//	}
	
	if ((n_new != NULL) && (proc_neighbours_new != NULL)) {
	
		ierr = MatGetRow( redA, rank_i_, &nc, &cols, &red_vals );CHKERRQ(ierr);
		
		_n_new = (PetscMPIInt)nc;
		_proc_neighbours_new = (PetscMPIInt*)malloc( sizeof(PetscMPIInt) * _n_new );
		
		for (j=0; j<nc; j++) {
			_proc_neighbours_new[j] = (PetscMPIInt)cols[j];
		}
		ierr = MatRestoreRow( redA, rank_i_, &nc, &cols, &red_vals );CHKERRQ(ierr);
		
		*n_new               = (PetscMPIInt)_n_new;
		*proc_neighbours_new = (PetscMPIInt*)_proc_neighbours_new;
	}
	
	ierr = MatDestroy(&redA);CHKERRQ(ierr);
	ierr = MatDestroy(&A);CHKERRQ(ierr);
	ierr = PetscFree(vals);CHKERRQ(ierr);
	ierr = PetscFree(proc_neighbours_);CHKERRQ(ierr);	

	ierr = MPI_Barrier(comm);CHKERRQ(ierr);
	PetscGetTime(&t1);
	PetscPrintf(PETSC_COMM_WORLD,"************************** Ending _DataExCompleteCommunicationMap [setup time: %1.4e (sec)] ************************** \n",t1-t0);
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DataExTopologyFinalize"
PetscErrorCode DataExTopologyFinalize(DataEx d)
{
	PetscMPIInt    symm_nn;
	PetscMPIInt   *symm_procs;
	PetscMPIInt    r0,n,st,rt;
	PetscMPIInt    nprocs;
	PetscErrorCode ierr;

	
	PetscFunctionBegin;
	if (d->topology_status != DEOBJECT_INITIALIZED) {
		SETERRQ( d->comm, PETSC_ERR_ARG_WRONGSTATE, "Topology must be intialised. Call DataExTopologyInitialize() first" );
	}
	
	/* given infomation about all my neighbours, make map symmetric */
	ierr = _DataExCompleteCommunicationMap( d->comm,d->n_neighbour_procs,d->neighbour_procs, &symm_nn, &symm_procs );CHKERRQ(ierr);
	/* update my arrays */
	free(d->neighbour_procs);
	
	d->n_neighbour_procs = symm_nn;
	d->neighbour_procs   = symm_procs;
	
	
	/* allocates memory */
	if (d->messages_to_be_sent == NULL) {
		d->messages_to_be_sent = (PetscInt*)malloc( sizeof(PetscInt) * d->n_neighbour_procs );
	}
	if (d->message_offsets == NULL) {
		d->message_offsets = (PetscInt*)malloc( sizeof(PetscInt) * d->n_neighbour_procs );
	}
	if (d->messages_to_be_recvieved == NULL) {
		d->messages_to_be_recvieved = (PetscInt*)malloc( sizeof(PetscInt) * d->n_neighbour_procs );
	}
	
	if (d->pack_cnt == NULL) {
		d->pack_cnt = (PetscInt*)malloc( sizeof(PetscInt) * d->n_neighbour_procs );
	}
	
	if (d->_stats == NULL) {
		d->_stats = (MPI_Status*)malloc( sizeof(MPI_Status) * 2*d->n_neighbour_procs );
	}
	if (d->_requests == NULL) {
		d->_requests = (MPI_Request*)malloc( sizeof(MPI_Request) * 2*d->n_neighbour_procs );
	}
	
	if (d->send_tags == NULL) {
		d->send_tags = (int*)malloc( sizeof(int) * d->n_neighbour_procs );
	}
	if (d->recv_tags == NULL) {
		d->recv_tags = (int*)malloc( sizeof(int) * d->n_neighbour_procs );
	}
	
	/* compute message tags */
	ierr = MPI_Comm_size(d->comm,&nprocs);CHKERRQ(ierr);
	r0 = d->rank;
	for (n=0; n<d->n_neighbour_procs; n++) {
		PetscMPIInt r1 = d->neighbour_procs[n];
		
		_get_tags( d->instance, nprocs, r0,r1, &st, &rt );
		
		d->send_tags[n] = (int)st;
		d->recv_tags[n] = (int)rt;
	}
	
	d->topology_status = DEOBJECT_FINALIZED;
	
	PetscFunctionReturn(0);
}

/* === Phase B === */
#undef __FUNCT__  
#define __FUNCT__ "_DataExConvertProcIdToLocalIndex"
PetscErrorCode _DataExConvertProcIdToLocalIndex(DataEx de,PetscMPIInt proc_id,PetscMPIInt *local)
{
	PetscMPIInt i,np;
	
	
	PetscFunctionBegin;
	np = de->n_neighbour_procs;
	
	*local = -1;
	for (i=0; i<np; i++) {
		if (proc_id == de->neighbour_procs[i]) {
			*local = i;
			break;
		}
	}
	PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DataExInitializeSendCount"
PetscErrorCode DataExInitializeSendCount(DataEx de)
{
	PetscMPIInt i;
	
	
	PetscFunctionBegin;
	if (de->topology_status != DEOBJECT_FINALIZED) {
		SETERRQ( de->comm, PETSC_ERR_ORDER, "Topology not finalized" );
	}
	
	de->message_lengths_status = DEOBJECT_INITIALIZED;
	
	for (i=0; i<de->n_neighbour_procs; i++) {
		de->messages_to_be_sent[i] = 0;
	}
	
	PetscFunctionReturn(0);
}

/*
1) only allows counters to be set on neighbouring cpus
*/
#undef __FUNCT__  
#define __FUNCT__ "DataExAddToSendCount"
PetscErrorCode DataExAddToSendCount(DataEx de,const PetscMPIInt proc_id,const PetscInt count)
{
	PetscMPIInt    i,np, valid_neighbour;
	PetscMPIInt    local_val;
	PetscErrorCode ierr;
	

	PetscFunctionBegin;
	if (de->message_lengths_status == DEOBJECT_FINALIZED) {
		SETERRQ( de->comm, PETSC_ERR_ORDER, "Message lengths have been defined. To modify these call DataExInitializeSendCount() first" );
	}
	else if (de->message_lengths_status != DEOBJECT_INITIALIZED) {
		SETERRQ( de->comm, PETSC_ERR_ORDER, "Message lengths must be defined. Call DataExInitializeSendCount() first" );
	}
	
	np = de->n_neighbour_procs;
	
	ierr = _DataExConvertProcIdToLocalIndex( de, proc_id, &local_val );CHKERRQ(ierr); 
	if (local_val == -1) {
		SETERRQ1( PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,"Proc %d is not a valid neighbour rank", proc_id );
	}
	
	de->messages_to_be_sent[local_val] = de->messages_to_be_sent[local_val] + count;
	PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DataExFinalizeSendCount"
PetscErrorCode DataExFinalizeSendCount(DataEx de)
{
	PetscFunctionBegin;
	if (de->message_lengths_status != DEOBJECT_INITIALIZED) {
		SETERRQ( de->comm, PETSC_ERR_ORDER, "Message lengths must be defined. Call DataExInitializeSendCount() first" );
	}
	de->message_lengths_status = DEOBJECT_FINALIZED;
	
	PetscFunctionReturn(0);
}

/* === Phase C === */
/*
 * zero out all send counts
 * free send and recv buffers
 * zeros out message length
 * zeros out all counters
 * zero out packed data counters
*/
#undef __FUNCT__  
#define __FUNCT__ "_DataExInitializeTmpStorage"
PetscErrorCode _DataExInitializeTmpStorage(DataEx de)
{
	PetscMPIInt i,np;
	
	
	PetscFunctionBegin;
	if (de->n_neighbour_procs < 0) {
		SETERRQ( PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Number of neighbour procs < 0");
	}
	if (de->neighbour_procs == NULL) {
		SETERRQ( PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Neighbour proc list is NULL" );
	}
	
	np = de->n_neighbour_procs;
	for (i=0; i<np; i++) {
	/*	de->messages_to_be_sent[i] = -1; */
		de->messages_to_be_recvieved[i] = -1;
	}
	
	if (de->send_message != NULL) {
		free(de->send_message);
		de->send_message = NULL;
	}
	if (de->recv_message != NULL) {
		free(de->recv_message);
		de->recv_message = NULL;
	}
	
	PetscFunctionReturn(0);
}

/*
*) Zeros out pack data counters
*) Ensures mesaage length is set
*) Checks send counts properly initialized
*) allocates space for pack data
*/
#undef __FUNCT__  
#define __FUNCT__ "DataExPackInitialize"
PetscErrorCode DataExPackInitialize(DataEx de,size_t unit_message_size)
{
	PetscMPIInt    i,np;
	PetscInt       total;
	PetscErrorCode ierr;
	
	
	PetscFunctionBegin;
	if (de->topology_status != DEOBJECT_FINALIZED) {
		SETERRQ( de->comm, PETSC_ERR_ORDER, "Topology not finalized" );
	}
	if (de->message_lengths_status != DEOBJECT_FINALIZED) {
		SETERRQ( de->comm, PETSC_ERR_ORDER, "Message lengths not finalized" );
	}
	
	de->packer_status = DEOBJECT_INITIALIZED;
	
	ierr = _DataExInitializeTmpStorage(de);CHKERRQ(ierr);
	
	np = de->n_neighbour_procs;
	
	de->unit_message_size = unit_message_size;
	
	total = 0;
	for (i=0; i<np; i++) {
		if (de->messages_to_be_sent[i] == -1) {
			PetscMPIInt proc_neighour = de->neighbour_procs[i];
			SETERRQ1( PETSC_COMM_SELF, PETSC_ERR_ORDER, "Messages_to_be_sent[neighbour_proc=%d] is un-initialised. Call DataExSetSendCount() first", proc_neighour );
		}
		total = total + de->messages_to_be_sent[i];
	}
	
	/* create space for the data to be sent */
	de->send_message = (void*)malloc( unit_message_size * (total + 1) );
	/* initialize memory */
	memset( de->send_message, 0, unit_message_size * (total + 1) );
	/* set total items to send */
	de->send_message_length = total;
	
	de->message_offsets[0] = 0;
	total = de->messages_to_be_sent[0];
	for (i=1; i<np; i++) {
		de->message_offsets[i] = total;
		total = total + de->messages_to_be_sent[i];
	}
	
	/* init the packer counters */
	de->total_pack_cnt = 0;
	for (i=0; i<np; i++) {
		de->pack_cnt[i] = 0;
	}
	
	PetscFunctionReturn(0);
}

/*
*) Ensures data gets been packed appropriately and no overlaps occur
*/
#undef __FUNCT__  
#define __FUNCT__ "DataExPackData"
PetscErrorCode DataExPackData(DataEx de,PetscMPIInt proc_id,PetscInt n,void *data)
{
	PetscMPIInt    i;
	PetscMPIInt    local;
	PetscInt       insert_location;
	void           *dest;
	PetscErrorCode ierr;
	
	
	PetscFunctionBegin;
	if (de->packer_status == DEOBJECT_FINALIZED) {
		SETERRQ( de->comm, PETSC_ERR_ORDER, "Packed data have been defined. To modify these call DataExInitializeSendCount(), DataExAddToSendCount(), DataExPackInitialize() first" );
	}
	else if (de->packer_status != DEOBJECT_INITIALIZED) {
		SETERRQ( de->comm, PETSC_ERR_ORDER, "Packed data must be defined. Call DataExInitializeSendCount(), DataExAddToSendCount(), DataExPackInitialize() first" );
	}
	
	
	if (de->send_message == NULL){
		SETERRQ( de->comm, PETSC_ERR_ORDER, "send_message is not initialized. Call DataExPackInitialize() first" );
	}
	
	
	ierr = _DataExConvertProcIdToLocalIndex( de, proc_id, &local );CHKERRQ(ierr);
	if (local == -1) {
		SETERRQ1( PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "proc_id %d is not registered neighbour", proc_id );
	}
	
	if (n+de->pack_cnt[local] > de->messages_to_be_sent[local]) {
		SETERRQ3( PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Trying to pack too many entries to be sent to proc %d. Space requested = %d: Attempt to insert %d", 
				proc_id, de->messages_to_be_sent[local], n+de->pack_cnt[local] );
		
		/* don't need this - the catch for too many messages will pick this up. Gives us more info though */
		if (de->packer_status == DEOBJECT_FINALIZED) {
			SETERRQ( de->comm, PETSC_ERR_ARG_WRONG, "Cannot insert any more data. DataExPackFinalize() has been called." );
		}
	}
	
	/* copy memory */
	insert_location = de->message_offsets[local] + de->pack_cnt[local];
	dest = ((char*)de->send_message) + de->unit_message_size*insert_location;
	memcpy( dest, data, de->unit_message_size * n );
	
	/* increment counter */
	de->pack_cnt[local] = de->pack_cnt[local] + n;
	
	PetscFunctionReturn(0);
}

/*
*) Ensures all data has been packed
*/
#undef __FUNCT__  
#define __FUNCT__ "DataExPackFinalize"
PetscErrorCode DataExPackFinalize(DataEx de)
{
	PetscMPIInt i,np;
	PetscInt    total;
	MPI_Status  stat;
	PetscErrorCode ierr;
	
	
	PetscFunctionBegin;
	if (de->packer_status != DEOBJECT_INITIALIZED) {
		SETERRQ( de->comm, PETSC_ERR_ORDER, "Packer has not been initialized. Must call DataExPackInitialize() first." );
	}
	
	np = de->n_neighbour_procs;
	
	for (i=0; i<np; i++) {
		if (de->pack_cnt[i] != de->messages_to_be_sent[i]) {
			SETERRQ3( PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Not all messages for neighbour[%d] have been packed. Expected %d : Inserted %d", 
					de->neighbour_procs[i], de->messages_to_be_sent[i], de->pack_cnt[i] );
		}
	}
	
	/* init */
	for (i=0; i<np; i++) {
		de->messages_to_be_recvieved[i] = -1;
	}
	
	/* figure out the recv counts here */
	for (i=0; i<np; i++) {
	//	MPI_Send( &de->messages_to_be_sent[i], 1, MPI_INT, de->neighbour_procs[i], de->send_tags[i], de->comm );
		ierr = MPI_Isend( &de->messages_to_be_sent[i], 1, MPIU_INT, de->neighbour_procs[i], de->send_tags[i], de->comm, &de->_requests[i] );CHKERRQ(ierr);
	//	MPI_Send( &de->messages_to_be_sent[i], 1, MPI_INT, de->neighbour_procs[i], 0, de->comm );
	}
	for (i=0; i<np; i++) {
	//	MPI_Recv( &de->messages_to_be_recvieved[i], 1, MPI_INT, de->neighbour_procs[i], de->recv_tags[i], de->comm, &stat );
		ierr = MPI_Irecv( &de->messages_to_be_recvieved[i], 1, MPIU_INT, de->neighbour_procs[i], de->recv_tags[i], de->comm, &de->_requests[np+i] );CHKERRQ(ierr);
	//	MPI_Recv( &de->messages_to_be_recvieved[i], 1, MPI_INT, de->neighbour_procs[i], 0, de->comm, &stat );
	}
	ierr = MPI_Waitall( 2*np, de->_requests, de->_stats );CHKERRQ(ierr);
	
	/* create space for the data to be recvieved */
	total = 0;
	for (i=0; i<np; i++) {
		total = total + de->messages_to_be_recvieved[i];
	}
	de->recv_message = (void*)malloc( de->unit_message_size * (total + 1) );
	/* initialize memory */
	memset( de->recv_message, 0, de->unit_message_size * (total + 1) );
	/* set total items to recieve */
	de->recv_message_length = total;
	
	de->packer_status = DEOBJECT_FINALIZED;
	
	de->communication_status = DEOBJECT_INITIALIZED;
	
	PetscFunctionReturn(0);
}

/* do the actual message passing now */
#undef __FUNCT__  
#define __FUNCT__ "DataExBegin"
PetscErrorCode DataExBegin(DataEx de)
{
	PetscMPIInt i,j,np;
	MPI_Status  stat;
	void       *dest;
	PetscInt    length,cnt;
	PetscErrorCode ierr;
	
	
	PetscFunctionBegin;
	if (de->topology_status != DEOBJECT_FINALIZED) {
		SETERRQ( de->comm, PETSC_ERR_ORDER, "Topology not finalized" );
	}
	if (de->message_lengths_status != DEOBJECT_FINALIZED) {
		SETERRQ( de->comm, PETSC_ERR_ORDER, "Message lengths not finalized" );
	}
	if (de->packer_status != DEOBJECT_FINALIZED) {
		SETERRQ( de->comm, PETSC_ERR_ORDER, "Packer not finalized" );
	}
	
	if (de->communication_status == DEOBJECT_FINALIZED) {
		SETERRQ( de->comm, PETSC_ERR_ORDER, "Communication has already been finalized. Must call DataExInitialize() first." );
	}
	
	if (de->recv_message == NULL) {
		SETERRQ( de->comm, PETSC_ERR_ORDER, "recv_message has not been initialized. Must call DataExPackFinalize() first" );
	}
	
	np = de->n_neighbour_procs;
	
	/* == NON BLOCKING == */
	for (i=0; i<np; i++) {
		length = de->messages_to_be_sent[i] * de->unit_message_size;
		dest = ((char*)de->send_message) + de->unit_message_size * de->message_offsets[i];
		ierr = MPI_Isend( dest, length, MPI_CHAR, de->neighbour_procs[i], de->send_tags[i], de->comm, &de->_requests[i] );CHKERRQ(ierr);
	}
	
	PetscFunctionReturn(0);
}

/* do the actual message passing now */
#undef __FUNCT__  
#define __FUNCT__ "DataExEnd"
PetscErrorCode DataExEnd(DataEx de)
{
	PetscMPIInt  i,j,np;
	PetscInt     total;
	MPI_Status   stat;
	PetscInt    *message_recv_offsets;
	void        *dest;
	PetscInt     length,cnt;
	PetscErrorCode ierr;
	
	
	PetscFunctionBegin;
	if (de->communication_status != DEOBJECT_INITIALIZED) {
		SETERRQ( de->comm, PETSC_ERR_ORDER, "Communication has not been initialized. Must call DataExInitialize() first." );
	}
	if (de->recv_message == NULL) {
		SETERRQ( de->comm, PETSC_ERR_ORDER, "recv_message has not been initialized. Must call DataExPackFinalize() first" );
	}
	
	np = de->n_neighbour_procs;
	
	message_recv_offsets = (PetscInt*)malloc( sizeof(PetscInt) * np );
	message_recv_offsets[0] = 0;
	total = de->messages_to_be_recvieved[0];
	for (i=1; i<np; i++) {
		message_recv_offsets[i] = total;
		total = total + de->messages_to_be_recvieved[i];
	}
	
	/* == NON BLOCKING == */
	for (i=0; i<np; i++) {
		length = de->messages_to_be_recvieved[i] * de->unit_message_size;
		dest = ((char*)de->recv_message) + de->unit_message_size * message_recv_offsets[i];
		ierr = MPI_Irecv( dest, length, MPI_CHAR, de->neighbour_procs[i], de->recv_tags[i], de->comm, &de->_requests[np+i] );CHKERRQ(ierr);
	}
	ierr = MPI_Waitall( 2*np, de->_requests, de->_stats );CHKERRQ(ierr);
	
	free(message_recv_offsets);
	
	de->communication_status = DEOBJECT_FINALIZED;
	PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DataExGetSendData"
PetscErrorCode DataExGetSendData(DataEx de,PetscInt *length,void **send)
{
	PetscFunctionBegin;
	if (de->packer_status != DEOBJECT_FINALIZED) {
		SETERRQ( de->comm, PETSC_ERR_ARG_WRONGSTATE, "Data has not finished being packed." );
	}
	*length = de->send_message_length;
	*send   = de->send_message;
	PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DataExGetRecvData"
PetscErrorCode DataExGetRecvData(DataEx de,PetscInt *length,void **recv)
{
	PetscFunctionBegin;
	if (de->communication_status != DEOBJECT_FINALIZED) {
		SETERRQ( de->comm, PETSC_ERR_ARG_WRONGSTATE, "Data has not finished being sent." );
	}
	*length = de->recv_message_length;
	*recv   = de->recv_message;
	PetscFunctionReturn(0);
}
