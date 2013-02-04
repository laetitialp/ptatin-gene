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
 **    Filename:      dmda_bcs.c
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

#include "petsc.h"
#include "petscdm.h"
#include "dmda_bcs.h"

#undef __FUNCT__
#define __FUNCT__ "BCListIsDirichlet"
PetscErrorCode BCListIsDirichlet(PetscInt value,PetscBool *flg)
{
	if (value==BCList_DIRICHLET) { *flg = PETSC_TRUE;  }
	else                         { *flg = PETSC_FALSE; }
	PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "BCListInitialize"
PetscErrorCode BCListInitialize(BCList list)
{
	PetscInt       n;
	PetscErrorCode ierr;
	
	for (n=0; n<list->L; n++) {
		list->dofidx_global[n] = 0;
		list->scale_global[n]  = 1.0;
	}
	
	for (n=0; n<list->L_local; n++) {
		list->dofidx_local[n] = 0.0;
		list->vals_local[n]   = 0.0;
	}
	
	PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "BCListCreate"
PetscErrorCode BCListCreate(BCList *list)
{
	BCList ll;
	PetscErrorCode ierr;
	
	ierr = PetscMalloc( sizeof(struct _p_BCList),&ll);CHKERRQ(ierr);
	ll->allEmpty = PETSC_FALSE;
	*list = ll;
	PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "BCListDestroy"
PetscErrorCode BCListDestroy(BCList *list)
{
	BCList         ll = *list;
	PetscErrorCode ierr;

	{
		PetscBool isdir;
		PetscInt n,cnt;

		cnt = 0;
		for (n=0; n<ll->L; n++) {
			ierr = BCListIsDirichlet(ll->dofidx_global[n],&isdir);CHKERRQ(ierr);
			if (isdir==PETSC_TRUE) { cnt++; }
		}
		//PetscPrintf(PETSC_COMM_WORLD,"BCList(global): only %D of %D entries need to be stored \n", cnt, ll->L );
		cnt = 0;
		for (n=0; n<ll->L_local; n++) {
			ierr = BCListIsDirichlet(ll->dofidx_local[n],&isdir);CHKERRQ(ierr);
			if (isdir==PETSC_TRUE) { cnt++; }
		}
		//PetscPrintf(PETSC_COMM_WORLD,"BCList(local): only %D of %D entries need to be stored \n", cnt, ll->L );
	}

	if (ll->vals_global != ll->vals_local) {
		ierr = PetscFree(ll->vals_global);CHKERRQ(ierr);
		ierr = PetscFree(ll->vals_local);CHKERRQ(ierr);
	} else {
		ierr = PetscFree(ll->vals_global);CHKERRQ(ierr);
	}

	if (ll->dofidx_global != ll->dofidx_local) {
		ierr = PetscFree(ll->dofidx_global);CHKERRQ(ierr);
		ierr = PetscFree(ll->dofidx_local);CHKERRQ(ierr);
	} else {
		ierr = PetscFree(ll->dofidx_global);CHKERRQ(ierr);
	}
	ierr = PetscFree(ll->scale_global);CHKERRQ(ierr);
	
	ierr = DMDestroy(&ll->dm);CHKERRQ(ierr);
	ierr = PetscFree(ll);CHKERRQ(ierr);
	*list = PETSC_NULL;
	PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "BCListSetSizes"
PetscErrorCode BCListSetSizes(BCList list,PetscInt bs,PetscInt N,PetscInt N_local)
{
	PetscReal mem_usage_min,mem_usage_max,mem_usage = 0.0;
	PetscErrorCode ierr;
	
	list->blocksize = bs;
	list->N  = N;
	list->L  = bs * N;
	ierr = PetscMalloc( sizeof(PetscInt)*list->L,    &list->dofidx_global);CHKERRQ(ierr);   mem_usage += (PetscReal)(sizeof(PetscInt)*list->L);
	ierr = PetscMalloc( sizeof(PetscScalar)*list->L, &list->vals_global);CHKERRQ(ierr);     mem_usage += (PetscReal)(sizeof(PetscScalar)*list->L);
	ierr = PetscMalloc( sizeof(PetscScalar)*list->L, &list->scale_global); CHKERRQ(ierr);   mem_usage += (PetscReal)(sizeof(PetscScalar)*list->L);
	
	list->N_local  = N_local;
	list->L_local  = bs * N_local;
	ierr = PetscMalloc( sizeof(PetscInt)*list->L_local,    &list->dofidx_local);CHKERRQ(ierr);  mem_usage += (PetscReal)(sizeof(PetscInt)*list->L_local);
	ierr = PetscMalloc( sizeof(PetscScalar)*list->L_local, &list->vals_local);CHKERRQ(ierr);    mem_usage += (PetscReal)(sizeof(PetscScalar)*list->L_local);
	
	ierr = BCListInitialize(list);CHKERRQ(ierr);
	
	mem_usage = mem_usage * 1.0e-6;
	ierr = MPI_Allreduce( &mem_usage,&mem_usage_min,1,MPIU_REAL,MPI_MIN,((PetscObject)list->dm)->comm);CHKERRQ(ierr);
	ierr = MPI_Allreduce( &mem_usage,&mem_usage_max,1,MPIU_REAL,MPI_MAX,((PetscObject)list->dm)->comm);CHKERRQ(ierr);
	PetscPrintf(PETSC_COMM_WORLD,"BCList: Mem. usage (min,max) = %1.2e,%1.2e (MB) \n", mem_usage_min, mem_usage_max );
	
	PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "BCListUpdateCache"
PetscErrorCode BCListUpdateCache(BCList list)
{
	PetscInt       n,cnt;
	PetscBool      isdir;
	PetscErrorCode ierr;
	
	cnt = 0;
	for (n=0; n<list->L; n++) {
		ierr = BCListIsDirichlet(list->dofidx_global[n],&isdir);CHKERRQ(ierr);
		if (!isdir) { cnt++; }
	}
	list->allEmpty = PETSC_FALSE;
	if (cnt==list->L) { list->allEmpty = PETSC_TRUE; }
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BCListInitGlobal"
PetscErrorCode BCListInitGlobal(BCList list)
{
	PetscInt i,max,*indices,lsize;
	Vec dindices,dindices_g;
	PetscScalar *_dindices;
	PetscErrorCode ierr;
	
	
	ierr = DMDAGetGlobalIndices(list->dm,&max,&indices);CHKERRQ(ierr);
	ierr = DMGetGlobalVector(list->dm,&dindices_g);CHKERRQ(ierr);
	ierr = DMGetLocalVector(list->dm,&dindices);CHKERRQ(ierr);
	ierr = VecGetLocalSize(dindices,&lsize);CHKERRQ(ierr);
	if (lsize!=max) { SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Sizes don't match 1"); }
	ierr = VecGetArray(dindices,&_dindices);CHKERRQ(ierr);
	/* convert to scalar */
	for (i=0; i<lsize; i++) {
		_dindices[i] = (PetscScalar)indices[i] + 1.0e-3;
	}
	ierr = VecRestoreArray(dindices,&_dindices);CHKERRQ(ierr);
	
	/* scatter (ignore ghosts) */
	ierr = DMLocalToGlobalBegin(list->dm,dindices,INSERT_VALUES,dindices_g);CHKERRQ(ierr);
	ierr = DMLocalToGlobalEnd(  list->dm,dindices,INSERT_VALUES,dindices_g);CHKERRQ(ierr);
	
	/* convert to int */
	ierr = VecGetLocalSize(dindices_g,&lsize);CHKERRQ(ierr);
	if (list->L!=lsize) { SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Sizes don't match 2"); }
	ierr = VecGetArray(dindices_g,&_dindices);CHKERRQ(ierr);
	for (i=0; i<lsize; i++) {
		list->dofidx_global[i] = (PetscInt)_dindices[i];
	}
	ierr = VecRestoreArray(dindices_g,&_dindices);CHKERRQ(ierr);
	
	ierr = DMRestoreGlobalVector(list->dm,&dindices_g);CHKERRQ(ierr);
	ierr = DMRestoreLocalVector(list->dm,&dindices);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BCListGlobalToLocal"
PetscErrorCode BCListGlobalToLocal(BCList list)
{
	PetscInt i,lsize;
	Vec dindices,dindices_g;
	PetscScalar *_dindices;
	PetscBool is_dirich;
	PetscErrorCode ierr;
	
	
	ierr = DMGetGlobalVector(list->dm,&dindices_g);CHKERRQ(ierr);
	ierr = DMGetLocalVector(list->dm,&dindices);CHKERRQ(ierr);
	
	/* clean up indices (global -> local) */
	ierr = VecGetLocalSize(dindices_g,&lsize);CHKERRQ(ierr);
	if (lsize!=list->L) { SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Sizes don't match 1"); }
	ierr = VecGetArray(dindices_g,&_dindices);CHKERRQ(ierr);
	/* convert to scalar and copy values */
	for (i=0; i<lsize; i++) {
		BCListIsDirichlet(list->dofidx_global[i],&is_dirich);
		if (is_dirich) {
			_dindices[i] = (PetscScalar)list->dofidx_global[i] - 1.0e-3;
		} else {
			_dindices[i] = (PetscScalar)list->dofidx_global[i] + 1.0e-3;
		}
	}
	ierr = VecRestoreArray(dindices_g,&_dindices);CHKERRQ(ierr);
	
	/* scatter (ignore ghosts) */
	ierr = DMGlobalToLocalBegin(list->dm,dindices_g,INSERT_VALUES,dindices);CHKERRQ(ierr);
	ierr = DMGlobalToLocalEnd(  list->dm,dindices_g,INSERT_VALUES,dindices);CHKERRQ(ierr);
	
	/* convert to int */
	ierr = VecGetLocalSize(dindices,&lsize);CHKERRQ(ierr);
	if (list->L_local!=lsize) { SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Sizes don't match 2"); }
	ierr = VecGetArray(dindices,&_dindices);CHKERRQ(ierr);
	for (i=0; i<lsize; i++) {
		list->dofidx_local[i] = (PetscInt)_dindices[i];
	}
	ierr = VecRestoreArray(dindices,&_dindices);CHKERRQ(ierr);
	
	
	/* setup values (global -> local) */
	ierr = VecGetLocalSize(dindices_g,&lsize);CHKERRQ(ierr);
	ierr = VecGetArray(dindices_g,&_dindices);CHKERRQ(ierr);
	/* copy global values into a vector */
	for (i=0; i<lsize; i++) {
		_dindices[i] = list->vals_global[i];
	}
	ierr = VecRestoreArray(dindices_g,&_dindices);CHKERRQ(ierr);
	
	/* scatter (ignore ghosts) */
	ierr = DMGlobalToLocalBegin(list->dm,dindices_g,INSERT_VALUES,dindices);CHKERRQ(ierr);
	ierr = DMGlobalToLocalEnd(  list->dm,dindices_g,INSERT_VALUES,dindices);CHKERRQ(ierr);
	
	/* retrieve entries */
	ierr = VecGetLocalSize(dindices,&lsize);CHKERRQ(ierr);
	ierr = VecGetArray(dindices,&_dindices);CHKERRQ(ierr);
	for (i=0; i<lsize; i++) {
		list->vals_local[i] = _dindices[i];
	}
	ierr = VecRestoreArray(dindices,&_dindices);CHKERRQ(ierr);
	
	ierr = DMRestoreGlobalVector(list->dm,&dindices_g);CHKERRQ(ierr);
	ierr = DMRestoreLocalVector(list->dm,&dindices);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDABCListCreate"
PetscErrorCode DMDABCListCreate(DM da,BCList *list)
{
	BCList ll;
	PetscInt bs,N,m,n,p,Ng,mg,ng,pg;
	PetscInt gidx,gdofidx,blockloc,loc,dof_idx,dim,ndof;
	PetscInt si,sj,sk,nx,ny,nz,gsi,gsj,gsk,gnx,gny,gnz,i,j,k;
	PetscInt max_index,*globalindices;
	PetscErrorCode ierr;
	
	ierr = DMDAGetInfo(da,0, 0,0,0, 0,0,0, &bs,0, 0,0,0, 0);CHKERRQ(ierr);
	ierr = DMDAGetCorners(da, 0,0,0, &m,&n,&p);CHKERRQ(ierr);
	ierr = DMDAGetGhostCorners(da, 0,0,0, &mg,&ng,&pg);CHKERRQ(ierr);
	N = m;
	if (n!=0) N = N * n; /* 2d */
	if (p!=0) N = N * p; /* 3d */
	Ng = mg;
	if (ng!=0) Ng = Ng * ng; /* 2d */
	if (pg!=0) Ng = Ng * pg; /* 3d */
	
	
	ierr = BCListCreate(&ll);CHKERRQ(ierr);
	ll->dm = da;
	ierr = PetscObjectReference((PetscObject)da);CHKERRQ(ierr);
	ierr = BCListSetSizes(ll,bs,N,Ng);CHKERRQ(ierr);
	ierr = BCListInitialize(ll);CHKERRQ(ierr);
	
	ierr = DMDAGetGhostCorners(da,&gsi,&gsj,&gsk,&gnx,&gny,&gnz);CHKERRQ(ierr);
	ierr = DMDAGetCorners(da,&si,&sj,&sk,&nx,&ny,&nz);CHKERRQ(ierr);
	
	ierr = DMDAGetGlobalIndices(da,&max_index,&globalindices);CHKERRQ(ierr);
	ierr = DMDAGetInfo(da,&dim, 0,0,0, 0,0,0, &ndof,0, 0,0,0, 0);CHKERRQ(ierr);
	
	ierr = BCListInitGlobal(ll);CHKERRQ(ierr);
	ierr = BCListGlobalToLocal(ll);CHKERRQ(ierr);
	
	*list = ll;
	PetscFunctionReturn(0);
}

/* read/write */
#undef __FUNCT__
#define __FUNCT__ "BCListGetGlobalIndices"
PetscErrorCode BCListGetGlobalIndices(BCList list,PetscInt *n,PetscInt **idx)
{
	if (n) {   *n   = list->L; }
	if (idx) { *idx = list->dofidx_global; }
	PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "BCListRestoreGlobalIndices"
PetscErrorCode BCListRestoreGlobalIndices(BCList list,PetscInt *n,PetscInt **idx)
{
	PetscErrorCode ierr;
	if (idx) { 
		if (*idx != list->dofidx_global) {
			SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"idx doesn't match");
		}
	}
	/* update cached info */
	ierr = BCListUpdateCache(list);CHKERRQ(ierr);
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BCListGetGlobalValues"
PetscErrorCode BCListGetGlobalValues(BCList list,PetscInt *n,PetscScalar **vals)
{
	if (n) {   *n     = list->L; }
	if (vals) { *vals = list->vals_global; }
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BCListGetDofIdx"
PetscErrorCode BCListGetDofIdx(BCList list,PetscInt *Lg,PetscInt **dofidx_global,PetscInt *Ll,PetscInt **dofidx_local)
{
	if (Lg)            { *Lg   = list->L; }
	if (dofidx_global) { *dofidx_global = list->dofidx_global; }
	if (Ll)            { *Ll   = list->L_local; }
	if (dofidx_local)  { *dofidx_local = list->dofidx_local; }
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BCListEvaluator_constant"
PetscBool BCListEvaluator_constant( PetscScalar position[], PetscScalar *value, void *ctx ) 
{
	PetscBool impose_dirichlet = PETSC_TRUE;
	PetscScalar dv = *((PetscScalar*)ctx);
	
	*value = dv;
	return impose_dirichlet;
}

/* matrix free stuff */
/*
 if (isbc_local[i] == dirch) y[i] = xbc[i]
 else                        y[i] = y[i] 
 */
#undef __FUNCT__
#define __FUNCT__ "BCListInsert"
PetscErrorCode BCListInsert(BCList list,Vec y)
{
	PetscInt m,k,L;
	PetscInt *idx;
	PetscScalar *LA_x,*LA_y;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	ierr = BCListGetGlobalIndices(list,&L,&idx);CHKERRQ(ierr);
	ierr = BCListGetGlobalValues(list,&L,&LA_x);CHKERRQ(ierr);
	ierr = VecGetArray(y,&LA_y);CHKERRQ(ierr);
	ierr = VecGetLocalSize(y,&m);CHKERRQ(ierr);
	for (k=0; k<m; k++) {
		if (idx[k]==BCList_DIRICHLET) {
			LA_y[k] = LA_x[k];
		}
	}
	ierr = VecRestoreArray(y,&LA_y);CHKERRQ(ierr);
	ierr = BCListRestoreGlobalIndices(list,&L,&idx);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BCListInsertValueIntoDirichletSlot"
PetscErrorCode BCListInsertValueIntoDirichletSlot(BCList list,PetscScalar value,Vec y)
{
	PetscInt m,k,L;
	PetscInt *idx;
	PetscScalar *LA_y;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	ierr = BCListGetGlobalIndices(list,&L,&idx);CHKERRQ(ierr);
	ierr = VecGetArray(y,&LA_y);CHKERRQ(ierr);
	ierr = VecGetLocalSize(y,&m);CHKERRQ(ierr);

	if (L!=m) {
		SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"L != m");
	}
	for (k=0; k<m; k++) {
		if (idx[k]==BCList_DIRICHLET) {
			LA_y[k] = value;
		}
	}
	ierr = VecRestoreArray(y,&LA_y);CHKERRQ(ierr);
	ierr = BCListRestoreGlobalIndices(list,&L,&idx);CHKERRQ(ierr);	

	PetscFunctionReturn(0);
}


/*
 if (isbc_local[i] == dirch) y[i] = 0.0
 else                        y[i] = y[i] 
 */
#undef __FUNCT__
#define __FUNCT__ "BCListInsertZero"
PetscErrorCode BCListInsertZero(BCList list,Vec y)
{
	PetscInt m,k,L;
	PetscInt *idx;
	PetscScalar *LA_x,*LA_y;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	ierr = BCListGetGlobalIndices(list,&L,&idx);CHKERRQ(ierr);
	ierr = BCListGetGlobalValues(list,&L,&LA_x);CHKERRQ(ierr);
	ierr = VecGetArray(y,&LA_y);CHKERRQ(ierr);
	ierr = VecGetLocalSize(y,&m);CHKERRQ(ierr);
	for (k=0; k<m; k++) {
		if (idx[k]==BCList_DIRICHLET) {
			LA_y[k] = 0.0;
		}
	}
	ierr = VecRestoreArray(y,&LA_y);CHKERRQ(ierr);
	ierr = BCListRestoreGlobalIndices(list,&L,&idx);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}

/*
 if (isbc_local[i] == dirch) y[i] = xbc[i]
 else                        y[i] = y[i] 
 */
#undef __FUNCT__
#define __FUNCT__ "BCListInsertLocal"
PetscErrorCode BCListInsertLocal(BCList list,Vec y)
{
	PetscInt M,k,L;
	const PetscInt *idx;
	PetscScalar *LA_x,*LA_y;
	PetscBool is_seq = PETSC_FALSE;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	L    = list->L_local;
	idx  = list->dofidx_local;
	LA_x = list->vals_local;
	ierr = VecGetArray(y,&LA_y);CHKERRQ(ierr);
	ierr = VecGetSize(y,&M);CHKERRQ(ierr);
	
	/* debug error checking */
	if (L!=M) { SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Sizes do not match"); };
	ierr = PetscTypeCompare((PetscObject)y,VECSEQ,&is_seq);CHKERRQ(ierr);
	if (!is_seq) { SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Vec must be VECSEQ, i.e. a local (ghosted) vec"); };
	
	for (k=0; k<M; k++) {
		if (idx[k]==BCList_DIRICHLET) {
			LA_y[k] = LA_x[k];
		}
	}
	ierr = VecRestoreArray(y,&LA_y);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}

/*
 if (isbc_local[i] == dirch) y[i] = 0.0
 else                        y[i] = y[i] 
 */
#undef __FUNCT__
#define __FUNCT__ "BCListInsertLocalZero"
PetscErrorCode BCListInsertLocalZero(BCList list,Vec y)
{
	PetscInt M,k,L;
	const PetscInt *idx;
	PetscScalar *LA_x,*LA_y;
	PetscBool is_seq = PETSC_FALSE;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	L    = list->L_local;
	idx  = list->dofidx_local;
	LA_x = list->vals_local;
	ierr = VecGetArray(y,&LA_y);CHKERRQ(ierr);
	ierr = VecGetSize(y,&M);CHKERRQ(ierr);
	
	/* debug error checking */
	if (L!=M) { SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Sizes do not match"); };
	ierr = PetscTypeCompare((PetscObject)y,VECSEQ,&is_seq);CHKERRQ(ierr);
	if (!is_seq) { SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Vec must be VECSEQ, i.e. a local (ghosted) vec"); };
	
	for (k=0; k<M; k++) {
		if (idx[k]==BCList_DIRICHLET) {
			LA_y[k] = 0.0;
		}
	}
	ierr = VecRestoreArray(y,&LA_y);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}

/*
 Apply's
 F = scale(X-phi) where ever a dirichlet is encountered.
 */
#undef __FUNCT__
#define __FUNCT__ "BCListResidualDirichlet"
PetscErrorCode BCListResidualDirichlet(BCList list,Vec X,Vec F)
{
	PetscInt m,k,L;
	const PetscInt *idx;
	PetscScalar *LA_S,*LA_X,*LA_F,*LA_phi;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	if (!X) { SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_NULL,"Vec X cannot be PETSC_NULL"); }
	if (!F) { SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_NULL,"Vec F cannot be PETSC_NULL"); }
	
	L      = list->L;
	idx    = list->dofidx_global;
	LA_phi = list->vals_global;
	LA_S   = list->scale_global;
	
	ierr = VecGetArray(X,&LA_X);CHKERRQ(ierr);
	ierr = VecGetArray(F,&LA_F);CHKERRQ(ierr);
	
	/* debug error checking */
	ierr = VecGetLocalSize(X,&m);CHKERRQ(ierr);
	if (L!=m) { SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Sizes do not match (X)"); };
	ierr = VecGetLocalSize(F,&m);CHKERRQ(ierr);
	if (L!=m) { SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Sizes do not match (F)"); };
	
	for (k=0; k<m; k++) {
		if (idx[k]==BCList_DIRICHLET) {
			LA_F[k] = LA_S[k]*(LA_X[k] - LA_phi[k]);
		}
	}
	ierr = VecRestoreArray(X,&LA_X);CHKERRQ(ierr);
	ierr = VecRestoreArray(F,&LA_F);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}

/*
 Apply's
 F = scale(X) where ever a dirichlet is encountered.
 */
#undef __FUNCT__
#define __FUNCT__ "BCListInsertDirichlet_MatMult"
PetscErrorCode BCListInsertDirichlet_MatMult(BCList list,Vec X,Vec F)
{
	PetscInt m,k,L;
	const PetscInt *idx;
	PetscScalar *LA_S,*LA_X,*LA_F,*LA_phi;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	if (!X) { SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_NULL,"Vec X cannot be PETSC_NULL"); }
	if (!F) { SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_NULL,"Vec F cannot be PETSC_NULL"); }
	
	L      = list->L;
	idx    = list->dofidx_global;
	LA_phi = list->vals_global;
	LA_S   = list->scale_global;
	
	ierr = VecGetArray(X,&LA_X);CHKERRQ(ierr);
	ierr = VecGetArray(F,&LA_F);CHKERRQ(ierr);
	
	/* debug error checking */
	ierr = VecGetLocalSize(X,&m);CHKERRQ(ierr);
	if (L!=m) { SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Sizes do not match (X)"); };
	ierr = VecGetLocalSize(F,&m);CHKERRQ(ierr);
	if (L!=m) { SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Sizes do not match (F)"); };
	
	for (k=0; k<m; k++) {
		if (idx[k]==BCList_DIRICHLET) {
			LA_F[k] = LA_S[k]*LA_X[k];
		}
	}
	ierr = VecRestoreArray(X,&LA_X);CHKERRQ(ierr);
	ierr = VecRestoreArray(F,&LA_F);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}

/*
 
 PetscBool *eval(PetscScalar*,PetscScalar*,void*)
 
 PetscBool my_bc_evaluator( PetscScalar position[], PetscScalar *value, void *ctx ) 
 {
 PetscBool impose_dirichlet = PETSC_FALSE;
 
 if (position[0]<1.0) {
 *value = 10.0;
 impose_dirichlet = PETSC_TRUE;
 }
 
 return impose_dirichlet;
 }
 
 */
#undef __FUNCT__
#define __FUNCT__ "DMDABCListTraverse3d"
PetscErrorCode DMDABCListTraverse3d(BCList list,DM da,DMDABCListConstraintLoc doflocation,PetscInt dof_idx,PetscBool (*eval)(PetscScalar*,PetscScalar*,void*),void *ctx)
{
	PetscInt i,j,k,si,sj,sk,m,n,p,M,N,P,ndof;
	DM cda;
	Vec coords;
	DMDACoor3d ***LA_coords;	
	PetscInt L,*idx;
	PetscScalar pos[3];
	PetscScalar *vals,bc_val;
	PetscBool impose_dirichlet;
	PetscErrorCode ierr;
	
	ierr = DMDAGetInfo(da,0, &M,&N,&P, 0,0,0, &ndof,0, 0,0,0, 0);CHKERRQ(ierr);
	if (dof_idx >= ndof) { SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"dof_index >= dm->blocksize"); }
	
	ierr = DMDAGetCorners(da,&si,&sj,&sk,&m,&n,&p);CHKERRQ(ierr);
	ierr = DMDAGetCoordinateDA(da,&cda);CHKERRQ(ierr);
	ierr = DMDAGetCoordinates(da,&coords);CHKERRQ(ierr);
	if (!coords) { SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Coordinates must be set"); }
	ierr = DMDAVecGetArray(cda,coords,&LA_coords);CHKERRQ(ierr);
	
	ierr = BCListGetGlobalIndices(list,&L,&idx);
	ierr = BCListGetGlobalValues(list,&L,&vals);
	
	
	switch (doflocation) {
		/* volume */
		case DMDABCList_INTERIOR_LOC:
			for (k=sk; k<sk+p; k++) {
				for (j=sj; j<sj+n; j++) {
					for (i=si; i<si+m; i++) {
						PetscInt blockloc = (i-si) + (j-sj)*m + (k-sk)*m*n;
						PetscInt loc = blockloc*ndof+dof_idx;
						pos[0] = LA_coords[k][j][i].x;
						pos[1] = LA_coords[k][j][i].y;
						pos[2] = LA_coords[k][j][i].z;
						
						impose_dirichlet = eval(pos,&bc_val,ctx);
						if (impose_dirichlet) {
							idx[loc] = BCList_DIRICHLET;
							vals[loc] = bc_val;
						}
						
					}
				}
			}
			break;
			
		/* i=0 plane (left) */
		case DMDABCList_IMIN_LOC:
			if (si==0) {
				i = 0;
				for (k=sk; k<sk+p; k++) {
					for (j=sj; j<sj+n; j++) {
						PetscInt blockloc = (i-si) + (j-sj)*m + (k-sk)*m*n;
						PetscInt loc = blockloc*ndof+dof_idx;
						pos[0] = LA_coords[k][j][i].x;
						pos[1] = LA_coords[k][j][i].y;
						pos[2] = LA_coords[k][j][i].z;
						
						impose_dirichlet = eval(pos,&bc_val,ctx);
						if (impose_dirichlet) {
							idx[loc] = BCList_DIRICHLET;
							vals[loc] = bc_val;
						}
						
					}
				}
			}
			break;
		/* i=si+m == M plane (right) */
		case DMDABCList_IMAX_LOC:
			if (si+m==M) {
				i = si+m-1;
				for (k=sk; k<sk+p; k++) {
					for (j=sj; j<sj+n; j++) {
						PetscInt blockloc = (i-si) + (j-sj)*m + (k-sk)*m*n;
						PetscInt loc = blockloc*ndof+dof_idx;
						pos[0] = LA_coords[k][j][i].x;
						pos[1] = LA_coords[k][j][i].y;
						pos[2] = LA_coords[k][j][i].z;
						
						impose_dirichlet = eval(pos,&bc_val,ctx);
						if (impose_dirichlet) {
							idx[loc] = BCList_DIRICHLET;
							vals[loc] = bc_val;
						}
						
					}
				}
			}
			break;
			
		/* j=0 plane (bottom) */
		case DMDABCList_JMIN_LOC:
			if (sj==0) {
				j = 0;
				for (k=sk; k<sk+p; k++) {
					for (i=si; i<si+m; i++) {
						PetscInt blockloc = (i-si) + (j-sj)*m + (k-sk)*m*n;
						PetscInt loc = blockloc*ndof+dof_idx;
						pos[0] = LA_coords[k][j][i].x;
						pos[1] = LA_coords[k][j][i].y;
						pos[2] = LA_coords[k][j][i].z;
						
						impose_dirichlet = eval(pos,&bc_val,ctx);
						if (impose_dirichlet) {
							idx[loc] = BCList_DIRICHLET;
							vals[loc] = bc_val;
						}
						
					}
				}
			}
			
			break;
		/* j=sj+n == N plane (top) */
		case DMDABCList_JMAX_LOC:
			if (sj+n==N) {
				j = sj+n-1;
				for (k=sk; k<sk+p; k++) {
					for (i=si; i<si+m; i++) {
						PetscInt blockloc = (i-si) + (j-sj)*m + (k-sk)*m*n;
						PetscInt loc = blockloc*ndof+dof_idx;
						pos[0] = LA_coords[k][j][i].x;
						pos[1] = LA_coords[k][j][i].y;
						pos[2] = LA_coords[k][j][i].z;
						
						impose_dirichlet = eval(pos,&bc_val,ctx);
						if (impose_dirichlet) {
							idx[loc] = BCList_DIRICHLET;
							vals[loc] = bc_val;
						}
					}	
				}
			}
			
			break;
			
		/* k=0 plane (back) */
		case DMDABCList_KMIN_LOC:
			if (sk==0) {
				k = 0;
				for (j=sj; j<sj+n; j++) {
					for (i=si; i<si+m; i++) {
						PetscInt blockloc = (i-si) + (j-sj)*m + (k-sk)*m*n;
						PetscInt loc = blockloc*ndof+dof_idx;
						pos[0] = LA_coords[k][j][i].x;
						pos[1] = LA_coords[k][j][i].y;
						pos[2] = LA_coords[k][j][i].z;
						
						impose_dirichlet = eval(pos,&bc_val,ctx);
						if (impose_dirichlet) {
							idx[loc] = BCList_DIRICHLET;
							vals[loc] = bc_val;
						}
						
					}
				}
			}
			break;
		/* k=sk+p == P plane (front) */
		case DMDABCList_KMAX_LOC:
			if (sk+p==P) {
				k = sk+p-1;
				for (j=sj; j<sj+n; j++) {
					for (i=si; i<si+m; i++) {
						PetscInt blockloc = (i-si) + (j-sj)*m + (k-sk)*m*n;
						PetscInt loc = blockloc*ndof+dof_idx;
						pos[0] = LA_coords[k][j][i].x;
						pos[1] = LA_coords[k][j][i].y;
						pos[2] = LA_coords[k][j][i].z;
						
						impose_dirichlet = eval(pos,&bc_val,ctx);
						if (impose_dirichlet) {
							idx[loc] = BCList_DIRICHLET;
							vals[loc] = bc_val;
						}
					}	
				}
			}					
			break;
			
		default:
			SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unknown Dirichlet boundary condition specified");
			break;
	}
	
	ierr = BCListRestoreGlobalIndices(list,&L,&idx);
	ierr = DMDAVecRestoreArray(cda,coords,&LA_coords);CHKERRQ(ierr);
	ierr = BCListGlobalToLocal(list);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}

/* FLATTENED VARIANT */
#undef __FUNCT__
#define __FUNCT__ "BCListFlattenedCreate"
PetscErrorCode BCListFlattenedCreate(BCList std,BCList *flat)
{
	PetscErrorCode ierr;
	BCList F;
	PetscMPIInt nproc;
	PetscInt count,k;
	PetscReal mem_usage_min,mem_usage_max,mem_usage = 0.0;
	
	PetscFunctionBegin;
	
	
	ierr = BCListCreate(&F);CHKERRQ(ierr);
	F->dm = std->dm;
	ierr = PetscObjectReference((PetscObject)F->dm);CHKERRQ(ierr);
	F->blocksize = std->blocksize;
	
	
	/* two pass */
	
	/* count global size */
	count = 0;
	for (k=0; k<std->L; k++) {
		if (std->dofidx_global[k] == BCList_DIRICHLET) {
			count++;
		}
	}
	
	ierr = PetscMalloc( sizeof(PetscInt)*count,&F->dofidx_global );CHKERRQ(ierr);     mem_usage += (PetscReal)(sizeof(PetscInt)*count);
	ierr = PetscMalloc( sizeof(PetscScalar)*count,&F->vals_global );CHKERRQ(ierr);    mem_usage += (PetscReal)(sizeof(PetscScalar)*count);
	ierr = PetscMalloc( sizeof(PetscScalar)*count,&F->scale_global );CHKERRQ(ierr);   mem_usage += (PetscReal)(sizeof(PetscScalar)*count);
	count = 0;
	for (k=0; k<std->L; k++) {
		if (std->dofidx_global[k] == BCList_DIRICHLET) {
			F->dofidx_global[count] = k;
			F->vals_global[count]   = std->vals_global[k];
			F->scale_global[count]  = std->scale_global[k];

			count++;
		}
	}
	F->L = count;
	
	ierr = MPI_Comm_size(((PetscObject)std->dm)->comm,&nproc);CHKERRQ(ierr);
	if (nproc==1) {
		F->vals_local   = F->vals_global;
		F->dofidx_local = F->dofidx_global;
		F->L_local      = F->L;
	} else {
	
		/* count global size */
		count = 0;
		for (k=0; k<std->L_local; k++) {
			if (std->dofidx_local[k] == BCList_DIRICHLET) {
				count++;
			}
		}
		
		ierr = PetscMalloc( sizeof(PetscInt)*count,&F->dofidx_local );CHKERRQ(ierr);    mem_usage += (PetscReal)(sizeof(PetscInt)*count);  
		ierr = PetscMalloc( sizeof(PetscScalar)*count,&F->vals_local );CHKERRQ(ierr);   mem_usage += (PetscScalar)(sizeof(PetscInt)*count);
		
		count = 0;
		for (k=0; k<std->L; k++) {
			if (std->dofidx_local[k] == BCList_DIRICHLET) {
				F->dofidx_local[count] = k;
				F->vals_local[count]   = std->vals_local[k];
				
				count++;
			}
		}
		F->L_local = count;
		
	}
	
	mem_usage = mem_usage * 1.0e-6;
	ierr = MPI_Allreduce( &mem_usage,&mem_usage_min,1,MPIU_REAL,MPI_MIN,((PetscObject)std->dm)->comm);CHKERRQ(ierr);
	ierr = MPI_Allreduce( &mem_usage,&mem_usage_max,1,MPIU_REAL,MPI_MAX,((PetscObject)std->dm)->comm);CHKERRQ(ierr);
	PetscPrintf(PETSC_COMM_WORLD,"BCListFlat: Mem. usage (min,max) = %1.2e,%1.2e (MB) \n", mem_usage_min, mem_usage_max );
	
	
	*flat = F;
	
	PetscFunctionReturn(0);
}

/* matrix free stuff */
/*
 if (isbc_local[i] == dirch) y[i] = xbc[i]
 else                        y[i] = y[i] 
 */
#undef __FUNCT__
#define __FUNCT__ "BCListFlatInsert"
PetscErrorCode BCListFlatInsert(BCList list,Vec y)
{
	PetscInt m,k,L;
	PetscInt *idx;
	PetscScalar *LA_x,*LA_y;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	ierr = BCListGetGlobalIndices(list,&L,&idx);CHKERRQ(ierr);
	ierr = BCListGetGlobalValues(list,&L,&LA_x);CHKERRQ(ierr);
	ierr = VecGetArray(y,&LA_y);CHKERRQ(ierr);
	ierr = VecGetLocalSize(y,&m);CHKERRQ(ierr);
	for (k=0; k<L; k++) {
		PetscInt dirichlet_idx = idx[k];

		LA_y[dirichlet_idx] = LA_x[dirichlet_idx];
	}
	ierr = VecRestoreArray(y,&LA_y);CHKERRQ(ierr);
	ierr = BCListRestoreGlobalIndices(list,&L,&idx);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}

/*
 if (isbc_local[i] == dirch) y[i] = xbc[i]
 else                        y[i] = y[i] 
 */
#undef __FUNCT__
#define __FUNCT__ "BCListFlatInsertLocal"
PetscErrorCode BCListFlatInsertLocal(BCList list,Vec y)
{
	PetscInt M,k,L;
	const PetscInt *idx;
	PetscScalar *LA_x,*LA_y;
	PetscBool is_seq = PETSC_FALSE;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	L    = list->L_local;
	idx  = list->dofidx_local;
	LA_x = list->vals_local;
	ierr = VecGetArray(y,&LA_y);CHKERRQ(ierr);
	ierr = VecGetSize(y,&M);CHKERRQ(ierr);
	
	/* debug error checking */
	ierr = PetscTypeCompare((PetscObject)y,VECSEQ,&is_seq);CHKERRQ(ierr);
	if (!is_seq) { SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Vec must be VECSEQ, i.e. a local (ghosted) vec"); };
	
	for (k=0; k<L; k++) {
		PetscInt dirichlet_idx = idx[k];

		LA_y[dirichlet_idx] = LA_x[dirichlet_idx];
	}
	ierr = VecRestoreArray(y,&LA_y);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}

/*
 Apply's
 F = scale(X-phi) where ever a dirichlet is encountered.
 */
#undef __FUNCT__
#define __FUNCT__ "BCListFlatResidualDirichlet"
PetscErrorCode BCListFlatResidualDirichlet(BCList list,Vec X,Vec F)
{
	PetscInt m,k,L;
	const PetscInt *idx;
	PetscScalar *LA_S,*LA_X,*LA_F,*LA_phi;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	if (!X) { SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_NULL,"Vec X cannot be PETSC_NULL"); }
	if (!F) { SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_NULL,"Vec F cannot be PETSC_NULL"); }
	
	L      = list->L;
	idx    = list->dofidx_global;
	LA_phi = list->vals_global;
	LA_S   = list->scale_global;
	
	ierr = VecGetArray(X,&LA_X);CHKERRQ(ierr);
	ierr = VecGetArray(F,&LA_F);CHKERRQ(ierr);
	
	/* debug error checking */
	for (k=0; k<L; k++) {
		PetscInt dirichlet_idx = idx[k];

		LA_F[dirichlet_idx] = LA_S[dirichlet_idx]*(LA_X[dirichlet_idx] - LA_phi[dirichlet_idx]);
	}
	ierr = VecRestoreArray(X,&LA_X);CHKERRQ(ierr);
	ierr = VecRestoreArray(F,&LA_F);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BCListApplyDirichletMask"
PetscErrorCode BCListApplyDirichletMask(PetscInt N_EQNS, PetscInt gidx[],BCList list)
{
	PetscInt k,L;
	PetscInt *idx;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	L   = list->L_local;
	idx = list->dofidx_local;
	for (k=0; k<L; k++) {
		if (idx[k]==BCList_DIRICHLET) {
			gidx[k] = - ( gidx[k] + 1 );
		}
	}
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BCListRemoveDirichletMask"
PetscErrorCode BCListRemoveDirichletMask(PetscInt N_EQNS, PetscInt gidx[],BCList list)
{
	PetscInt k,L;
	PetscInt *idx;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	L   = list->L_local;
	idx = list->dofidx_local;
	for (k=0; k<L; k++) {
		if (idx[k]==BCList_DIRICHLET) {
			gidx[k] = - gidx[k] - 1;
		}
	}
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BCListInsertScaling"
PetscErrorCode BCListInsertScaling(Mat A,PetscInt N_EQNS, PetscInt gidx[],BCList list)
{
	PetscInt k,L;
	PetscInt *idx;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
#if 0
	L   = list->L;
	idx = list->dofidx_global;
	for (k=0; k<L; k++) {
		if (idx[k]==BCList_DIRICHLET) {
			//printf("local index %d is dirichlet--->inserted into %d\n", k,gidx[k]);
			ierr = MatSetValue(A,gidx[k],gidx[k],list->scale_global[k],INSERT_VALUES);CHKERRQ(ierr);
		}
	}
#endif
	
	L   = list->L_local;
	idx = list->dofidx_local;
	for (k=0; k<L; k++) {
		if (idx[k]==BCList_DIRICHLET) {
			//printf("local index %d is dirichlet--->inserted into %d\n", k,gidx[k]);
			ierr = MatSetValue(A,gidx[k],gidx[k],1.0,INSERT_VALUES);CHKERRQ(ierr);
		}
	}
	
	PetscFunctionReturn(0);
}



