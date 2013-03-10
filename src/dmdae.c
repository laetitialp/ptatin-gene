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
 **    Filename:      dmdae.c
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

#include "dmdae.h"


#undef __FUNCT__  
#define __FUNCT__ "DMDAECreate"
PetscErrorCode DMDAECreate(DMDAE *dae)
{
	DMDAE d;
	PetscErrorCode ierr;
	
  PetscFunctionBegin;
	ierr = PetscMalloc(sizeof(struct _p_DMDAE),&d);CHKERRQ(ierr);
	ierr = PetscMemzero(d,sizeof(struct _p_DMDAE));CHKERRQ(ierr);
	*dae = d;
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMDAEDestroy"
PetscErrorCode DMDAEDestroy(DMDAE *dae)
{
	DMDAE d;
	PetscErrorCode ierr;
	
  PetscFunctionBegin;
	if (!dae){ PetscFunctionReturn(0); }
	d = (*dae);
	ierr = PetscFree(d);CHKERRQ(ierr);
	*dae = PETSC_NULL;
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMAttachDMDAE"
PetscErrorCode DMAttachDMDAE(DM dm)
{
	PetscContainer container;
	DMDAE dae;
	PetscErrorCode ierr;

  PetscFunctionBegin;
	ierr = DMDAECreate(&dae);CHKERRQ(ierr);
	
  ierr = PetscContainerCreate(PETSC_COMM_WORLD,&container);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(container,(void*)dae);CHKERRQ(ierr);
	
	ierr = PetscObjectCompose((PetscObject)dm,"DMDAEobject",(PetscObject)container);CHKERRQ(ierr);

	
	PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMGetDMDAE"
PetscErrorCode DMGetDMDAE(DM dm,DMDAE *dae)
{
	DMDAE d;
	PetscContainer container;
	PetscErrorCode ierr;
	
  PetscFunctionBegin;
	ierr = PetscObjectQuery((PetscObject)dm,"DMDAEobject",(PetscObject*)&container);CHKERRQ(ierr);
	if (!container) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"No data with name \"DMDAEobject\" was composed with DM");
	ierr = PetscContainerGetPointer(container,(void**)&d);CHKERRQ(ierr);
	*dae = d;
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMDestroyDMDAE"
PetscErrorCode DMDestroyDMDAE(DM dm)
{
	DMDAE d;
	PetscContainer container;
	PetscErrorCode ierr;
	
  PetscFunctionBegin;
	ierr = PetscObjectQuery((PetscObject)dm,"DMDAEobject",(PetscObject*)&container);CHKERRQ(ierr);
	if (!container) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"No data with name \"DMDAEobject\" was composed with DM");
	ierr = PetscContainerGetPointer(container,(void**)&d);CHKERRQ(ierr);

	ierr = DMDAEDestroy(&d);CHKERRQ(ierr);
	ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMDAEGetOwnershipRanges"
PetscErrorCode DMDAEGetOwnershipRanges(DM dm,
																			 PetscInt *m,PetscInt *n,PetscInt *p,
																			 PetscInt **si,PetscInt **sj,PetscInt **sk,
																			 PetscInt **mx,PetscInt **my,PetscInt **mz)
{
	DMDAE dae;
	PetscInt pM,pN,pP;
	PetscErrorCode ierr;
	
  PetscFunctionBegin;
	ierr = DMGetDMDAE(dm,&dae);CHKERRQ(ierr);
	
	ierr = DMDAGetInfo(dm,0,0,0,0,&pM,&pN,&pP, 0, 0, 0,0,0, 0 );CHKERRQ(ierr);
	if (m) { *m = pM; }
	if (n) { *n = pN; }
	if (p) { *p = pP; }

	if (si) { *si = dae->lsip; }
	if (sj) { *sj = dae->lsjp; }
	if (sk) { *sk = dae->lskp; }

	if (mx) { *mx = dae->lmxp; }
	if (my) { *my = dae->lmyp; }
	if (mz) { *mz = dae->lmzp; }
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMDAEGetCornersElement"
PetscErrorCode DMDAEGetCornersElement(DM dm,PetscInt *esi,PetscInt *esj,PetscInt *esk,PetscInt *mx,PetscInt *my,PetscInt *mz)
{
	DMDAE dae;
	PetscInt pM,pN,pP;
	PetscErrorCode ierr;
	
  PetscFunctionBegin;
	ierr = DMGetDMDAE(dm,&dae);CHKERRQ(ierr);
	if (esi) { *esi = dae->sgi; }
	if (esj) { *esj = dae->sgj; }
	if (esk) { *esk = dae->sgk; }

	if (mx) { *mx = dae->lmx; }
	if (my) { *my = dae->lmy; }
	if (mz) { *mz = dae->lmz; }
	
	PetscFunctionReturn(0);	
}




