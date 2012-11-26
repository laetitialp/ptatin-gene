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
 **    Filename:      dmda_element_q1.c
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
#include "ptatin3d_defs.h"
#include "dmdae.h"
#include "dmda_element_q2p1.h"
#include "dmda_element_q1.h"




#undef __FUNCT__
#define __FUNCT__ "DMDAGetElements_DA_Q1_3D"
PetscErrorCode DMDAGetElements_DA_Q1_3D(DM dm,PetscInt *nel,PetscInt *npe,const PetscInt **eidx)
{
  DM_DA          *da = (DM_DA*)dm->data;
	const PetscInt order = 1;
	PetscErrorCode ierr;
	PetscInt *idx,mx,my,mz,_npe, M,N,P;
	PetscInt ei,ej,ek,i,j,k,elcnt,esi,esj,esk,gsi,gsj,gsk,nid[8],n,d,X,Y,Z,width;
	PetscInt *el;
	int rank;
	PetscFunctionBegin;
	
	MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
	ierr = DMDAGetInfo(dm,0, &M,&N,&P, 0,0,0, 0,&width, 0,0,0, 0);CHKERRQ(ierr);
	if (width!=1) {
		SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Stencil width must be 1 for Q1");
	}
	
	_npe = (order + 1)*(order + 1)*(order + 1);
  if (!da->e) {
		DMDAE dmdae;
		
		ierr = DMGetDMDAE(dm,&dmdae);CHKERRQ(ierr);
		mx = dmdae->lmx;
		my = dmdae->lmy;
		mz = dmdae->lmz;
		
		esi = dmdae->si;
		esj = dmdae->sj;
		esk = dmdae->sk;
		
		ierr = PetscMalloc(sizeof(PetscInt)*(mx*my*mz*_npe+1),&idx);CHKERRQ(ierr);
		ierr = DMDAGetGhostCorners(dm,&gsi,&gsj,&gsk, &X,&Y,&Z);CHKERRQ(ierr);
		
		elcnt = 0;
		for (ek=0; ek<mz; ek++) {
			k = (esk-gsk) + ek;

			for (ej=0; ej<my; ej++) {
				j = (esj-gsj) + ej;
			
				for (ei=0; ei<mx; ei++) {
					i = (esi-gsi) + ei;
				
					el = &idx[_npe*elcnt];
					
					nid[0] = (i  ) + (j  ) *X  + (k  ) *X*Y;
					nid[1] = (i+1) + (j  ) *X  + (k  ) *X*Y;
					
					nid[2] = (i  ) + (j+1) *X  + (k  ) *X*Y;
					nid[3] = (i+1) + (j+1) *X  + (k  ) *X*Y;
					
					nid[4] = (i  ) + (j  ) *X  + (k+1) *X*Y;
					nid[5] = (i+1) + (j  ) *X  + (k+1) *X*Y;
					
					nid[6] = (i  ) + (j+1) *X  + (k+1) *X*Y;
					nid[7] = (i+1) + (j+1) *X  + (k+1) *X*Y;

					
					for (n=0; n<_npe; n++) {
						if (nid[n]>M*N*P) { 
							SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Local indexing exceeds number of global nodes");
						}

						if (nid[n]>X*Y*Z) { 
							SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Local indexing exceeds number of local nodes");
						}
						
						el[n] = nid[n]; //gidx[dof*nid[n]+0]/dof;
					}
					
					elcnt++;
				}
			}
		}
		
		da->e  = idx;
		da->ne = elcnt;
	}
	
	*eidx = da->e;
	*npe = _npe;
	*nel = da->ne;
	
	PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMDASetElementType_Q1"
PetscErrorCode  DMDASetElementType_Q1(DM da)
{
  DM_DA          *dd = (DM_DA*)da->data;
  PetscErrorCode ierr;
	
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  if (dd->elementtype) {
    ierr = PetscFree(dd->e);CHKERRQ(ierr);
    dd->ne          = 0; 
    dd->e           = PETSC_NULL;
  }
	
  PetscFunctionReturn(0);
}



/* constructors */
#undef __FUNCT__  
#define __FUNCT__ "DMDACreateOverlappingQ1FromQ2"
PetscErrorCode DMDACreateOverlappingQ1FromQ2(DM dmq2,PetscInt ndofs,DM *dmq1)
{
	DM dm;
	DMDAE dae;
	PetscInt sei,sej,sek,lmx,lmy,lmz,MX,MY,MZ,Mp,Np,Pp,i,j,k,n;
	PetscInt *siq2,*sjq2,*skq2,*lmxq2,*lmyq2,*lmzq2,*lxq1,*lyq1,*lzq1;
	PetscInt *lsip,*lsjp,*lskp;
	PetscErrorCode ierr;
	int rank;
	PetscFunctionBegin;
	
	
	ierr = DMDAGetCornersElementQ2(dmq2,&sei,&sej,&sek,&lmx,&lmy,&lmz);CHKERRQ(ierr);
	ierr = DMDAGetSizeElementQ2(dmq2,&MX,&MY,&MZ);CHKERRQ(ierr);
	ierr = DMDAGetInfo(dmq2,0,0,0,0,&Mp,&Np,&Pp,0,0, 0,0,0, 0);CHKERRQ(ierr);
	
	ierr = DMDAGetOwnershipRangesElementQ2(dmq2,0,0,0,&siq2,&sjq2,&skq2,&lmxq2,&lmyq2,&lmzq2);CHKERRQ(ierr);
	
	
	ierr = PetscMalloc(sizeof(PetscInt)*Mp,&lsip);CHKERRQ(ierr);
	ierr = PetscMalloc(sizeof(PetscInt)*Np,&lsjp);CHKERRQ(ierr);
	ierr = PetscMalloc(sizeof(PetscInt)*Pp,&lskp);CHKERRQ(ierr);
	
	for (i=0; i<Mp; i++) {
		lsip[i] = siq2[i]/2;
	}
	
	for (j=0; j<Np; j++) {
		lsjp[j] = sjq2[j]/2;
	}

	for (k=0; k<Pp; k++) {
		lskp[k] = skq2[k]/2;
	}
	
	ierr = PetscMalloc(sizeof(PetscInt)*Mp,&lxq1);CHKERRQ(ierr);
	ierr = PetscMalloc(sizeof(PetscInt)*Np,&lyq1);CHKERRQ(ierr);
	ierr = PetscMalloc(sizeof(PetscInt)*Pp,&lzq1);CHKERRQ(ierr);
	
	for (i=0; i<Mp-1; i++) {
		lxq1[i] = siq2[i+1]/2 - siq2[i]/2;
	}
	lxq1[Mp-1] = (2*MX - siq2[Mp-1])/2;
	lxq1[Mp-1]++;
	
	for (j=0; j<Np-1; j++) {
		lyq1[j] = sjq2[j+1]/2 - sjq2[j]/2;
	}
	lyq1[Np-1] = (2*MY - sjq2[Np-1])/2;
	lyq1[Np-1]++;

	for (k=0; k<Pp-1; k++) {
		lzq1[k] = skq2[k+1]/2 - skq2[k]/2;
	}
	lzq1[Pp-1] = (2*MZ - skq2[Pp-1])/2;
	lzq1[Pp-1]++;
	
	
	
	MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
	if (rank==0) {
		for (i=0; i<Mp; i++) {
			printf("rank[%d] startI = %d: lmxq1 = %d \n",i,lsip[i],lxq1[i]);
		}
		for (j=0; j<Np; j++) {
			printf("rank[%d] startJ = %d: lmyq1 = %d \n",j,lsjp[j],lyq1[j]);
		}
		for (k=0; k<Pp; k++) {
			printf("rank[%d] startK = %d: lmzq1 = %d \n",k,lskp[k],lzq1[k]);
		}
	}
	
	
	ierr = DMDACreate3d(((PetscObject)dmq2)->comm,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE, DMDA_STENCIL_BOX, MX+1,MY+1,MZ+1, Mp,Np,Pp, ndofs,1, lxq1,lyq1,lzq1, &dm );CHKERRQ(ierr);
	
	
	/* add the space for the data structure */
	ierr = DMAttachDMDAE(dm);CHKERRQ(ierr);
	/* fetch the data structure */
	ierr = DMGetDMDAE(dm,&dae);CHKERRQ(ierr);
	dae->ne = MX * MY * MZ;
	dae->lne = lmx * lmy * lmz;
	
	dae->mx = MX;
	dae->my = MY;
	dae->mz = MZ;
	dae->lmx = lmx;
	dae->lmy = lmy;
	dae->lmz = lmz;
	
	dae->si = sei/2;
	dae->sj = sej/2;
	dae->sk = sek/2;
	
	dae->npe = 8;
	dae->nps = 2;
	dae->overlap = 0;
	
	for (i=0; i<Mp; i++) {
		lxq1[i] = lmxq2[i];
	}
	for (j=0; j<Np; j++) {
		lyq1[j] = lmyq2[j];
	}
	for (k=0; k<Pp; k++) {
		lzq1[k] = lmzq2[k];
	}
	dae->lmxp = lxq1;
	dae->lmyp = lyq1;
	dae->lmzp = lzq1;
	
	dae->lsip = lsip;
	dae->lsjp = lsjp;
	dae->lskp = lskp;
	
	*dmq1 = dm;
	
	
	PetscFree(siq2);
	PetscFree(sjq2);
	PetscFree(skq2);
	PetscFree(lmxq2);
	PetscFree(lmyq2);
	PetscFree(lmzq2);
	
	/* force element creation using MY numbering */
	{
		PetscInt nel,nen;
		const PetscInt *els;

		ierr = DMDASetElementType_Q1(dm);CHKERRQ(ierr);
		ierr = DMDAGetElements_DA_Q1_3D(dm,&nel,&nen,&els);CHKERRQ(ierr);
	}	
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DMDACreateNestedQ1FromQ2"
PetscErrorCode DMDACreateNestedQ1FromQ2(DM dmq2,PetscInt ndofs,DM *dmq1)
{
	DM dm;
	DMDAE dae;
	PetscInt sei,sej,sek,lmx,lmy,lmz,MX,MY,MZ,Mp,Np,Pp,i,j,k,n;
	PetscInt *siq2,*sjq2,*skq2,*lmxq2,*lmyq2,*lmzq2,*lxq1,*lyq1,*lzq1;
	PetscInt *lsip,*lsjp,*lskp;
	PetscErrorCode ierr;
	int rank;
	PetscFunctionBegin;
	
	
	ierr = DMDAGetCornersElementQ2(dmq2,&sei,&sej,&sek,&lmx,&lmy,&lmz);CHKERRQ(ierr);
	ierr = DMDAGetSizeElementQ2(dmq2,&MX,&MY,&MZ);CHKERRQ(ierr);
	ierr = DMDAGetInfo(dmq2,0,0,0,0,&Mp,&Np,&Pp,0,0, 0,0,0, 0);CHKERRQ(ierr);
	
	ierr = DMDAGetOwnershipRangesElementQ2(dmq2,0,0,0,&siq2,&sjq2,&skq2,&lmxq2,&lmyq2,&lmzq2);CHKERRQ(ierr);
	
	
	ierr = PetscMalloc(sizeof(PetscInt)*Mp,&lsip);CHKERRQ(ierr);
	ierr = PetscMalloc(sizeof(PetscInt)*Np,&lsjp);CHKERRQ(ierr);
	ierr = PetscMalloc(sizeof(PetscInt)*Pp,&lskp);CHKERRQ(ierr);
	
	for (i=0; i<Mp; i++) {
		lsip[i] = siq2[i];
	}
	
	for (j=0; j<Np; j++) {
		lsjp[j] = sjq2[j];
	}
	
	for (k=0; k<Pp; k++) {
		lskp[k] = skq2[k];
	}
	
	ierr = PetscMalloc(sizeof(PetscInt)*Mp,&lxq1);CHKERRQ(ierr);
	ierr = PetscMalloc(sizeof(PetscInt)*Np,&lyq1);CHKERRQ(ierr);
	ierr = PetscMalloc(sizeof(PetscInt)*Pp,&lzq1);CHKERRQ(ierr);
	
	for (i=0; i<Mp-1; i++) {
		lxq1[i] = siq2[i+1] - siq2[i];
	}
	lxq1[Mp-1] = 2*MX - siq2[Mp-1];
	lxq1[Mp-1]++;
	
	for (j=0; j<Np-1; j++) {
		lyq1[j] = sjq2[j+1] - sjq2[j];
	}
	lyq1[Np-1] = 2*MY - sjq2[Np-1];
	lyq1[Np-1]++;
	
	for (k=0; k<Pp-1; k++) {
		lzq1[k] = skq2[k+1] - skq2[k];
	}
	lzq1[Pp-1] = 2*MZ - skq2[Pp-1];
	lzq1[Pp-1]++;
	
	
	
	MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
	if (rank==0) {
		for (i=0; i<Mp; i++) {
			printf("rank[%d] startI = %d: lmxq1 = %d \n",i,lsip[i],lxq1[i]);
		}
		for (j=0; j<Np; j++) {
			printf("rank[%d] startJ = %d: lmyq1 = %d \n",j,lsjp[j],lyq1[j]);
		}
		for (k=0; k<Pp; k++) {
			printf("rank[%d] startK = %d: lmzq1 = %d \n",k,lskp[k],lzq1[k]);
		}
	}
	
	
	ierr = DMDACreate3d(((PetscObject)dmq2)->comm,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE, DMDA_STENCIL_BOX, 2*MX+1,2*MY+1,2*MZ+1, Mp,Np,Pp, ndofs,1, lxq1,lyq1,lzq1, &dm );CHKERRQ(ierr);
	
	
	/* add the space for the data structure */
	ierr = DMAttachDMDAE(dm);CHKERRQ(ierr);
	/* fetch the data structure */
	ierr = DMGetDMDAE(dm,&dae);CHKERRQ(ierr);
	dae->ne = 8 * MX * MY * MZ;
	dae->lne = 8 * lmx * lmy * lmz;
	
	dae->mx = 2*MX;
	dae->my = 2*MY;
	dae->mz = 2*MZ;
	dae->lmx = 2*lmx;
	dae->lmy = 2*lmy;
	dae->lmz = 2*lmz;
	
	dae->si = sei;
	dae->sj = sej;
	dae->sk = sek;
	
	dae->npe = 8;
	dae->nps = 2;
	dae->overlap = 0;
	
	for (i=0; i<Mp; i++) {
		lxq1[i] = lmxq2[i];
	}
	for (j=0; j<Np; j++) {
		lyq1[j] = lmyq2[j];
	}
	for (k=0; k<Pp; k++) {
		lzq1[k] = lmzq2[k];
	}
	dae->lmxp = lxq1;
	dae->lmyp = lyq1;
	dae->lmzp = lzq1;
	
	dae->lsip = lsip;
	dae->lsjp = lsjp;
	dae->lskp = lskp;
	
	*dmq1 = dm;
	
	
	PetscFree(siq2);
	PetscFree(sjq2);
	PetscFree(skq2);
	PetscFree(lmxq2);
	PetscFree(lmyq2);
	PetscFree(lmzq2);
	
	/* force element creation using MY numbering */
	{
		PetscInt nel,nen;
		const PetscInt *els;

		ierr = DMDASetElementType_Q1(dm);CHKERRQ(ierr);
		ierr = DMDAGetElements_DA_Q1_3D(dm,&nel,&nen,&els);CHKERRQ(ierr);
	}	
	
	PetscFunctionReturn(0);
}



/* element helpers */
#undef __FUNCT__
#define __FUNCT__ "DMDAEQ1_GetElementCoordinates_3D"
PetscErrorCode DMDAEQ1_GetElementCoordinates_3D(PetscScalar elcoords[],PetscInt elnid[],PetscScalar LA_gcoords[])
{
	PetscInt n;
	PetscErrorCode ierr;
	PetscFunctionBegin;
	for (n=0; n<8; n++) {
		elcoords[3*n  ] = LA_gcoords[3*elnid[n]  ];
		elcoords[3*n+1] = LA_gcoords[3*elnid[n]+1];
		elcoords[3*n+2] = LA_gcoords[3*elnid[n]+2];
	}
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAEQ1_GetScalarElementField_3D"
PetscErrorCode DMDAEQ1_GetScalarElementField_3D(PetscScalar elfield[],PetscInt elnid[],PetscScalar LA_gfield[])
{
	PetscInt n;
	PetscErrorCode ierr;
	PetscFunctionBegin;
	for (n=0; n<8; n++) {
		elfield[n] = LA_gfield[elnid[n]];
	}
	PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "DMDAEQ1_GetVectorElementField_3D"
PetscErrorCode DMDAEQ1_GetVectorElementField_3D(PetscScalar elfield[],PetscInt elnid[],PetscScalar LA_gfield[])
{
	PetscInt n;
	PetscErrorCode ierr;
	PetscFunctionBegin;
	for (n=0; n<8; n++) {
		elfield[3*n  ] = LA_gfield[3*elnid[n]  ];
		elfield[3*n+1] = LA_gfield[3*elnid[n]+1];
		elfield[3*n+2] = LA_gfield[3*elnid[n]+2];
	}
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAEQ1_SetValuesLocalStencil_AddValues_DOF"
PetscErrorCode DMDAEQ1_SetValuesLocalStencil_AddValues_DOF(PetscScalar *fields_F,PetscInt ndof,PetscInt eqn[],PetscScalar Fe[])
{
  PetscInt n,d,el_idx,idx;
	
  PetscFunctionBegin;
	for (d=0; d<ndof; d++) {
		for (n = 0; n<8; n++) {
			el_idx = ndof*n + d;
			idx    = eqn[el_idx];
			fields_F[idx] += Fe[el_idx];
		}
	}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAEQ1_GetElementLocalIndicesDOF"
PetscErrorCode DMDAEQ1_GetElementLocalIndicesDOF(PetscInt el_localIndices[],PetscInt ndof,PetscInt elnid[])
{
	PetscInt n,d;
	PetscErrorCode ierr;
	PetscFunctionBegin;
	for (d=0; d<ndof; d++) {
		for (n=0; n<8; n++) {
			el_localIndices[ndof*n+d] = ndof*elnid[n]+d;
		}
	}		
	PetscFunctionReturn(0);
}

