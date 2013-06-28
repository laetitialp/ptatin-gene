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
 **    Filename:      test_stokes_operators.c
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

static const char help[] = "Stokes solver using Q2-Pm1 mixed finite elements.\n"
"3D prototype of the (p)ragmatic version of Tatin. (pTatin3d_v0.0)\n\n";


#include "ptatin3d.h"
#include "private/ptatin_impl.h"

#include "material_point_utils.h"
#include "material_point_std_utils.h"
#include "ptatin_models.h"
#include "ptatin_utils.h"
#include "element_utils_q2.h"
#include "stokes_form_function.h"
#include "stokes_assembly.h"
#include "stokes_operators.h"


#undef __FUNCT__  
#define __FUNCT__ "_GenerateTestVector"
PetscErrorCode _GenerateTestVector(DM da,PetscInt dofs,PetscInt index,Vec x)
{
	PetscErrorCode ierr;
	Vec tmp;
	DMDACoor3d ***coors;
	PetscInt i,j,k,mstart,nstart,pstart,m,n,p;
	DM cda;
	PetscInt NUM_GINDICES, *GINDICES;
	
	
	
	ierr = DMDAGetGlobalIndices( da, &NUM_GINDICES, &GINDICES );CHKERRQ(ierr);
	
	ierr = DMDAGetCoordinateDA(da,&cda);CHKERRQ(ierr);
	ierr = DMDAGetGhostedCoordinates(da,&tmp);CHKERRQ(ierr);
	ierr = DMDAGetGhostCorners(cda,&mstart,&nstart,&pstart,&m,&n,&p);CHKERRQ(ierr);
	
	ierr = DMDAVecGetArray(cda,tmp,&coors);CHKERRQ(ierr);
	for (i=mstart; i<mstart+m; i++) {
		for (j=nstart; j<nstart+n; j++) {
			for (k=pstart; k<pstart+p; k++) {
				PetscInt LIDX = (i-mstart) + (j-nstart)*m + (k-pstart)*m*n;
				PetscInt GIDX = GINDICES[ dofs*LIDX + index ];
				PetscScalar f,XX,YY,ZZ;
				
				XX = coors[k][j][i].x;
				YY = coors[k][j][i].y;
				ZZ = coors[k][j][i].z;
				
				f = XX + YY + ZZ + (PetscScalar)index + 3.0;
				
				ierr = VecSetValue(x,GIDX,f,INSERT_VALUES);CHKERRQ(ierr);
			}}}
	ierr = DMDAVecRestoreArray(cda,tmp,&coors);CHKERRQ(ierr);
	
	ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
	ierr = VecAssemblyEnd(x);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "_GenerateTestVectorDAP"
PetscErrorCode _GenerateTestVectorDAP(DM da,PetscInt dofs,PetscInt index,Vec x)
{
	PetscErrorCode ierr;
	Vec tmp;
	PetscInt i,j,k,mstart,nstart,pstart,m,n,p;
	PetscInt NUM_GINDICES, *GINDICES;
	PetscInt M,N,P;
	PetscReal dx,dy,dz;
	
	
	ierr = DMDAGetInfo(da,0,&M,&N,&P,0,0,0, 0,0,0,0,0,0);CHKERRQ(ierr);
	dx = 2.0/(PetscReal)(M-1);
	dy = 2.0/(PetscReal)(N-1);
	dz = 2.0/(PetscReal)(P-1);
	
	
	ierr = DMDAGetGlobalIndices(da,&NUM_GINDICES,&GINDICES);CHKERRQ(ierr);
	
	ierr = DMDAGetGhostCorners(da,&mstart,&nstart,&pstart,&m,&n,&p);CHKERRQ(ierr);
	
	for (i=mstart; i<mstart+m; i++) {
		for (j=nstart; j<nstart+n; j++) {
			for (k=pstart; k<pstart+p; k++) {
				PetscInt LIDX = (i-mstart) + (j-nstart)*m + (k-pstart)*m*n;
				PetscInt GIDX = GINDICES[ dofs*LIDX + index ];
				PetscScalar f,XX,YY,ZZ;
				
				XX = -1.0 + i * dx;
				YY = -1.0 + j * dy;
				ZZ = -1.0 + k * dz;
				
				f = XX + YY + ZZ + (PetscScalar)index + 3.0;
				
				ierr = VecSetValue(x,GIDX,f,INSERT_VALUES);CHKERRQ(ierr);
			}}}
	
	ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
	ierr = VecAssemblyEnd(x);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ass_A11"
PetscErrorCode ass_A11(PhysCompStokes stk)
{
	PetscErrorCode ierr;
	DM da;
	Mat B;
	PetscBool same;
	Vec x,y;
	
	
	PetscFunctionBegin;
	
	PetscPrintf(PETSC_COMM_WORLD,"+  Test [%s]: Mesh %D x %D x %D \n", __FUNCT__,stk->mx,stk->my,stk->mz );
	da = stk->dav;

	ierr = DMGetMatrix(da,MATAIJ,&B);CHKERRQ(ierr);
	
	ierr = PetscTypeCompare((PetscObject)B,MATSBAIJ,&same);CHKERRQ(ierr);
	if (same) {
		ierr = MatSetOption(B,MAT_IGNORE_LOWER_TRIANGULAR,PETSC_TRUE);CHKERRQ(ierr);
	}
	
	ierr = MatAssemble_StokesA_AUU(B,da,stk->u_bclist,stk->volQ);CHKERRQ(ierr);
	
	ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);
	ierr = VecDuplicate(x,&y);CHKERRQ(ierr);
	
	ierr = VecSet(x,0.0);CHKERRQ(ierr);
	ierr = _GenerateTestVector(da,3,0,x);CHKERRQ(ierr);
	ierr = _GenerateTestVector(da,3,1,x);CHKERRQ(ierr);
	ierr = _GenerateTestVector(da,3,2,x);CHKERRQ(ierr);
	
	ierr = MatMult(B,x,y);CHKERRQ(ierr);
	
	ierr = VecDestroy(&x);CHKERRQ(ierr);
	ierr = VecDestroy(&y);CHKERRQ(ierr);
	ierr = MatDestroy(&B);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ass_B22"
PetscErrorCode ass_B22(PhysCompStokes stk)
{
	PetscErrorCode ierr;
	DM dau,dap;
	Mat B;
	PetscBool same;
	Vec x,y;
	
	
	PetscFunctionBegin;
	
	PetscPrintf(PETSC_COMM_WORLD,"+  Test [%s]: Mesh %D x %D x %D \n", __FUNCT__,stk->mx,stk->my,stk->mz );
	dau = stk->dav;
	dap = stk->dap;
	
	ierr = DMGetMatrix(dap,MATAIJ,&B);CHKERRQ(ierr);
	
	ierr = PetscTypeCompare((PetscObject)B,MATSBAIJ,&same);CHKERRQ(ierr);
	if (same) {
		ierr = MatSetOption(B,MAT_IGNORE_LOWER_TRIANGULAR,PETSC_TRUE);CHKERRQ(ierr);
	}
	
	ierr = MatAssemble_StokesPC_ScaledMassMatrix(B,dau,dap,stk->p_bclist,stk->volQ);CHKERRQ(ierr);
	
	ierr = DMCreateGlobalVector(dap,&x);CHKERRQ(ierr);
	ierr = VecDuplicate(x,&y);CHKERRQ(ierr);

	/*
	ierr = VecSet(x,0.0);CHKERRQ(ierr);
	ierr = _GenerateTestVector(da,3,0,x);CHKERRQ(ierr);
	ierr = _GenerateTestVector(da,3,1,x);CHKERRQ(ierr);
	ierr = _GenerateTestVector(da,3,2,x);CHKERRQ(ierr);
	
	ierr = MatMult(B,x,y);CHKERRQ(ierr);
	*/
	
	ierr = VecDestroy(&x);CHKERRQ(ierr);
	ierr = VecDestroy(&y);CHKERRQ(ierr);
	ierr = MatDestroy(&B);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "compare_mf_A11"
PetscErrorCode compare_mf_A11(PhysCompStokes user)
{
	MatStokesMF    StkCtx;
	MatA11MF       A11Ctx;
	Mat            Auu,B;
	Vec            x,y,y2;
	DM             da;
	PetscScalar    min,max;
	PetscReal      cmp;
	PetscLogDouble t0,t1;
	double tl,timeMIN,timeMAX;
	PetscErrorCode ierr;
	
	
	PetscFunctionBegin;

	PetscPrintf(PETSC_COMM_WORLD,"\n+  Test [%s]: Mesh %D x %D x %D \n", __FUNCT__,user->mx,user->my,user->mz );

	/* create the mf operators */
	da = user->dav;
	ierr = MatStokesMFCreate(&StkCtx);CHKERRQ(ierr);
	ierr = MatStokesMFSetup(StkCtx,user);CHKERRQ(ierr);
	ierr = MatCopy_StokesMF_A11MF(StkCtx,&A11Ctx);CHKERRQ(ierr);
	
	ierr = StokesQ2P1CreateMatrix_MFOperator_A11(A11Ctx,&Auu);CHKERRQ(ierr);
	
	/* matrix free */
	ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);
	ierr = VecDuplicate(x,&y);CHKERRQ(ierr);
	
	ierr = VecSet(x,0.0);CHKERRQ(ierr);
	ierr = _GenerateTestVector(da,3,0,x);CHKERRQ(ierr);
	ierr = _GenerateTestVector(da,3,1,x);CHKERRQ(ierr);
	ierr = _GenerateTestVector(da,3,2,x);CHKERRQ(ierr);
	
	PetscGetTime(&t0);
	ierr = MatMult(Auu,x,y);CHKERRQ(ierr);
	PetscGetTime(&t1);
	tl = (double)(t1 - t0);
	ierr = MPI_Allreduce(&tl,&timeMIN,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD);CHKERRQ(ierr);
	ierr = MPI_Allreduce(&tl,&timeMAX,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD);CHKERRQ(ierr);
	PetscPrintf(PETSC_COMM_WORLD,"MatMultA11(MF):      time %1.4e (sec): ratio %1.4e%%: min/max %1.4e %1.4e (sec)\n",tl,100.0*(timeMIN/timeMAX),timeMIN,timeMAX);


	/* assembled */
	ierr = VecDuplicate(x,&y2);CHKERRQ(ierr);

	ierr = DMGetMatrix(da,MATAIJ,&B);CHKERRQ(ierr);
	PetscGetTime(&t0);
	ierr = MatAssemble_StokesA_AUU(B,da,user->u_bclist,user->volQ);CHKERRQ(ierr);
	PetscGetTime(&t1);
	tl = (double)(t1 - t0);
	ierr = MPI_Allreduce(&tl,&timeMIN,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD);CHKERRQ(ierr);
	ierr = MPI_Allreduce(&tl,&timeMAX,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD);CHKERRQ(ierr);
	PetscPrintf(PETSC_COMM_WORLD,"MatAssemblyA11(ASM): time %1.4e (sec): ratio %1.4e%%: min/max %1.4e %1.4e (sec)\n",tl,100.0*(timeMIN/timeMAX),timeMIN,timeMAX);
	
	PetscGetTime(&t0);
	ierr = MatMult(B,x,y2);CHKERRQ(ierr);
	PetscGetTime(&t1);
	tl = (double)(t1 - t0);
	ierr = MPI_Allreduce(&tl,&timeMIN,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD);CHKERRQ(ierr);
	ierr = MPI_Allreduce(&tl,&timeMAX,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD);CHKERRQ(ierr);
	PetscPrintf(PETSC_COMM_WORLD,"MatMultA11(ASM):     time %1.4e (sec): ratio %1.4e%%: min/max %1.4e %1.4e (sec)\n",tl,100.0*(timeMIN/timeMAX),timeMIN,timeMAX);

	/*
	PetscPrintf(PETSC_COMM_WORLD,"y_mfo\n");
	ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
	PetscPrintf(PETSC_COMM_WORLD,"y_asm\n");
	ierr = VecView(y2,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
	*/
	
	/* compare result */
	ierr = VecDot(y,y,&cmp);CHKERRQ(ierr);
	PetscPrintf(PETSC_COMM_WORLD,"  y.y    = %+1.8e [mfo]\n", cmp );
	ierr = VecDot(y2,y2,&cmp);CHKERRQ(ierr);
	PetscPrintf(PETSC_COMM_WORLD,"  y2.y2  = %+1.8e [asm]\n", cmp );
		
	ierr = VecAXPY(y2,-1.0,y);CHKERRQ(ierr); /* y2 = y2 - y */
	ierr = VecMin(y2,PETSC_NULL,&min);CHKERRQ(ierr);
	ierr = VecMax(y2,PETSC_NULL,&max);CHKERRQ(ierr);
	PetscPrintf(PETSC_COMM_WORLD,"  min[A11_mfo.x-A11_asm.x]  = %+1.8e \n", min );
	PetscPrintf(PETSC_COMM_WORLD,"  max[A11_mfo.x-A11_asm.x]  = %+1.8e \n", max );

	ierr = VecDestroy(&x);CHKERRQ(ierr);
	ierr = VecDestroy(&y);CHKERRQ(ierr);
	ierr = VecDestroy(&y2);CHKERRQ(ierr);
	
	ierr = MatA11MFDestroy(&A11Ctx);CHKERRQ(ierr);
	ierr = MatStokesMFDestroy(&StkCtx);CHKERRQ(ierr);
	ierr = MatDestroy(&Auu);CHKERRQ(ierr);
	ierr = MatDestroy(&B);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "compare_mf_A21"
PetscErrorCode compare_mf_A21(PhysCompStokes user)
{
	MatStokesMF    StkCtx;
	MatA11MF       A11Ctx;
	Mat            A21,B;
	Vec            x,y,y2;
	DM             dav,dap;
	PetscScalar    min,max;
	PetscReal      cmp;
	PetscErrorCode ierr;
	
	
	PetscFunctionBegin;
	
	PetscPrintf(PETSC_COMM_WORLD,"\n+  Test [%s]: Mesh %D x %D x %D \n", __FUNCT__,user->mx,user->my,user->mz );
	
	/* create the mf operators */
	dav = user->dav;
	dap = user->dap;
	ierr = MatStokesMFCreate(&StkCtx);CHKERRQ(ierr);
	ierr = MatStokesMFSetup(StkCtx,user);CHKERRQ(ierr);
	ierr = MatCopy_StokesMF_A11MF(StkCtx,&A11Ctx);CHKERRQ(ierr);
	
	ierr = StokesQ2P1CreateMatrix_MFOperator_A21(StkCtx,&A21);CHKERRQ(ierr);
	
	/* matrix free */
	ierr = DMCreateGlobalVector(dav,&x);CHKERRQ(ierr);
	ierr = DMCreateGlobalVector(dap,&y);CHKERRQ(ierr);
	
	ierr = VecSet(x,0.0);CHKERRQ(ierr);
	ierr = _GenerateTestVector(dav,3,0,x);CHKERRQ(ierr);
	ierr = _GenerateTestVector(dav,3,1,x);CHKERRQ(ierr);
	ierr = _GenerateTestVector(dav,3,2,x);CHKERRQ(ierr);
	
	ierr = MatMult(A21,x,y);CHKERRQ(ierr);
	
	/* assembled */
	ierr = VecDuplicate(y,&y2);CHKERRQ(ierr);
	
	ierr = StokesQ2P1CreateMatrix_A21(user,&B);CHKERRQ(ierr);
	ierr = MatAssemble_StokesA_A21(B,dav,dap,user->u_bclist,user->p_bclist,user->volQ);CHKERRQ(ierr);

	ierr = MatMult(B,x,y2);CHKERRQ(ierr);


	/* compare result */
	ierr = VecDot(y,y,&cmp);CHKERRQ(ierr);
	PetscPrintf(PETSC_COMM_WORLD,"  y.y    = %+1.8e [mfo]\n", cmp );
	ierr = VecDot(y2,y2,&cmp);CHKERRQ(ierr);
	PetscPrintf(PETSC_COMM_WORLD,"  y2.y2  = %+1.8e [asm]\n", cmp );
	
	ierr = VecAXPY(y2,-1.0,y);CHKERRQ(ierr); /* y2 = y2 - y */
	ierr = VecMin(y2,PETSC_NULL,&min);CHKERRQ(ierr);
	ierr = VecMax(y2,PETSC_NULL,&max);CHKERRQ(ierr);
	PetscPrintf(PETSC_COMM_WORLD,"  min[A21_mfo.x-A21_asm.x]  = %+1.8e \n", min );
	PetscPrintf(PETSC_COMM_WORLD,"  max[A21_mfo.x-A21_asm.x]  = %+1.8e \n", max );
	
	//	ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
	
	ierr = VecDestroy(&x);CHKERRQ(ierr);
	ierr = VecDestroy(&y);CHKERRQ(ierr);
	ierr = VecDestroy(&y2);CHKERRQ(ierr);
	
	ierr = MatA11MFDestroy(&A11Ctx);CHKERRQ(ierr);
	ierr = MatStokesMFDestroy(&StkCtx);CHKERRQ(ierr);
	ierr = MatDestroy(&A21);CHKERRQ(ierr);
	ierr = MatDestroy(&B);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "compare_mf_A12"
PetscErrorCode compare_mf_A12(PhysCompStokes user)
{
	MatStokesMF    StkCtx;
	MatA11MF       A11Ctx;
	Mat            A12,B;
	Vec            x,y,y2;
	DM             dav,dap;
	PetscScalar    min,max;
	PetscReal      cmp;
	PetscErrorCode ierr;
	
	
	PetscFunctionBegin;
	
	PetscPrintf(PETSC_COMM_WORLD,"\n+  Test [%s]: Mesh %D x %D x %D \n", __FUNCT__,user->mx,user->my,user->mz );
	
	/* create the mf operators */
	dav = user->dav;
	dap = user->dap;
	ierr = MatStokesMFCreate(&StkCtx);CHKERRQ(ierr);
	ierr = MatStokesMFSetup(StkCtx,user);CHKERRQ(ierr);
	ierr = MatCopy_StokesMF_A11MF(StkCtx,&A11Ctx);CHKERRQ(ierr);
	
	ierr = StokesQ2P1CreateMatrix_MFOperator_A12(StkCtx,&A12);CHKERRQ(ierr);
	
	/* matrix free */
	ierr = DMCreateGlobalVector(dap,&x);CHKERRQ(ierr);
	ierr = DMCreateGlobalVector(dav,&y);CHKERRQ(ierr);
	
	ierr = VecSet(x,0.0);CHKERRQ(ierr);
	ierr = _GenerateTestVectorDAP(dap,4,0,x);CHKERRQ(ierr);
	ierr = _GenerateTestVectorDAP(dap,4,1,x);CHKERRQ(ierr);
	ierr = _GenerateTestVectorDAP(dap,4,2,x);CHKERRQ(ierr);
	ierr = _GenerateTestVectorDAP(dap,4,3,x);CHKERRQ(ierr);
	
	ierr = MatMult(A12,x,y);CHKERRQ(ierr);
	
	/* assembled */
	ierr = VecDuplicate(y,&y2);CHKERRQ(ierr);
	
	ierr = StokesQ2P1CreateMatrix_A12(user,&B);CHKERRQ(ierr);
	ierr = MatAssemble_StokesA_A12(B,dav,dap,user->u_bclist,user->p_bclist,user->volQ);CHKERRQ(ierr);
	
	ierr = MatMult(B,x,y2);CHKERRQ(ierr);
	
	
	/* compare result */
	ierr = VecDot(y,y,&cmp);CHKERRQ(ierr);
	PetscPrintf(PETSC_COMM_WORLD,"  y.y    = %+1.8e [mfo]\n", cmp );
	ierr = VecDot(y2,y2,&cmp);CHKERRQ(ierr);
	PetscPrintf(PETSC_COMM_WORLD,"  y2.y2  = %+1.8e [asm]\n", cmp );
	
	ierr = VecAXPY(y2,-1.0,y);CHKERRQ(ierr); /* y2 = y2 - y */
	ierr = VecMin(y2,PETSC_NULL,&min);CHKERRQ(ierr);
	ierr = VecMax(y2,PETSC_NULL,&max);CHKERRQ(ierr);
	PetscPrintf(PETSC_COMM_WORLD,"  min[A12_mfo.x-A12_asm.x]  = %+1.8e \n", min );
	PetscPrintf(PETSC_COMM_WORLD,"  max[A12_mfo.x-A12_asm.x]  = %+1.8e \n", max );
	
	ierr = VecDestroy(&x);CHKERRQ(ierr);
	ierr = VecDestroy(&y);CHKERRQ(ierr);
	ierr = VecDestroy(&y2);CHKERRQ(ierr);
	
	ierr = MatA11MFDestroy(&A11Ctx);CHKERRQ(ierr);
	ierr = MatStokesMFDestroy(&StkCtx);CHKERRQ(ierr);
	ierr = MatDestroy(&A12);CHKERRQ(ierr);
	ierr = MatDestroy(&B);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "compare_mf_A"
PetscErrorCode compare_mf_A(PhysCompStokes user)
{
	Mat            A,B;
	Vec            x,xu,xp,y,y2;
	DM             dav,dap,pack;
	PetscScalar    min,max;
	PetscReal      cmp;
	PetscErrorCode ierr;
	
	
	PetscFunctionBegin;
	
	PetscPrintf(PETSC_COMM_WORLD,"\n+  Test [%s]: Mesh %D x %D x %D \n", __FUNCT__,user->mx,user->my,user->mz );
	
	/* create the mf operators */
	dav = user->dav;
	dap = user->dap;
	pack = user->stokes_pack;
	
	/* matrix free */
	ierr = DMCreateGlobalVector(pack,&x);CHKERRQ(ierr);
	ierr = DMCompositeGetAccess(pack,x,&xu,&xp);CHKERRQ(ierr);
		ierr = VecSetRandom(xu,PETSC_NULL);CHKERRQ(ierr);
		ierr = VecSetRandom(xp,PETSC_NULL);CHKERRQ(ierr);
//		ierr = VecZeroEntries(xp);CHKERRQ(ierr);
	ierr = DMCompositeRestoreAccess(pack,x,&xu,&xp);CHKERRQ(ierr);
	
	ierr = VecDuplicate(x,&y);CHKERRQ(ierr);
	
	ierr = StokesQ2P1CreateMatrix_Operator(user,&A);CHKERRQ(ierr);
	ierr = MatMult(A,x,y);CHKERRQ(ierr);

//	PetscPrintf(PETSC_COMM_WORLD,"y_mfo\n");
//	ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
	
	/* assembled */
	ierr = VecDuplicate(x,&y2);CHKERRQ(ierr);
	
	ierr = StokesQ2P1CreateMatrixNest_Operator(user,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,&B);CHKERRQ(ierr);
	ierr = MatMult(B,x,y2);CHKERRQ(ierr);
	
//	 PetscPrintf(PETSC_COMM_WORLD,"y_asm\n");
//	 ierr = VecView(y2,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

	
	/* compare result */
	ierr = VecDot(y,y,&cmp);CHKERRQ(ierr);
	PetscPrintf(PETSC_COMM_WORLD,"  y.y    = %+1.8e [mfo]\n", cmp );
	ierr = VecDot(y2,y2,&cmp);CHKERRQ(ierr);
	PetscPrintf(PETSC_COMM_WORLD,"  y2.y2  = %+1.8e [asm]\n", cmp );
	
	ierr = VecAXPY(y2,-1.0,y);CHKERRQ(ierr); /* y2 = y2 - y */
	ierr = VecMin(y2,PETSC_NULL,&min);CHKERRQ(ierr);
	ierr = VecMax(y2,PETSC_NULL,&max);CHKERRQ(ierr);
	PetscPrintf(PETSC_COMM_WORLD,"  min[A_mfo.x-A_asm.x]  = %+1.8e \n", min );
	PetscPrintf(PETSC_COMM_WORLD,"  max[A_mfo.x-A_asm.x]  = %+1.8e \n", max );
	
	ierr = VecDestroy(&x);CHKERRQ(ierr);
	ierr = VecDestroy(&y);CHKERRQ(ierr);
	ierr = VecDestroy(&y2);CHKERRQ(ierr);
	
	ierr = MatDestroy(&A);CHKERRQ(ierr);
	ierr = MatDestroy(&B);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "compare_mf_diagA11"
PetscErrorCode compare_mf_diagA11(PhysCompStokes user)
{
	MatStokesMF    StkCtx;
	MatA11MF       A11Ctx;
	Mat            Auu,B;
	Vec            y,y2;
	DM             da;
	PetscScalar    min,max;
	PetscReal      cmp;
	PetscErrorCode ierr;
	
	
	PetscFunctionBegin;
	
	PetscPrintf(PETSC_COMM_WORLD,"\n+  Test [%s]: Mesh %D x %D x %D \n", __FUNCT__,user->mx,user->my,user->mz );
	
	/* create the mf operators */
	da = user->dav;
	ierr = MatStokesMFCreate(&StkCtx);CHKERRQ(ierr);
	ierr = MatStokesMFSetup(StkCtx,user);CHKERRQ(ierr);
	ierr = MatCopy_StokesMF_A11MF(StkCtx,&A11Ctx);CHKERRQ(ierr);
	
	ierr = StokesQ2P1CreateMatrix_MFOperator_A11(A11Ctx,&Auu);CHKERRQ(ierr);
	
	/* matrix free */
	ierr = DMCreateGlobalVector(da,&y);CHKERRQ(ierr);
	
	ierr = MatGetDiagonal(Auu,y);CHKERRQ(ierr);
	
	/* assembled */
	ierr = VecDuplicate(y,&y2);CHKERRQ(ierr);
	
	ierr = DMGetMatrix(da,MATAIJ,&B);CHKERRQ(ierr);
	ierr = MatAssemble_StokesA_AUU(B,da,user->u_bclist,user->volQ);CHKERRQ(ierr);
	
	ierr = MatGetDiagonal(B,y2);CHKERRQ(ierr);
	
	/* compare result */
	ierr = VecDot(y,y,&cmp);CHKERRQ(ierr);
	PetscPrintf(PETSC_COMM_WORLD,"  y.y    = %+1.8e [mfo]\n", cmp );
	ierr = VecDot(y2,y2,&cmp);CHKERRQ(ierr);
	PetscPrintf(PETSC_COMM_WORLD,"  y2.y2  = %+1.8e [asm]\n", cmp );
	
	ierr = VecAXPY(y2,-1.0,y);CHKERRQ(ierr); /* y2 = y2 - y */
	ierr = VecMin(y2,PETSC_NULL,&min);CHKERRQ(ierr);
	ierr = VecMax(y2,PETSC_NULL,&max);CHKERRQ(ierr);
	PetscPrintf(PETSC_COMM_WORLD,"  min[diagA11_mfo-diagA11_asm]  = %+1.8e \n", min );
	PetscPrintf(PETSC_COMM_WORLD,"  max[diagA11_mfo-diagA11_asm]  = %+1.8e \n", max );
	
	ierr = VecDestroy(&y);CHKERRQ(ierr);
	ierr = VecDestroy(&y2);CHKERRQ(ierr);
	
	ierr = MatA11MFDestroy(&A11Ctx);CHKERRQ(ierr);
	ierr = MatStokesMFDestroy(&StkCtx);CHKERRQ(ierr);
	ierr = MatDestroy(&Auu);CHKERRQ(ierr);
	ierr = MatDestroy(&B);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "apply_mf_A11"
PetscErrorCode apply_mf_A11(PhysCompStokes user)
{
	MatStokesMF    StkCtx;
	MatA11MF       A11Ctx;
	Mat            Auu;
	Vec            x,y;
	DM             da;
	PetscScalar    min,max;
	PetscLogDouble t0,t1;
	double         tl,timeMIN,timeMAX;
	PetscInt       ii,iterations;
	PetscErrorCode ierr;
	
	
	PetscFunctionBegin;
	
	PetscPrintf(PETSC_COMM_WORLD,"\n+  Test [%s]: Mesh %D x %D x %D \n", __FUNCT__,user->mx,user->my,user->mz );
	iterations = 5;
	ierr = PetscOptionsGetInt(PETSC_NULL,"-iterations",&iterations,0);CHKERRQ(ierr);
	
	/* create the mf operators */
	da = user->dav;
	ierr = MatStokesMFCreate(&StkCtx);CHKERRQ(ierr);
	ierr = MatStokesMFSetup(StkCtx,user);CHKERRQ(ierr);
	ierr = MatCopy_StokesMF_A11MF(StkCtx,&A11Ctx);CHKERRQ(ierr);
	
	ierr = StokesQ2P1CreateMatrix_MFOperator_A11(A11Ctx,&Auu);CHKERRQ(ierr);
	
	/* matrix free */
	ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);
	ierr = VecDuplicate(x,&y);CHKERRQ(ierr);
	
	ierr = VecSet(x,0.0);CHKERRQ(ierr);
	ierr = _GenerateTestVector(da,3,0,x);CHKERRQ(ierr);
	ierr = _GenerateTestVector(da,3,1,x);CHKERRQ(ierr);
	ierr = _GenerateTestVector(da,3,2,x);CHKERRQ(ierr);
	
	PetscGetTime(&t0);
	for (ii=0; ii<iterations; ii++) {
		ierr = MatMult(Auu,x,y);CHKERRQ(ierr);
	}
	PetscGetTime(&t1);
	tl = (double)(t1 - t0);
	ierr = MPI_Allreduce(&tl,&timeMIN,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD);CHKERRQ(ierr);
	ierr = MPI_Allreduce(&tl,&timeMAX,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD);CHKERRQ(ierr);

	PetscPrintf(PETSC_COMM_WORLD,"MatMultA11(MF): iterations %.6d     time %1.4e (sec): ratio %1.4e%%: min/max %1.4e %1.4e (sec)\n",iterations,tl,100.0*(timeMIN/timeMAX),timeMIN,timeMAX);
	PetscPrintf(PETSC_COMM_WORLD,"MatMultA11(MF): average               time %1.4e (sec): ratio %1.4e%%: min/max %1.4e %1.4e (sec)\n",tl/((double)iterations),100.0*(timeMIN/timeMAX),timeMIN/((double)iterations),timeMAX/((double)iterations));
	
	
	ierr = VecDestroy(&x);CHKERRQ(ierr);
	ierr = VecDestroy(&y);CHKERRQ(ierr);
	
	ierr = MatA11MFDestroy(&A11Ctx);CHKERRQ(ierr);
	ierr = MatStokesMFDestroy(&StkCtx);CHKERRQ(ierr);
	ierr = MatDestroy(&Auu);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "apply_asm_A11"
PetscErrorCode apply_asm_A11(PhysCompStokes user)
{
	Mat            B;
	Vec            x,y;
	DM             da;
	PetscScalar    min,max;
	PetscLogDouble t0,t1;
	double         tl,timeMIN,timeMAX;
	PetscInt       ii,iterations;
	PetscErrorCode ierr;
	
	
	PetscFunctionBegin;
	
	PetscPrintf(PETSC_COMM_WORLD,"\n+  Test [%s]: Mesh %D x %D x %D \n", __FUNCT__,user->mx,user->my,user->mz );
	iterations = 5;
	ierr = PetscOptionsGetInt(PETSC_NULL,"-iterations",&iterations,0);CHKERRQ(ierr);
	
	/* create the assembled operator */
	da = user->dav;
	
	/* assembled matrix */
	ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);
	ierr = VecDuplicate(x,&y);CHKERRQ(ierr);
	
	ierr = VecSet(x,0.0);CHKERRQ(ierr);
	ierr = _GenerateTestVector(da,3,0,x);CHKERRQ(ierr);
	ierr = _GenerateTestVector(da,3,1,x);CHKERRQ(ierr);
	ierr = _GenerateTestVector(da,3,2,x);CHKERRQ(ierr);
	
	ierr = DMGetMatrix(da,MATAIJ,&B);CHKERRQ(ierr);
	PetscGetTime(&t0);
	ierr = MatAssemble_StokesA_AUU(B,da,user->u_bclist,user->volQ);CHKERRQ(ierr);
	PetscGetTime(&t1);
	tl = (double)(t1 - t0);
	ierr = MPI_Allreduce(&tl,&timeMIN,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD);CHKERRQ(ierr);
	ierr = MPI_Allreduce(&tl,&timeMAX,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD);CHKERRQ(ierr);
	PetscPrintf(PETSC_COMM_WORLD,"MatAssemblyA11(ASM):                   time %1.4e (sec): ratio %1.4e%%: min/max %1.4e %1.4e (sec)\n",tl,100.0*(timeMIN/timeMAX),timeMIN,timeMAX);
	
	PetscGetTime(&t0);
	for (ii=0; ii<iterations; ii++) {
		ierr = MatMult(B,x,y);CHKERRQ(ierr);
	}
	PetscGetTime(&t1);
	tl = (double)(t1 - t0);
	ierr = MPI_Allreduce(&tl,&timeMIN,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD);CHKERRQ(ierr);
	ierr = MPI_Allreduce(&tl,&timeMAX,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD);CHKERRQ(ierr);

	PetscPrintf(PETSC_COMM_WORLD,"MatMultA11(ASM): iterations %.6d     time %1.4e (sec): ratio %1.4e%%: min/max %1.4e %1.4e (sec)\n",iterations,tl,100.0*(timeMIN/timeMAX),timeMIN,timeMAX);
	PetscPrintf(PETSC_COMM_WORLD,"MatMultA11(ASM): average               time %1.4e (sec): ratio %1.4e%%: min/max %1.4e %1.4e (sec)\n",tl/((double)iterations),100.0*(timeMIN/timeMAX),timeMIN/((double)iterations),timeMAX/((double)iterations));
	
	ierr = VecDestroy(&x);CHKERRQ(ierr);
	ierr = VecDestroy(&y);CHKERRQ(ierr);
	
	ierr = MatDestroy(&B);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "perform_viscous_solve"
PetscErrorCode perform_viscous_solve(PhysCompStokes user)
{
	Mat            B;
	Vec            x,y;
	DM             da;
	PetscScalar    min,max;
	PetscLogDouble t0,t1;
	double         tl,timeMIN,timeMAX;
	PetscInt       its;
	PetscInt       ii,iterations;
	KSP            ksp;
	PetscErrorCode ierr;
	
	
	PetscFunctionBegin;
	
	PetscPrintf(PETSC_COMM_WORLD,"\n+  Test [%s]: Mesh %D x %D x %D \n", __FUNCT__,user->mx,user->my,user->mz );
	iterations = 5;
	ierr = PetscOptionsGetInt(PETSC_NULL,"-iterations",&iterations,0);CHKERRQ(ierr);
	
	/* create the assembled operator */
	da = user->dav;
	
	/* assembled matrix */
	ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);
	ierr = VecDuplicate(x,&y);CHKERRQ(ierr);
	
	ierr = VecSet(x,0.0);CHKERRQ(ierr);
	ierr = _GenerateTestVector(da,3,0,x);CHKERRQ(ierr);
	ierr = _GenerateTestVector(da,3,1,x);CHKERRQ(ierr);
	ierr = _GenerateTestVector(da,3,2,x);CHKERRQ(ierr);
	
	ierr = DMGetMatrix(da,MATAIJ,&B);CHKERRQ(ierr);
	PetscGetTime(&t0);
	ierr = MatAssemble_StokesA_AUU(B,da,user->u_bclist,user->volQ);CHKERRQ(ierr);
	PetscGetTime(&t1);
	tl = (double)(t1 - t0);
	ierr = MPI_Allreduce(&tl,&timeMIN,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD);CHKERRQ(ierr);
	ierr = MPI_Allreduce(&tl,&timeMAX,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD);CHKERRQ(ierr); 
	PetscPrintf(PETSC_COMM_WORLD,"MatAssemblyA11(ASM):                time %1.4e (sec): ratio %1.4e%%: min/max %1.4e %1.4e (sec)\n",tl,100.0*(timeMIN/timeMAX),timeMIN,timeMAX);
	
	ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
	ierr = KSPSetOperators(ksp,B,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
	ierr = KSPSetTolerances(ksp,1.0e-20,PETSC_DEFAULT,PETSC_DEFAULT,30);CHKERRQ(ierr);
	ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

	PetscGetTime(&t0);
	ierr = KSPSetUp(ksp);CHKERRQ(ierr);
	PetscGetTime(&t1);
	tl = (double)(t1 - t0);
	ierr = MPI_Allreduce(&tl,&timeMIN,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD);CHKERRQ(ierr);
	ierr = MPI_Allreduce(&tl,&timeMAX,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD);CHKERRQ(ierr); 

	PetscPrintf(PETSC_COMM_WORLD,"KSPSetUpA11:                        time %1.4e (sec): ratio %1.4e%%: min/max %1.4e %1.4e (sec)\n",tl,100.0*(timeMIN/timeMAX),timeMIN,timeMAX);
	
	PetscGetTime(&t0);
	for (ii=0; ii<iterations; ii++) {
		ierr = KSPSolve(ksp,x,y);CHKERRQ(ierr);
	}
	PetscGetTime(&t1);
	tl = (double)(t1 - t0);
	ierr = MPI_Allreduce(&tl,&timeMIN,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD);CHKERRQ(ierr);
	ierr = MPI_Allreduce(&tl,&timeMAX,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD);CHKERRQ(ierr);

	ierr = KSPGetTolerances(ksp,PETSC_NULL,PETSC_NULL,PETSC_NULL,&its);CHKERRQ(ierr);

	PetscPrintf(PETSC_COMM_WORLD,"KSPSolveA11(its = %d,cycles = %d)   time %1.4e (sec): ratio %1.4e%%: min/max %1.4e %1.4e (sec)\n",its,iterations,tl,100.0*(timeMIN/timeMAX),timeMIN,timeMAX);
	PetscPrintf(PETSC_COMM_WORLD,"KSPSolveA11: average                time %1.4e (sec): ratio %1.4e%%: min/max %1.4e %1.4e (sec)\n",tl/((double)iterations),100.0*(timeMIN/timeMAX),timeMIN/((double)iterations),timeMAX/((double)iterations));
	ierr = KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
	
	ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
	ierr = VecDestroy(&x);CHKERRQ(ierr);
	ierr = VecDestroy(&y);CHKERRQ(ierr);
	ierr = MatDestroy(&B);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "pTatin3d_assemble_stokes"
PetscErrorCode pTatin3d_assemble_stokes(int argc,char **argv)
{
	PetscErrorCode  ierr;
	DM              multipys_pack,dav,dap;
	pTatinCtx       user;
	PetscBool       found;

	PetscFunctionBegin;
	
	ierr = pTatin3dCreateContext(&user);CHKERRQ(ierr);
	ierr = pTatin3dParseOptions(user);CHKERRQ(ierr);

	/* Register all models */
	ierr = pTatinModelRegisterAll();CHKERRQ(ierr);
	/* Load model, call an initialization routines */
	ierr = pTatinModelLoad(user);CHKERRQ(ierr);
	
	ierr = pTatinModel_Initialize(user->model,user);CHKERRQ(ierr);
	
	/* Generate physics modules */
	ierr = pTatin3d_PhysCompStokesCreate(user);CHKERRQ(ierr);

	/* Pack all physics together */
	/* Here it's simple, we don't need a DM for this, just assign the pack DM to be equal to the stokes DM */
	ierr = PetscObjectReference((PetscObject)user->stokes_ctx->stokes_pack);CHKERRQ(ierr);
	user->pack = user->stokes_ctx->stokes_pack;

	/* fetch some local variables */
	multipys_pack = user->pack;
	dav           = user->stokes_ctx->dav;
	dap           = user->stokes_ctx->dap;
	
	ierr = pTatin3dCreateMaterialPoints(user,dav);CHKERRQ(ierr);
	
	/* mesh geometry */
	ierr = pTatinModel_ApplyInitialMeshGeometry(user->model,user);CHKERRQ(ierr);
	
	/* interpolate point coordinates (needed if mesh was modified) */
	//ierr = QuadratureStokesCoordinateSetUp(user->stokes_ctx->Q,dav);CHKERRQ(ierr);
	//for (e=0; e<QUAD_EDGES; e++) {
	//	ierr = SurfaceQuadratureStokesGeometrySetUp(user->stokes_ctx->surfQ[e],dav);CHKERRQ(ierr);
	//}
	/* interpolate material point coordinates (needed if mesh was modified) */
	ierr = MaterialPointCoordinateSetUp(user,dav);CHKERRQ(ierr);
	
	/* material geometry */
	ierr = pTatinModel_ApplyInitialMaterialGeometry(user->model,user);CHKERRQ(ierr);
	
	/* boundary conditions */
	ierr = pTatinModel_ApplyBoundaryCondition(user->model,user);CHKERRQ(ierr);


	/* update markers = >> gauss points */
	{
		int               npoints;
		DataField         PField_std;
		DataField         PField_stokes;
		MPntStd           *mp_std;
		MPntPStokes       *mp_stokes;
		
		DataBucketGetDataFieldByName(user->materialpoint_db, MPntStd_classname     , &PField_std);
		DataBucketGetDataFieldByName(user->materialpoint_db, MPntPStokes_classname , &PField_stokes);
		
		DataBucketGetSizes(user->materialpoint_db,&npoints,PETSC_NULL,PETSC_NULL);
		mp_std    = PField_std->data; /* should write a function to do this */
		mp_stokes = PField_stokes->data; /* should write a function to do this */
		
		ierr = SwarmUpdateGaussPropertiesLocalL2Projection_Q1_MPntPStokes(npoints,mp_std,mp_stokes,user->stokes_ctx->dav,user->stokes_ctx->volQ);CHKERRQ(ierr);
	}
	
	/* perform tests */
	PetscPrintf(PETSC_COMM_WORLD,"\n\n\n====================================================================\n");
	
//	ierr = ass_A11(user->stokes_ctx);CHKERRQ(ierr);
//	ierr = ass_B22(user->stokes_ctx);CHKERRQ(ierr);


	found  = PETSC_FALSE;
	ierr = PetscOptionsGetBool(PETSC_NULL,"-compare_operators",&found,0);CHKERRQ(ierr);
	if (found) {
		ierr = compare_mf_A11(user->stokes_ctx);CHKERRQ(ierr);

		ierr = compare_mf_A21(user->stokes_ctx);CHKERRQ(ierr);
		ierr = compare_mf_A12(user->stokes_ctx);CHKERRQ(ierr);
		
		ierr = compare_mf_A(user->stokes_ctx);CHKERRQ(ierr);

		ierr = compare_mf_diagA11(user->stokes_ctx);CHKERRQ(ierr);
	}
	
	found  = PETSC_FALSE;
	ierr = PetscOptionsGetBool(PETSC_NULL,"-apply_A11mf_operator",&found,PETSC_NULL);CHKERRQ(ierr);
	if (found) {
		ierr = apply_mf_A11(user->stokes_ctx);CHKERRQ(ierr);
	}

	found  = PETSC_FALSE;
	ierr = PetscOptionsGetBool(PETSC_NULL,"-apply_A11asm_operator",&found,PETSC_NULL);CHKERRQ(ierr);
	if (found) {
		ierr = apply_asm_A11(user->stokes_ctx);CHKERRQ(ierr);		
	}

	found  = PETSC_FALSE;
	ierr = PetscOptionsGetBool(PETSC_NULL,"-perform_viscous_solve_A11asm_operator",&found,PETSC_NULL);CHKERRQ(ierr);
	if (found) {
		ierr = perform_viscous_solve(user->stokes_ctx);CHKERRQ(ierr);		
	}
	
	
	
	PetscPrintf(PETSC_COMM_WORLD,"\n\n\n====================================================================\n");


	ierr = pTatin3dDestroyContext(&user);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
	PetscErrorCode ierr;
	
	ierr = PetscInitialize(&argc,&argv,0,help);CHKERRQ(ierr);
	
	ierr = pTatin3d_assemble_stokes(argc,argv);CHKERRQ(ierr);
	
	ierr = PetscFinalize();CHKERRQ(ierr);
	return 0;
}
