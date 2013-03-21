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
 **    Filename:      model_utils.c
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

#include "ptatin3d.h"
#include "ptatin3d_defs.h"
#include "private/ptatin_impl.h"
#include "ptatin3d_stokes.h"
#include "ptatin3d_energy.h"
#include "mesh_quality_metrics.h"
#include "mesh_update.h"
#include "mesh_deformation.h"

#include "dmda_bcs.h"
#include "dmda_iterator.h"
#include "dmda_redundant.h"
#include "dmda_remesh.h"
#include "dmda_update_coords.h"
#include "dmda_duplicate.h"
#include "dmdae.h"
#include "dmda_element_q2p1.h"
#include "dmda_element_q1.h"
#include "element_type_Q2.h"
#include "element_utils_q2.h"
#include "element_utils_q1.h"

#include "swarm_fields.h"
#include "MPntStd_def.h"
#include "MPntPStokes_def.h"
#include "MPntPStokesPl_def.h"
#include "MPntPEnergy_def.h"
#include "ptatin_std_dirichlet_boundary_conditions.h"

#include "model_utils.h"


#undef __FUNCT__
#define __FUNCT__ "MPntGetField_global_element_IJKindex"
PetscErrorCode MPntGetField_global_element_IJKindex(DM da, MPntStd *material_point, PetscInt *I, PetscInt *J, PetscInt *K)
{
	PetscInt    li, lj, lk,lmx, lmy, lmz, si, sj, sk, localeid;	
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	MPntStdGetField_local_element_index(material_point,&localeid);
	ierr = DMDAGetCornersElementQ2(da,&si,&sj,&sk,&lmx,&lmy,&lmz);CHKERRQ(ierr);
    
	si = si/2; 
	sj = sj/2;
	sk = sk/2;
    //	lmx -= si;
    //	lmy -= sj;
    //	lmz -= sk;
	//global/localrank = mx*my*k + mx*j + i;
	lk = (PetscInt)localeid/(lmx*lmy);
	lj = (PetscInt)(localeid - lk*(lmx*lmy))/lmx;
	li = localeid - lk*(lmx*lmy) - lj*lmx;
    
	if ( (li < 0) || (li>=lmx) ) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"I computed incorrectly"); }
	if ( (lj < 0) || (lj>=lmy) ) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"J computed incorrectly"); }
	if ( (lk < 0) || (lk>=lmz) ) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"K computed incorrectly"); }
	//printf("li,lj,lk %d %d %d \n", li,lj,lk );
	
	*K = lk + sk;
	*J = lj + sj;
	*I = li + si;
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "pTatinModelGetOptionReal"
PetscErrorCode pTatinModelGetOptionReal(const char option[],PetscReal *val,
																				const char error[],
																				const char default_opt[],
																				PetscBool essential)
{
	PetscBool flg;
	PetscErrorCode ierr;

	PetscFunctionBegin;
	ierr = PetscOptionsGetReal(PETSC_NULL,option,val,&flg);CHKERRQ(ierr);
	if (essential) {
		if (!flg) {
			if (!default_opt) {
				SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"ModelOptionMissing(%s): \n\t\t%s ",option,error);
			} else {
				SETERRQ3(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"ModelOptionMissing(%s): \n\t\t%s : Suggested default values %s ",option,error,default_opt);
			}
		}
	}
	PetscFunctionReturn(0);
}


