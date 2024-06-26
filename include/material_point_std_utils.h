/*@ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 **
 **    Copyright (c) 2012
 **        Dave A. May [dave.may@erdw.ethz.ch]
 **        Institute of Geophysics
 **        ETH Zürich
 **        Sonneggstrasse 5
 **        CH-8092 Zürich
 **        Switzerland
 **
 **    project:    pTatin3d
 **    filename:   material_point_std_utils.h
 **
 **
 **    pTatin3d is free software: you can redistribute it and/or modify
 **    it under the terms of the GNU General Public License as published
 **    by the Free Software Foundation, either version 3 of the License,
 **    or (at your option) any later version.
 **
 **    pTatin3d is distributed in the hope that it will be useful,
 **    but WITHOUT ANY WARRANTY; without even the implied warranty of
 **    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 **    See the GNU General Public License for more details.
 **
 **    You should have received a copy of the GNU General Public License
 **    along with pTatin3d. If not, see <http://www.gnu.org/licenses/>.
 **
 ** ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ @*/

#ifndef __MATERIAL_POINT_STD_UTILS_H__
#define __MATERIAL_POINT_STD_UTILS_H__

#include "data_bucket.h"
#include "data_exchanger.h"
#include "MPntStd_def.h"

#define MATERIAL_POINT_PHASE_UNASSIGNED -7

PetscErrorCode SwarmMPntStd_AssignUniquePointIdentifiers(MPI_Comm comm,DataBucket db,int start_pid,int end_pid);
PetscErrorCode SwarmMPntStd_CoordAssignment_LatticeLayout3d(DM da,PetscInt Nxp[],PetscReal perturb,DataBucket db);
PetscErrorCode SwarmMPntStd_CoordAssignment_GaussLayout3d(DM da,DataBucket db);
PetscErrorCode SwarmMPntStd_CoordAssignment_RandomLayout3d(DM da,PetscInt nPerCell,DataBucket db);
PetscErrorCode SwarmOutputParaView_MPntStd(DataBucket db,const char path[],const char prefix[]);
PetscErrorCode SwarmMPntStd_CoordAssignment_FaceLatticeLayout3d(DM da,PetscInt Nxp[],PetscReal perturb,PetscInt face_idx,DataBucket db);
PetscErrorCode SwarmMPntStd_CoordAssignment_FaceLatticeLayout3d_epsilon(DM da,PetscInt Nxp[],PetscReal perturb, PetscReal epsilon, PetscInt face_idx,DataBucket db);
PetscErrorCode SwarmMPntStd_CoordAssignmentFromElementList_FaceLatticeLayout3d(DM da,PetscInt Nxp[],PetscReal perturb,PetscInt face_idx,PetscInt ncells_list,PetscInt cell_list[],PetscInt *start_pidx,PetscInt *n_pidx,DataBucket db);
PetscErrorCode SwarmMPntStd_CoordAssignment_InsertWithinPlane(DataBucket db,DM dav,PetscInt Nxp2[],PetscInt region_idx,PetscReal vert_coord[]);
PetscErrorCode SwarmMPntStd_CoordAssignment_InsertFromList(DataBucket db,DM dav,PetscInt nlist,PetscReal coor[],PetscInt region_idx,PetscBool use_natural_index);

#endif
