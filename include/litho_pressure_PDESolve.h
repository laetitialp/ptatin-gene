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
 **    filename:   litho_pressure_PDESolve.h
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

#ifndef __litho_pressure_PDESolve_h__
#define __litho_pressure_PDESolve_h__

#include <petscdm.h>
#include "ptatin3d.h"
#include "quadrature.h"
#include "dmda_bcs.h"

struct _p_PDESolveLithoP {
  PetscInt                mesh_type; /* 0-std dmda, 1-overlap, 2-nested */
  PhysCompStokes          stokes;
  SNES                    snes;
  PetscInt                mx,my,mz; /* global mesh size */
  DM                      da;
  BCList                  bclist;
  Quadrature              volQ;
  Vec                     F; /* residual vector */
  Vec                     X; /* solution vector */
};

PetscErrorCode PhysCompCreate_LithoP(PDESolveLithoP *LP);
PetscErrorCode PhysCompDestroy_LithoP(PDESolveLithoP *LP);
PetscErrorCode PhysCompCreateMesh_LithoP(PDESolveLithoP LP,DM dav,PetscInt mx,PetscInt my, PetscInt mz,PetscInt mesh_generator_type);
PetscErrorCode PhysCompCreateBoundaryList_LithoP(PDESolveLithoP LP);
PetscErrorCode PhysCompCreateVolumeQuadrature_LithoP(PDESolveLithoP LP);
PetscErrorCode PhysCompNew_LithoP(DM dav,PetscInt mx,PetscInt my, PetscInt mz,PetscInt mesh_generator_type,PDESolveLithoP *LP);
PetscErrorCode pTatinGetContext_LithoP(pTatinCtx ctx,PDESolveLithoP *LP);
PetscErrorCode pTatinContextValid_LithoP(pTatinCtx ctx,PetscBool *exists);
PetscErrorCode pTatinPhysCompCreate_LithoP(pTatinCtx user);
PetscErrorCode pTatinPhysCompActivate_LithoP(pTatinCtx user,PetscBool load);
PetscErrorCode SNESSolve_LithoPressure(PDESolveLithoP LP,Mat J,Vec X, Vec F, pTatinCtx user);
PetscErrorCode ApplyLithostaticPressure_SurfQuadratureStokes_FullFace(PhysCompStokes stokes,DM da,Vec X,HexElementFace face_location[],PetscInt face_list_n);
PetscErrorCode ModelApplyTractionFromLithoPressure(pTatinCtx pctx, Vec X_stokes);
PetscErrorCode AverageVolumeQuadraturePointsToSurfaceQuadraturePointsStokes(pTatinCtx user);
PetscErrorCode ModelApplyTractionFromStokesVolumeQuadrature(pTatinCtx user, Vec X_stokes);
PetscErrorCode InterpolateIsostaticDisplacementQ1OnQ2Mesh(DM da_q1, Vec u_q1, DM da_q2, Vec u_q2);
PetscErrorCode ComputeIsostaticDisplacementQ1(PDESolveLithoP poisson_pressure, PetscInt jnode_compensation, PetscReal gravity_norm, PetscReal density_ref, Vec u_iso_q1);
PetscErrorCode IsostaticFullLagrangianAdvectionFromPoissonPressure(pTatinCtx ptatin, PetscReal density_ref, PetscInt jnode_compensation);


#endif
