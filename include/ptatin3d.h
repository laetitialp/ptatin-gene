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
 **    filename:   ptatin3d.h
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


#ifndef __ptatin3d_h__
#define __ptatin3d_h__

#include "petsc.h"
#include "petscvec.h"
#include "petscdm.h"


extern PetscClassId PTATIN_CLASSID;

typedef struct _p_pTatinCtx *pTatinCtx;
typedef struct _p_pTatinModel *pTatinModel;
typedef struct _p_PhysCompStokes *PhysCompStokes;
typedef struct _p_PhysCompEnergy *PhysCompEnergy;
typedef struct _p_Quadrature *Quadrature;
typedef struct _p_SurfaceQuadrature *SurfaceQuadrature;
typedef struct _p_PDESolveLithoP *PDESolveLithoP;

typedef enum { LINE_QUAD=0,SURFACE_QUAD,VOLUME_QUAD } QuadratureType;

typedef struct _p_RheologyConstants RheologyConstants;


/*
#include "dmda_bcs.h"
#include "dmda_checkpoint.h"
#include "dmda_compare.h"
#include "dmda_duplicate.h"
#include "dmda_element_q2p1.h"
#include "dmda_project_coords.h"
#include "dmda_redundant.h"
#include "dmda_remesh.h"
#include "dmda_update_coords.h"
#include "dmda_view_petscvtk.h"

#include "data_bucket.h"
#include "data_exchanger.h"

#include "ptatin3d_defs.h"
#include "ptatin_utils.h"
#include "ptatin3d_stokes.h"
#include "ptatin_models.h"
#include "rheology.h"
*/

#include "data_bucket.h"
#include "data_exchanger.h"
//#include "rheology.h"
#include "material_point_load.h"
#include "material_point_utils.h"


PetscErrorCode pTatin3d_PhysCompStokesCreate(pTatinCtx user);
PetscErrorCode pTatin3d_ModelOutput_VelocityPressure_Stokes(pTatinCtx ctx,Vec X,const char prefix[]);
PetscErrorCode pTatin3d_ModelOutputLite_Velocity_Stokes(pTatinCtx ctx,Vec X,const char prefix[]);
PetscErrorCode pTatin3d_ModelOutputPetscVec_VelocityPressure_Stokes(pTatinCtx ctx,Vec X,const char prefix[]);

PetscErrorCode pTatin3dCreateMaterialPoints(pTatinCtx ctx,DM dav);
PetscErrorCode MaterialPointCoordinateSetUp(pTatinCtx ctx,DM da);
PetscErrorCode pTatin3d_ModelOutput_MPntStd(pTatinCtx ctx,const char prefix[]);

PetscErrorCode pTatin3dCreateContext(pTatinCtx *ctx);
PetscErrorCode pTatin3dDestroyContext(pTatinCtx *ctx);
PetscErrorCode pTatin3dSetFromOptions(pTatinCtx ctx);
PetscErrorCode pTatinModelLoad(pTatinCtx ctx);

PetscErrorCode pTatinGetTime(pTatinCtx ctx,PetscReal *time);
PetscErrorCode pTatinGetTimestep(pTatinCtx ctx,PetscReal *dt);
PetscErrorCode pTatinGetMaterialPoints(pTatinCtx ctx,DataBucket *db,DataEx *de);
PetscErrorCode pTatinGetModel(pTatinCtx ctx,pTatinModel *m);
PetscErrorCode pTatinGetRheology(pTatinCtx ctx,RheologyConstants **r);
PetscErrorCode pTatinGetStokesContext(pTatinCtx ctx,PhysCompStokes *s);
PetscErrorCode pTatinGetMaterialConstants(pTatinCtx ctx,DataBucket *db);

PetscErrorCode pTatin3dCheckpoint(pTatinCtx ctx,Vec X,const char prefix[]);

PetscErrorCode pTatinCtxGetModelData(pTatinCtx ctx,const char name[],void **data);
PetscErrorCode pTatinCtxAttachModelData(pTatinCtx ctx,const char name[],void *data);
PetscErrorCode pTatinCtxGetModelDataPetscObject(pTatinCtx ctx,const char name[],PetscObject *data);
PetscErrorCode pTatinCtxAttachModelDataPetscObject(pTatinCtx ctx,const char name[],PetscObject data);

PetscErrorCode pTatin3dCheckpointManager(pTatinCtx ctx,Vec X);
PetscErrorCode pTatin3dCheckpointManagerFV(pTatinCtx ctx,Vec Xs);
PetscErrorCode DMCoarsenHierarchy2_DA(DM da,PetscInt nlevels,DM dac[]);

PetscErrorCode pTatin_SetTimestep(pTatinCtx ctx,const char timescale_name[],PetscReal dt_trial);

PetscErrorCode pTatinCtxCheckpointWrite(pTatinCtx ctx,const char path[],const char prefix[],
                                        DM dms,DM dme,
                                        PetscInt nfields,const char *dmnames[],DM dmlist[],
                                        Vec Xs,Vec Xe,const char *fieldnames[],Vec veclist[]);
PetscErrorCode pTatinCtxCheckpointWriteFV(pTatinCtx ctx,const char path[],const char prefix[],
                                          DM dms,DM dme,
                                          PetscInt nfields,const char *dmnames[],DM dmlist[],
                                          Vec Xs,Vec Xe,const char *fieldnames[],Vec veclist[]);
PetscErrorCode pTatin3dLoadContext_FromFile(pTatinCtx *_ctx);
PetscErrorCode pTatin3dLoadState_FromFile(pTatinCtx ctx,DM dmstokes,DM dmenergy,Vec Xs,Vec Xt);
PetscErrorCode pTatin3dLoadState_FromFile_FV(pTatinCtx ctx,DM dmstokes,DM dmenergy,Vec Xs,Vec Xt);
PetscErrorCode pTatin3d_PhysCompStokesLoad_FromFile(pTatinCtx ctx);
PetscErrorCode pTatin3dLoadMaterialPoints_FromFile(pTatinCtx ctx,DM dmv);
PetscErrorCode pTatinPhysCompActivate_Energy_FromFile(pTatinCtx ctx);

#endif
