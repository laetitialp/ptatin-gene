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
 **    filename:   model_gene3d_ctx.h
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


#ifndef __ptatinmodel_GENE3D_ctx_h__
#define __ptatinmodel_GENE3D_ctx_h__

#include "pswarm.h"

typedef enum { GENEBC_FreeSlip=0, GENEBC_NoSlip, GENEBC_FreeSlipFreeSurface, GENEBC_NoSlipFreeSurface } GENE3DBC;
typedef enum { GENE_LayeredCake=0, GENE_ExtrudeFromMap, GENE_ReadFromCAD} GENE3DINIGEOM;
enum {LAYER_MAX = 100};

typedef struct {
  /* bounding box */
  PetscReal L[3],O[3];
  /* regions */
  PetscInt  n_regions;
  PetscInt  *regions_table;
  char      mesh_file[PETSC_MAX_PATH_LEN],region_file[PETSC_MAX_PATH_LEN];
  /* viscosity cutoff */
  PetscReal eta_max,eta_min;
  PetscBool eta_cutoff;
  /* surface processes */
  PetscBool surface_diffusion;
  PetscReal diffusivity_spm;
  /* passive markers */
  PetscBool passive_markers;
  PSwarm    pswarm;
  /* Output */
  PetscBool output_markers;
  /* Scaling values */
  PetscReal length_bar,viscosity_bar,velocity_bar;
  PetscReal time_bar,pressure_bar,density_bar,acceleration_bar;
  PetscReal cm_per_year2m_per_sec,Myr2sec;
  /* poisson pressure */
  PetscInt  prev_step;
  Mat       poisson_Jacobian;
  PetscReal surface_pressure;
  PetscBool poisson_pressure_active;
  /* bcs */
  PetscInt          bc_nfaces;
  PetscInt          *bc_tag_table;
  SurfaceConstraint *bc_sc;
  /* General Navier slip */
  PetscReal epsilon_s[6],t1_hat[3],n_hat[3];

  PetscBool u_dot_n_flow;
  PetscReal u_bc[6*3];
  GENE3DBC  boundary_conditon_type; /* [ 0 free slip | 1 no slip | 2 free surface + free slip | 3 free surface + no slip ] */
  GENE3DINIGEOM  initial_geom;
} ModelGENE3DCtx;

typedef struct {
  PetscReal epsilon_s[6];
  PetscReal mcal_H[6];
  PetscReal n_hat[3];
  PetscReal t1_hat[3];
} GenNavierSlipCtx;

typedef struct {
  te_expr     *expression;
  PetscScalar *x,*y,*z,*t;
  PetscReal   length_scale;
} ExpressionCtx;

PetscErrorCode ModelSetMarkerIndexLayeredCake_GENE3D(pTatinCtx c,void *ctx);
PetscErrorCode ModelSetMarkerIndexFromMap_GENE3D(pTatinCtx c,void *ctx);
PetscErrorCode ModelSetInitialStokesVariableOnMarker_GENE3D(pTatinCtx c,void *ctx);

#endif
