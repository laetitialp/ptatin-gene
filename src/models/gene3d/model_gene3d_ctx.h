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

typedef enum { MESH_EULERIAN=0, MESH_ALE } GeneMeshType;

struct _p_ScalingCtx {
  /* Scaling values */
  PetscReal length_bar;
  PetscReal viscosity_bar;
  PetscReal velocity_bar;
  PetscReal time_bar;
  PetscReal pressure_bar;
  PetscReal density_bar;
  PetscReal acceleration_bar;
  PetscReal cm_per_year2m_per_sec;
  PetscReal Myr2sec;
};
typedef struct _p_ScalingCtx *ScalingCtx;

typedef struct {
  /* bounding box */
  PetscReal L[3],O[3];
  /* regions */
  PetscInt  n_regions;
  PetscInt  *regions_table;
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
  ScalingCtx scale;
  /* poisson pressure */
  Mat       poisson_Jacobian;
  PetscReal surface_pressure;
  PetscBool poisson_pressure_active;
  /* bcs */
  PetscInt  prev_step;
  PetscInt  bc_nfaces;
  PetscInt  *bc_tag_table;
  PetscBool bc_debug;
  /* General Navier slip */
  PetscReal epsilon_s[6],t1_hat[3],n_hat[3];
  /* Meshes */
  Mesh *mesh_facets;
  GeneMeshType mesh_type;
} ModelGENE3DCtx;

typedef struct {
  te_expr     *expression;
  PetscScalar *x,*y,*z,*t,*p;
  ScalingCtx  scale;
} ExpressionCtx;

typedef struct {
  PetscInt       nen,m[3];
  const PetscInt *elnidx;
  const PetscInt *elnidx_q2;
  PetscReal      *pressure;
  PetscScalar    *coor;
  ExpressionCtx  *expr_ctx;
} NeumannCtx;

typedef struct {
  PetscReal epsilon_s[6];
  PetscReal mcal_H[6];
  PetscReal n_hat[3];
  PetscReal t1_hat[3];
} GenNavierSlipCtx;

#endif
