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
 **    filename:   rift3D_T_ctx.h
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

#ifndef __ptatinmodel_kinematic_TFV_ctx_h__
#define __ptatinmodel_kinematic_TFV_ctx_h__

typedef struct {
  PetscInt  nmaterials;
  PetscBool runmises;
  PetscReal Lx,Ly,Lz,Ox,Oy,Oz,vx,vy,vz,rho0;
  PetscReal Tbottom,Ttop,h_prod,y_prod,k,qm,Tlitho,ylab;
  PetscBool dimensional;
  PetscReal density_bar;
  PetscReal length_bar;
  PetscReal viscosity_bar;
  PetscReal velocity_bar;
  PetscReal time_bar;
  PetscReal pressure_bar;
  PetscBool use_semi_eulerian_mesh;
  PetscBool output_markers;
  GeometryObject G[100];
  PetscInt  ngo;
  PetscInt  n_phase_map,nlayers,n_phases;
  PetscBool from_map;
} Modelkinematic_TFVCtx;

typedef struct _p_BC_RotationPole *BC_RotationPole;
struct _p_BC_RotationPole {
  PetscInt  component;
  PetscReal xp,zp;
  PetscReal v0;
};

typedef struct _p_BC_RotationPoleFreeslip *BC_RotationPoleFreeslip;
struct _p_BC_RotationPoleFreeslip {
  PetscInt  normal,component;
  PetscReal xp,zp;
  PetscReal x0,x1;
  PetscReal v0;
};

typedef struct _p_BC_RotationPoleFreeslipLinear *BC_RotationPoleFreeslipLinear;
struct _p_BC_RotationPoleFreeslipLinear {
  PetscInt  normal,component;
  PetscReal xp,zp;
  PetscReal x0,x1,x2,x3;
  PetscReal v0;
};

typedef struct _p_BC_RotationPoleSpherical *BC_RotationPoleSpherical;
struct _p_BC_RotationPoleSpherical {
  PetscInt  component;
  PetscReal v0;
  PetscReal lat_pole,lon_pole,phi_0;
};

typedef struct _p_BC_RotationPoleFreeslipSpherical *BC_RotationPoleFreeslipSpherical;
struct _p_BC_RotationPoleFreeslipSpherical {
  PetscInt  normal,component;
  PetscReal x0,x1,x2,x3;
  PetscReal v0;
  PetscReal lat_pole,lon_pole,phi_0;
};
#endif
