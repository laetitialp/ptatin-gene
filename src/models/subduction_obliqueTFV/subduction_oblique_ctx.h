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
 **    filename:   subduction_oblique_ctx.h
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

#ifndef __ptatin3d_subduction_oblique_ctx_h__
#define __ptatin3d_subduction_oblique_ctx_h__

/* define user model */
typedef struct {
  PetscReal length_bar, viscosity_bar, velocity_bar, time_bar, pressure_bar, density_bar, acceleration_bar;
  PetscReal Lx, Ly, Lz, Ox, Oy, Oz;
  PetscReal y_continent[3], y_ocean[4];
  PetscReal y0, alpha_subd, theta_subd, wz;
  PetscReal normV, angle_v;
  PetscReal Ttop,Tbottom;
  PetscBool oblique_IC, oblique_BC, output_markers;
  PetscInt  n_phases;
  PetscBool subduction_temp_ic_steadystate_analytics;
  PetscReal qm,k,h_prod,y_prod,Tlitho; /* Initial continental geotherm params */
  PetscReal age; /* Initial oceanic geotherm params */
  
} ModelSubductionObliqueCtx;

typedef struct _p_BC_Lithosphere *BC_Lithosphere;
struct _p_BC_Lithosphere {
  PetscReal y_lab,v;
};

#endif
