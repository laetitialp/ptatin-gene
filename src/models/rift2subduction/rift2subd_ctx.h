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

#ifndef __ptatin3d_rift2subd_ctx_h__
#define __ptatin3d_rift2subd_ctx_h__

/* define user model */
typedef struct {
  PetscReal length_bar, viscosity_bar, velocity_bar, time_bar, pressure_bar, density_bar, acceleration_bar;
  PetscReal Lx, Ly, Lz, Ox, Oy, Oz;
  PetscReal y_continent[3];
  PetscReal BC_time[4];
  PetscReal wz,Kero;
  PetscReal normV, angle_v, v_extension, vy;
  PetscReal Ttop, Tbottom;
  PetscBool output_markers,is_2D,open_base;
  PetscBool freeslip_z,notches,use_v_dot_n,straight_wz,one_notch;
  PetscBool litho_plitho_z,full_face_plitho_z,full_face_plithoKMAX,litho_plithoKMAX;
  PetscInt  n_phases;
  
} ModelRiftSubdCtx;

typedef struct _p_BC_Lithosphere *BC_Lithosphere;
struct _p_BC_Lithosphere {
  PetscReal y_lab,v;
};

#endif