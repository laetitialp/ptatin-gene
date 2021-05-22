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
 **    filename:   ocean_ctx.h
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

#ifndef __ptatinmodel_ocean_ctx_h__
#define __ptatinmodel_ocean_ctx_h__

typedef struct {
  PetscInt  n_phases;
  PetscReal Ox,Lx,Oy,Ly,Oz,Lz;
  PetscReal ocean_floor,moho;
  PetscReal v_spreading,v_top,v_bot;
  PetscReal T_top,T_bot;
  PetscReal w_offset,w_width,w_length;
  PetscReal length_bar,viscosity_bar,velocity_bar;
  PetscReal time_bar,pressure_bar,density_bar,acceleration_bar;
  PetscBool output_markers;
} ModelOceanCtx;

#endif
