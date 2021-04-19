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
 **    filename:   litho_pressure_assembly.h
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

#ifndef __litho_pressure_assembly_h__
#define __litho_pressure_assembly_h__

#include <petscdm.h>
#include "dmda_bcs.h"

struct _p_PDESolveLithoP {
  PetscInt                LP_mesh_type; /* 0-std dmda, 1-overlap, 2-nested */
  PhysCompStokes          stokes;
  SNES                    snesLP;
  PetscInt                mx,my,mz; /* global mesh size */
  DM                      daLP;
  BCList                  LP_bclist;
  Quadrature              volQ;
  //  SurfaceQuadratureEnergy surfQ[QUAD_EDGES]; /* four edges */
  Vec                     F; /* residue vector */
  Vec                     X; /* solution vector */
};

PetscErrorCode Element_FormFunction_LithoPressure(PetscScalar Re[],
                                                  PetscScalar el_coords[],
                                                  PetscScalar el_phi[],
                                                  PetscScalar gp_rho[],
                                                  PetscInt ngp,
                                                  PetscScalar gp_xi[],
                                                  PetscScalar gp_weight[]);


#endif