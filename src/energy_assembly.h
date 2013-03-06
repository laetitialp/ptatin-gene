/*@ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 **
 **    Copyright (c) 2012, 
 **        Dave A. May [dave.may@erdw.ethz.ch]
 **        Geophysical Fluid Dynamics, 
 **        Department of Earth Sciences,
 **        ETH Zürich,
 **        Sonneggstrasse 5,
 **        CH-8092 Zurich,
 **        Switzerland
 **
 **    Project:       pTatin3d
 **    Filename:      energy_assembly.h
 **
 **
 **    pTatin3d is free software: you can redistribute it and/or modify
 **    it under the terms of the GNU General Public License as published by
 **    the Free Software Foundation, either version 3 of the License, or
 **    (at your option) any later version.
 **
 **    pTatin3d is distributed in the hope that it will be useful,
 **    but WITHOUT ANY WARRANTY; without even the implied warranty of
 **    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 **    GNU General Public License for more details.
 **
 **    You should have received a copy of the GNU General Public License
 **    along with pTatin3d.  If not, see <http://www.gnu.org/licenses/>.
 **
 **
 **    $Id$
 **
 ** ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~@*/

#ifndef __energy_assembly_h__
#define __energy_assembly_h__

#include "QPntVolCoefEnergy_def.h"

double AdvDiffResidualForceTerm_UpwindXiExact(double pecletNumber);
double AdvDiffResidualForceTerm_UpwindXiDoublyAsymptoticAssumption(double pecletNumber);
double AdvDiffResidualForceTerm_UpwindXiCriticalAssumption(double pecletNumber);
PetscErrorCode DASUPG3dComputeAverageCellSize( PetscScalar el_coords[], PetscScalar DX[]);
PetscErrorCode DASUPG3dComputeElementPecletNumber_qp( PetscScalar el_coords[],PetscScalar u[],
																										 PetscScalar kappa_el,
																										 PetscScalar *alpha);
PetscErrorCode DASUPG3dComputeElementTimestep_qp(PetscScalar el_coords[],PetscScalar u[],PetscScalar kappa_el,PetscScalar *dta,PetscScalar *dtd);
PetscErrorCode DASUPG3dComputeElementStreamlineDiffusion_qp(PetscScalar el_coords[],PetscScalar u[],
																														PetscInt nqp, PetscScalar qp_detJ[], PetscScalar qp_w[],
																														PetscScalar qp_kappa[],
																														PetscScalar *khat);
void ConstructNiSUPG_Q1_3D(PetscScalar Up[],PetscScalar kappa_hat,PetscScalar Ni[],PetscScalar GNx[NSD][NODES_PER_EL_Q1_3D],PetscScalar Ni_supg[]);
PetscErrorCode AElement_FormJacobian_T( PetscScalar Re[],PetscReal dt,PetscScalar el_coords[],
																			 PetscScalar gp_kappa[],
																			 PetscScalar el_V[],
																			 PetscInt ngp,PetscScalar gp_xi[],PetscScalar gp_weight[] );
PetscErrorCode FormJacobianEnergy(PetscReal time,Vec X,PetscReal dt,Mat *A,Mat *B,MatStructure *mstr,void *ctx);


#endif
