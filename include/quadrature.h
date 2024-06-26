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
 **    filename:   quadrature.h
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

#ifndef __ptatin_quadrature_h__
#define __ptatin_quadrature_h__

#include "element_type_Q2.h"


PetscErrorCode QuadratureCreate(Quadrature *quadrature);
PetscErrorCode QuadratureDestroy(Quadrature *quadrature);
PetscErrorCode QuadratureView(Quadrature q);
PetscErrorCode QuadratureSetSize(Quadrature Q);

void QuadratureCreateGauss_2pnt_3D(PetscInt *ngp,PetscReal **_q_coor,PetscReal **_q_weight);
void QuadratureCreateGauss_3pnt_3D(PetscInt *ngp,PetscReal **_q_coor,PetscReal **_q_weight);

PetscErrorCode VolumeQuadratureCreateGaussLegendre(PetscInt dim,PetscInt ncells,PetscInt np_per_dim,Quadrature *quadrature);

PetscErrorCode SurfaceQuadratureCreate(SurfaceQuadrature *quadrature);
PetscErrorCode SurfaceQuadratureDestroy(SurfaceQuadrature *quadrature);
PetscErrorCode SurfaceQuadratureSetSize(SurfaceQuadrature Q);

PetscErrorCode SurfaceQuadratureGetQuadratureInfo(SurfaceQuadrature q,HexElementFace faceid,PetscInt *nqp,QPoint2d **qp2,QPoint3d **qp3);
PetscErrorCode SurfaceQuadratureGetFaceInfo(SurfaceQuadrature q,PetscInt *nfaces);

PetscErrorCode SurfaceQuadratureCreateGaussLegendre(PetscInt dim,PetscInt nfacets,SurfaceQuadrature *quadrature);

#endif
