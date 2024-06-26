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
 **    filename:   mesh_deformation.c
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

/*
Deform mesh based on analytic functions.
Typically just used to test robustness of multigrid solver w.r.t to aspect ratio,
grid deformation etc.
*/

#include "dmda_update_coords.h"
#include "mesh_deformation.h"


/*
 Apply a bump to the surface
 Domain is set to be [-1,1]^3
 "bump" is defined in y-direction and given by the function

 gbump_amp * exp(gbump_lambda*(x*x+z*z))+1

 */
PetscErrorCode MeshDeformation_GaussianBump_YMAX(DM da,PetscReal gbump_amp,PetscReal gbump_lambda)
{
  PetscErrorCode ierr;
  PetscInt si,sj,sk,nx,ny,nz,i,j,k,MY;
  DM cda;
  Vec coord;
  DMDACoor3d ***_coord;
  PetscReal y_height,dy;
  PetscReal Gmin[3],Gmax[3];
  PetscReal xshift = 0.0,zshift = 0.0;
  PetscBool flg;

  PetscFunctionBegin;

  {
    PetscReal tmp;

    ierr = PetscOptionsGetReal(NULL,NULL,"-gbump_amp",&tmp,&flg);CHKERRQ(ierr);
    if (flg) {
      gbump_amp    = tmp;
    }

    ierr = PetscOptionsGetReal(NULL,NULL,"-gbump_lambda",&tmp,&flg);CHKERRQ(ierr);
    if (flg) {
      gbump_lambda = tmp;
    }
  }
  ierr = PetscOptionsGetReal(NULL,NULL,"-gbump_xorigin",&xshift,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-gbump_zorigin",&zshift,&flg);CHKERRQ(ierr);


  ierr = DMGetBoundingBox(da,Gmin,Gmax);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,-1.0,1.0, -1.0,1.0, -1.0,1.0);CHKERRQ(ierr);

  ierr = DMDAGetInfo(da,0,0,&MY,0, 0,0,0, 0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners( da, &si,&sj,&sk, &nx,&ny,&nz );CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinates(da,&coord);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(cda,coord,&_coord);CHKERRQ(ierr);

  /* apply bump */
  for( j=sj; j<sj+ny; j++ ) {
    for( k=sk; k<sk+nz; k++ ) {
      for( i=si; i<si+nx; i++ ) {
        PetscReal xn,zn;

        xn = _coord[k][j][i].x - xshift;
        zn = _coord[k][j][i].z - zshift;

        /* scale amplitude with depth */
        //fac_scale = (yn-(-1.0))/2.0;

        /* constant spacing */
        y_height = gbump_amp * exp(gbump_lambda*(xn*xn+zn*zn))+1.0;
        dy = (y_height+1.0)/(PetscReal)(MY-1);

        _coord[k][j][i].y = -1.0 + j * dy;
      }
    }
  }
  /* rescale */
  for( j=sj; j<sj+ny; j++ ) {
    for( k=sk; k<sk+nz; k++ ) {
      for( i=si; i<si+nx; i++ ) {
        PetscReal xn,yn,zn;

        xn = _coord[k][j][i].x;
        yn = _coord[k][j][i].y;
        zn = _coord[k][j][i].z;

        _coord[k][j][i].x = (Gmax[0]-Gmin[0])*(xn + 1.0)/2.0 + Gmin[0];
        _coord[k][j][i].y = (Gmax[1]-Gmin[1])*(yn + 1.0)/2.0 + Gmin[1];
        _coord[k][j][i].z = (Gmax[2]-Gmin[2])*(zn + 1.0)/2.0 + Gmin[2];
      }
    }
  }

  ierr = DMDAVecRestoreArray(cda,coord,&_coord);CHKERRQ(ierr);

  /* update */
  ierr = DMDAUpdateGhostedCoordinates(da);CHKERRQ(ierr);


  PetscFunctionReturn(0);
}

PetscErrorCode MeshDeformation_Sinusodial_ZMAX(DM da)
{
  PetscErrorCode ierr;
  PetscReal      amp,theta,phi,offset;
  PetscInt       si,sj,sk,nx,ny,nz,i,j,k,MZ;
  DM             cda;
  Vec            coord;
  DMDACoor3d     ***_coord;
  PetscReal      MeshMin[3],MeshMax[3];

  PetscFunctionBegin;
  amp    = 0.2;
  theta  = 0.7;
  phi    = 1.2;
  offset = 0.1;
  ierr = PetscOptionsGetReal(NULL,NULL,"-offset",&offset,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-amp",&amp,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-theta",&theta,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-phi",&phi,NULL);CHKERRQ(ierr);

  ierr = DMGetBoundingBox(da,MeshMin,MeshMax);CHKERRQ(ierr);


  ierr = DMDAGetInfo(da,0,0,0,&MZ, 0,0,0, 0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners( da, &si,&sj,&sk, &nx,&ny,&nz );CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinates(da,&coord);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(cda,coord,&_coord);CHKERRQ(ierr);

  if ((sk+nz) == MZ) {
    k = MZ - 1;
    for( j=sj; j<sj+ny; j++ ) {
      for( i=si; i<si+nx; i++ ) {
        PetscReal xn,yn;

        xn = _coord[k][j][i].x;
        yn = _coord[k][j][i].y;

        _coord[k][j][i].z = MeshMax[2] + offset + amp * sin( theta * M_PI * xn ) * cos( phi * M_PI * (xn+yn) );
      }
    }
  }
  ierr = DMDAVecRestoreArray(cda,coord,&_coord);CHKERRQ(ierr);

  /* update */
  ierr = DMDAUpdateGhostedCoordinates(da);CHKERRQ(ierr);


  PetscFunctionReturn(0);
}

PetscErrorCode MeshDeformation_ShearXY(DM da)
{
  PetscErrorCode ierr;
  PetscInt si,sj,sk,nx,ny,nz,i,j,k,MY;
  DM cda;
  Vec coord;
  DMDACoor3d ***_coord;
  PetscReal Ly,theta,y_displacement;
  PetscReal MeshMin[3],MeshMax[3];

  PetscFunctionBegin;
  y_displacement = 0.5;
  ierr = PetscOptionsGetReal(NULL,NULL,"-y_displacement",&y_displacement,NULL);CHKERRQ(ierr);


  //ierr = DMDASetUniformCoordinates(da,-1.0,1.0, -1.0,1.0, -1.0,1.0);CHKERRQ(ierr);
  ierr = DMGetBoundingBox(da,MeshMin,MeshMax);CHKERRQ(ierr);
  //Lx = (MeshMax[0] - MeshMin[0]);
  Ly = (MeshMax[1] - MeshMin[1]);
  theta = atan( y_displacement / Ly );

  ierr = DMDAGetInfo(da,0,0,&MY,0, 0,0,0, 0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners( da, &si,&sj,&sk, &nx,&ny,&nz );CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinates(da,&coord);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(cda,coord,&_coord);CHKERRQ(ierr);

  for( j=sj; j<sj+ny; j++ ) {
    for( k=sk; k<sk+nz; k++ ) {
      for( i=si; i<si+nx; i++ ) {
        PetscReal xn,yn;

        xn = _coord[k][j][i].x;
        yn = _coord[k][j][i].y;

        _coord[k][j][i].x = xn + tan(theta) * yn;
      }
    }
  }
  ierr = DMDAVecRestoreArray(cda,coord,&_coord);CHKERRQ(ierr);

  /* update */
  ierr = DMDAUpdateGhostedCoordinates(da);CHKERRQ(ierr);


  PetscFunctionReturn(0);
}

PetscErrorCode DMDASetUniformCoordinates1D(DM da,PetscInt dir,PetscReal X0,PetscReal X1)
{
  PetscErrorCode ierr;
  PetscInt si,sj,sk,nx,ny,nz,i,j,k,M,N,P,ML;
  DM cda;
  Vec coord;
  DMDACoor3d ***_coord;
  PetscReal delta;

  PetscFunctionBegin;

  ierr = DMDAGetInfo(da,0,&M,&N,&P, 0,0,0, 0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&si,&sj,&sk,&nx,&ny,&nz);CHKERRQ(ierr);

  ML = 0;
  switch (dir) {
    case 0:
      ML = M;
      break;
    case 1:
      ML = N;
      break;
    case 2:
      ML = P;
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"direction specified must be 0, 1, 2");
      break;
  }
  delta = (X1-X0)/((PetscReal)ML - 1.0);

  ierr = DMGetCoordinateDM(da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinates(da,&coord);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(cda,coord,&_coord);CHKERRQ(ierr);
  for( k=sk; k<sk+nz; k++ ) {
    for( j=sj; j<sj+ny; j++ ) {
      for( i=si; i<si+nx; i++ ) {

        switch (dir) {
          case 0:
            _coord[k][j][i].x = X0 + delta * i;
            break;
          case 1:
            _coord[k][j][i].y = X0 + delta * j;
            break;
          case 2:
            _coord[k][j][i].z = X0 + delta * k;
            break;
        }

      }
    }
  }
  ierr = DMDAVecRestoreArray(cda,coord,&_coord);CHKERRQ(ierr);

  ierr = DMDAUpdateGhostedCoordinates(da);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*
PetscInt dir     : Value of 0, 1, or 2 indicating in which diection you wish refiniment to occur along
PetscInt side    : Value of 0 or 1, indicating the direction in which you wish to refine.
                   Suppose coords in one-direction are in the range [c0,c1], then side=0 indicates refinement
                   will occur towards c0. side=1 implies refinement occurs towards c1.
PetscReal factor : Controls aggressiveness of coarsening. Larger values cause very rapid coarsening.
*/
PetscErrorCode DMDASetGraduatedCoordinates1D(DM da,PetscInt dir,PetscInt side,PetscReal factor)
{
  PetscErrorCode ierr;
  PetscInt si,sj,sk,nx,ny,nz,i,j,k,M,N,P;
  DM cda;
  Vec coord;
  DMDACoor3d ***_coord;
  PetscReal MeshMin[3],MeshMax[3],Lx[3],xp,x,f,f0,f1;


  PetscFunctionBegin;

  if ((dir < 0) || (dir > 3)) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Value \"dir\" must be one of {0,1,2}");
  }
  if ((side < 0) || (side > 1)) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Value \"side\" must be one of {0,1}");
  }
  if (factor < 0.0) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Value \"factor\" must be > 0.0");
  }

  ierr = DMDAGetInfo(da,0,&M,&N,&P, 0,0,0, 0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&si,&sj,&sk,&nx,&ny,&nz);CHKERRQ(ierr);

  ierr = DMGetBoundingBox(da,MeshMin,MeshMax);CHKERRQ(ierr);
  Lx[0] = (MeshMax[0] - MeshMin[0]);
  Lx[1] = (MeshMax[1] - MeshMin[1]);
  Lx[2] = (MeshMax[2] - MeshMin[2]);

  ierr = DMGetCoordinateDM(da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinates(da,&coord);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(cda,coord,&_coord);CHKERRQ(ierr);

  if (side == 0) {
    f0 = exp(MeshMin[dir]*factor);
    f1 = exp(MeshMax[dir]*factor) - f0;
  } else {
    x = MeshMin[dir];
    xp = MeshMax[dir] - x;
    f0 = exp(xp*factor);

    x = MeshMax[dir];
    xp = MeshMax[dir] - x;
    f1 = exp(xp*factor) - f0;
  }

  for (k=sk; k<sk+nz; k++) {
    for (j=sj; j<sj+ny; j++) {
      for (i=si; i<si+nx; i++) {
        PetscScalar pos[3];

        pos[0] = _coord[k][j][i].x;
        pos[1] = _coord[k][j][i].y;
        pos[2] = _coord[k][j][i].z;

        if (side == 0) {
          f = (exp(pos[dir]*factor) - f0)/f1;
        } else {
          x = pos[dir];
          xp = MeshMax[dir] - x;

          f = (exp(xp*factor) - f0)/f1;
        }

        pos[dir] = f;
        pos[dir] = pos[dir] * Lx[dir] + MeshMin[dir];

        _coord[k][j][i].x = pos[0];
        _coord[k][j][i].y = pos[1];
        _coord[k][j][i].z = pos[2];
      }
    }
  }
  ierr = DMDAVecRestoreArray(cda,coord,&_coord);CHKERRQ(ierr);

  ierr = DMDAUpdateGhostedCoordinates(da);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


/*
 PetscInt dir     : Value of 0, 1, or 2 indicating in which diection you wish refiniment to occur along
 PetscReal factor : Controls aggressiveness of refinement in central section of domain. Values larger than one incur refinement.
 PetscReal x0,x3 : Define the start end point of the final domain
 PetscReal x1,x2 : Define the start end point of the sector in the domain here refinement will occur
 Domain is mapped like this

 xprime = slope * (x - x_ref) + xprime_ref

 */
PetscErrorCode DMDASetCoordinatesCentralSqueeze1D(DM da,PetscInt dir,PetscReal factor,PetscReal x0,PetscReal x1,PetscReal x2,PetscReal x3)
{
  PetscErrorCode ierr;
  PetscInt si,sj,sk,nx,ny,nz,i,j,k,M,N,P;
  DM cda;
  Vec coord;
  DMDACoor3d ***_coord;
  PetscReal x0prime,x1prime,x2prime,x3prime;


  PetscFunctionBegin;

  if ((dir < 0) || (dir > 3)) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Value \"dir\" must be one of {0,1,2}");
  }
  if (factor < 1.0) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Value \"factor\" must be >= 1.0");
  }

  ierr = DMDAGetInfo(da,0,&M,&N,&P, 0,0,0, 0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&si,&sj,&sk,&nx,&ny,&nz);CHKERRQ(ierr);

  x0prime = x0;
  x1prime = x1;
  // x = [(x2 - x1)/fac] * (xprime - x1prime) + x1
  // x2prime
  //x2prime = factor + x1prime;
  x2prime = (x2-x1)*factor + x1;
  // x3prime
  x3prime = x3 - x2 + x2prime;

  ierr = DMDASetUniformCoordinates1D(da,dir,x0prime,x3prime);CHKERRQ(ierr);



  ierr = DMGetCoordinateDM(da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinates(da,&coord);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(cda,coord,&_coord);CHKERRQ(ierr);

  for (k=sk; k<sk+nz; k++) {
    for (j=sj; j<sj+ny; j++) {
      for (i=si; i<si+nx; i++) {
        PetscScalar pos[3],coord_prime,new_coord;

        pos[0] = _coord[k][j][i].x;
        pos[1] = _coord[k][j][i].y;
        pos[2] = _coord[k][j][i].z;

        coord_prime = pos[dir];

        if (pos[dir] <= x1prime) {
          new_coord = coord_prime;
        } else if (pos[dir] > x1prime && pos[dir] < x2prime) {
          //new_coord = (x2-x1) * (coord_prime - x1prime)/(factor) + x1;
          new_coord = (1.0) * (coord_prime - x1prime)/(factor) + x1;
        } else {
          new_coord = 1.0 * (coord_prime - x2prime) + x2;
        }

        pos[dir] = new_coord;

        _coord[k][j][i].x = pos[0];
        _coord[k][j][i].y = pos[1];
        _coord[k][j][i].z = pos[2];
      }
    }
  }
  ierr = DMDAVecRestoreArray(cda,coord,&_coord);CHKERRQ(ierr);

  ierr = DMDAUpdateGhostedCoordinates(da);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

