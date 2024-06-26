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
 **    filename:   model_utils.c
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

#include "ptatin3d.h"
#include "ptatin3d_defs.h"
#include "private/ptatin_impl.h"
#include "ptatin3d_stokes.h"
#include "ptatin3d_energy.h"
#include "mesh_quality_metrics.h"
#include "mesh_update.h"
#include "mesh_deformation.h"

#include "dmda_bcs.h"
#include "dmda_iterator.h"
#include "dmda_redundant.h"
#include "dmda_remesh.h"
#include "dmda_update_coords.h"
#include "dmda_duplicate.h"
#include "dmdae.h"
#include "dmda_element_q2p1.h"
#include "dmda_element_q1.h"
#include "element_type_Q2.h"
#include "element_utils_q2.h"
#include "element_utils_q1.h"

#include "data_bucket.h"
#include "MPntStd_def.h"
#include "MPntPStokes_def.h"
#include "MPntPStokesPl_def.h"
#include "MPntPEnergy_def.h"
#include "ptatin_std_dirichlet_boundary_conditions.h"
#include <ptatin3d_energyfv.h>
#include <ptatin3d_energyfv_impl.h>

#include "parse.h"
#include "point_in_tetra.h"

#include "model_utils.h"

PetscErrorCode MPntGetField_global_element_nInJnKindex(DM da, MPntStd *material_point, PetscInt *nI, PetscInt *nJ, PetscInt *nK)
{
  PetscInt       li, lj, lk,lmx, lmy, lmz, si, sj, sk;
  int            localeid;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  MPntStdGetField_local_element_index(material_point,&localeid);
  ierr = DMDAGetCornersElementQ2(da,&si,&sj,&sk,&lmx,&lmy,&lmz);CHKERRQ(ierr);

  si = si/2;
  sj = sj/2;
  sk = sk/2;
  //  lmx -= si;
  //  lmy -= sj;
  //  lmz -= sk;
  //global/localrank = mx*my*k + mx*j + i;
  lk = (PetscInt)localeid/(lmx*lmy);
  lj = (PetscInt)(localeid - lk*(lmx*lmy))/lmx;
  li = localeid - lk*(lmx*lmy) - lj*lmx;

  if ( (li < 0) || (li>=lmx) ) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"nI computed incorrectly"); }
  if ( (lj < 0) || (lj>=lmy) ) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"nJ computed incorrectly"); }
  if ( (lk < 0) || (lk>=lmz) ) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"nK computed incorrectly"); }
  //printf("li,lj,lk %d %d %d \n", li,lj,lk );

  if (nK) { *nK = lk + sk; }
  if (nJ) { *nJ = lj + sj; }
  if (nI) { *nI = li + si; }
  PetscFunctionReturn(0);
}

PetscErrorCode DMDAConvertLocalElementIndex2GlobalnInJnK(DM da,PetscInt localeid,PetscInt *nI,PetscInt *nJ,PetscInt *nK)
{
  PetscInt       li,lj,lk,lmx,lmy,lmz,si,sj,sk;
  PetscErrorCode ierr;


  PetscFunctionBegin;
  ierr = DMDAGetCornersElementQ2(da,&si,&sj,&sk,&lmx,&lmy,&lmz);CHKERRQ(ierr);

  si = si/2;
  sj = sj/2;
  sk = sk/2;

  //global/localrank = mx*my*k + mx*j + i;
  lk = (PetscInt)localeid/(lmx*lmy);
  lj = (PetscInt)(localeid - lk*(lmx*lmy))/lmx;
  li = localeid - lk*(lmx*lmy) - lj*lmx;

  if ( (li < 0) || (li >= lmx) ) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"nI computed incorrectly"); }
  if ( (lj < 0) || (lj >= lmy) ) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"nJ computed incorrectly"); }
  if ( (lk < 0) || (lk >= lmz) ) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"nK computed incorrectly"); }
  if (nK) { *nK = lk + sk; }
  if (nJ) { *nJ = lj + sj; }
  if (nI) { *nI = li + si; }

  PetscFunctionReturn(0);
}

PetscErrorCode DMDAConvertLocalElementIndex2LocalnInJnK(DM da,PetscInt localeid,PetscInt *nI,PetscInt *nJ,PetscInt *nK)
{
  PetscInt       li,lj,lk,lmx,lmy,lmz,si,sj,sk;
  PetscErrorCode ierr;


  PetscFunctionBegin;
  ierr = DMDAGetCornersElementQ2(da,&si,&sj,&sk,&lmx,&lmy,&lmz);CHKERRQ(ierr);

  //global/localrank = mx*my*k + mx*j + i;
  lk = (PetscInt)localeid/(lmx*lmy);
  lj = (PetscInt)(localeid - lk*(lmx*lmy))/lmx;
  li = localeid - lk*(lmx*lmy) - lj*lmx;

  if ( (li < 0) || (li >= lmx) ) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"nI computed incorrectly"); }
  if ( (lj < 0) || (lj >= lmy) ) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"nJ computed incorrectly"); }
  if ( (lk < 0) || (lk >= lmz) ) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"nK computed incorrectly"); }
  if (nK) { *nK = lk; }
  if (nJ) { *nJ = lj; }
  if (nI) { *nI = li; }

  PetscFunctionReturn(0);
}

PetscErrorCode DMDAConvertLocalNodeIndex2GlobalnInJnK(DM da,PetscInt localnid,PetscInt *nI,PetscInt *nJ,PetscInt *nK)
{
  PetscInt       li,lj,lk,lnx,lny,lnz,si,sj,sk;
  PetscErrorCode ierr;


  PetscFunctionBegin;
  ierr = DMDAGetCorners(da,&si,&sj,&sk,&lnx,&lny,&lnz);CHKERRQ(ierr);

  //global/localrank = mx*my*k + mx*j + i;
  lk = (PetscInt)localnid/(lnx*lny);
  lj = (PetscInt)(localnid - lk*(lnx*lny))/lnx;
  li = localnid - lk*(lnx*lny) - lj*lnx;

  if ( (li < 0) || (li >= lnx) ) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"nI computed incorrectly"); }
  if ( (lj < 0) || (lj >= lny) ) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"nJ computed incorrectly"); }
  if ( (lk < 0) || (lk >= lnz) ) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"nK computed incorrectly"); }
  if (nK) { *nK = lk + sk; }
  if (nJ) { *nJ = lj + sj; }
  if (nI) { *nI = li + si; }

  PetscFunctionReturn(0);
}

PetscErrorCode DMDAConvertLocalGhostNodeIndex2GlobalnInJnK(DM da,PetscInt localnid,PetscInt *nI,PetscInt *nJ,PetscInt *nK)
{
  PetscInt       li,lj,lk,lnx,lny,lnz,si,sj,sk;
  PetscErrorCode ierr;


  PetscFunctionBegin;
  ierr = DMDAGetGhostCorners(da,&si,&sj,&sk,&lnx,&lny,&lnz);CHKERRQ(ierr);

  //global/localrank = mx*my*k + mx*j + i;
  lk = (PetscInt)localnid/(lnx*lny);
  lj = (PetscInt)(localnid - lk*(lnx*lny))/lnx;
  li = localnid - lk*(lnx*lny) - lj*lnx;

  if ( (li < 0) || (li >= lnx) ) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"nI computed incorrectly"); }
  if ( (lj < 0) || (lj >= lny) ) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"nJ computed incorrectly"); }
  if ( (lk < 0) || (lk >= lnz) ) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"nK computed incorrectly"); }
  if (nK) { *nK = lk + sk; }
  if (nJ) { *nJ = lj + sj; }
  if (nI) { *nI = li + si; }

  PetscFunctionReturn(0);
}

PetscErrorCode pTatinModelGetOptionReal(const char option[],PetscReal *val,
    const char error[],
    const char default_opt[],
    PetscBool essential)
{
  PetscBool flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsGetReal(NULL,NULL,option,val,&flg);CHKERRQ(ierr);
  if (essential) {
    if (!flg) {
      if (!default_opt) {
        SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"ModelOptionMissing(%s): \n\t\t%s ",option,error);
      } else {
        SETERRQ3(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"ModelOptionMissing(%s): \n\t\t%s : Suggested default values %s ",option,error,default_opt);
      }
    }
  }
  PetscFunctionReturn(0);
}

/* Absolute value for double in C - it's called fabs(double a) or PetscAbsReal(PetscRea a) */
PetscReal absolute(PetscReal a)
{
  if (a < 0) {
    return(-1.0 * a);
  } else {
    return(a);
  }
}

/*
   Remove the linear trend
   Assume a regular spacing and chose an appropriate coordinate system such
   sum x_i = 0
   */
PetscErrorCode detrend(PetscReal array[],PetscInt n)
{
  PetscReal x,y,a,b;
  PetscReal sy = 0.0,sxy = 0.0,sxx = 0.0;
  PetscInt  i;

  for (i=0, x=(-n/2.0+0.5); i<n; i++, x+=1.0) {
    y = array[i];
    sy += y;
    sxy += x * y;
    sxx += x * x;
  }
  a = sxy/sxx;
  b = sy/(PetscReal)n;

  for (i=0, x=(-n/2.0+0.5); i<n; i++, x+=1.0) {
    array[i] -= (a*x+b);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode rednoise(PetscReal rnoise[],PetscInt n,PetscInt seed)
{
  PetscInt       i;
  PetscReal      maxi;

  PetscFunctionBegin;

  srand(seed);
  rnoise[0] = 2.0 * rand()/(RAND_MAX+1.0) - 1.0;

  for (i=1; i<n; i++) {
    rnoise[i] = rnoise[i-1] + 2.0 * rand()/(RAND_MAX+1.0) - 1.0;
  }

  detrend(rnoise,n);

  maxi = 0.0;
  for (i=1; i<n; i++) {
    if (PetscAbsReal(rnoise[i]) > maxi) {
      maxi = PetscAbsReal(rnoise[i]);
    }
  }
  for (i=1; i<n; i++) {
    rnoise[i] /= maxi;
  }

  PetscFunctionReturn(0);
}

PetscBool DMDAVecTraverse_InitialThermalField3D(PetscScalar pos[],PetscScalar *val,void *ctx)
{
  DMDA_thermalfield_init_params *thermalparams;
  PetscInt i,klay,nlt;
  PetscReal y,yldep,dtemp;

  thermalparams = (DMDA_thermalfield_init_params*)ctx;
  y   = pos[1] * thermalparams->lscale;
  nlt = thermalparams->nlayers;
  //  Which layer contains the current node?
  klay = 0;

  for (i=0; i <= nlt-1; i++) {
    if (y <= thermalparams->ytop[i] && y >= thermalparams->ytop[i+1]) {
      klay = i;
      break;
    }
  }
  yldep = thermalparams->ytop[klay] - y;
  dtemp = thermalparams->hp[klay] * yldep * (thermalparams->thick[klay]-yldep/2.0e0) + thermalparams->qbase[klay]*yldep;
  dtemp = dtemp / thermalparams->cond[klay];
  *val  = thermalparams->ttop[klay] + dtemp;
  /*    if (pos[0]==0&&pos[2]==0) {
        printf("y=%1.4e, T=%1.4e \n",y,*val);
        printf("ytop0=%1.4e\n",thermalparams->ytop[0]);
        printf("ytop1=%1.4e\n",thermalparams->ytop[1]);
        printf("ytop1=%1.4e\n",thermalparams->ytop[2]);
        printf("klay=%d\n",klay);
        }
  */

  return PETSC_TRUE;
}

PetscErrorCode DMDAComputeMeshVolume(DM dm,PetscReal *value)
{
  DM              cda;
  Vec             gcoords;
  PetscReal       *LA_gcoords;
  PetscInt        nel,nen,e,p;
  const PetscInt  *el_nidx;
  PetscReal       el_coords[3*Q2_NODES_PER_EL_3D];
  PetscInt        ngp;
  PetscReal       WEIGHT[NQP],XI[NQP][3],NI[NQP][NPE],GNI[NQP][3][NPE];
  PetscReal       _value,detJ[NQP],dNudx[NQP][NPE],dNudy[NQP][NPE],dNudz[NQP][NPE];
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  /* setup quadrature */
  ngp = 27;
  P3D_prepare_elementQ2(ngp,WEIGHT,XI,NI,GNI);

  /* setup for coords */
  ierr = DMGetCoordinateDM(dm,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm,&gcoords);CHKERRQ(ierr);
  ierr = VecGetArray(gcoords,&LA_gcoords);CHKERRQ(ierr);
  ierr = DMDAGetElements_pTatinQ2P1(dm,&nel,&nen,&el_nidx);CHKERRQ(ierr);

  _value = 0.0;
  for (e=0; e<nel; e++) {
    PetscReal el_vol;

    ierr = DMDAGetElementCoordinatesQ2_3D(el_coords,(PetscInt*)&el_nidx[nen*e],LA_gcoords);CHKERRQ(ierr);
    P3D_evaluate_geometry_elementQ2(ngp,el_coords,GNI, detJ,dNudx,dNudy,dNudz);

    el_vol = 0.0;
    for (p=0; p<ngp; p++) {
      el_vol = el_vol + 1.0 * WEIGHT[p] * detJ[p];
    }
    _value += el_vol;
  }
  ierr = VecRestoreArray(gcoords,&LA_gcoords);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&_value,value,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode pTatin3d_DefineVelocityMeshQuasi2D(pTatinCtx c)
{
  c->mz = 1;
  PetscFunctionReturn(0);
}

PetscErrorCode pTatin3d_DefineVelocityMeshGeometryQuasi2D(pTatinCtx c)
{
  PhysCompStokes stokes;
  DM             stokes_pack,dav,dap;
  PetscReal      Lz,min_dl[3],max_dl[3];
  PetscBool      geom_max;
  PetscErrorCode ierr;

  ierr = pTatinGetStokesContext(c,&stokes);CHKERRQ(ierr);
  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);

  /* determine min/max dx,dy,dz for mesh */
  ierr = DMDAComputeQ2ElementBoundingBox(dav,min_dl,max_dl);CHKERRQ(ierr);

  geom_max = PETSC_FALSE;
  Lz = 1.0e32;
  Lz = PetscMin(Lz,min_dl[0]);
  Lz = PetscMin(Lz,min_dl[1]);

  PetscOptionsGetBool(NULL,NULL,"-ptatin_geometry_quasi_2d_max",&geom_max,NULL);
  if (geom_max) {
    Lz = 1.0e-32;
    Lz = PetscMax(Lz,max_dl[0]);
    Lz = PetscMax(Lz,max_dl[1]);
  }

  if (geom_max) {
    PetscPrintf(PETSC_COMM_WORLD,"[[pTatin3d_DefineVelocityMeshGeometryQuasi2D]] Using Lz = %1.4e from max(dx,dy) \n",Lz );
  } else {
    PetscPrintf(PETSC_COMM_WORLD,"[[pTatin3d_DefineVelocityMeshGeometryQuasi2D]] Using Lz = %1.4e from min(dx,dy) \n",Lz );
  }

  ierr = DMDASetUniformCoordinates1D(dav,2,0.0,Lz);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode DMDAComputeQ2ElementBoundingBox(DM dm,PetscReal gmin[],PetscReal gmax[])
{
  DM              cda;
  Vec             gcoords;
  PetscReal       *LA_gcoords;
  PetscInt        nel,nen,e;
  const PetscInt  *el_nidx;
  PetscReal       el_coords[3*Q2_NODES_PER_EL_3D];
  PetscReal       dx,dy,dz,dl_min[3],dl_max[3];
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  /* setup for coords */
  ierr = DMGetCoordinateDM(dm,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm,&gcoords);CHKERRQ(ierr);
  ierr = VecGetArray(gcoords,&LA_gcoords);CHKERRQ(ierr);

  ierr = DMDAGetElements_pTatinQ2P1(dm,&nel,&nen,&el_nidx);CHKERRQ(ierr);

  dl_min[0] = dl_min[1] = dl_min[2] = PETSC_MAX_REAL;
  dl_max[0] = dl_max[1] = dl_max[2] = PETSC_MIN_REAL;

  for (e=0;e<nel;e++) {
    ierr = DMDAGetElementCoordinatesQ2_3D(el_coords,(PetscInt*)&el_nidx[nen*e],LA_gcoords);CHKERRQ(ierr);

    dx = fabs( el_coords[3*Q2_FACE_NODE_EAST +0] - el_coords[3*Q2_FACE_NODE_WEST +0]  );
    dy = fabs( el_coords[3*Q2_FACE_NODE_NORTH+1] - el_coords[3*Q2_FACE_NODE_SOUTH+1] );
    dz = fabs( el_coords[3*Q2_FACE_NODE_FRONT+2] - el_coords[3*Q2_FACE_NODE_BACK +2]  );

    if (dx < dl_min[0]) { dl_min[0] = dx; }
    if (dy < dl_min[1]) { dl_min[1] = dy; }
    if (dz < dl_min[2]) { dl_min[2] = dz; }

    if (dx > dl_max[0]) { dl_max[0] = dx; }
    if (dy > dl_max[1]) { dl_max[1] = dy; }
    if (dz > dl_max[2]) { dl_max[2] = dz; }

  }
  ierr = VecRestoreArray(gcoords,&LA_gcoords);CHKERRQ(ierr);

  ierr = MPI_Allreduce(dl_min,gmin,3,MPIU_REAL,MPIU_MIN,PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
  ierr = MPI_Allreduce(dl_max,gmax,3,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode DMDAComputeQ2LocalBoundingBox(DM dm,PetscReal gmin[],PetscReal gmax[])
{
  DM              cda;
  Vec             gcoords;
  PetscReal       *LA_gcoords;
  PetscInt        nel,nen,e,k;
  const PetscInt  *el_nidx;
  PetscReal       el_coords[3*Q2_NODES_PER_EL_3D];
  PetscReal       xp,yp,zp,min[3],max[3];
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  /* setup for coords */
  ierr = DMGetCoordinateDM(dm,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm,&gcoords);CHKERRQ(ierr);
  ierr = VecGetArray(gcoords,&LA_gcoords);CHKERRQ(ierr);

  ierr = DMDAGetElements_pTatinQ2P1(dm,&nel,&nen,&el_nidx);CHKERRQ(ierr);

  min[0] = min[1] = min[2] = PETSC_MAX_REAL;
  max[0] = max[1] = max[2] = PETSC_MIN_REAL;

  for (e=0;e<nel;e++) {
    ierr = DMDAGetElementCoordinatesQ2_3D(el_coords,(PetscInt*)&el_nidx[nen*e],LA_gcoords);CHKERRQ(ierr);

    for (k=0; k<nen; k++) {
      xp = el_coords[3*k];
      yp = el_coords[3*k+1];
      zp = el_coords[3*k+2];

      if (xp < min[0]) { min[0] = xp; }
      if (yp < min[1]) { min[1] = yp; }
      if (zp < min[2]) { min[2] = zp; }

      if (xp > max[0]) { max[0] = xp; }
      if (yp > max[1]) { max[1] = yp; }
      if (zp > max[2]) { max[2] = zp; }
    }

  }
  ierr = VecRestoreArray(gcoords,&LA_gcoords);CHKERRQ(ierr);

  gmin[0] = min[0];
  gmin[1] = min[1];
  gmin[2] = min[2];

  gmax[0] = max[0];
  gmax[1] = max[1];
  gmax[2] = max[2];

  PetscFunctionReturn(0);
}

/*
   This should only ever be used for debugging.
   We scatter the vector created from a DMDA into the natural i+j*nx+k*nx*ny ordering,
   then we scatter this i,j,k ordered vector onto rank 0 and write the contents out in ascii.
   */
PetscErrorCode DMDAFieldViewAscii(DM dm,Vec field,const char filename[])
{
  PetscErrorCode ierr;
  Vec natural_field,natural_field_red;
  VecScatter ctx;
  FILE *fp;
  PetscInt M,N,P,dofs;
  const char *oname = NULL;
  MPI_Comm comm;
  PetscMPIInt rank;
  PetscInt i,n;
  PetscScalar *LA_field;


  PetscFunctionBegin;

  ierr = DMDACreateNaturalVector(dm,&natural_field);CHKERRQ(ierr);
  ierr = DMDAGlobalToNaturalBegin(dm,field,INSERT_VALUES,natural_field);CHKERRQ(ierr);
  ierr = DMDAGlobalToNaturalEnd(dm,field,INSERT_VALUES,natural_field);CHKERRQ(ierr);

  ierr = VecScatterCreateToZero(natural_field,&ctx,&natural_field_red);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,natural_field,natural_field_red,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,natural_field,natural_field_red,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ctx);CHKERRQ(ierr);

  ierr = VecDestroy(&natural_field);CHKERRQ(ierr);

  /*
# DMDAFieldViewAscii:
# DMDA Vec (name)
# M N P x y z
# dofs x
*/
  ierr = DMDAGetInfo(dm,0,&M,&N,&P, 0,0,0, &dofs,0, 0,0,0, 0);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)field,&oname);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  if (rank == 0) {

    PetscFOpen(PETSC_COMM_SELF,filename,"w",&fp);
    if (fp == NULL) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Unable to open file %s on rank 0",filename); }

    PetscFPrintf(PETSC_COMM_SELF,fp,"# DMDAFieldViewAscii\n");
    if (oname) {
      PetscFPrintf(PETSC_COMM_SELF,fp,"# DMDA Vec %s\n",oname);
    } else {
      PetscFPrintf(PETSC_COMM_SELF,fp,"# DMDA Vec\n");
    }
    PetscFPrintf(PETSC_COMM_SELF,fp,"# M N P %D %D %D\n",M,N,P);
    PetscFPrintf(PETSC_COMM_SELF,fp,"# dofs %D\n",dofs);
    ierr = VecGetSize(natural_field_red,&n);CHKERRQ(ierr);
    ierr = VecGetArray(natural_field_red,&LA_field);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      PetscFPrintf(PETSC_COMM_SELF,fp,"%1.6e\n",LA_field[i]);
    }
    ierr = VecRestoreArray(natural_field_red,&LA_field);CHKERRQ(ierr);

    PetscFClose(PETSC_COMM_SELF,fp);
  }
  ierr = VecDestroy(&natural_field_red);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode ModelUtilsComputeAiryIsostaticHeights_SEQ(PhysCompStokes stokes)
{
  DM              stokes_pack,dav,dap,cda;
  Vec             gcoords;
  PetscReal       *LA_gcoords;
  PetscInt        nel,nen,e,q;
  const PetscInt  *el_nidx;
  PetscReal       el_coords[3*Q2_NODES_PER_EL_3D];
  PetscInt        nqp;
  PetscReal       WEIGHT[NQP],XI[NQP][3],NI[NQP][NPE],GNI[NQP][3][NPE];
  PetscReal       detJ[NQP],dNudx[NQP][NPE],dNudy[NQP][NPE],dNudz[NQP][NPE];
  QPntVolCoefStokes *all_gausspoints,*cell_gausspoints;
  PetscReal         *vol_col,*rho_col;
  PetscInt          idx,nI,nJ,nK,MX,MY,MZ;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  /* setup quadrature */
  nqp = 27;
  P3D_prepare_elementQ2(nqp,WEIGHT,XI,NI,GNI);

  /* get dav */
  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);

  /* setup for coords */
  ierr = DMGetCoordinateDM(dav,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dav,&gcoords);CHKERRQ(ierr);
  ierr = VecGetArray(gcoords,&LA_gcoords);CHKERRQ(ierr);
  ierr = DMDAGetElements_pTatinQ2P1(dav,&nel,&nen,&el_nidx);CHKERRQ(ierr);

  /* quadrature for all cells */
  ierr = VolumeQuadratureGetAllCellData_Stokes(stokes->volQ,&all_gausspoints);CHKERRQ(ierr);

  ierr = DMDAGetSizeElementQ2(dav,&MX,&MY,&MZ);CHKERRQ(ierr);

  PetscMalloc(sizeof(PetscReal)*MX*MZ,&vol_col);  PetscMemzero(vol_col,sizeof(PetscReal)*MX*MZ);
  PetscMalloc(sizeof(PetscReal)*MX*MZ,&rho_col);  PetscMemzero(rho_col,sizeof(PetscReal)*MX*MZ);

  for (e=0; e<nel; e++) {
    PetscReal el_rho,el_vol;

    ierr = DMDAGetElementCoordinatesQ2_3D(el_coords,(PetscInt*)&el_nidx[nen*e],LA_gcoords);CHKERRQ(ierr);
    P3D_evaluate_geometry_elementQ2(nqp,el_coords,GNI, detJ,dNudx,dNudy,dNudz);

    ierr = VolumeQuadratureGetCellData_Stokes(stokes->volQ,all_gausspoints,e,&cell_gausspoints);CHKERRQ(ierr);

    el_rho = 0.0;
    el_vol = 0.0;
    for (q=0; q<nqp; q++) {
      el_rho = el_rho + cell_gausspoints[q].rho * WEIGHT[q] * detJ[q];
      el_vol = el_vol + 1.0 * WEIGHT[q] * detJ[q];
    }

    ierr = DMDAConvertLocalElementIndex2GlobalnInJnK(dav,e,&nI,&nJ,&nK);CHKERRQ(ierr);
    idx = nI + nK*MX;
    rho_col[idx] = rho_col[idx] + el_rho;
    vol_col[idx] = vol_col[idx] + el_vol;

  }
  ierr = VecRestoreArray(gcoords,&LA_gcoords);CHKERRQ(ierr);

  for (e=0; e<MX*MZ; e++) {
    rho_col[e] = rho_col[e] / vol_col[e];
  }

  PetscFree(vol_col);
  PetscFree(rho_col);

  PetscFunctionReturn(0);
}

PetscErrorCode ModelUtilsComputeAiryIsostaticHeights(PhysCompStokes stokes)
{
  PetscErrorCode ierr;


  ierr = ModelUtilsComputeAiryIsostaticHeights_SEQ(stokes);CHKERRQ(ierr);


  PetscFunctionReturn(0);
}

PetscErrorCode MPntStdComputeBoundingBox(DataBucket materialpoint_db,PetscReal gmin[],PetscReal gmax[])
{
  MPAccess         mpX;
  int              p,n_mpoints;
  PetscReal        min[3],max[3];
  double           *pos_p;
  PetscErrorCode   ierr;

  PetscFunctionBegin;

  /* initialize */
  min[0] = PETSC_MAX_REAL;
  min[1] = PETSC_MAX_REAL;
  min[2] = PETSC_MAX_REAL;
  max[0] = PETSC_MIN_REAL;
  max[1] = PETSC_MIN_REAL;
  max[2] = PETSC_MIN_REAL;

  DataBucketGetSizes(materialpoint_db,&n_mpoints,0,0);
  ierr = MaterialPointGetAccess(materialpoint_db,&mpX);CHKERRQ(ierr);
  for (p=0; p<n_mpoints; p++) {
    ierr = MaterialPointGet_global_coord(mpX,p,&pos_p);CHKERRQ(ierr);

    min[0] = PetscMin(min[0],pos_p[0]);
    min[1] = PetscMin(min[1],pos_p[1]);
    min[2] = PetscMin(min[2],pos_p[2]);

    max[0] = PetscMax(max[0],pos_p[0]);
    max[1] = PetscMax(max[1],pos_p[1]);
    max[2] = PetscMax(max[2],pos_p[2]);
  }
  ierr = MaterialPointRestoreAccess(materialpoint_db,&mpX);CHKERRQ(ierr);

  ierr = MPI_Allreduce(min,gmin,3,MPIU_REAL,MPIU_MIN,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Allreduce(max,gmax,3,MPIU_REAL,MPIU_MAX,PETSC_COMM_WORLD);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode MPntStdComputeBoundingBoxInRange(DataBucket materialpoint_db,PetscReal rmin[],PetscReal rmax[],PetscReal gmin[],PetscReal gmax[])
{
  MPAccess         mpX;
  int              p,n_mpoints;
  PetscReal        min[3],max[3];
  double           *pos_p;
  PetscErrorCode   ierr;

  PetscFunctionBegin;

  /* initialize */
  min[0] = PETSC_MAX_REAL;
  min[1] = PETSC_MAX_REAL;
  min[2] = PETSC_MAX_REAL;
  max[0] = PETSC_MIN_REAL;
  max[1] = PETSC_MIN_REAL;
  max[2] = PETSC_MIN_REAL;

  DataBucketGetSizes(materialpoint_db,&n_mpoints,0,0);
  ierr = MaterialPointGetAccess(materialpoint_db,&mpX);CHKERRQ(ierr);
  for (p=0; p<n_mpoints; p++) {
    PetscReal p_i, range_min, range_max;
    PetscInt  idx;

    ierr = MaterialPointGet_global_coord(mpX,p,&pos_p);CHKERRQ(ierr);

    idx = 0;
    range_min = -1.0e32; if (rmin) { range_min = rmin[idx]; }
    range_max =  1.0e32; if (rmax) { range_max = rmax[idx]; }
    p_i = pos_p[idx];
    if ( (p_i >= range_min) && (p_i <= range_max) ) {
      min[idx] = PetscMin(min[idx],p_i);
      max[idx] = PetscMax(max[idx],p_i);
    }

    idx = 1;
    range_min = -1.0e32; if (rmin) { range_min = rmin[idx]; }
    range_max =  1.0e32; if (rmax) { range_max = rmax[idx]; }
    p_i = pos_p[idx];
    if ( (p_i >= range_min) && (p_i <= range_max) ) {
      min[idx] = PetscMin(min[idx],p_i);
      max[idx] = PetscMax(max[idx],p_i);
    }

    idx = 2;
    range_min = -1.0e32; if (rmin) { range_min = rmin[idx]; }
    range_max =  1.0e32; if (rmax) { range_max = rmax[idx]; }
    p_i = pos_p[idx];
    if ( (p_i >= range_min) && (p_i <= range_max) ) {
      min[idx] = PetscMin(min[idx],p_i);
      max[idx] = PetscMax(max[idx],p_i);
    }

  }
  ierr = MaterialPointRestoreAccess(materialpoint_db,&mpX);CHKERRQ(ierr);

  ierr = MPI_Allreduce(min,gmin,3,MPIU_REAL,MPIU_MIN,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Allreduce(max,gmax,3,MPIU_REAL,MPIU_MAX,PETSC_COMM_WORLD);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode MPntStdComputeBoundingBoxInRangeInRegion(DataBucket materialpoint_db,PetscReal rmin[],PetscReal rmax[],PetscInt region_idx,PetscReal gmin[],PetscReal gmax[])
{
  MPAccess         mpX;
  int              p,n_mpoints;
  PetscReal        min[3],max[3];
  double           *pos_p;
  int              region_p;
  PetscInt         found_region = 0,found_region_g;
  PetscErrorCode   ierr;

  PetscFunctionBegin;

  /* initialize */
  min[0] = PETSC_MAX_REAL;
  min[1] = PETSC_MAX_REAL;
  min[2] = PETSC_MAX_REAL;
  max[0] = PETSC_MIN_REAL;
  max[1] = PETSC_MIN_REAL;
  max[2] = PETSC_MIN_REAL;

  DataBucketGetSizes(materialpoint_db,&n_mpoints,0,0);
  ierr = MaterialPointGetAccess(materialpoint_db,&mpX);CHKERRQ(ierr);
  for (p=0; p<n_mpoints; p++) {
    PetscReal p_i, range_min, range_max;
    PetscInt  idx;

    ierr = MaterialPointGet_global_coord(mpX,p,&pos_p);CHKERRQ(ierr);
    ierr = MaterialPointGet_phase_index(mpX,p,&region_p);CHKERRQ(ierr);

    //if ( (region_idx == -1) && (region_p != region_idx) ) { continue; }
    if (region_p != region_idx) {
      if (region_idx != -1) {
        continue;
      }
    }

    idx = 0;
    range_min = PETSC_MIN_REAL; if (rmin) { range_min = rmin[idx]; }
    range_max = PETSC_MAX_REAL; if (rmax) { range_max = rmax[idx]; }
    p_i = pos_p[idx];
    if ( (p_i >= range_min) && (p_i <= range_max) ) {
      min[idx] = PetscMin(min[idx],p_i);
      max[idx] = PetscMax(max[idx],p_i);
    }

    idx = 1;
    range_min = PETSC_MIN_REAL; if (rmin) { range_min = rmin[idx]; }
    range_max = PETSC_MAX_REAL; if (rmax) { range_max = rmax[idx]; }
    p_i = pos_p[idx];
    if ( (p_i >= range_min) && (p_i <= range_max) ) {
      min[idx] = PetscMin(min[idx],p_i);
      max[idx] = PetscMax(max[idx],p_i);
    }

    idx = 2;
    range_min = PETSC_MIN_REAL; if (rmin) { range_min = rmin[idx]; }
    range_max = PETSC_MAX_REAL; if (rmax) { range_max = rmax[idx]; }
    p_i = pos_p[idx];
    if ( (p_i >= range_min) && (p_i <= range_max) ) {
      min[idx] = PetscMin(min[idx],p_i);
      max[idx] = PetscMax(max[idx],p_i);
    }

    found_region = 1;
  }
  ierr = MaterialPointRestoreAccess(materialpoint_db,&mpX);CHKERRQ(ierr);

  ierr = MPI_Allreduce(min,gmin,3,MPIU_REAL,MPIU_MIN,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Allreduce(max,gmax,3,MPIU_REAL,MPIU_MAX,PETSC_COMM_WORLD);CHKERRQ(ierr);

  ierr = MPI_Allreduce(&found_region,&found_region_g,1,MPIU_INT,MPI_MAX,PETSC_COMM_WORLD);CHKERRQ(ierr);

  /* no particles of the desired region were found on any processors, set min/max accordingly */
  if (found_region_g == 0) {
    gmin[0] = gmin[1] = gmin[2] = NAN;
    gmax[0] = gmax[1] = gmax[2] = NAN;
  }

  PetscFunctionReturn(0);
}

PetscErrorCode DMDAComputeBoundingBoxBoundaryFace(DM dav,BoundaryFaceType ft,PetscReal min[],PetscReal max[])
{
  DM cda;
  Vec coords;
  PetscInt i,j,k,si,sj,sk,ni,nj,nk,M,N,P;
  DMDACoor3d ***LA_coords;
  PetscReal gmin[3],gmax[3];
  PetscErrorCode ierr;

  ierr = DMGetCoordinateDM(dav,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinates(dav,&coords);CHKERRQ(ierr);
  ierr = DMDAGetInfo(dav,0,&M,&N,&P,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(dav,&si,&sj,&sk,&ni,&nj,&nk);CHKERRQ(ierr);

  gmax[0] = gmax[1] = gmax[2] = PETSC_MIN_REAL;
  gmin[0] = gmin[1] = gmin[2] = PETSC_MAX_REAL;

  ierr = DMDAVecGetArray(cda,coords,&LA_coords);CHKERRQ(ierr);

  switch (ft) {
    case NORTH_FACE:
      if (sj+nj == N) {
        j = N-1;
        for (k=sk; k<sk+nk; k++) {
          for (i=si; i<si+ni; i++) {
            gmin[0] = PetscMin(gmin[0],LA_coords[k][j][i].x);
            gmin[1] = PetscMin(gmin[1],LA_coords[k][j][i].y);
            gmin[2] = PetscMin(gmin[2],LA_coords[k][j][i].z);

            gmax[0] = PetscMax(gmax[0],LA_coords[k][j][i].x);
            gmax[1] = PetscMax(gmax[1],LA_coords[k][j][i].y);
            gmax[2] = PetscMax(gmax[2],LA_coords[k][j][i].z);
          }
        }
      }
      break;

    case SOUTH_FACE:
      if (sj == 0) {
        j = 0;
        for (k=sk; k<sk+nk; k++) {
          for (i=si; i<si+ni; i++) {
            gmin[0] = PetscMin(gmin[0],LA_coords[k][j][i].x);
            gmin[1] = PetscMin(gmin[1],LA_coords[k][j][i].y);
            gmin[2] = PetscMin(gmin[2],LA_coords[k][j][i].z);

            gmax[0] = PetscMax(gmax[0],LA_coords[k][j][i].x);
            gmax[1] = PetscMax(gmax[1],LA_coords[k][j][i].y);
            gmax[2] = PetscMax(gmax[2],LA_coords[k][j][i].z);
          }
        }
      }
      break;

    case EAST_FACE:
      if (si+ni == N) {
        i = N-1;
        for (k=sk; k<sk+nk; k++) {
          for (j=sj; j<sj+nj; j++) {
            gmin[0] = PetscMin(gmin[0],LA_coords[k][j][i].x);
            gmin[1] = PetscMin(gmin[1],LA_coords[k][j][i].y);
            gmin[2] = PetscMin(gmin[2],LA_coords[k][j][i].z);

            gmax[0] = PetscMax(gmax[0],LA_coords[k][j][i].x);
            gmax[1] = PetscMax(gmax[1],LA_coords[k][j][i].y);
            gmax[2] = PetscMax(gmax[2],LA_coords[k][j][i].z);
          }
        }
      }
      break;

    case WEST_FACE:
      if (si == 0) {
        i = 0;
        for (k=sk; k<sk+nk; k++) {
          for (j=sj; j<sj+nj; j++) {
            gmin[0] = PetscMin(gmin[0],LA_coords[k][j][i].x);
            gmin[1] = PetscMin(gmin[1],LA_coords[k][j][i].y);
            gmin[2] = PetscMin(gmin[2],LA_coords[k][j][i].z);

            gmax[0] = PetscMax(gmax[0],LA_coords[k][j][i].x);
            gmax[1] = PetscMax(gmax[1],LA_coords[k][j][i].y);
            gmax[2] = PetscMax(gmax[2],LA_coords[k][j][i].z);
          }
        }
      }
      break;

    case FRONT_FACE:
      if (sk+nk == P) {
        k = P-1;
        for (j=sj; j<sj+nj; j++) {
          for (i=si; i<si+ni; i++) {
            gmin[0] = PetscMin(gmin[0],LA_coords[k][j][i].x);
            gmin[1] = PetscMin(gmin[1],LA_coords[k][j][i].y);
            gmin[2] = PetscMin(gmin[2],LA_coords[k][j][i].z);

            gmax[0] = PetscMax(gmax[0],LA_coords[k][j][i].x);
            gmax[1] = PetscMax(gmax[1],LA_coords[k][j][i].y);
            gmax[2] = PetscMax(gmax[2],LA_coords[k][j][i].z);
          }
        }
      }
      break;

    case BACK_FACE:
      if (sk == 0) {
        k = 0;
        for (j=sj; j<sj+nj; j++) {
          for (i=si; i<si+ni; i++) {
            gmin[0] = PetscMin(gmin[0],LA_coords[k][j][i].x);
            gmin[1] = PetscMin(gmin[1],LA_coords[k][j][i].y);
            gmin[2] = PetscMin(gmin[2],LA_coords[k][j][i].z);

            gmax[0] = PetscMax(gmax[0],LA_coords[k][j][i].x);
            gmax[1] = PetscMax(gmax[1],LA_coords[k][j][i].y);
            gmax[2] = PetscMax(gmax[2],LA_coords[k][j][i].z);
          }
        }
      }
      break;
  }
  ierr = DMDAVecRestoreArray(cda,coords,&LA_coords);CHKERRQ(ierr);

  if (min) { ierr = MPI_Allreduce(gmin,min,3,MPIU_REAL,MPIU_MIN,PetscObjectComm((PetscObject)dav));CHKERRQ(ierr); }
  if (max) { ierr = MPI_Allreduce(gmax,max,3,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)dav));CHKERRQ(ierr); }

  PetscFunctionReturn(0);
}

PetscErrorCode DMDAComputeCoordinateAverageBoundaryFace(DM dav,BoundaryFaceType ft,PetscReal avg[])
{
  DM             cda;
  Vec            coords;
  PetscInt       i,j,k,si,sj,sk,ni,nj,nk,M,N,P,n_face;
  DMDACoor3d     ***LA_coords;
  PetscReal      gavg[3];
  PetscErrorCode ierr;

  ierr = DMGetCoordinateDM(dav,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinates(dav,&coords);CHKERRQ(ierr);
  ierr = DMDAGetInfo(dav,0,&M,&N,&P,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(dav,&si,&sj,&sk,&ni,&nj,&nk);CHKERRQ(ierr);

  gavg[0] = gavg[1] = gavg[2] = 0.0;

  ierr = DMDAVecGetArray(cda,coords,&LA_coords);CHKERRQ(ierr);

  n_face = 0;
  switch (ft) {
    case NORTH_FACE:
      if (sj+nj == N) {
        j = N-1;
        for (k=sk; k<sk+nk; k++) {
          for (i=si; i<si+ni; i++) {
            gavg[0] += LA_coords[k][j][i].x;
            gavg[1] += LA_coords[k][j][i].y;
            gavg[2] += LA_coords[k][j][i].z;
          }
        }
      }
      n_face = M * P;
      break;

    case SOUTH_FACE:
      if (sj == 0) {
        j = 0;
        for (k=sk; k<sk+nk; k++) {
          for (i=si; i<si+ni; i++) {
            gavg[0] += LA_coords[k][j][i].x;
            gavg[1] += LA_coords[k][j][i].y;
            gavg[2] += LA_coords[k][j][i].z;
          }
        }
      }
      n_face = M * P;
      break;

    case EAST_FACE:
      if (si+ni == N) {
        i = N-1;
        for (k=sk; k<sk+nk; k++) {
          for (j=sj; j<sj+nj; j++) {
            gavg[0] += LA_coords[k][j][i].x;
            gavg[1] += LA_coords[k][j][i].y;
            gavg[2] += LA_coords[k][j][i].z;
          }
        }
      }
      n_face = N * P;
      break;

    case WEST_FACE:
      if (si == 0) {
        i = 0;
        for (k=sk; k<sk+nk; k++) {
          for (j=sj; j<sj+nj; j++) {
            gavg[0] += LA_coords[k][j][i].x;
            gavg[1] += LA_coords[k][j][i].y;
            gavg[2] += LA_coords[k][j][i].z;
          }
        }
      }
      n_face = N * P;
      break;

    case FRONT_FACE:
      if (sk+nk == P) {
        k = P-1;
        for (j=sj; j<sj+nj; j++) {
          for (i=si; i<si+ni; i++) {
            gavg[0] += LA_coords[k][j][i].x;
            gavg[1] += LA_coords[k][j][i].y;
            gavg[2] += LA_coords[k][j][i].z;
          }
        }
      }
      n_face = M * N;
      break;

    case BACK_FACE:
      if (sk == 0) {
        k = 0;
        for (j=sj; j<sj+nj; j++) {
          for (i=si; i<si+ni; i++) {
            gavg[0] += LA_coords[k][j][i].x;
            gavg[1] += LA_coords[k][j][i].y;
            gavg[2] += LA_coords[k][j][i].z;
          }
        }
      }
      n_face = M * N;
      break;
  }
  ierr = DMDAVecRestoreArray(cda,coords,&LA_coords);CHKERRQ(ierr);

  if (avg) {
    ierr = MPI_Allreduce(gavg,avg,3,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)dav));CHKERRQ(ierr);
    avg[0] = avg[0] / ((PetscReal)n_face);
    avg[1] = avg[1] / ((PetscReal)n_face);
    avg[2] = avg[2] / ((PetscReal)n_face);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode StokesComputeVRMS(DM dav,Vec v,PetscReal *value_vrms,PetscReal *value_vol)
{
  DM              cda;
  Vec             gcoords;
  PetscScalar    *LA_gcoords;
  PetscInt        nel,nen,e,i,p;
  const PetscInt  *elnidx;
  PetscReal       el_coords[3*Q2_NODES_PER_EL_3D];
  PetscInt        nqp;
  PetscReal       WEIGHT[NQP],XI[NQP][3],NI[NQP][NPE],GNI[NQP][3][NPE];
  PetscReal       el_v[3*Q2_NODES_PER_EL_3D];
  PetscReal       _value_vol,_value_vrms,detJ[NQP],dNudx[NQP][NPE],dNudy[NQP][NPE],dNudz[NQP][NPE];
  Vec             v_local;
  PetscScalar     *LA_v;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  /* setup quadrature */
  nqp = 27;
  P3D_prepare_elementQ2(nqp,WEIGHT,XI,NI,GNI);

  /* setup local coords */
  ierr = DMGetCoordinateDM(dav,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dav,&gcoords);CHKERRQ(ierr);
  ierr = VecGetArray(gcoords,&LA_gcoords);CHKERRQ(ierr);

  /* setup local velocity */
  ierr = DMGetLocalVector(dav,&v_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dav,v,INSERT_VALUES,v_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(  dav,v,INSERT_VALUES,v_local);CHKERRQ(ierr);
  ierr = VecGetArray(v_local,&LA_v);CHKERRQ(ierr);

  ierr = DMDAGetElements_pTatinQ2P1(dav,&nel,&nen,&elnidx);CHKERRQ(ierr);

  _value_vol  = 0.0;
  _value_vrms = 0.0;
  for (e=0; e<nel; e++) {
    PetscReal el_vol,el_vrms;

    ierr = DMDAGetElementCoordinatesQ2_3D(el_coords,(PetscInt*)&elnidx[nen*e],LA_gcoords);CHKERRQ(ierr);

    ierr = DMDAGetVectorElementFieldQ2_3D(el_v,(PetscInt*)&elnidx[nen*e],LA_v);CHKERRQ(ierr);

    P3D_evaluate_geometry_elementQ2(nqp,el_coords,GNI, detJ,dNudx,dNudy,dNudz);

    el_vol = 0.0;
    for (p=0; p<nqp; p++) {
      el_vol = el_vol + 1.0 * WEIGHT[p] * detJ[p];
    }
    _value_vol += el_vol;

    el_vrms = 0.0;
    for (p=0; p<nqp; p++) {
      PetscReal vx_q,vy_q,vz_q;

      vx_q = vy_q = vz_q = 0.0;
      for (i=0; i<Q2_NODES_PER_EL_3D; i++) {
        vx_q = vx_q + NI[p][i] * el_v[3*i + 0];
        vy_q = vy_q + NI[p][i] * el_v[3*i + 1];
        vz_q = vz_q + NI[p][i] * el_v[3*i + 2];
      }

      el_vrms = el_vrms + WEIGHT[p] * (vx_q*vx_q + vy_q*vy_q + vz_q*vz_q) * detJ[p];
    }
    _value_vrms += el_vrms;

  }
  ierr = VecRestoreArray(gcoords,&LA_gcoords);CHKERRQ(ierr);
  ierr = VecRestoreArray(v_local,&LA_v);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dav,&v_local);CHKERRQ(ierr);

  ierr = MPI_Allreduce(&_value_vol, value_vol, 1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)dav));CHKERRQ(ierr);
  ierr = MPI_Allreduce(&_value_vrms,value_vrms,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)dav));CHKERRQ(ierr);
  //*value_vrms = PetscSqrtReal(*value_vrms);

  PetscFunctionReturn(0);
}

/*
   Viscous dissipiation is defined as

   Phi = 0.5 ( \int (\tau - pI) : \epsilon' dV )
   */
PetscErrorCode StokesComputeViscousDissipation(DM dav,DM dap,Vec sv,Vec sp,Quadrature volQ,PetscInt stress_type,PetscReal *value)
{
  DM              cda;
  Vec             gcoords,sv_local,sp_local;
  PetscScalar     *LA_gcoords,*LA_sv,*LA_sp;
  PetscInt        nel,nen_u,nen_p,e,p,k;
  const PetscInt  *elnidx_u;
  const PetscInt  *elnidx_p;
  PetscReal       el_coords[3*Q2_NODES_PER_EL_3D],el_v[3*Q2_NODES_PER_EL_3D],el_p[P_BASIS_FUNCTIONS];
  PetscReal       ux[Q2_NODES_PER_EL_3D],uy[Q2_NODES_PER_EL_3D],uz[Q2_NODES_PER_EL_3D];
  PetscInt        nqp;
  PetscReal       WEIGHT[NQP],XI[NQP][3],NI[NQP][NPE],GNI[NQP][3][NPE],NIp[NQP][P_BASIS_FUNCTIONS];
  PetscReal       value_local,detJ[NQP],dNudx[NQP][NPE],dNudy[NQP][NPE],dNudz[NQP][NPE];
  QPntVolCoefStokes *quadraturepoints,*cell_quadraturepoints;
  PetscErrorCode    ierr;


  PetscFunctionBegin;

  /* setup quadrature */
  nqp = volQ->npoints;
  P3D_prepare_elementQ2(nqp,WEIGHT,XI,NI,GNI);

  /* setup local coords */
  ierr = DMGetCoordinateDM(dav,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dav,&gcoords);CHKERRQ(ierr);
  ierr = VecGetArray(gcoords,&LA_gcoords);CHKERRQ(ierr);

  /* setup local velocity */
  ierr = DMGetLocalVector(dav,&sv_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dav,sv,INSERT_VALUES,sv_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(  dav,sv,INSERT_VALUES,sv_local);CHKERRQ(ierr);
  ierr = VecGetArray(sv_local,&LA_sv);CHKERRQ(ierr);

  /* setup local pressure */
  ierr = DMGetLocalVector(dap,&sp_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dap,sp,INSERT_VALUES,sp_local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(  dap,sp,INSERT_VALUES,sp_local);CHKERRQ(ierr);
  ierr = VecGetArray(sp_local,&LA_sp);CHKERRQ(ierr);

  ierr = DMDAGetElements_pTatinQ2P1(dav,&nel,&nen_u,&elnidx_u);CHKERRQ(ierr);
  ierr = DMDAGetElements_pTatinQ2P1(dap,&nel,&nen_p,&elnidx_p);CHKERRQ(ierr);

  ierr = VolumeQuadratureGetAllCellData_Stokes(volQ,&quadraturepoints);CHKERRQ(ierr);

  value_local = 0.0;
  for (e=0; e<nel; e++) {
    PetscReal value_element;

    ierr = DMDAGetElementCoordinatesQ2_3D(el_coords,(PetscInt*)&elnidx_u[nen_u*e],LA_gcoords);CHKERRQ(ierr);

    ierr = DMDAGetVectorElementFieldQ2_3D(el_v,(PetscInt*)&elnidx_u[nen_u*e],LA_sv);CHKERRQ(ierr);
    for (k=0; k<Q2_NODES_PER_EL_3D; k++ ) {
      ux[k] = el_v[3*k  ];
      uy[k] = el_v[3*k+1];
      uz[k] = el_v[3*k+2];
    }

    ierr = DMDAGetScalarElementField(el_p,nen_p,(PetscInt*)&elnidx_p[nen_p*e],LA_sp);CHKERRQ(ierr);

    ierr = VolumeQuadratureGetCellData_Stokes(volQ,quadraturepoints,e,&cell_quadraturepoints);CHKERRQ(ierr);

    for (p=0; p<nqp; p++) {
      PetscScalar xip[] = { XI[p][0], XI[p][1], XI[p][2] };
      ConstructNi_pressure(xip,el_coords,NIp[p]);
    }

    P3D_evaluate_geometry_elementQ2(nqp,el_coords,GNI,detJ,dNudx,dNudy,dNudz);

    value_element = 0.0;
    for (p=0; p<nqp; p++) {
      PetscReal eta_qp,pressure_qp,E_qp[3][3],sigma_qp[3][3],phi_qp;
      /* PetscReal divu_qp */
      PetscInt ii,jj;

      /* pressure */
      pressure_qp = 0.0;
      for (k=0; k<P_BASIS_FUNCTIONS; k++) {
        pressure_qp += NIp[p][k] * el_p[k];
      }

      /* strain rate, e_ij = 0.5(u_{i,j} + u_{j,i}) */
      E_qp[0][0] = E_qp[1][1] = E_qp[2][2] = 0.0;
      E_qp[0][1] = E_qp[0][2] = E_qp[1][2] = 0.0;
      for (k=0; k<Q2_NODES_PER_EL_3D; k++) {
        E_qp[0][0] += (dNudx[p][k] * ux[k]);
        E_qp[1][1] += (dNudy[p][k] * uy[k]);
        E_qp[2][2] += (dNudz[p][k] * uz[k]);

        E_qp[0][1] += 0.5 * (dNudy[p][k] * ux[k] + dNudx[p][k] * uy[k]);
        E_qp[0][2] += 0.5 * (dNudz[p][k] * ux[k] + dNudx[p][k] * uz[k]);
        E_qp[1][2] += 0.5 * (dNudz[p][k] * uy[k] + dNudy[p][k] * uz[k]);
      }
      E_qp[1][0] = E_qp[0][1];
      E_qp[2][0] = E_qp[0][2];
      E_qp[2][1] = E_qp[1][2];

      /* divu_qp = (E_qp[0][0] + E_qp[1][1] + E_qp[2][2]); */

      /* constitutive */
      eta_qp = cell_quadraturepoints[p].eta;

      if (stress_type == 0) {
        /* total stress: 2 eta e_{ij} - p \delta_{ij} */
        for (ii=0; ii<3; ii++) {
          for (jj=0; jj<3; jj++) {
            sigma_qp[ii][jj] = 2.0 * eta_qp * E_qp[ii][jj];
          }
        }
        for (ii=0; ii<3; ii++) {
          sigma_qp[ii][ii] = sigma_qp[ii][ii] - pressure_qp;
        }
      } else if (stress_type == 1) {
        /* deviatoric stress: 2 eta e_{ij} */
        for (ii=0; ii<3; ii++) {
          for (jj=0; jj<3; jj++) {
            sigma_qp[ii][jj] = 2.0 * eta_qp * E_qp[ii][jj];
          }
        }
      } else {
        /* pressure: p \delta_{ij} */
        for (ii=0; ii<3; ii++) {
          for (jj=0; jj<3; jj++) {
            sigma_qp[ii][jj] = 0.0;
          }
        }
        for (ii=0; ii<3; ii++) {
          sigma_qp[ii][ii] = sigma_qp[ii][ii] - pressure_qp;
        }
      }


      /* contraction, sigma_{ij}.e_{ij} */
      phi_qp = 0.0;
      for (ii=0; ii<3; ii++) {
        for (jj=0; jj<3; jj++) {
          phi_qp = phi_qp + sigma_qp[ii][jj] * E_qp[ii][jj];
        }
      }
      value_element = value_element + WEIGHT[p] * (phi_qp) * detJ[p];
    }

    value_local = value_local + value_element;
  }
  value_local = 0.5 * value_local;

  ierr = VecRestoreArray(gcoords,&LA_gcoords);CHKERRQ(ierr);
  ierr = VecRestoreArray(sv_local,&LA_sv);CHKERRQ(ierr);
  ierr = VecRestoreArray(sp_local,&LA_sp);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dav,&sv_local);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dap,&sp_local);CHKERRQ(ierr);

  ierr = MPI_Allreduce(&value_local, value, 1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)dav));CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*
Notes:
- This function will identify the material point (index and rank) within "tolerance" distance of coord[].
- The user can optionally mask out coordinates of the material point from the distance test.
- If multiple material points on a given sub-domain (rank) are within "tolerance" distance of coord[], the point returned will be  the "closest" to the target (coord[]).
- If no coordinate is within tol of target (coor[]), the returned values are _pidx = -1, _rank = -1
*/

struct MPI_PairedValueRank {
  double distance;
  int    rank;
};

PetscErrorCode MPntStdIdentifyFromPosition(DataBucket materialpoint_db,PetscReal coord[],PetscBool mask[],PetscInt region_idx,PetscReal tolerance,int *_pidx,PetscMPIInt *_rank)
{
  MPAccess         mpX;
  int              p,n_mpoints;
  double           *pos_p;
  int              region_p;
  PetscReal        sep2,tol2,min2;
  int              p_mine,p_found,rank;
  struct MPI_PairedValueRank input,output;
  PetscErrorCode   ierr;

  PetscFunctionBegin;

  tol2 = tolerance*tolerance;
  min2 = PETSC_MAX_REAL;

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  p_found  = 0;
  p_mine   = -1;

  DataBucketGetSizes(materialpoint_db,&n_mpoints,0,0);
  ierr = MaterialPointGetAccess(materialpoint_db,&mpX);CHKERRQ(ierr);
  for (p=0; p<n_mpoints; p++) {

    ierr = MaterialPointGet_global_coord(mpX,p,&pos_p);CHKERRQ(ierr);
    ierr = MaterialPointGet_phase_index(mpX,p,&region_p);CHKERRQ(ierr);

    if (region_idx != -1) {
      if (region_p != region_idx) {
        continue;
      }
    }

    sep2 = 0.0;
    if (mask) {
      if (mask[0]) { sep2 += (pos_p[0]-coord[0])*(pos_p[0]-coord[0]); }
      if (mask[1]) { sep2 += (pos_p[1]-coord[1])*(pos_p[1]-coord[1]); }
      if (mask[2]) { sep2 += (pos_p[2]-coord[2])*(pos_p[2]-coord[2]); }
    } else {
      sep2 += (pos_p[0]-coord[0])*(pos_p[0]-coord[0]);
      sep2 += (pos_p[1]-coord[1])*(pos_p[1]-coord[1]);
      sep2 += (pos_p[2]-coord[2])*(pos_p[2]-coord[2]);
    }

    if (sep2 < min2) {
      p_mine   = p;
      p_found++;
      min2 = sep2;
    }
  }
  ierr = MaterialPointRestoreAccess(materialpoint_db,&mpX);CHKERRQ(ierr);

  /*
  http://mpi-forum.org/docs/mpi-1.1/mpi-11-html/node79.html
  */
  input.distance = min2;
  input.rank     = rank;

  /* intialise output struct members */
  output.distance = 0.0;
  output.rank     = -1;

  ierr = MPI_Reduce( &input, &output, 1, MPI_DOUBLE_INT, MPI_MINLOC, 0, PETSC_COMM_WORLD );CHKERRQ(ierr);

  /* Answer resides on process root - broadcast rank with the minimum value */
  ierr = MPI_Bcast(&output.rank,1,MPI_INT,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(&output.distance,1,MPI_DOUBLE,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  min2 = output.distance;

  /* broadcast p_mine from the root = output.rank (that with the minimum value) */
  ierr = MPI_Bcast(&p_mine,1,MPI_INT,output.rank,PETSC_COMM_WORLD);CHKERRQ(ierr);

  if (min2 > tol2) {
    *_pidx = -1;
    *_rank = -1;
  } else {
    *_pidx = p_mine;
    *_rank = (PetscMPIInt)output.rank;
  }

  PetscFunctionReturn(0);
}

PetscErrorCode MPntStdCoordinateMinIdentifyPointIndex(DataBucket materialpoint_db,
    int region_idx,
    int pmin_x[],int pmin_y[],int pmin_z[])
{
  MPAccess         mpX;
  int              d,p,n_mpoints;
  double           *pos_p;
  int              region_p;
  double           min_coord[] = { PETSC_MAX_REAL, PETSC_MAX_REAL, PETSC_MAX_REAL };
  int              p_mine[3],rank;
  struct MPI_PairedValueRank input[3],output[3];
  int              *collection[3];
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  collection[0] = pmin_x;
  collection[1] = pmin_y;
  collection[2] = pmin_z;

  p_mine[0] = p_mine[1] = p_mine[2] = -1;

  DataBucketGetSizes(materialpoint_db,&n_mpoints,0,0);
  ierr = MaterialPointGetAccess(materialpoint_db,&mpX);CHKERRQ(ierr);
  for (p=0; p<n_mpoints; p++) {
    ierr = MaterialPointGet_phase_index(mpX,p,&region_p);CHKERRQ(ierr);
    if (region_idx != -1) {
      if (region_p != region_idx) {
        continue;
      }
    }
    ierr = MaterialPointGet_global_coord(mpX,p,&pos_p);CHKERRQ(ierr);

    for (d=0; d<3; d++) {
      if (collection[d]) {
        if (pos_p[d] < min_coord[d]) {
          min_coord[d] = pos_p[d];
          p_mine[d]     = p;
        }
      }
    }
  }
  ierr = MaterialPointRestoreAccess(materialpoint_db,&mpX);CHKERRQ(ierr);

  for (d=0; d<3; d++) {
    if (collection[d]) {
      input[d].distance = min_coord[d];
      input[d].rank     = rank;

      /* intialise output struct members */
      output[d].distance = 0.0;
      output[d].rank     = -1;

      ierr = MPI_Reduce( &input[d], &output[d], 1, MPI_DOUBLE_INT, MPI_MINLOC, 0, PETSC_COMM_WORLD );CHKERRQ(ierr);

      /* Answer resides on process root - broadcast rank with the minimum value */
      ierr = MPI_Bcast(&output[d].rank,1,MPI_INT,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
      ierr = MPI_Bcast(&output[d].distance,1,MPI_DOUBLE,0,PETSC_COMM_WORLD);CHKERRQ(ierr);

      /* broadcast p_mine from the root = output.rank (that with the minimum value) */
      ierr = MPI_Bcast(&p_mine[d],1,MPI_INT,output[d].rank,PETSC_COMM_WORLD);CHKERRQ(ierr);

      collection[d][0] = p_mine[d];
      collection[d][1] = (PetscMPIInt)output[d].rank;
    }
  }

  PetscFunctionReturn(0);
}

PetscErrorCode MPntStdCoordinateMaxIdentifyPointIndex(DataBucket materialpoint_db,
    int region_idx,
    int pmax_x[],int pmax_y[],int pmax_z[])
{
  MPAccess         mpX;
  int              d,p,n_mpoints;
  double           *pos_p;
  int              region_p;
  double           max_coord[] = { PETSC_MIN_REAL, PETSC_MIN_REAL, PETSC_MIN_REAL };
  int              p_mine[3],rank;
  struct MPI_PairedValueRank input[3],output[3];
  int              *collection[3];
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  collection[0] = pmax_x;
  collection[1] = pmax_y;
  collection[2] = pmax_z;

  p_mine[0] = p_mine[1] = p_mine[2] = -1;

  DataBucketGetSizes(materialpoint_db,&n_mpoints,0,0);
  ierr = MaterialPointGetAccess(materialpoint_db,&mpX);CHKERRQ(ierr);
  for (p=0; p<n_mpoints; p++) {
    ierr = MaterialPointGet_phase_index(mpX,p,&region_p);CHKERRQ(ierr);
    if (region_idx != -1) {
      if (region_p != region_idx) {
        continue;
      }
    }
    ierr = MaterialPointGet_global_coord(mpX,p,&pos_p);CHKERRQ(ierr);

    for (d=0; d<3; d++) {
      if (collection[d]) {
        if (pos_p[d] > max_coord[d]) {
          max_coord[d] = pos_p[d];
          p_mine[d]     = p;
        }
      }
    }
  }
  ierr = MaterialPointRestoreAccess(materialpoint_db,&mpX);CHKERRQ(ierr);

  for (d=0; d<3; d++) {
    if (collection[d]) {
      input[d].distance = max_coord[d];
      input[d].rank     = rank;

      /* intialise output struct members */
      output[d].distance = 0.0;
      output[d].rank     = -1;

      ierr = MPI_Reduce( &input[d], &output[d], 1, MPI_DOUBLE_INT, MPI_MAXLOC, 0, PETSC_COMM_WORLD );CHKERRQ(ierr);

      /* Answer resides on process root - broadcast rank with the minimum value */
      ierr = MPI_Bcast(&output[d].rank,1,MPI_INT,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
      ierr = MPI_Bcast(&output[d].distance,1,MPI_DOUBLE,0,PETSC_COMM_WORLD);CHKERRQ(ierr);

      /* broadcast p_mine from the root = output.rank (that with the minimum value) */
      ierr = MPI_Bcast(&p_mine[d],1,MPI_INT,output[d].rank,PETSC_COMM_WORLD);CHKERRQ(ierr);

      collection[d][0] = p_mine[d];
      collection[d][1] = (PetscMPIInt)output[d].rank;
    }
  }

  PetscFunctionReturn(0);
}

/*
 Computes
   \int_S \vec u \cdot \vec n dS
 over each (six) faces of the physical domain.

 Collective on stokes->dau

 [Notes]
 * u_dot_n[] must be of length HEX_EDGES = 6.
 * int_u_dot_n[] must be valid on all ranks, even those which have a sub-domain which does not intersect the physical boundary.
 * For convienence the ordering of the output array does not follow
     typedef enum { HEX_FACE_Pxi=0, HEX_FACE_Nxi, HEX_FACE_Peta, HEX_FACE_Neta, HEX_FACE_Pzeta, HEX_FACE_Nzeta } HexElementFace;
   rather it uses
     typedef enum { FRONT_FACE=1, BACK_FACE, EAST_FACE, NORTH_FACE, SOUTH_FACE, WEST_FACE } BoundaryFaceType;
   However (unfortunately), this enum starts at 1 (rather than zero),
   hence if you want the NORTH value of \int_S u.n dS you access it via
     PetscReal flux = int_u_dot_n[ NORTH_FACE - 1 ].
*/
PetscErrorCode StokesComputeVdotN(PhysCompStokes stokes,Vec u,PetscReal int_u_dot_n[])
{
  PetscErrorCode     ierr;
  DM                 dau;
  PetscInt           q,d,k,e,edge,fe,face_id;
  Vec                ul,gcoords;
  PetscInt           nel,nen_u;
  const PetscInt     *elnidx_u;
  PetscReal          elcoords[3*Q2_NODES_PER_EL_3D];
  PetscReal          elu[3*Q2_NODES_PER_EL_3D],elu_face[3*Q2_NODES_PER_EL_2D];
  PetscInt           vel_el_lidx[3*U_BASIS_FUNCTIONS];
  QPntSurfCoefStokes *quadpoints,*cell_quadpoints;
  PetscReal          NIu_surf[NQP][Q2_NODES_PER_EL_2D];
  SurfaceQuadrature  surfQ;
  const PetscScalar  *_ufield,*_coords;
  PetscReal          _int_u_dot_n[HEX_EDGES];
  MeshFacetInfo      mfi;

  PetscFunctionBegin;
  ierr = PhysCompStokesGetDMs(stokes,&dau,NULL);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dau,&ul);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dau,u,INSERT_VALUES,ul);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (dau,u,INSERT_VALUES,ul);CHKERRQ(ierr);
  ierr = VecGetArrayRead(ul,&_ufield);CHKERRQ(ierr);

  ierr = DMGetCoordinatesLocal(dau,&gcoords);CHKERRQ(ierr);
  ierr = VecGetArrayRead(gcoords,&_coords);CHKERRQ(ierr);

  ierr = DMDAGetElements_pTatinQ2P1(dau,&nel,&nen_u,&elnidx_u);CHKERRQ(ierr);

  surfQ = stokes->surfQ;
  mfi   = stokes->mfi;

  ierr = PetscMemzero(_int_u_dot_n,sizeof(PetscReal)*HEX_EDGES);CHKERRQ(ierr);
  for (edge=0; edge<HEX_EDGES; edge++) {
    ConformingElementFamily element;
    int                     *face_local_indices;
    PetscInt                nqp,start,end;
    QPoint2d                *qp2;

    element = mfi->element;
    qp2     = surfQ->gp2[edge];
    nqp     = surfQ->npoints;

    start = mfi->facet_label_offset[edge];
    end   = mfi->facet_label_offset[edge+1];

    ierr = SurfaceQuadratureGetAllCellData_Stokes(surfQ,&quadpoints);CHKERRQ(ierr);

    /* evaluate the quadrature points using the 1D basis for this edge */
    for (q=0; q<nqp; q++) {
      element->basis_NI_2D(&qp2[q],NIu_surf[q]);
    }

    face_local_indices = element->face_node_list[edge];

    for (fe=start; fe<end; fe++) { /* for all elements on this domain face */
      /* get element index of the face element we want to integrate */
      e = mfi->facet_cell_index[fe];
      face_id = mfi->facet_label[fe];

      ierr = StokesVelocity_GetElementLocalIndices(vel_el_lidx,(PetscInt*)&elnidx_u[nen_u*e]);CHKERRQ(ierr);
      ierr = DMDAGetElementCoordinatesQ2_3D(elcoords,(PetscInt*)&elnidx_u[nen_u*e],(PetscScalar*)_coords);CHKERRQ(ierr);

      ierr = DMDAGetVectorElementFieldQ2_3D(elu,(PetscInt*)&elnidx_u[nen_u*e],(PetscScalar*)_ufield);CHKERRQ(ierr);
      for (k=0; k<Q2_NODES_PER_EL_2D; k++) {
        int nidx3d;

        /* map 2D index over element face to 3D element space */
        nidx3d = face_local_indices[k];
        elu_face[3*k  ] = elu[3*nidx3d  ];
        elu_face[3*k+1] = elu[3*nidx3d+1];
        elu_face[3*k+2] = elu[3*nidx3d+2];
      }

      ierr = SurfaceQuadratureGetCellData_Stokes(surfQ,quadpoints,fe,&cell_quadpoints);CHKERRQ(ierr);

      for (q=0; q<nqp; q++) {
        PetscScalar fac,surfJ_q,u_q[] = {0,0,0},u_dot_n_q = 0;

        element->compute_surface_geometry_3D(
                                             element,
                                             elcoords,    // should contain 27 points with dimension 3 (x,y,z) //
                                             face_id,   // edge index 0,...,7 //
                                             &qp2[q], // should contain 1 point with dimension 2 (xi,eta)   //
                                             NULL,NULL, &surfJ_q ); // n0[],t0 contains 1 point with dimension 3 (x,y,z) //
        fac = qp2[q].w * surfJ_q;

        /* interpolate v to face quadrature point */
        for (k=0; k<Q2_NODES_PER_EL_2D; k++) {
          for (d=0; d<3; d++) {
            u_q[d] += NIu_surf[q][k] * elu_face[3*k+d];
          }
        }

        /* compute v.n at quadrature point */
        for (d=0; d<3; d++) {
          u_dot_n_q += u_q[d] * cell_quadpoints[q].normal[d];
        }

        /* accumulate integrand */
        _int_u_dot_n[edge] += fac * u_dot_n_q;
      }
    }
  }

  ierr = VecRestoreArrayRead(gcoords,&_coords);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(ul,&_ufield);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dau,&ul);CHKERRQ(ierr);

  /* perform reduction */
  ierr = MPI_Allreduce(MPI_IN_PLACE,_int_u_dot_n,HEX_EDGES,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)dau));CHKERRQ(ierr);
  int_u_dot_n[ NORTH_FACE -1 ] = _int_u_dot_n[ HEX_FACE_Peta  ];
  int_u_dot_n[ EAST_FACE  -1 ] = _int_u_dot_n[ HEX_FACE_Pxi   ];
  int_u_dot_n[ SOUTH_FACE -1 ] = _int_u_dot_n[ HEX_FACE_Neta  ];
  int_u_dot_n[ WEST_FACE  -1 ] = _int_u_dot_n[ HEX_FACE_Nxi   ];
  int_u_dot_n[ FRONT_FACE -1 ] = _int_u_dot_n[ HEX_FACE_Pzeta ];
  int_u_dot_n[ BACK_FACE  -1 ] = _int_u_dot_n[ HEX_FACE_Nzeta ];

  PetscFunctionReturn(0);
}

PetscErrorCode MPntStdSetRegionIndexFromMesh(DataBucket material_points, Mesh mesh, long int *region_idx, PetscInt method, PetscReal length_scale)
{ 
  DataField      PField_std;
  int            p,n_mp_points;
  PetscFunctionBegin;

  DataBucketGetSizes(material_points,&n_mp_points,0,0);
  DataBucketGetDataFieldByName(material_points,MPntStd_classname,&PField_std);
  DataFieldGetAccess(PField_std);

  /* Loop over material points */
  for (p=0; p<n_mp_points; p++) {
    MPntStd  *marker_std;
    int      d;
    long int np = 1,found;
    long int ep[] = {-1};
    double   xip[] = {0.0,0.0,0.0};
    double   point_coor[NSD];

    DataFieldAccessPoint(PField_std,p,(void**)&marker_std);

    for (d=0; d<NSD; d++) {
      point_coor[d] = marker_std->coor[d] * length_scale;
    }

    /* locate point */
    switch (method) {
      case 0:
        PointLocation_BruteForce(mesh,np,(const double*)point_coor,ep,xip,&found);
        break;
      case 1:
        PointLocation_PartitionedBoundingBox(mesh,np,(const double*)point_coor,ep,xip,&found);
        break;
      default:
        PointLocation_PartitionedBoundingBox(mesh,np,(const double*)point_coor,ep,xip,&found);
        break;
    }
    /* assign marker phase */
    marker_std->phase = region_idx[ep[0]];
  }
  DataFieldRestoreAccess(PField_std);
  
  PetscFunctionReturn(0);
}

PetscErrorCode pTatin_MPntStdSetRegionIndexFromMesh(pTatinCtx ptatin, const char mesh_file[], const char region_file[], PetscInt method, PetscReal length_scale)
{
  Mesh           mesh;
  DataBucket     material_points;
  long int       *region_idx = NULL;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  /* Get user mesh from file */
  parse_mesh(PETSC_COMM_WORLD,mesh_file,&mesh);
  if (!mesh) { SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"mesh from file %s = NULL. Aborting.\n",mesh_file); }
  /* Get region index from file */
  parse_field(PETSC_COMM_WORLD,mesh,region_file,'c',(void**)&region_idx);
  /* Get material points data bucket */
  ierr = pTatinGetMaterialPoints(ptatin,&material_points,NULL);CHKERRQ(ierr);
  /* Assign marker phase */
  ierr = MPntStdSetRegionIndexFromMesh(material_points,mesh,region_idx,method,length_scale);CHKERRQ(ierr);

  /* Free region array and mesh from files */
  free(region_idx);
  MeshDestroy(&mesh);

  PetscFunctionReturn(0);
}


PetscErrorCode pTatin_ModelLoadTemperatureInitialSolution_FromFile(pTatinCtx ptatin, const char model_name[])
{
  PhysCompEnergyFV energy;
  PetscBool        flg = PETSC_FALSE,temperature_ic_from_file = PETSC_FALSE;
  char             fname[PETSC_MAX_PATH_LEN],temperature_file[PETSC_MAX_PATH_LEN];
  PetscViewer      viewer;
  PetscErrorCode   ierr;

  PetscFunctionBegin;

  ierr = pTatinGetContext_EnergyFV(ptatin,&energy);CHKERRQ(ierr);
  /* If job is restarted skip that part (Temperature is loaded from checkpointed file) */
  if (!ptatin->restart_from_file) {
    ierr = PetscOptionsGetBool(NULL,model_name,"-ic_temperature_from_file",&temperature_ic_from_file,NULL);CHKERRQ(ierr);
    if (temperature_ic_from_file) {
      /* Check if a file is provided */
      ierr = PetscOptionsGetString(NULL,model_name,"-temperature_file",temperature_file,PETSC_MAX_PATH_LEN-1,&flg);CHKERRQ(ierr);
      if (flg) {
        ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,temperature_file,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
      } else {
        PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s/temperature_steady.pbvec",ptatin->outputpath);
        ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,fname,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
      }
      ierr = VecLoad(energy->T,viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Providing a temperature file for initial state is required\n");
    }
  }
  PetscFunctionReturn(0);
}
