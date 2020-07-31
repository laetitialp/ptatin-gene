
#include <ptatin3d.h>
#include <private/ptatin_impl.h>
#include <ptatin3d_stokes.h>
#include <dmda_element_q2p1.h>
#include <element_utils_q2.h>
#include <phys_comp_energy.h>

#include <ptatin3d_energyfv.h>
#include <ptatin3d_energyfv_impl.h>
#include <finite_volume/fvda_impl.h>
#include <finite_volume/fvda.h>
#include <finite_volume/fvda_utils.h>
#include <finite_volume/fvda_private.h>
#include <finite_volume/kdtree.h>

PetscErrorCode fvgeometry_dmda3d_create_from_element_partition(MPI_Comm comm,PetscInt target_decomp[],const PetscInt m[],DM *dm);
PetscErrorCode _cart_convert_index_to_ijk(PetscInt r,const PetscInt mp[],PetscInt rijk[]);
PetscErrorCode _cart_convert_ijk_to_index(const PetscInt rijk[],const PetscInt mp[],PetscInt *r);

PetscErrorCode PhysCompEnergyFVDestroy(PhysCompEnergyFV *energy)
{
  PhysCompEnergyFV e;
  PetscInt         i;
  PetscErrorCode   ierr;
  
  PetscFunctionBegin;
  if (!energy) PetscFunctionReturn(0);
  e = *energy;
  if (!e) PetscFunctionReturn(0);
  
  ierr = PetscFree(e->xi_macro);CHKERRQ(ierr);
  for (i=0; i<e->npoints_macro; i++) {
    ierr = PetscFree(e->basis_macro[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(e->basis_macro);CHKERRQ(ierr);
  ierr = DMDestroy(&e->dmv);CHKERRQ(ierr);
  ierr = VecDestroy(&e->velocity);CHKERRQ(ierr);
  ierr = VecDestroy(&e->T);CHKERRQ(ierr);
  ierr = VecDestroy(&e->Told);CHKERRQ(ierr);
  ierr = VecDestroy(&e->Xold);CHKERRQ(ierr);
  ierr = VecDestroy(&e->F);CHKERRQ(ierr);
  ierr = MatDestroy(&e->J);CHKERRQ(ierr);
  ierr = SNESDestroy(&e->snes);CHKERRQ(ierr);
  ierr = FVDADestroy(&e->fv);CHKERRQ(ierr);
  ierr = PetscFree(e);CHKERRQ(ierr);
  *energy = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode PhysCompEnergyFVCreate(MPI_Comm comm,PhysCompEnergyFV *energy)
{
  PhysCompEnergyFV e;
  PetscErrorCode   ierr;
  
  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(struct _p_PhysCompEnergyFV),&e);CHKERRQ(ierr);
  ierr = PetscMemzero(e,sizeof(struct _p_PhysCompEnergyFV));CHKERRQ(ierr);
  e->nsubdivision[0] = -1;
  e->nsubdivision[1] = -1;
  e->nsubdivision[2] = -1;
  ierr = FVDACreate(comm,&e->fv);CHKERRQ(ierr);
  ierr = FVDASetDimension(e->fv,3);CHKERRQ(ierr);
  *energy = e;
  PetscFunctionReturn(0);
}

PetscErrorCode PhysCompEnergyFVSetParams(PhysCompEnergyFV energy,PetscReal time,PetscReal dt,PetscInt nsub[])
{
  PetscFunctionBegin;
  energy->time = time;
  energy->dt = dt;
  if (nsub) {
    energy->nsubdivision[0] = nsub[0];
    energy->nsubdivision[1] = nsub[1];
    energy->nsubdivision[2] = nsub[2];
    
    energy->npoints_macro = (energy->nsubdivision[0] + 1)*(energy->nsubdivision[1] + 1)*(energy->nsubdivision[2] + 1);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PhysCompEnergyFVSetFromOptions(PhysCompEnergyFV energy)
{
  PetscErrorCode ierr;
  PetscInt n=3,sub[]={0,0,0};
  PetscBool found = PETSC_FALSE;
  PetscFunctionBegin;
  ierr = PetscOptionsGetIntArray(NULL,NULL,"-ptatin_energyfv_nsub",sub,&n,&found);CHKERRQ(ierr);
  if (found) {
    if (n != 3) SETERRQ(energy->fv->comm,PETSC_ERR_USER,"Must provide 3 values for option -ptatin_energyfv_nsub");
    ierr = PhysCompEnergyFVSetParams(energy,energy->time,energy->dt,sub);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PhysCompEnergyFVSetUp(PhysCompEnergyFV energy,pTatinCtx pctx)
{
  PetscInt q2_mi[]={0,0,0};
  PetscInt fv_mi[]={0,0,0},mi[]={0,0,0},Mi[]={0,0,0};
  PetscInt decomp[]={0,0,0};
  PhysCompStokes stokes;
  DM stokes_dmv,fv_dmgeom;
  PetscInt d;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = pTatinGetStokesContext(pctx,&stokes);CHKERRQ(ierr);
  ierr = PhysCompStokesGetDMs(stokes,&stokes_dmv,NULL);CHKERRQ(ierr);

  /* fetch the parallel decomposition */
  ierr = DMDAGetInfo(stokes_dmv,NULL,NULL,NULL,NULL,&decomp[0],&decomp[1],&decomp[2],NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);

  /* fetch local size of the Q2 mesh */
  ierr = DMDAGetLocalSizeElementQ2(stokes_dmv,&q2_mi[0],&q2_mi[1],&q2_mi[2]);CHKERRQ(ierr);
  energy->mi_parent[0] = q2_mi[0];
  energy->mi_parent[1] = q2_mi[1];
  energy->mi_parent[2] = q2_mi[2];
  
  /* Build a compatable DMDA for velocity */
  fv_mi[0] = q2_mi[0] * energy->nsubdivision[0];
  fv_mi[1] = q2_mi[1] * energy->nsubdivision[1];
  fv_mi[2] = q2_mi[2] * energy->nsubdivision[2];
  
  ierr = fvgeometry_dmda3d_create_from_element_partition(energy->fv->comm,decomp,fv_mi,&energy->dmv);CHKERRQ(ierr);
  
  ierr = DMDAGetElementsSizes(energy->dmv,&mi[0],&mi[1],&mi[2]);CHKERRQ(ierr);
  /* check mi[] == fv_mi[] */
  for (d=0; d<3; d++) {
    if (mi[d] != fv_mi[d]) SETERRQ1(energy->fv->comm,PETSC_ERR_USER,"DMDA for FV has inconsistent number of elements (direction %D)",d);
  }
  
  /* Set the sizes for the FV mesh */
  Mi[0] = pctx->mx * energy->nsubdivision[0];
  Mi[1] = pctx->my * energy->nsubdivision[1];
  Mi[2] = pctx->mz * energy->nsubdivision[2];

  ierr = FVDASetSizes(energy->fv,mi,Mi);CHKERRQ(ierr);
  
  ierr = FVDASetProblemType(energy->fv,PETSC_TRUE,FVDA_PARABOLIC,0,0);CHKERRQ(ierr);

  /* Setup geometry DM and coordinates for FVDA */
  ierr = DMGetCoordinateDM(energy->dmv,&fv_dmgeom);CHKERRQ(ierr);
  ierr = FVDASetGeometryDM(energy->fv,fv_dmgeom);CHKERRQ(ierr);
  {
    Vec gcoor;
    
    ierr = DMGetCoordinates(energy->dmv,&gcoor);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)gcoor);CHKERRQ(ierr);
    energy->fv->vertex_coor_geometry = gcoor;
  }
  
  /* Finalize setup */
  ierr = FVDASetUp(energy->fv);CHKERRQ(ierr);

  //ierr = FVDASetup_TimeDep(energy->fv);CHKERRQ(ierr);
  ierr = FVDASetup_ALE(energy->fv);CHKERRQ(ierr);


  ierr = FVDARegisterFaceProperty(energy->fv,"v",3);CHKERRQ(ierr);
  ierr = FVDARegisterFaceProperty(energy->fv,"xDot",3);CHKERRQ(ierr);

  ierr = FVDARegisterFaceProperty(energy->fv,"v.n",1);CHKERRQ(ierr);
  ierr = FVDARegisterFaceProperty(energy->fv,"xDot.n",1);CHKERRQ(ierr);
  ierr = FVDARegisterFaceProperty(energy->fv,"k",1);CHKERRQ(ierr);
  
  ierr = FVDARegisterCellProperty(energy->fv,"rho*cp",1);CHKERRQ(ierr);
  ierr = FVDARegisterCellProperty(energy->fv,"k",1);CHKERRQ(ierr);
  ierr = FVDARegisterCellProperty(energy->fv,"H",1);CHKERRQ(ierr);
  
  
  /* PhysCompEnergyFV internals */
  {
    PetscInt ii,jj,kk,d,cnt=0;
    PetscReal dxi[]={0,0,0};
    
    ierr = PetscCalloc1(3*energy->npoints_macro,&energy->xi_macro);CHKERRQ(ierr);
    for (d=0; d<3; d++) {
      dxi[d] = 2.0 / ((PetscReal)energy->nsubdivision[d]);
    }
    for (kk=0; kk<energy->nsubdivision[2]+1; kk++) {
      for (jj=0; jj<energy->nsubdivision[1]+1; jj++) {
        for (ii=0; ii<energy->nsubdivision[0]+1; ii++) {
          energy->xi_macro[3*cnt+0] = -1.0 + ii * dxi[0];
          energy->xi_macro[3*cnt+1] = -1.0 + jj * dxi[1];
          energy->xi_macro[3*cnt+2] = -1.0 + kk * dxi[2];
          cnt++;
        }
      }
    }
    
    ierr = PetscCalloc1(energy->npoints_macro,&energy->basis_macro);CHKERRQ(ierr);
    for (d=0; d<energy->npoints_macro; d++) {
      ierr = PetscCalloc1(Q2_NODES_PER_EL_3D,&energy->basis_macro[d]);CHKERRQ(ierr);
    }
    for (d=0; d<energy->npoints_macro; d++) {
      P3D_ConstructNi_Q2_3D(&energy->xi_macro[3*d],energy->basis_macro[d]);CHKERRQ(ierr);
    }
  }
  
  ierr = DMCreateGlobalVector(energy->fv->dm_fv,&energy->T);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(energy->fv->dm_fv,&energy->Told);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(energy->fv->dm_fv,&energy->F);CHKERRQ(ierr);
  ierr = DMCreateMatrix(energy->fv->dm_fv,&energy->J);CHKERRQ(ierr);
  {
    ierr = DMCreateGlobalVector(energy->dmv,&energy->velocity);CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(fv_dmgeom,&energy->Xold);CHKERRQ(ierr);
  }

  /* PhysCompEnergyFV snes configuration for adv-diffusion */
  ierr = SNESCreate(energy->fv->comm,&energy->snes);CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(energy->snes,"energyfv_");CHKERRQ(ierr);
  ierr = SNESSetDM(energy->snes,energy->fv->dm_fv);CHKERRQ(ierr);
  ierr = SNESSetSolution(energy->snes,energy->T);CHKERRQ(ierr);
  ierr = SNESSetApplicationContext(energy->snes,(void*)energy->fv);CHKERRQ(ierr);
  //ierr = SNESSetApplicationContext(energy->snes,(void*)energy);CHKERRQ(ierr);
  
  //ierr = SNESSetFunction(energy->snes,energy->F,fvda_eval_F_timedep,NULL);CHKERRQ(ierr);
  //ierr = SNESSetFunction(energy->snes,energy->F,fvda_highres_eval_F_timedep,NULL);CHKERRQ(ierr);
  //ierr = SNESSetJacobian(energy->snes,energy->J,energy->J,fvda_eval_J_timedep,NULL);CHKERRQ(ierr);

  ierr = SNESSetFunction(energy->snes,energy->F,fvda_highres_eval_F_forward_ale,NULL);CHKERRQ(ierr);
  ierr = SNESSetJacobian(energy->snes,energy->J,energy->J,fvda_eval_J_forward_ale,NULL);CHKERRQ(ierr);

  
  ierr = SNESSetFromOptions(energy->snes);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/*
 
*/
PetscErrorCode PhysCompEnergyFVUpdateGeometry(PhysCompEnergyFV energy,PhysCompStokes stokes)
{
  DM               dmv,dmv_fv,dmc,dmc_fv;
  PetscInt         nel,nen_u,nel_fv,nen_fv,e,k,m,ii,jj,kk,cnt;
  const PetscInt   *elnidx_u,*elnidx_fv;
  PetscReal        elcoords[3*Q2_NODES_PER_EL_3D];
  Vec              coor,coor_fv;
  const PetscReal  *_coor;
  PetscReal        *_coor_fv;
  PetscInt         *fvindex;
  PetscInt         gni[]={0,0,0};
  PetscErrorCode   ierr;
  
  PetscFunctionBegin;
  
  
  ierr = PhysCompStokesGetDMs(stokes,&dmv,NULL);CHKERRQ(ierr);
  dmv_fv = energy->dmv;
  
  ierr = DMGetCoordinateDM(dmv,&dmc);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dmv,&coor);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coor,&_coor);CHKERRQ(ierr);

  dmc_fv = energy->fv->dm_geometry;
  ierr = DMGetCoordinatesLocal(dmv_fv,&coor_fv);CHKERRQ(ierr);
  ierr = VecGetArray(coor_fv,&_coor_fv);CHKERRQ(ierr);

  ierr = PetscCalloc1(energy->npoints_macro,&fvindex);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(dmv_fv,NULL,NULL,NULL,&gni[0],&gni[1],&gni[2]);CHKERRQ(ierr);

  ierr = DMDAGetElements_pTatinQ2P1(dmv,&nel,&nen_u,&elnidx_u);CHKERRQ(ierr);
  ierr = DMDAGetElements(dmv_fv,&nel_fv,&nen_fv,&elnidx_fv);CHKERRQ(ierr);
  for (e=0; e<nel; e++) {
    PetscInt fv_start[]={0,0,0};
    PetscInt q2_start[]={0,0,0};
    PetscInt fv_cell,start_ijk,fv_i_start[]={0,0,0};
    
    ierr = DMDAGetElementCoordinatesQ2_3D(elcoords,(PetscInt*)&elnidx_u[nen_u*e],(PetscReal*)_coor);CHKERRQ(ierr);

    ierr = _cart_convert_index_to_ijk(e,(const PetscInt*)energy->mi_parent,q2_start);CHKERRQ(ierr);
    
    /* determine the macro lower/left corner point */
    fv_start[0] = q2_start[0] * energy->nsubdivision[0];
    fv_start[1] = q2_start[1] * energy->nsubdivision[1];
    fv_start[2] = q2_start[2] * energy->nsubdivision[2];
    
    ierr = _cart_convert_ijk_to_index((const PetscInt*)fv_start,(const PetscInt*)energy->fv->mi,&fv_cell);CHKERRQ(ierr);
    start_ijk = elnidx_fv[8 * fv_cell + 0];
    
    ierr = _cart_convert_index_to_ijk(start_ijk,(const PetscInt*)gni,fv_i_start);CHKERRQ(ierr);

    cnt = 0;
    for (kk=0; kk<energy->nsubdivision[2]+1; kk++) {
      for (jj=0; jj<energy->nsubdivision[1]+1; jj++) {
        for (ii=0; ii<energy->nsubdivision[0]+1; ii++) {
          fvindex[cnt] = (fv_i_start[0] + ii) + (fv_i_start[1] + jj)*gni[0] + (fv_i_start[2] + kk)*gni[0]*gni[1];
          cnt++;
        }
      }
    }

    for (m=0; m<energy->npoints_macro; m++) {
      PetscReal x[]={0,0,0};
      
      /* interpolate Q2 coordinates to macro points */
      x[0] = x[1] = x[2] = 0;
      for (k=0; k<Q2_NODES_PER_EL_3D; k++) {
        x[0] += energy->basis_macro[m][k] * elcoords[3*k+0];
        x[1] += energy->basis_macro[m][k] * elcoords[3*k+1];
        x[2] += energy->basis_macro[m][k] * elcoords[3*k+2];
      }
      
      /* insert macro point into the local coordinate array for fv geometry */
      _coor_fv[3*fvindex[m] + 0] = x[0];
      _coor_fv[3*fvindex[m] + 1] = x[1];
      _coor_fv[3*fvindex[m] + 2] = x[2];
    }
    
  }

  ierr = VecRestoreArray(coor_fv,&_coor_fv);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(coor,&_coor);CHKERRQ(ierr);

  ierr = DMLocalToGlobal(dmc_fv,coor_fv,INSERT_VALUES,energy->fv->vertex_coor_geometry);CHKERRQ(ierr);
  
  ierr = FVDAUpdateGeometry(energy->fv);CHKERRQ(ierr);
  
  ierr = PetscFree(fvindex);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode PhysCompEnergyFVInterpolateMacroQ2ToSubQ1(DM dmv,Vec X,PhysCompEnergyFV energy,DM dmv_fv,Vec X_fv)
{
  Vec              Xl,Xl_fv;
  PetscInt         nel,nen,nel_fv,nen_fv,e,k,m,ii,jj,kk,cnt,bs,b;
  const PetscInt   *elnidx,*elnidx_fv;
  PetscReal        elfield[3*Q2_NODES_PER_EL_3D];
  const PetscReal  *_X;
  PetscReal        *_X_fv;
  PetscInt         *fvindex;
  PetscInt         gni[]={0,0,0};
  PetscErrorCode   ierr;
  
  
  PetscFunctionBegin;
  ierr = VecGetBlockSize(X,&bs);CHKERRQ(ierr);
  if (bs > 3) SETERRQ(energy->fv->comm,PETSC_ERR_SUP,"Blocksize > 3 not supported - requires a small patch/fix to be applied (trivial)");
  
  ierr = DMCreateLocalVector(dmv,&Xl);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dmv_fv,&Xl_fv);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dmv,X,INSERT_VALUES,Xl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dmv_fv,X_fv,INSERT_VALUES,Xl_fv);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Xl,&_X);CHKERRQ(ierr);
  ierr = VecGetArray(Xl_fv,&_X_fv);CHKERRQ(ierr);
  
  ierr = PetscCalloc1(energy->npoints_macro,&fvindex);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(dmv_fv,NULL,NULL,NULL,&gni[0],&gni[1],&gni[2]);CHKERRQ(ierr);
  
  ierr = DMDAGetElements_pTatinQ2P1(dmv,&nel,&nen,&elnidx);CHKERRQ(ierr);
  ierr = DMDAGetElements(dmv_fv,&nel_fv,&nen_fv,&elnidx_fv);CHKERRQ(ierr);
  for (e=0; e<nel; e++) {
    PetscInt fv_start[]={0,0,0};
    PetscInt q2_start[]={0,0,0};
    PetscInt fv_cell,start_ijk,fv_i_start[]={0,0,0};
    
    for (k=0; k<Q2_NODES_PER_EL_3D; k++) {
      const PetscInt *element = (const PetscInt*)&elnidx[nen*e];
      for (b=0; b<bs; b++) {
        elfield[bs*k+b] = _X[bs*element[k]+b];
      }
    }
    
    ierr = _cart_convert_index_to_ijk(e,(const PetscInt*)energy->mi_parent,q2_start);CHKERRQ(ierr);
    
    /* determine the macro lower/left corner point */
    fv_start[0] = q2_start[0] * energy->nsubdivision[0];
    fv_start[1] = q2_start[1] * energy->nsubdivision[1];
    fv_start[2] = q2_start[2] * energy->nsubdivision[2];
    
    ierr = _cart_convert_ijk_to_index((const PetscInt*)fv_start,(const PetscInt*)energy->fv->mi,&fv_cell);CHKERRQ(ierr);
    start_ijk = elnidx_fv[8 * fv_cell + 0]; /* 8 is the number of basis per hex, 0 is the first vertex */
    
    ierr = _cart_convert_index_to_ijk(start_ijk,(const PetscInt*)gni,fv_i_start);CHKERRQ(ierr);
    
    cnt = 0;
    for (kk=0; kk<energy->nsubdivision[2]+1; kk++) {
      for (jj=0; jj<energy->nsubdivision[1]+1; jj++) {
        for (ii=0; ii<energy->nsubdivision[0]+1; ii++) {
          fvindex[cnt] = (fv_i_start[0] + ii) + (fv_i_start[1] + jj)*gni[0] + (fv_i_start[2] + kk)*gni[0]*gni[1];
          cnt++;
        }
      }
    }
    
    for (m=0; m<energy->npoints_macro; m++) {
      PetscReal field[]={0,0,0};
      
      /* interpolate Q2 coordinates to macro points */
      for (b=0; b<bs; b++) {
        field[b] = 0;
      }
      for (k=0; k<Q2_NODES_PER_EL_3D; k++) {
        for (b=0; b<bs; b++) {
          field[b] += energy->basis_macro[m][k] * elfield[bs*k+b];
        }
      }
      
      /* insert macro point into the local coordinate array for fv geometry */
      for (b=0; b<bs; b++) {
        _X_fv[bs*fvindex[m] + b] = field[b];
      }
    }
    
  }
  
  ierr = VecRestoreArray(Xl_fv,&_X_fv);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Xl,&_X);CHKERRQ(ierr);
  
  ierr = DMLocalToGlobal(dmv_fv,Xl_fv,INSERT_VALUES,X_fv);CHKERRQ(ierr);

  ierr = VecDestroy(&Xl);CHKERRQ(ierr);
  ierr = VecDestroy(&Xl_fv);CHKERRQ(ierr);
  ierr = PetscFree(fvindex);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/*
 Scatter X to local
 for each face
   get valid element
   get element values
   get face label
   get face normal
   average element values onto face 
   compute dot product with normal
   store
 
*/
PetscErrorCode PhysCompEnergyFVInterpolateNormalVectorToFace(PhysCompEnergyFV energy,Vec X,const char face_field_name[])
{
  const PetscInt  nsd = 3;
  PetscReal       *fielddotn_face;
  PetscInt        f,cellid,i,d;
  FVDA            fv;
  PetscReal       elfield[3*DACELL3D_Q1_SIZE];
  Vec             Xl;
  const PetscReal *_X,*n_face;
  DM              dm;
  PetscInt        dm_nel,dm_nen;
  const PetscInt  *dm_element,*element;
  DACellFace      cell_face_label;
  PetscInt        fidx[DACELL3D_FACE_VERTS];
  PetscErrorCode  ierr;
  
  PetscFunctionBegin;
  fv = energy->fv;
  dm = energy->dmv;
  ierr = FVDAGetFacePropertyByNameArray(fv,face_field_name,&fielddotn_face);CHKERRQ(ierr);
  ierr = FVDAGetFaceInfo(fv,NULL,NULL,NULL,&n_face,NULL);CHKERRQ(ierr);
  
  ierr = DMCreateLocalVector(dm,&Xl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm,X,INSERT_VALUES,Xl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Xl,&_X);CHKERRQ(ierr);
  
  ierr = DMDAGetElements(dm,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  for (f=0; f<fv->nfaces; f++) {
    PetscReal avg_v[] = {0,0,0};
    
    ierr = FVDAGetValidElement(fv,f,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];

    for (i=0; i<DACELL3D_VERTS; i++) {
      for (d=0; d<nsd; d++) {
        elfield[nsd*i+d] = _X[nsd*element[i]+d];
      }
    }
    
    cell_face_label = fv->face_type[f];
    ierr = DACellGeometry3d_GetFaceIndices(NULL,cell_face_label,fidx);CHKERRQ(ierr);
    
    for (i=0; i<DACELL3D_FACE_VERTS; i++) {
      for (d=0; d<nsd; d++) {
        avg_v[d] += elfield[nsd*fidx[i]+d];
      }
    }
    for (d=0; d<nsd; d++) {
      avg_v[d] = avg_v[d] * 0.25; /* four vertices per face in 3D */
    }
    
    fielddotn_face[f] = 0;
    for (d=0; d<nsd; d++) {
      fielddotn_face[f] += avg_v[d] * n_face[3*f+d];
    }
  }
  
  ierr = VecRestoreArrayRead(Xl,&_X);CHKERRQ(ierr);
  ierr = VecDestroy(&Xl);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode PhysCompEnergyFVInterpolateVectorToFace(PhysCompEnergyFV energy,Vec X,const char face_field_name[])
{
  const PetscInt  nsd = 3;
  PetscReal       *fielddotn_face;
  PetscInt        f,cellid,i,d;
  FVDA            fv;
  PetscReal       elfield[3*DACELL3D_Q1_SIZE];
  Vec             Xl;
  const PetscReal *_X;
  DM              dm;
  PetscInt        dm_nel,dm_nen;
  const PetscInt  *dm_element,*element;
  DACellFace      cell_face_label;
  PetscInt        fidx[DACELL3D_FACE_VERTS];
  PetscErrorCode  ierr;
  
  PetscFunctionBegin;
  fv = energy->fv;
  dm = energy->dmv;
  ierr = FVDAGetFacePropertyByNameArray(fv,face_field_name,&fielddotn_face);CHKERRQ(ierr);
  
  ierr = DMCreateLocalVector(dm,&Xl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm,X,INSERT_VALUES,Xl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Xl,&_X);CHKERRQ(ierr);
  
  ierr = DMDAGetElements(dm,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  for (f=0; f<fv->nfaces; f++) {
    PetscReal avg_v[] = {0,0,0};
    
    ierr = FVDAGetValidElement(fv,f,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
    
    for (i=0; i<DACELL3D_VERTS; i++) {
      for (d=0; d<nsd; d++) {
        elfield[nsd*i+d] = _X[nsd*element[i]+d];
      }
    }
    
    cell_face_label = fv->face_type[f];
    ierr = DACellGeometry3d_GetFaceIndices(NULL,cell_face_label,fidx);CHKERRQ(ierr);
    
    for (i=0; i<DACELL3D_FACE_VERTS; i++) {
      for (d=0; d<nsd; d++) {
        avg_v[d] += elfield[nsd*fidx[i]+d];
      }
    }
    for (d=0; d<nsd; d++) {
      avg_v[d] = avg_v[d] * 0.25; /* four vertices per face in 3D */
    }
    
    for (d=0; d<nsd; d++) {
      fielddotn_face[3*f+d] = avg_v[d];
    }
  }
  
  ierr = VecRestoreArrayRead(Xl,&_X);CHKERRQ(ierr);
  ierr = VecDestroy(&Xl);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/* utils */

/*
 mp[] : length 3 : defines number of ranks in each i-j-k direction
 mx[] : length 3 : defines the number of points in i-j-k for the current rank
 _li[] : length mp[0] : defines number of points in each sub-domain in the i-direction
 _lj[] : length mp[1] : defines number of points in each sub-domain in the j-direction
 _lk[] : length mp[2] : defines number of points in each sub-domain in the k-direction
 */
PetscErrorCode _ijk_get_ownership_ranges_3d(MPI_Comm comm,const PetscInt mp[],const PetscInt mx[],
                                            PetscInt *_li[],PetscInt *_lj[],PetscInt *_lk[])
{
  PetscErrorCode ierr;
  PetscInt       mxr[3];
  PetscInt       *li,*lj,*lk;
  PetscMPIInt    r,crank,csize,rij,rijk[3];
  MPI_Status     stat;
  
  
  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&csize);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&crank);CHKERRQ(ierr);
  ierr = PetscCalloc1(mp[0],&li);CHKERRQ(ierr);
  ierr = PetscCalloc1(mp[1],&lj);CHKERRQ(ierr);
  ierr = PetscCalloc1(mp[2],&lk);CHKERRQ(ierr);
  
  if (crank == 0) {
    li[0] = mx[0];
    lj[0] = mx[1];
    lk[0] = mx[2];
    for (r=1; r<csize; r++) {
      ierr = MPI_Recv(mxr,3,MPIU_INT,r,r,comm,&stat);CHKERRQ(ierr);
      
      rijk[2] = r / (mp[0] * mp[1]);
      rij = r - rijk[2] * mp[0] * mp[1];
      rijk[1] = rij/mp[0];
      rijk[0] = rij - rijk[1] * mp[0];
      
      if (r != rijk[0] + rijk[1]*mp[0] + rijk[2]*mp[0]*mp[1]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"rijk conversion failed");
      
      if (rijk[0] > mp[0]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"rijk[0] conversion failed");
      if (rijk[1] > mp[1]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"rijk[1] conversion failed");
      if (rijk[2] > mp[2]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"rijk[2] conversion failed");
      
      if (li[ rijk[0] ] != 0 && li[ rijk[0] ] != mxr[0]) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_USER,"Input in i-direction is inconsistent. Ranks mapping to same i-decomp slot define different values. i-decomp[%d] had value %D received value %D from rank %d",(int)rijk[0],li[ rijk[0] ],mxr[0],(int)r);
      if (lj[ rijk[1] ] != 0 && lj[ rijk[1] ] != mxr[1]) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_USER,"Input in j-direction is inconsistent. Ranks mapping to same j-decomp slot define different values. j-decomp[%d] had value %D received value %D from rank %d",(int)rijk[1],lj[ rijk[1] ],mxr[1],(int)r);
      if (lk[ rijk[2] ] != 0 && lk[ rijk[2] ] != mxr[2]) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_USER,"Input in k-direction is inconsistent. Ranks mapping to same k-decomp slot define different values. k-decomp[%d] had value %D received value %D from rank %d",(int)rijk[2],lk[ rijk[2] ],mxr[2],(int)r);
      
      li[ rijk[0] ] = mxr[0];
      lj[ rijk[1] ] = mxr[1];
      lk[ rijk[2] ] = mxr[2];
    }
  } else {
    ierr = MPI_Send(mx,3,MPIU_INT,0,crank,comm);CHKERRQ(ierr);
  }
  
  ierr = MPI_Bcast(li,mp[0],MPIU_INT,0,comm);CHKERRQ(ierr);
  ierr = MPI_Bcast(lj,mp[1],MPIU_INT,0,comm);CHKERRQ(ierr);
  ierr = MPI_Bcast(lk,mp[2],MPIU_INT,0,comm);CHKERRQ(ierr);
  
  if (_li) { *_li = li; } else { ierr = PetscFree(li);CHKERRQ(ierr); }
  if (_lj) { *_lj = lj; } else { ierr = PetscFree(lj);CHKERRQ(ierr); }
  if (_lk) { *_lk = lk; } else { ierr = PetscFree(lk);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

/*
 Create DMDA which has the desired number of cells in each direction (in parallel) with overlap 1.
 
 Studying the PETSc function
 PetscErrorCode DMDAGetElements_3D(DM dm,PetscInt *nel,PetscInt *nen,const PetscInt *e[])
 we infer the following
 
 Suppose you are told
 (i) There are 3 ranks in direction i
 (ii) On each rank in the i direction we request the following number of elements {mx0,mx1,mx2}.
 
 To define a vertex based DMDA which will yield that many elements in each direction,
 we define the DMDA to have
 * (mx0 + mx1 + mx2) + 1 points in the i direction
 * We set the layout of the points on each rank to be {mx0+1,mx1,mx2}.
 The special item to note is that the left-most rank as +1 points cf the other ranks.
 
*/
PetscErrorCode fvgeometry_dmda3d_create_from_element_partition(MPI_Comm comm,PetscInt target_decomp[],const PetscInt m[],DM *dm)
{
  PetscErrorCode ierr;
  PetscMPIInt    commsize,commsize2,commrank;
  PetscInt       i,*ni,*nj,*nk,M[]={0,0,0};
  PetscInt       *target_mi,*target_mj,*target_mk;
  
  ierr = MPI_Comm_size(comm,&commsize);CHKERRQ(ierr);
  commsize2 = target_decomp[0] * target_decomp[1] * target_decomp[2];
  if (commsize != commsize2) SETERRQ(comm,PETSC_ERR_USER,"Communicator size does not match request i-j-k decomposition");
  
  ierr = _ijk_get_ownership_ranges_3d(comm,target_decomp,m,&target_mi,&target_mj,&target_mk);CHKERRQ(ierr);
  
  for (i=0; i<target_decomp[0]; i++) { M[0] += target_mi[i]; }
  for (i=0; i<target_decomp[1]; i++) { M[1] += target_mj[i]; }
  for (i=0; i<target_decomp[2]; i++) { M[2] += target_mk[i]; }
  M[0]++;
  M[1]++;
  M[2]++;
  
  ierr = PetscMalloc1(target_decomp[0],&ni);CHKERRQ(ierr);
  ierr = PetscMalloc1(target_decomp[1],&nj);CHKERRQ(ierr);
  ierr = PetscMalloc1(target_decomp[2],&nk);CHKERRQ(ierr);
  
  ierr = PetscMemcpy(ni,target_mi,sizeof(PetscInt)*target_decomp[0]);CHKERRQ(ierr);
  ierr = PetscMemcpy(nj,target_mj,sizeof(PetscInt)*target_decomp[1]);CHKERRQ(ierr);
  ierr = PetscMemcpy(nk,target_mk,sizeof(PetscInt)*target_decomp[2]);CHKERRQ(ierr);
  ni[0]++;
  nj[0]++;
  nk[0]++;
  
  ierr = DMDACreate3d(comm,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,
                      M[0],M[1],M[2],
                      target_decomp[0],target_decomp[1],target_decomp[2],
                      3,1, /* [NOTE] stencil width 1 */
                      ni,nj,nk,dm);CHKERRQ(ierr);
  ierr = DMSetUp(*dm);CHKERRQ(ierr);
  
  ierr = DMDASetUniformCoordinates(*dm,0.0,1.0,0.0,1.0,0.0,1.0);CHKERRQ(ierr);
  
  ierr = DMDASetElementType(*dm,DMDA_ELEMENT_Q1);CHKERRQ(ierr);
  {
    PetscInt       nel,nen;
    const PetscInt *e;
    
    ierr = DMDAGetElements(*dm,&nel,&nen,&e);CHKERRQ(ierr);
  }
  
  ierr = MPI_Comm_rank(comm,&commrank);CHKERRQ(ierr);
  
  PetscPrintf(comm,"  M %D x %D x %D\n",M[0]-1,M[1]-1,M[2]-1);
  PetscPrintf(comm,"  N %D x %D x %D\n",M[0],M[1],M[2]);
  
  PetscSynchronizedPrintf(comm,"  [rank %d]\n",(int)commrank);
  PetscSynchronizedPrintf(comm,"    mx,my,mz  %D x %D x %D (local)[input]\n",m[0],m[1],m[2]);
  {
    PetscInt       mi[]={0,0,0};
    ierr = DMDAGetElementsSizes(*dm,&mi[0],&mi[1],&mi[2]);CHKERRQ(ierr);
    PetscSynchronizedPrintf(comm,"    mx,my,mz  %D x %D x %D (local)[output]\n",mi[0],mi[1],mi[2]);
  }
  
  for (i=0; i<target_decomp[0]; i++) {
    PetscSynchronizedPrintf(comm,"      <i-dir %D> ni %D (local)\n",i,ni[i]);
  }
  for (i=0; i<target_decomp[1]; i++) {
    PetscSynchronizedPrintf(comm,"      <j-dir %D> nj %D (local)\n",i,nj[i]);
  }
  for (i=0; i<target_decomp[2]; i++) {
    PetscSynchronizedPrintf(comm,"      <k-dir %D> nk %D (local)\n",i,nk[i]);
  }
  
  PetscSynchronizedFlush(comm,PETSC_STDOUT);
  
  ierr = PetscFree(target_mi);CHKERRQ(ierr);
  ierr = PetscFree(target_mj);CHKERRQ(ierr);
  ierr = PetscFree(target_mk);CHKERRQ(ierr);
  ierr = PetscFree(ni);CHKERRQ(ierr);
  ierr = PetscFree(nj);CHKERRQ(ierr);
  ierr = PetscFree(nk);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode _cart_convert_index_to_ijk(PetscInt r,const PetscInt mp[],PetscInt rijk[])
{
  PetscInt rij;
  PetscFunctionBegin;
  rijk[2] = r / (mp[0] * mp[1]);
  rij = r - rijk[2] * mp[0] * mp[1];
  rijk[1] = rij/mp[0];
  rijk[0] = rij - rijk[1] * mp[0];
  if (r != rijk[0] + rijk[1]*mp[0] + rijk[2]*mp[0]*mp[1]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"_cart_convert_index_to_ijk() conversion failed");
  PetscFunctionReturn(0);
}

PetscErrorCode _cart_convert_ijk_to_index(const PetscInt rijk[],const PetscInt mp[],PetscInt *r)
{
  PetscFunctionBegin;
  *r = rijk[0] + rijk[1]*mp[0] + rijk[2]*mp[0]*mp[1];
  if (*r != rijk[0] + rijk[1]*mp[0] + rijk[2]*mp[0]*mp[1]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"_cart_convert_ijk_to_index() conversion failed");
  PetscFunctionReturn(0);
}

/* ptatin-PhysCompEnergyFV functionality */
PetscErrorCode pTatinPhysCompActivate_EnergyFV(pTatinCtx user,PetscBool load)
{
  PetscErrorCode   ierr;
  PhysCompEnergyFV energy;
  
  PetscFunctionBegin;
  if (load && (user->energyfv_ctx == NULL)) {
    PetscInt nsub[] = {3,3,3};
    
    ierr = PhysCompEnergyFVCreate(PETSC_COMM_WORLD,&energy);CHKERRQ(ierr);
    ierr = PhysCompEnergyFVSetParams(energy,0,0,nsub);CHKERRQ(ierr);
    ierr = PhysCompEnergyFVSetFromOptions(energy);CHKERRQ(ierr);
    ierr = PhysCompEnergyFVSetUp(energy,user);CHKERRQ(ierr);
    
    if (user->restart_from_file) {
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"pTatinPhysCompActivate_EnergyFV should not be called during restart");
    } else {
      ierr = PhysCompAddMaterialPointCoefficients_Energy(user->materialpoint_db);CHKERRQ(ierr);
    }
    
    ierr = PhysCompEnergyFVUpdateGeometry(energy,user->stokes_ctx);CHKERRQ(ierr);
    ierr = FVDAView_CellData(energy->fv,energy->T,PETSC_TRUE,"xcell");CHKERRQ(ierr);
    user->energyfv_ctx = energy;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode pTatinGetContext_EnergyFV(pTatinCtx ctx,PhysCompEnergyFV *e)
{
  PetscFunctionBegin;
  if (e) { *e = ctx->energyfv_ctx; }
  PetscFunctionReturn(0);
}

PetscErrorCode pTatinContextValid_EnergyFV(pTatinCtx ctx,PetscBool *exists)
{
  PetscFunctionBegin;
  *exists = PETSC_FALSE;
  if (ctx->energyfv_ctx) {
    *exists = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode pTatinPhysCompEnergyFV_Initialise(PhysCompEnergyFV e,Vec T)
{
  Vec            gcoor;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  
  /* update solution */
  ierr = VecCopy(T,e->Told);CHKERRQ(ierr);
  
  /* update coordinates */
  ierr = DMGetCoordinates(e->dmv,&gcoor);CHKERRQ(ierr);
  ierr = VecCopy(gcoor,e->Xold);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/* interpolate */

/*
 Input 
 q2_cell:     fe cell index
 q2_m[]:      local size of fe cells in i,j,k
 sub_m[]:     fv sub-division per fe cell
 xi[]:        local coordinated wrt fe cell, xi[] in [-1,1]
 sub_fv_cell: fv sub-cell index (local to fe cell)
 Convert xi[] in [-1,1] into an fv sub-cell index
*/
PetscErrorCode ptatin_macro_point_location_sub(PetscInt q2_cell,const PetscInt q2_m[],
                                               const PetscInt sub_m[],const PetscReal xi[],
                                               PetscInt *sub_fv_cell)
{
  PetscInt  q2[3];
  PetscInt  ii[3];
  PetscReal dxi[3];
  
  /* convert q2_cell index into i,j,k */
  _cart_convert_index_to_ijk(q2_cell,q2_m,q2);
  
  dxi[0] = 2.0 / ((PetscReal)sub_m[0]);
  dxi[1] = 2.0 / ((PetscReal)sub_m[1]);
  dxi[2] = 2.0 / ((PetscReal)sub_m[2]);
  
  /* convert xi index into i,j,k relative to a single macro q2 (sub-divided by fv cells) */
  ii[0] = (PetscInt)((xi[0] + 1.0) / dxi[0]); if (ii[0] == sub_m[0]) { ii[0]--; }
  ii[1] = (PetscInt)((xi[1] + 1.0) / dxi[1]); if (ii[1] == sub_m[1]) { ii[1]--; }
  ii[2] = (PetscInt)((xi[2] + 1.0) / dxi[2]); if (ii[2] == sub_m[2]) { ii[2]--; }
  
  _cart_convert_ijk_to_index(ii,sub_m,sub_fv_cell);
  PetscFunctionReturn(0);
}

/*
 Given a macro Q2 cell (q2_cell), get the indices of all nested FV cells (fv_cell) in terms of rank-local indices
*/
PetscErrorCode ptatin_macro_get_nested_fv_rank_local(PetscInt q2_cell,const PetscInt q2_m[],const PetscInt sub_m[],
                                                     const PetscInt fv_m[],PetscInt fv_cell[])
{
  PetscInt q2[3],fv[3],ii,jj,kk,c;
  
  /* convert q2_cell index into i,j,k */
  _cart_convert_index_to_ijk(q2_cell,q2_m,q2);
  c = 0;
  for (kk=0; kk<sub_m[2]; kk++) {
    for (jj=0; jj<sub_m[1]; jj++) {
      for (ii=0; ii<sub_m[0]; ii++) {
        fv[0] = sub_m[0]*q2[0] + ii;
        fv[1] = sub_m[1]*q2[1] + jj;
        fv[2] = sub_m[2]*q2[2] + kk;
        _cart_convert_ijk_to_index(fv,fv_m,&fv_cell[c]);
        c++;
      }
    }
  }
  PetscFunctionReturn(0);
}

/*
void ProjectFV2MP_P0DG(void) {}
void ProjectFV2MP_P1DG(void) {}
void ProjectFV2MP_Q1CG(void) {}
void ProjectFV2MP_Q1P1CG(void) {}
*/
 
/* coefficients */
#include "material_constants_energy.h"

PetscErrorCode EnergyFVEvaluateCoefficients_MaterialPoints(pTatinCtx user,PetscReal time,PhysCompEnergyFV efv,PetscScalar LA_T[],PetscScalar LA_U[])
{
  PetscErrorCode ierr;
  DataBucket     material_constants,material_points;
  DataField      PField_MatConsts,PField_SourceConst,PField_SourceDecay,PField_SourceAdiAdv,PField_ConductivityConst,PField_ConductivityThreshold;
  DataField      PField_std,PField_energy;
  EnergyMaterialConstants        *mat_consts;
  EnergySourceConst              *source_const;
  EnergySourceDecay              *source_decay;
  EnergySourceAdiabaticAdvection *source_adi_adv;
  EnergyConductivityConst        *k_const;
  EnergyConductivityThreshold    *k_threshold;
  int       pidx,n_mp_points;
  PhysCompStokes stokes;
  PetscReal *grav_vec;
  
  PetscFunctionBegin;
  
  /* Get bucket of material constants */
  ierr = pTatinGetMaterialConstants(user,&material_constants);CHKERRQ(ierr);
  
  ierr = pTatinGetStokesContext(user,&stokes);CHKERRQ(ierr);
  grav_vec = stokes->gravity_vector;
  
  /* fetch array to data for material constants */
  DataBucketGetDataFieldByName(material_constants,EnergyMaterialConstants_classname,&PField_MatConsts);
  DataFieldGetEntries(PField_MatConsts,(void**)&mat_consts);
  
  /* fetch array to data for source method */
  DataBucketGetDataFieldByName(material_constants, EnergySourceConst_classname, &PField_SourceConst );
  DataFieldGetEntries(PField_SourceConst,(void**)&source_const);
  DataBucketGetDataFieldByName(material_constants, EnergySourceDecay_classname, &PField_SourceDecay );
  DataFieldGetEntries(PField_SourceDecay,(void**)&source_decay);
  DataBucketGetDataFieldByName(material_constants, EnergySourceAdiabaticAdvection_classname, &PField_SourceAdiAdv );
  DataFieldGetEntries(PField_SourceAdiAdv,(void**)&source_adi_adv);
  
  /* fetch array to data for conductivity method */
  DataBucketGetDataFieldByName(material_constants, EnergyConductivityConst_classname, &PField_ConductivityConst );
  DataFieldGetEntries(PField_ConductivityConst,(void**)&k_const);
  DataBucketGetDataFieldByName(material_constants, EnergyConductivityThreshold_classname, &PField_ConductivityThreshold );
  DataFieldGetEntries(PField_ConductivityThreshold,(void**)&k_threshold);
  
  /* Get bucket of material points */
  ierr = pTatinGetMaterialPoints(user,&material_points,NULL);CHKERRQ(ierr);
  DataBucketGetSizes(material_points,&n_mp_points,0,0);
  
  DataBucketGetDataFieldByName(material_points,MPntStd_classname,&PField_std);
  DataFieldGetAccess(PField_std);
  DataBucketGetDataFieldByName(material_points,MPntPEnergy_classname,&PField_energy);
  DataFieldGetAccess(PField_energy);
  
  for (pidx=0; pidx<n_mp_points; pidx++) {
    MPntStd       *mp_std;
    MPntPEnergy   *mpp_energy;
    double        *xi_mp,T_mp,u_mp[3];
    int           t,eidx,region_idx;
    double        rho_mp,conductivity_mp,diffusivity_mp,H_mp,Cp;
    int           density_type,conductivity_type;
    int           *source_type;
    
    DataFieldAccessPoint(PField_std,    pidx,(void**)&mp_std);
    DataFieldAccessPoint(PField_energy, pidx,(void**)&mpp_energy);
    
    /* Get index of element containing this marker */
    eidx = mp_std->wil;
    /* Get marker local coordinate (for interpolation) */
    xi_mp = mp_std->xi;
    
    /* Get region index */
    region_idx = mp_std->phase;
    
    T_mp = 0.0;
    
    u_mp[0] = u_mp[1] = u_mp[2] = 0.0;
    
    density_type      = mat_consts[ region_idx ].density_type;
    conductivity_type = mat_consts[ region_idx ].conductivity_type;
    source_type       = mat_consts[ region_idx ].source_type;
    
    /* Fetch value for Cp */
    Cp = mat_consts[ region_idx ].Cp;
    
    /* Compute density */
    rho_mp = 1.0;
    switch (density_type) {
      case ENERGYDENSITY_NONE:
        break;
        
      case ENERGYDENSITY_USE_MATERIALPOINT_VALUE:
        SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"[region %D] ENERGYDENSITY_USE_MATERIALPOINT_VALUE is not available",region_idx);
        break;
        
      case ENERGYDENSITY_CONSTANT:
        rho_mp = mat_consts[ region_idx ].rho_ref;
        break;
        
      case ENERGYDENSITY_BOUSSINESQ:
        SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"[region %D] ENERGYDENSITY_BOUSSINESQ is not available - sorry email GD for help",region_idx);
        break;
    }
    
    /* Compute conductivity */
    conductivity_mp = 1.0;
    switch (conductivity_type) {
      case ENERGYCONDUCTIVITY_NONE:
        SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"[region %D] A valid conductivity type must be specified",region_idx);
        break;
        
      case ENERGYCONDUCTIVITY_USE_MATERIALPOINT_VALUE:
        conductivity_mp = mpp_energy->diffusivity;
        break;
        
      case ENERGYCONDUCTIVITY_CONSTANT:
        conductivity_mp = k_const[ region_idx ].k0;
        break;
        
      case ENERGYCONDUCTIVITY_TEMP_DEP_THRESHOLD:
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"[ENERGYCONDUCTIVITY_TEMP_DEP_THRESHOLD] Not supported with FV - easy to fix - just need to interpolate T to material point");
        /*
         conductivity_mp = k_threshold[ region_idx ].k0;
         if (T_mp >= k_threshold[ region_idx ].T0) {
         conductivity_mp = k_threshold[ region_idx ].k1;
         }
         */
        conductivity_mp = k_threshold[ region_idx ].k0;
        if (T_mp >= k_threshold[ region_idx ].T_threshold) {
          conductivity_mp = k_threshold[ region_idx ].k1;
        } else if (k_threshold[ region_idx ].T_threshold - T_mp < k_threshold[ region_idx ].dT) {
          double shift_T = T_mp - (k_threshold[ region_idx ].T_threshold - k_threshold[ region_idx ].dT);
          double dk = k_threshold[ region_idx ].k1 - k_threshold[ region_idx ].k0;
          
          conductivity_mp = k_threshold[ region_idx ].k0 + (dk/k_threshold[ region_idx ].dT)*shift_T;
        }
        break;
    }
    
    /*
     Compute heat sources
     Note: We want to allow multiple heat sources to exists.
     Presently 6 choices are available, we loop through all
     possible cases and sum the resulting source
     */
    H_mp = 0.0;
    for (t=0; t<7; t++) {
      switch (source_type[t]) {
        case ENERGYSOURCE_NONE:
          break;
          
        case ENERGYSOURCE_USE_MATERIALPOINT_VALUE:
          H_mp += mpp_energy->heat_source;
          break;
          
        case ENERGYSOURCE_CONSTANT:
          H_mp += source_const[ region_idx ].H;
          break;
          
        case ENERGYSOURCE_SHEAR_HEATING:
          SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"[region %D] SHEAR-HEATING is not available",region_idx);
          break;
          
        case ENERGYSOURCE_DECAY:
          H_mp += source_decay[ region_idx ].H0 * exp( -time * source_decay[ region_idx ].lambda );
          break;
          
          /*
           Taken from T. Gerya, "Introduction ot numerical geodynamic modelling"
           page 156-157
           */
        case ENERGYSOURCE_ADIABATIC:
        {
          double g_dot_v; /* g_i * u_i */
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"[ENERGYSOURCE_ADIABATIC] Not supported with FV - easy to fix - just need to interpolate T, V to material point");
          
          //g_dot_v = -(1.0)*u_mp[1]; /* todo - needs to be generalized to use gravity vector */
          
          g_dot_v = -( grav_vec[0]*u_mp[0] + grav_vec[1]*u_mp[1] + grav_vec[2]*u_mp[2] );
          
          H_mp += T_mp * mat_consts[ region_idx ].alpha * rho_mp * g_dot_v;
        }
          break;
          
          /*
           vector u point in the direction of gravity
           */
        case ENERGYSOURCE_ADIABATIC_ADVECTION:
        {
          double grav_nrm,u_vertical;
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"[ENERGYSOURCE_ADIABATIC_ADVECTION] Not supported with FV - easy to fix - just need to interpolate T, V to material point");
          
          //u_vertical = u_mp[1]; /* todo - needs to be generalized to use gravity vector */
          
          grav_nrm = PetscSqrtReal( grav_vec[0]*grav_vec[0] + grav_vec[1]*grav_vec[1] + grav_vec[2]*grav_vec[2] );
          u_vertical = -(u_mp[0]*grav_vec[0] + u_mp[1]*grav_vec[1] + u_mp[2]*grav_vec[2])/grav_nrm;
          
          H_mp += rho_mp * Cp * u_vertical * ( source_adi_adv[ region_idx ].dTdy );
          
        }
          break;
      }
    }
    
    diffusivity_mp = conductivity_mp / (rho_mp * Cp);
    
    H_mp = H_mp / (rho_mp * Cp);
    
    MPntPEnergySetField_diffusivity(mpp_energy,diffusivity_mp);
    MPntPEnergySetField_heat_source(mpp_energy,H_mp);
  }
  
  DataFieldRestoreAccess(PField_std);
  DataFieldRestoreAccess(PField_energy);
  
  PetscFunctionReturn(0);
}

PetscErrorCode EnergyFVEvaluateCoefficients(pTatinCtx user,PetscReal time,PhysCompEnergyFV efv,PetscScalar LA_T[],PetscScalar LA_U[])
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  
  /* Evaluate physics on material points */
  ierr = EnergyFVEvaluateCoefficients_MaterialPoints(user,time,efv,LA_T,LA_U);CHKERRQ(ierr);
  
  /* Project effective diffusivity and source from material points to fv cells and faces */
  
  PetscFunctionReturn(0);
}

PetscErrorCode MaterialPointOrderingCreate_Cellwise(int nkeys,
                                                    int L,const MPntStd point[],
                                                    int offset[],int order[])
{
  int            i,k,sum,r,*cnt = offset;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = PetscMemzero(cnt,sizeof(int)*(nkeys+1));CHKERRQ(ierr);
  ierr = PetscMemzero(order,sizeof(int)*L);CHKERRQ(ierr);
  
  for (i=0; i<L; i++) {
    int _key = point[i].wil;
    
#if defined(PETSC_USE_DEBUG)
    if (_key < 0 || _key >= nkeys) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_USER,"key[%d] = %d outside range [0,%d]",i,point[i].wil,nkeys-1);
#endif
    cnt[_key]++;
  }
  
  /* convert cnt[] into offset[] */
  sum = cnt[0];
  cnt[0] = 0;
  for (k=1; k<nkeys; k++) {
    r = cnt[k];
    cnt[k] = sum;
    sum += r;
  }
  cnt[nkeys] = sum;
  
  /* traverse list and fill */
  for (i=0; i<L; i++) {
    int _key = point[i].wil;
    order[ cnt[_key] ] = i;
    cnt[_key]++; /* convert offset[] into cnt[] */
  }
  
  /* convert cnt[] into offset[] */
  sum = cnt[0];
  cnt[0] = 0;
  for (k=1; k<nkeys; k++) {
    r = cnt[k];
    cnt[k] = sum; /* note that there is no += here, only assignement (=) */
    sum = r;
  }
  cnt[nkeys] = sum;
#if defined(PETSC_USE_DEBUG)
  if (sum != L) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Final offset traversal failed");
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode _FVGetCorners(PhysCompEnergyFV efv,const PetscInt offset[],const PetscInt fvm[],PetscInt corners[])
{
  PetscErrorCode ierr;
  PetscInt i,j,k;
  
  PetscFunctionBegin;
  for (k=0; k<efv->mi_parent[2]; k++) {
    for (j=0; j<efv->mi_parent[1]; j++) {
      for (i=0; i<efv->mi_parent[0]; i++) {
        PetscInt fecellid,feijk[] = {i,j,k};
        PetscInt fvcellid,fvijk[] = {0,0,0};
        
        ierr = _cart_convert_ijk_to_index(feijk,efv->mi_parent,&fecellid);CHKERRQ(ierr);
        
        fvijk[0] = feijk[0] * efv->nsubdivision[0] + offset[0];
        fvijk[1] = feijk[1] * efv->nsubdivision[1] + offset[1];
        fvijk[2] = feijk[2] * efv->nsubdivision[2] + offset[2];
        ierr = _cart_convert_ijk_to_index(fvijk,fvm,&fvcellid);CHKERRQ(ierr);

        corners[fecellid] = fvcellid;
      }
    }
  }
  
  PetscFunctionReturn(0);
}

PetscErrorCode pTatinPhysCompEnergyFV_CreateGetCornersCoefficient(PhysCompEnergyFV efv,PetscInt *n,PetscInt **_c)
{
  PetscErrorCode ierr;
  PetscInt *corners;
  PetscInt offset[] = {0,0,0},fvm[] = {0,0,0};
  PetscInt nfecells;

  PetscFunctionBegin;
  if (!_c) SETERRQ(efv->fv->comm,PETSC_ERR_ARG_NULL,"Arg 2 must be non-NULL");
  nfecells = efv->mi_parent[0]*efv->mi_parent[1]*efv->mi_parent[2];
  if (!*_c) {
    ierr = PetscCalloc1(nfecells,&corners);
    *_c = corners;
  } else {
    corners = *_c;
  }
  
  ierr = DMDAGetCorners(efv->fv->dm_fv,NULL,NULL,NULL,&fvm[0],&fvm[1],&fvm[2]);CHKERRQ(ierr);
  ierr = _FVGetCorners(efv,(const PetscInt*)offset,(const PetscInt*)fvm,corners);CHKERRQ(ierr);
  if (n) { *n = nfecells; }
  PetscFunctionReturn(0);
}

PetscErrorCode pTatinPhysCompEnergyFV_CreateGetCornersFVCell(PhysCompEnergyFV efv,PetscInt *n,PetscInt **_c)
{
  PetscErrorCode ierr;
  PetscInt *corners;
  PetscInt fv_start[3],fv_start_local[3],fv_ghost_offset[3],fv_ghost_range[3];
  PetscInt nfecells;
  
  PetscFunctionBegin;
  if (!_c) SETERRQ(efv->fv->comm,PETSC_ERR_ARG_NULL,"Arg 2 must be non-NULL");
  nfecells = efv->mi_parent[0]*efv->mi_parent[1]*efv->mi_parent[2];
  if (!*_c) {
    ierr = PetscCalloc1(nfecells,&corners);
    *_c = corners;
  } else {
    corners = *_c;
  }

  ierr = DMDAGetCorners(efv->fv->dm_fv,&fv_start[0],&fv_start[1],&fv_start[2],NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(efv->fv->dm_fv,&fv_start_local[0],&fv_start_local[1],&fv_start_local[2],&fv_ghost_range[0],&fv_ghost_range[1],&fv_ghost_range[2]);CHKERRQ(ierr);
  fv_ghost_offset[0] = fv_start[0] - fv_start_local[0];
  fv_ghost_offset[1] = fv_start[1] - fv_start_local[1];
  fv_ghost_offset[2] = fv_start[2] - fv_start_local[2];
  ierr = _FVGetCorners(efv,(const PetscInt*)fv_ghost_offset,(const PetscInt*)fv_ghost_range,corners);CHKERRQ(ierr);
  if (n) { *n = nfecells; }
  PetscFunctionReturn(0);
}

/*
 
 Project s
   mp.diffusivity => fv.k(cell)
   mp.H => fv.H(cell)
 Set 
   fv.rhoCp(cell) = 1
 Compute
   fv.k(face) from fv.k(cell) using harmonic averaging
*/
PetscErrorCode pTatinPhysCompEnergyFV_MPProjection_MacroP0DG(void)
{
  
  PetscFunctionReturn(0);
}

/*
 
 sort points based on fe cell index (wil)
 for each fe cell, e
   collect global coords for points in e
   collect local coords for points in e
   convert local coords into fv cell index (wrt to macro cell)
   insert global coords into kdtree
   for each fv sub-cell
     perform distance distance between sub-cell centroid and point global coords
     assign k, H based on nearest point
 
*/
PetscErrorCode pTatinPhysCompEnergyFV_MPProjection_FVP0DG(PhysCompEnergyFV efv,pTatinCtx pctx)
{
  PetscInt        e,se,nfecells,*fvcorners = NULL;
  DataBucket      point_db;
  DataField       pfield,pfield_energy;
  const MPntStd   *point;
  MPntPEnergy     *point_energy;
  int             npoints,*order,*offset;
  PetscLogDouble  t0,t1,cumtime[] = {0,0,0,0},st0,st1;
  double          *subfv_coor;
  PetscInt        *subfv_idx,fv_m[] = {0,0,0},nsubfv;
  Vec             fv_local_x;
  const PetscReal *_fv_local_x;
  KDTree          kdtree;
  PetscReal       *fv_cell_k,*fv_cell_h;
  PetscErrorCode  ierr;
  
  PetscFunctionBegin;
  nfecells = efv->mi_parent[0]*efv->mi_parent[1]*efv->mi_parent[2];
  ierr = pTatinPhysCompEnergyFV_CreateGetCornersCoefficient(efv,&nfecells,&fvcorners);CHKERRQ(ierr);

  ierr = pTatinGetMaterialPoints(pctx,&point_db,NULL);CHKERRQ(ierr);
  DataBucketGetSizes(point_db,&npoints,NULL,NULL);
  DataBucketGetDataFieldByName(point_db,MPntStd_classname,&pfield);
  DataFieldGetEntries(pfield,(void**)&point);
  DataBucketGetDataFieldByName(point_db,MPntPEnergy_classname,&pfield_energy);
  DataFieldGetEntries(pfield_energy,(void**)&point_energy);

  ierr = PetscCalloc1((nfecells+1),&offset);CHKERRQ(ierr);
  ierr = PetscCalloc1(npoints,&order);CHKERRQ(ierr);

  PetscTime(&t0);
  ierr = MaterialPointOrderingCreate_Cellwise((int)nfecells,npoints,point,offset,order);CHKERRQ(ierr);
  PetscTime(&t1);
  printf("[MaterialPointOrderingCreate_Cellwise] time %1.2e (sec)\n",t1-t0);
  
  nsubfv = efv->nsubdivision[0] * efv->nsubdivision[1] * efv->nsubdivision[2];
  ierr = PetscCalloc1(nsubfv*3,&subfv_coor);CHKERRQ(ierr);
  ierr = PetscCalloc1(nsubfv,&subfv_idx);CHKERRQ(ierr);
  fv_m[0] = efv->mi_parent[0] * efv->nsubdivision[0];
  fv_m[1] = efv->mi_parent[1] * efv->nsubdivision[1];
  fv_m[2] = efv->mi_parent[2] * efv->nsubdivision[2];
  
  ierr = DMGetCoordinates(efv->fv->dm_fv,&fv_local_x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(fv_local_x,&_fv_local_x);CHKERRQ(ierr);
  
  ierr = FVDAGetCellPropertyByNameArray(efv->fv,"k",&fv_cell_k);CHKERRQ(ierr);
  ierr = FVDAGetCellPropertyByNameArray(efv->fv,"H",&fv_cell_h);CHKERRQ(ierr);

  
  KDTreeCreate(3,&kdtree);
  
  PetscTime(&t0);
  for (e=0; e<nfecells; e++) {
    int     ppc,p,start,end;
    kd_node node;
    
    /* get indices */
    ierr = ptatin_macro_get_nested_fv_rank_local(e,(const PetscInt*)efv->mi_parent,
                                                 (const PetscInt*)efv->nsubdivision,
                                                 (const PetscInt*)fv_m,subfv_idx);CHKERRQ(ierr);
    /*
    printf("cell %d: fv-corner %d\n",e,fvcorners[e]);
    for (se=0; se<nsubfv; se++) {
      printf("  sub[%d] -> %d\n",se,subfv_idx[se]);
    }
    */
    for (se=0; se<nsubfv; se++) {
      PetscInt d;
      
      /* fill coords */
      for (d=0; d<3; d++) {
        subfv_coor[3*se+d] = (double)_fv_local_x[3*subfv_idx[se]+d];
      }
    }

    start = offset[e];
    end   = offset[e+1];
    ppc = end - start;

    /* set points */
    KDTreeSetPoints(kdtree,ppc);
    
    /* fill points + labels */
    PetscTime(&st0);
    KDTreeGetPoints(kdtree,NULL,&node);
    for (p=start; p<end; p++) {
      node[p-start].x[0]  = point[p].coor[0];
      node[p-start].x[1]  = point[p].coor[1];
      node[p-start].x[2]  = point[p].coor[2];
      node[p-start].index = p;
    }
    PetscTime(&st1);
    cumtime[0] += (st1 - st0);
    
    /* setup */
    PetscTime(&st0);
    KDTreeSetup(kdtree);
    PetscTime(&st1);
    cumtime[1] += (st1 - st0);
    
    
    /* get nearest and assign props */
    PetscTime(&st0);
    for (se=0; se<nsubfv; se++) {
      kd_node nearest;
      double  *target = &subfv_coor[3*se];
      
      KDTreeFindNearest(kdtree,target,&nearest,NULL);
      
      fv_cell_k[ subfv_idx[se] ] = point_energy[nearest->index].diffusivity;
      fv_cell_h[ subfv_idx[se] ] = point_energy[nearest->index].heat_source;
    }
    PetscTime(&st1);
    cumtime[2] += (st1 - st0);
    
    
    PetscTime(&st0);
    PetscTime(&st1);
    cumtime[3] += (st1 - st0);
    
    
    /* reset */
    KDTreeReset(kdtree);
    
  }
  PetscTime(&t1);
  printf("[kdtree][fill points] time %1.2e (sec)\n",cumtime[0]);
  printf("[kdtree][setup] time %1.2e (sec)\n",cumtime[1]);
  printf("[kdtree][get nearest] time %1.2e (sec)\n",cumtime[2]);
  printf("[kdtree][assign props] time %1.2e (sec)\n",cumtime[3]);
  printf("[FindNearest & Assign] time %1.2e (sec)\n",t1-t0);

  KDTreeDestroy(&kdtree);
  ierr = VecRestoreArrayRead(fv_local_x,&_fv_local_x);CHKERRQ(ierr);
  ierr = PetscFree(subfv_idx);CHKERRQ(ierr);
  ierr = PetscFree(subfv_coor);CHKERRQ(ierr);
  
  DataFieldRestoreEntries(pfield,(void**)&point);
  DataFieldRestoreEntries(pfield_energy,(void**)&point_energy);
  ierr = PetscFree(fvcorners);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode pTatinPhysCompEnergyFV_MPProjection(PhysCompEnergyFV efv,pTatinCtx pctx)
{
  PetscErrorCode ierr;
  
  ierr = pTatinPhysCompEnergyFV_MPProjection_FVP0DG(efv,pctx);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/* ale */

PetscErrorCode pTatinPhysCompEnergyFV_ComputeALESource(FVDA fv,Vec xk,Vec xk1,PetscReal dt,Vec S,PetscBool forward)
{
  PetscErrorCode  ierr;
  Vec               geometry_coorl,geometry_target_coorl;
  const PetscScalar *_geom_coor,*_geom_target_coor;
  PetscInt          c,row,offset;
  PetscInt          dm_nel,dm_nen;
  const PetscInt    *dm_element,*element;
  PetscReal         cell_coor[3*DACELL3D_VERTS],dV0,dV1;
  
  
  ierr = VecZeroEntries(S);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(S,&offset,NULL);CHKERRQ(ierr);
  
  ierr = DMGetLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_geometry,xk,INSERT_VALUES,geometry_coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
  
  ierr = DMGetLocalVector(fv->dm_geometry,&geometry_target_coorl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_geometry,xk1,INSERT_VALUES,geometry_target_coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(geometry_target_coorl,&_geom_target_coor);CHKERRQ(ierr);
  
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  for (c=0; c<fv->ncells; c++) {
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * c];
    
    ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateCellVolume3d(cell_coor,&dV0);
    
    ierr = DACellGeometry3d_GetCoordinates(element,_geom_target_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateCellVolume3d(cell_coor,&dV1);
    
    row = offset + c;
    /* forward divides by dV0, backward divides by dV1 */
    if (forward) {
      ierr = VecSetValue(S,row,((dV1-dV0)/dt)/dV0,INSERT_VALUES);CHKERRQ(ierr);
    } else {
      ierr = VecSetValue(S,row,((dV1-dV0)/dt)/dV1,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  
  ierr = VecRestoreArrayRead(geometry_target_coorl,&_geom_target_coor);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(fv->dm_geometry,&geometry_target_coorl);CHKERRQ(ierr);
  
  ierr = VecRestoreArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
  
  ierr = VecAssemblyBegin(S);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(S);CHKERRQ(ierr);
  
  //printf("[source \\int_V]\n");
  //VecView(S,PETSC_VIEWER_STDOUT_WORLD);
  
  PetscFunctionReturn(0);
}

/*
 Computes
 v = (x1 - x0)/dt
 */
PetscErrorCode pTatinPhysCompEnergyFV_ComputeALEVelocity(DM dmg,Vec x0,Vec x1,PetscReal dt,Vec v)
{
  PetscErrorCode  ierr;
  PetscInt        k,d,len;
  const PetscReal *_x0,*_x1;
  PetscReal       *_v;
  
  ierr = VecGetLocalSize(x0,&len);CHKERRQ(ierr);
  len = len / 3;
  ierr = VecGetArrayRead(x0,&_x0);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x1,&_x1);CHKERRQ(ierr);
  ierr = VecGetArray(v,&_v);CHKERRQ(ierr);
  for (k=0; k<len; k++) {
    for (d=0; d<3; d++) {
      _v[3*k+d] = (_x1[3*k+d] - _x0[3*k+d])/dt;
    }
  }
  ierr = VecRestoreArray(v,&_v);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x1,&_x1);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x0,&_x0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode pTatinPhysCompEnergyFV_ComputeAdvectiveTimestep(PhysCompEnergyFV energy,Vec X,PetscReal *_dt)
{
  const PetscInt  nsd = 3;
  PetscInt        cellid,i,d;
  FVDA            fv;
  PetscReal       elfield[3*DACELL3D_Q1_SIZE],*k_cell;
  Vec             Xl,geometry_coorl;
  const PetscReal *_X,*_geom_coor;
  DM              dm;
  PetscInt        dm_nel,dm_nen;
  const PetscInt  *dm_element,*element;
  PetscReal       dl[3],dh,dt_a = 1.0e32,dt_d = 1.0e32,dt;
  const PetscReal eps = 1.0e-32;
  PetscReal       cell_coor[3*DACELL3D_VERTS];
  PetscErrorCode  ierr;
  
  PetscFunctionBegin;
  fv = energy->fv;
  dm = energy->dmv;
  
  ierr = DMCreateLocalVector(dm,&Xl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm,X,INSERT_VALUES,Xl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Xl,&_X);CHKERRQ(ierr);
  
  ierr = DMGetLocalVector(energy->fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(energy->fv->dm_geometry,energy->fv->vertex_coor_geometry,INSERT_VALUES,geometry_coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);

  ierr = FVDAGetCellPropertyByNameArray(energy->fv,"k",&k_cell);CHKERRQ(ierr);
  ierr = DMDAGetElements(dm,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  for (cellid=0; cellid<fv->ncells; cellid++) {
    PetscReal avg_v[] = {0,0,0};
    
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
    
    ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);

    {
      const PetscReal xi[] = {0,0,0};
      PetscReal       pA[3],pB[3];
      
      dl[0] = dl[1] = dl[2] = 0.0;
      ierr = _EvaluateFaceCoord3d(DACELL_FACE_E,(const PetscReal*)cell_coor,xi,pB);CHKERRQ(ierr);
      ierr = _EvaluateFaceCoord3d(DACELL_FACE_W,(const PetscReal*)cell_coor,xi,pA);CHKERRQ(ierr);
      // east-west
      for (d=0; d<nsd; d++) { dl[0] += (pB[d] - pA[d])*(pB[d] - pA[d]); }
      
      ierr = _EvaluateFaceCoord3d(DACELL_FACE_N,(const PetscReal*)cell_coor,xi,pB);CHKERRQ(ierr);
      ierr = _EvaluateFaceCoord3d(DACELL_FACE_S,(const PetscReal*)cell_coor,xi,pA);CHKERRQ(ierr);
      // north-south
      for (d=0; d<nsd; d++) { dl[1] += (pB[d] - pA[d])*(pB[d] - pA[d]); }

      ierr = _EvaluateFaceCoord3d(DACELL_FACE_F,(const PetscReal*)cell_coor,xi,pB);CHKERRQ(ierr);
      ierr = _EvaluateFaceCoord3d(DACELL_FACE_B,(const PetscReal*)cell_coor,xi,pA);CHKERRQ(ierr);
      // front-back
      for (d=0; d<nsd; d++) { dl[2] += (pB[d] - pA[d])*(pB[d] - pA[d]); }
    }
    dh = PetscMin(dl[0],dl[1]);
    dh = PetscMin(dh,dl[2]);
    dh = PetscSqrtReal(dh);
    
    //printf("cell %d dl %g %g %g dh %g\n",cellid,dl[0],dl[1],dl[2],dh);
    
    for (i=0; i<DACELL3D_VERTS; i++) {
      for (d=0; d<nsd; d++) {
        elfield[nsd*i+d] = _X[nsd*element[i]+d];
      }
    }
    
    for (i=0; i<DACELL3D_VERTS; i++) {
      for (d=0; d<nsd; d++) {
        avg_v[d] += elfield[nsd*i+d];
      }
    }
    for (d=0; d<nsd; d++) {
      avg_v[d] = PetscAbsReal(avg_v[d]);
      avg_v[d] = avg_v[d] * 0.125; /* eight vertices per cell in 3D */
    }
    
    for (d=0; d<nsd; d++) {
      dt_a = PetscMin(dt_a,dh/(avg_v[d] + eps));
    }
    dt_d = PetscMin(dt_d,dh*dh/(k_cell[cellid] + eps));
    //printf("dt_a %g dt_d %g \n",dt_a,dt_d);
  }
  dt = PetscMin(dt_a,dt_d);
  
  ierr = VecRestoreArrayRead(Xl,&_X);CHKERRQ(ierr);
  ierr = VecDestroy(&Xl);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);

  ierr = MPI_Allreduce(MPI_IN_PLACE,&dt,1,MPIU_REAL,MPIU_MIN,PetscObjectComm((PetscObject)X));CHKERRQ(ierr);
  *_dt = dt;
  
  PetscFunctionReturn(0);
}




