
#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <fvda_impl.h>
#include <fvda.h>
#include <fvda_utils.h>


PetscErrorCode eval_F_central_local(FVDA fv,const PetscReal domain_geom_coor[],const PetscReal X[],PetscReal F[]);

PetscErrorCode eval_F_upwind_local(FVDA fv,const PetscReal domain_geom_coor[],const PetscReal X[],PetscReal F[]);
PetscErrorCode eval_F_diffusion_7point_local(FVDA fv,const PetscReal domain_geom_coor[],const PetscReal fv_coor[],const PetscReal X[],PetscReal F[]);

PetscErrorCode eval_F_upwind_hr_local(FVDA fv,const PetscReal domain_geom_coor[],const PetscReal fv_coor[],const PetscReal X[],PetscReal F[]);
PetscErrorCode eval_F_diffusion_7point_hr_local_store_MPI(FVDA fv,const PetscReal domain_geom_coor[],const PetscReal fv_coor[],const PetscReal X[],PetscReal F[]);

PetscErrorCode eval_J_upwind_local(FVDA fv,const PetscReal domain_geom_coor[],const PetscReal X[],Mat J);
PetscErrorCode eval_J_diffusion_7point_local(FVDA fv,const PetscReal domain_geom_coor[],const PetscReal fv_coor[],const PetscReal X[],Mat J);

PetscErrorCode eval_F_upwind_hr_bound_local(FVDA fv,const PetscReal range[],const PetscReal domain_geom_coor[],const PetscReal fv_coor[],const PetscReal X[],PetscReal F[]);

static PetscErrorCode FVDADestroy_ALE(FVDA fv);


PetscErrorCode FVDASetup_ALE(FVDA fv)
{
  PetscErrorCode ierr;
  FVALE          ctx = NULL;

  
  PetscFunctionBegin;
  if (!fv->setup) SETERRQ(fv->comm,PETSC_ERR_ORDER,"Must call FVDASetup_ALE() after FVDASetUp()");
  if (fv->ctx) SETERRQ(PetscObjectComm((PetscObject)fv->dm_fv),PETSC_ERR_USER,"FVDA has a non-NULL context already");
  if (!fv->q_dot) SETERRQ(PetscObjectComm((PetscObject)fv->dm_fv),PETSC_ERR_USER,"FVDA solver must be configured as time-dependent to use this residual evaluator");

  ierr = PetscMalloc(sizeof(struct _p_FVALE),&ctx);CHKERRQ(ierr);
  ierr = PetscMemzero(ctx,sizeof(struct _p_FVALE));CHKERRQ(ierr);
  ctx->dt = 0;
  ierr = DMCreateGlobalVector(fv->dm_fv,&ctx->Q_k);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(fv->dm_geometry,&ctx->vertex_coor_geometry_target);CHKERRQ(ierr);
  fv->ctx = (void*)ctx;
  fv->ctx_destroy = FVDADestroy_ALE;
  PetscFunctionReturn(0);
}

PetscErrorCode FVDAAccessData_ALE(FVDA fv,PetscReal **dt,Vec *Qk,Vec *coor_target)
{
  FVALE ctx = NULL;
  
  
  PetscFunctionBegin;
  if (!fv->setup) SETERRQ(fv->comm,PETSC_ERR_ORDER,"Must call FVDAAccessData_ALE() after FVDASetUp()");
  if (!fv->ctx) SETERRQ(PetscObjectComm((PetscObject)fv->dm_fv),PETSC_ERR_USER,"FVDA has a NULL context. Must call FVDASetupTimeDep() first");
  ctx = (FVALE)fv->ctx;
  if (dt) {
    *dt = &ctx->dt;
  }
  if (Qk) {
    *Qk = ctx->Q_k;
  }
  if (coor_target) {
    *coor_target = ctx->vertex_coor_geometry_target;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode FVDADestroy_ALE(FVDA fv)
{
  PetscErrorCode ierr;
  FVALE          ctx = NULL;
  
  
  PetscFunctionBegin;
  if (!fv->setup) SETERRQ(fv->comm,PETSC_ERR_ORDER,"Must call FVDADestroy_ALE() after FVDASetUp()");
  if (!fv->ctx) SETERRQ(PetscObjectComm((PetscObject)fv->dm_fv),PETSC_ERR_USER,"FVDA has a NULL context. Must call FVDASetupTimeDep() first");
  ctx = (FVALE)fv->ctx;
  ierr = VecDestroy(&ctx->vertex_coor_geometry_target);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->Q_k);CHKERRQ(ierr);
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  fv->ctx = NULL;
  PetscFunctionReturn(0);
}

/*
Q^k+1 - Q^k + F
*/
PetscErrorCode fvda_eval_F_forward_ale(SNES snes,Vec X,Vec F,void *data)
{
  PetscErrorCode    ierr;
  Vec               Xl,Fl,coorl,geometry_coorl,geometry_target_coorl;
  const PetscScalar *_X,*_Xk,*_fv_coor,*_geom_coor,*_geom_target_coor;
  PetscScalar       *_F;
  DM                dm;
  FVDA              fv = NULL;
  FVALE             ctx = NULL;
  PetscInt          k,m,c;
  PetscInt          dm_nel,dm_nen;
  const PetscInt    *dm_element,*element;
  PetscReal         cell_coor[3*DACELL3D_VERTS],dV0,dV1;

  
  PetscFunctionBegin;
  ierr = SNESGetApplicationContext(snes,(void*)&fv);CHKERRQ(ierr);
  if (!fv->q_dot) SETERRQ(PetscObjectComm((PetscObject)fv->dm_fv),PETSC_ERR_USER,"FVDA must be configured as time-dependent/ALE to use this residual evaluator");
  ctx = (FVALE)fv->ctx;
  if (!ctx) SETERRQ(PetscObjectComm((PetscObject)fv->dm_fv),PETSC_ERR_USER,"FVDA has a NULL context. Must call FVDASetup_ALE() first");
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  dm = fv->dm_fv;
  
  ierr = DMGetLocalVector(dm,&Xl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm,X,INSERT_VALUES,Xl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Xl,&_X);CHKERRQ(ierr);
  
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&Fl);CHKERRQ(ierr);
  ierr = VecZeroEntries(Fl);CHKERRQ(ierr);
  ierr = VecGetArray(Fl,&_F);CHKERRQ(ierr);
  
  ierr = DMGetCoordinatesLocal(dm,&coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coorl,&_fv_coor);CHKERRQ(ierr);
  
  ierr = DMGetLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_geometry,fv->vertex_coor_geometry,INSERT_VALUES,geometry_coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);

  ierr = DMGetLocalVector(fv->dm_geometry,&geometry_target_coorl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_geometry,ctx->vertex_coor_geometry_target,INSERT_VALUES,geometry_target_coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(geometry_target_coorl,&_geom_target_coor);CHKERRQ(ierr);
  
  ierr = VecGetSize(Fl,&m);CHKERRQ(ierr);
  {
    if (fv->equation_type == FVDA_HYPERBOLIC || fv->equation_type == FVDA_PARABOLIC) {
      ierr = eval_F_upwind_local(fv,_geom_coor,_X,_F);CHKERRQ(ierr);
      //ierr = eval_F_central_local(fv,_geom_coor,_X,_F);CHKERRQ(ierr);
      {
        for (k=0; k<m; k++) {
          _F[k] *= -1.0;
        }
      }
    }
    
    if (fv->equation_type == FVDA_ELLIPTIC|| fv->equation_type == FVDA_PARABOLIC) {
      ierr = eval_F_diffusion_7point_local(fv,_geom_coor,_fv_coor,_X,_F);CHKERRQ(ierr);
    }

    if (fv->equation_type == FVDA_ADV_DIFF) {
      ierr = eval_F_upwind_diffusion_7point_local(fv,_geom_coor,_fv_coor,_X,_F);CHKERRQ(ierr);
    }
  }
 
  /* scale everything by dt, put in volume contributions */
  for (k=0; k<m; k++) {
    _F[k] *= ctx->dt;
  }
  
  ierr = VecRestoreArrayRead(coorl,&_fv_coor);CHKERRQ(ierr);
  ierr = VecRestoreArray(Fl,&_F);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Xl,&_X);CHKERRQ(ierr);
  
  ierr = DMLocalToGlobal(dm,Fl,INSERT_VALUES,F);CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(dm,&Fl);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Xl);CHKERRQ(ierr);
  
  ierr = VecGetArray(F,&_F);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&_X);CHKERRQ(ierr);
  ierr = VecGetArrayRead(ctx->Q_k,&_Xk);CHKERRQ(ierr);
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  for (c=0; c<fv->ncells; c++) {
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * c];
    
    ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateCellVolume3d(cell_coor,&dV0);

    ierr = DACellGeometry3d_GetCoordinates(element,_geom_target_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateCellVolume3d(cell_coor,&dV1);

    //_F[c] = _F[c] - (_X[c] - _Xk[c]) * dV; /* from time-dep as a reference */
    _F[c] = -_F[c] + (_X[c] * dV1 - _Xk[c] * dV0);
  }
  ierr = VecRestoreArrayRead(ctx->Q_k,&_Xk);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&_X);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&_F);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(geometry_target_coorl,&_geom_target_coor);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(fv->dm_geometry,&geometry_target_coorl);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode fvda_eval_J_forward_ale(SNES snes,Vec X,Mat Ja,Mat Jb,void *data)
{
  PetscErrorCode    ierr;
  Vec               Xl,coorl,geometry_coorl,geometry_target_coorl;
  const PetscScalar *_X,*_fv_coor,*_geom_coor,*_geom_target_coor;
  DM                dm;
  FVDA              fv = NULL;
  FVALE             ctx = NULL;
  PetscInt          c,row,offset;
  PetscInt          dm_nel,dm_nen;
  const PetscInt    *dm_element,*element;
  PetscReal         cell_coor[3*DACELL3D_VERTS],dV0,dV1;

  
  PetscFunctionBegin;
  ierr = SNESGetApplicationContext(snes,(void*)&fv);CHKERRQ(ierr);
  if (!fv->q_dot) SETERRQ(PetscObjectComm((PetscObject)fv->dm_fv),PETSC_ERR_USER,"FVDA must be configured as time-dependent/ALE to use this Jacobain evaluator");
  ctx = (FVALE)fv->ctx;
  if (!ctx) SETERRQ(PetscObjectComm((PetscObject)fv->dm_fv),PETSC_ERR_USER,"FVDA has a NULL context. Must call FVDASetup_ALE() first");
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  dm = fv->dm_fv;
  
  ierr = DMGetLocalVector(dm,&Xl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm,X,INSERT_VALUES,Xl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Xl,&_X);CHKERRQ(ierr);
  
  ierr = MatZeroEntries(Jb);CHKERRQ(ierr);
  
  ierr = DMGetCoordinatesLocal(dm,&coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coorl,&_fv_coor);CHKERRQ(ierr);
  
  ierr = DMGetLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_geometry,fv->vertex_coor_geometry,INSERT_VALUES,geometry_coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
  
  ierr = DMGetLocalVector(fv->dm_geometry,&geometry_target_coorl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_geometry,ctx->vertex_coor_geometry_target,INSERT_VALUES,geometry_target_coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(geometry_target_coorl,&_geom_target_coor);CHKERRQ(ierr);

  {
    if (fv->equation_type == FVDA_HYPERBOLIC || fv->equation_type == FVDA_PARABOLIC) {
      ierr = eval_J_upwind_local(fv,_geom_coor,_X,Jb);CHKERRQ(ierr);
    }
    
    if (fv->equation_type == FVDA_ELLIPTIC|| fv->equation_type == FVDA_PARABOLIC) {
      ierr = eval_J_diffusion_7point_local(fv,_geom_coor,_fv_coor,_X,Jb);CHKERRQ(ierr);
    }

    if (fv->equation_type == FVDA_ADV_DIFF) {
      ierr = eval_J_upwind_diffusion_7point_local(fv,_geom_coor,_fv_coor,_X,Jb);CHKERRQ(ierr);
    }
  }
  
  ierr = VecRestoreArrayRead(coorl,&_fv_coor);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Xl,&_X);CHKERRQ(ierr);
  
  ierr = MatAssemblyBegin(Jb,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Jb,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  //
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Jb,&offset,NULL);CHKERRQ(ierr);
  
  /*
  // 1
  // _F[c] = _F[c] - (_X[c] * dV1 - _Xk[c] * dV0)
  ierr = MatScale(Jb,ctx->dt);CHKERRQ(ierr);
  for (c=0; c<fv->ncells; c++) {
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * c];
    ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateCellVolume3d(cell_coor,&dV);
    
    row = offset + c;
    ierr = MatSetValue(Jb,row,row,-dV,ADD_VALUES);CHKERRQ(ierr);
  }
  */
  
  // 2
  // _F[c] = -_F[c] + (_X[c] * dV1 - _Xk[c] * dV0);
  ierr = MatScale(Jb,-ctx->dt);CHKERRQ(ierr);
  for (c=0; c<fv->ncells; c++) {
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * c];
    
    ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateCellVolume3d(cell_coor,&dV0);
    
    ierr = DACellGeometry3d_GetCoordinates(element,_geom_target_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateCellVolume3d(cell_coor,&dV1);
    
    row = offset + c;
    ierr = MatSetValue(Jb,row,row,dV1,ADD_VALUES);CHKERRQ(ierr);
  }
  
  ierr = MatAssemblyBegin(Jb,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Jb,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  if (Ja != Jb) {
    ierr = MatAssemblyBegin(Ja,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Ja,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  
  ierr = DMRestoreLocalVector(dm,&Xl);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(geometry_target_coorl,&_geom_target_coor);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(fv->dm_geometry,&geometry_target_coorl);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode fvda_eval_F_backward_ale(SNES snes,Vec X,Vec F,void *data)
{
  PetscErrorCode    ierr;
  Vec               Xl,Fl,coorl,geometry_coorl,geometry_target_coorl;
  const PetscScalar *_X,*_Xk,*_fv_coor,*_geom_coor,*_geom_target_coor;
  PetscScalar       *_F;
  DM                dm;
  FVDA              fv = NULL;
  FVALE             ctx = NULL;
  PetscInt          k,m,c;
  PetscInt          dm_nel,dm_nen;
  const PetscInt    *dm_element,*element;
  PetscReal         cell_coor[3*DACELL3D_VERTS],dV0,dV1;
  
  
  PetscFunctionBegin;
  ierr = SNESGetApplicationContext(snes,(void*)&fv);CHKERRQ(ierr);
  if (!fv->q_dot) SETERRQ(PetscObjectComm((PetscObject)fv->dm_fv),PETSC_ERR_USER,"FVDA must be configured as time-dependent/ALE to use this residual evaluator");
  ctx = (FVALE)fv->ctx;
  if (!ctx) SETERRQ(PetscObjectComm((PetscObject)fv->dm_fv),PETSC_ERR_USER,"FVDA has a NULL context. Must call FVDASetup_ALE() first");
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  dm = fv->dm_fv;

  ierr = DMGetLocalVector(dm,&Xl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm,X,INSERT_VALUES,Xl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Xl,&_X);CHKERRQ(ierr);
  
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&Fl);CHKERRQ(ierr);
  ierr = VecZeroEntries(Fl);CHKERRQ(ierr);
  ierr = VecGetArray(Fl,&_F);CHKERRQ(ierr);
  
  ierr = DMGetCoordinatesLocal(dm,&coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coorl,&_fv_coor);CHKERRQ(ierr);
  
  ierr = DMGetLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_geometry,fv->vertex_coor_geometry,INSERT_VALUES,geometry_coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
  
  ierr = DMGetLocalVector(fv->dm_geometry,&geometry_target_coorl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_geometry,ctx->vertex_coor_geometry_target,INSERT_VALUES,geometry_target_coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(geometry_target_coorl,&_geom_target_coor);CHKERRQ(ierr);
  
  
  ierr = VecGetSize(Fl,&m);CHKERRQ(ierr);
  {
    if (fv->equation_type == FVDA_HYPERBOLIC || fv->equation_type == FVDA_PARABOLIC) {
      ierr = eval_F_upwind_local(fv,_geom_target_coor,_X,_F);CHKERRQ(ierr);
      //ierr = eval_F_central_local(fv,_geom_coor,_X,_F);CHKERRQ(ierr);
      {
        for (k=0; k<m; k++) {
          _F[k] *= -1.0;
        }
      }
    }
    
    if (fv->equation_type == FVDA_ELLIPTIC|| fv->equation_type == FVDA_PARABOLIC) {
      ierr = eval_F_diffusion_7point_local(fv,_geom_target_coor,_fv_coor,_X,_F);CHKERRQ(ierr);
    }
  }
  
  /* scale everything by dt, put in volume contributions */
  for (k=0; k<m; k++) {
    _F[k] *= ctx->dt;
  }
  
  ierr = VecRestoreArrayRead(coorl,&_fv_coor);CHKERRQ(ierr);
  ierr = VecRestoreArray(Fl,&_F);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Xl,&_X);CHKERRQ(ierr);
  
  ierr = DMLocalToGlobal(dm,Fl,INSERT_VALUES,F);CHKERRQ(ierr);
  
  ierr = DMRestoreLocalVector(dm,&Fl);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Xl);CHKERRQ(ierr);
  
  ierr = VecGetArray(F,&_F);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&_X);CHKERRQ(ierr);
  ierr = VecGetArrayRead(ctx->Q_k,&_Xk);CHKERRQ(ierr);
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  for (c=0; c<fv->ncells; c++) {
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * c];
    
    ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateCellVolume3d(cell_coor,&dV0);
    
    ierr = DACellGeometry3d_GetCoordinates(element,_geom_target_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateCellVolume3d(cell_coor,&dV1);
    
    //_F[c] = _F[c] - (_X[c] - _Xk[c]) * dV; /* from time-dep as a reference */
    _F[c] = -_F[c] + (_X[c] * dV1 - _Xk[c] * dV0);
  }
  ierr = VecRestoreArrayRead(ctx->Q_k,&_Xk);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&_X);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&_F);CHKERRQ(ierr);
  
  ierr = VecRestoreArrayRead(geometry_target_coorl,&_geom_target_coor);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(fv->dm_geometry,&geometry_target_coorl);CHKERRQ(ierr);
  
  ierr = VecRestoreArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode fvda_highres_eval_F_forward_ale(SNES snes,Vec X,Vec F,void *data)
{
  PetscErrorCode    ierr;
  Vec               Xl,Fl,coorl,geometry_coorl,geometry_target_coorl;
  const PetscScalar *_X,*_Xk,*_fv_coor,*_geom_coor,*_geom_target_coor;
  PetscScalar       *_F;
  DM                dm;
  FVDA              fv = NULL;
  FVALE             ctx = NULL;
  PetscInt          k,m,c;
  PetscInt          dm_nel,dm_nen;
  const PetscInt    *dm_element,*element;
  PetscReal         cell_coor[3*DACELL3D_VERTS],dV0,dV1;
  
  
  PetscFunctionBegin;
  ierr = SNESGetApplicationContext(snes,(void*)&fv);CHKERRQ(ierr);
  if (!fv->q_dot) SETERRQ(PetscObjectComm((PetscObject)fv->dm_fv),PETSC_ERR_USER,"FVDA must be configured as time-dependent/ALE to use this residual evaluator");
  ctx = (FVALE)fv->ctx;
  if (!ctx) SETERRQ(PetscObjectComm((PetscObject)fv->dm_fv),PETSC_ERR_USER,"FVDA has a NULL context. Must call FVDASetup_ALE() first");
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  dm = fv->dm_fv;

  ierr = DMGetLocalVector(dm,&Xl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm,X,INSERT_VALUES,Xl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Xl,&_X);CHKERRQ(ierr);
  
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&Fl);CHKERRQ(ierr);
  ierr = VecZeroEntries(Fl);CHKERRQ(ierr);
  ierr = VecGetArray(Fl,&_F);CHKERRQ(ierr);
  
  ierr = DMGetCoordinatesLocal(dm,&coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coorl,&_fv_coor);CHKERRQ(ierr);
  
  ierr = DMGetLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_geometry,fv->vertex_coor_geometry,INSERT_VALUES,geometry_coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
  
  ierr = DMGetLocalVector(fv->dm_geometry,&geometry_target_coorl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_geometry,ctx->vertex_coor_geometry_target,INSERT_VALUES,geometry_target_coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(geometry_target_coorl,&_geom_target_coor);CHKERRQ(ierr);
  
  ierr = VecGetSize(Fl,&m);CHKERRQ(ierr);
  {
    if (fv->equation_type == FVDA_HYPERBOLIC || fv->equation_type == FVDA_PARABOLIC) {
      //ierr = eval_F_upwind_local(fv,_geom_coor,_X,_F);CHKERRQ(ierr);
      ierr = eval_F_upwind_hr_local(fv,_geom_coor,_fv_coor,_X,_F);CHKERRQ(ierr);
      {
        for (k=0; k<m; k++) {
          _F[k] *= -1.0;
        }
      }
    }
    
    if (fv->equation_type == FVDA_ELLIPTIC|| fv->equation_type == FVDA_PARABOLIC) {
      ierr = eval_F_diffusion_7point_hr_local_store_MPI(fv,_geom_coor,_fv_coor,_X,_F);CHKERRQ(ierr);
    }
    
    /*
    
     switch (fv->equation_type) {
        case FVDA_HYPERBOLIC:
          //ierr = eval_F_upwind_local(fv,_geom_coor,_X,_F);CHKERRQ(ierr);
          ierr = eval_F_upwind_hr_local(fv,_geom_coor,_fv_coor,_X,_F);CHKERRQ(ierr);
          for (k=0; k<m; k++) { _F[k] *= -1.0; }
        break;
     
        case FVDA_PARABOLIC:
          //ierr = eval_F_upwind_local(fv,_geom_coor,_X,_F);CHKERRQ(ierr);
          ierr = eval_F_upwind_hr_local(fv,_geom_coor,_fv_coor,_X,_F);CHKERRQ(ierr);
          for (k=0; k<m; k++) { _F[k] *= -1.0; }
          ierr = eval_F_diffusion_7point_hr_local_store_MPI(fv,_geom_coor,_fv_coor,_X,_F);CHKERRQ(ierr);
        break;
     
        case FVDA_ELLIPTIC:
          ierr = eval_F_diffusion_7point_hr_local_store_MPI(fv,_geom_coor,_fv_coor,_X,_F);CHKERRQ(ierr);
        break;
     }
     
    */
  }
  
  /* scale everything by dt, put in volume contributions */
  for (k=0; k<m; k++) {
    _F[k] *= ctx->dt;
  }
  
  ierr = VecRestoreArrayRead(coorl,&_fv_coor);CHKERRQ(ierr);
  ierr = VecRestoreArray(Fl,&_F);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Xl,&_X);CHKERRQ(ierr);
  
  ierr = DMLocalToGlobal(dm,Fl,ADD_VALUES,F);CHKERRQ(ierr);
  
  ierr = DMRestoreLocalVector(dm,&Fl);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Xl);CHKERRQ(ierr);
  
  ierr = VecGetArray(F,&_F);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&_X);CHKERRQ(ierr);
  ierr = VecGetArrayRead(ctx->Q_k,&_Xk);CHKERRQ(ierr);
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  for (c=0; c<fv->ncells; c++) {
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * c];
    
    ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateCellVolume3d(cell_coor,&dV0);
    
    ierr = DACellGeometry3d_GetCoordinates(element,_geom_target_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateCellVolume3d(cell_coor,&dV1);
    
    //_F[c] = _F[c] - (_X[c] - _Xk[c]) * dV; /* from time-dep as a reference */
    _F[c] = -_F[c] + (_X[c] * dV1 - _Xk[c] * dV0);
  }
  ierr = VecRestoreArrayRead(ctx->Q_k,&_Xk);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&_X);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&_F);CHKERRQ(ierr);
  
  ierr = VecRestoreArrayRead(geometry_target_coorl,&_geom_target_coor);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(fv->dm_geometry,&geometry_target_coorl);CHKERRQ(ierr);
  
  ierr = VecRestoreArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}


/////////////////////

PetscErrorCode fvda_highres_eval_L_forward_ale(SNES snes,const PetscReal range[],Vec X,Vec F)
{
  PetscErrorCode    ierr;
  Vec               Xl,Fl,coorl,geometry_coorl,geometry_target_coorl;
  const PetscScalar *_X,*_fv_coor,*_geom_coor,*_geom_target_coor;
  PetscScalar       *_F;
  DM                dm;
  FVDA              fv = NULL;
  FVALE             ctx = NULL;
  PetscInt          k,m;
  
  PetscFunctionBegin;
  ierr = SNESGetApplicationContext(snes,(void*)&fv);CHKERRQ(ierr);
  if (!fv->q_dot) SETERRQ(PetscObjectComm((PetscObject)fv->dm_fv),PETSC_ERR_USER,"FVDA must be configured as time-dependent/ALE to use this residual evaluator");
  ctx = (FVALE)fv->ctx;
  if (!ctx) SETERRQ(PetscObjectComm((PetscObject)fv->dm_fv),PETSC_ERR_USER,"FVDA has a NULL context. Must call FVDASetup_ALE() first");
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  dm = fv->dm_fv;

  ierr = DMGetLocalVector(dm,&Xl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm,X,INSERT_VALUES,Xl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Xl,&_X);CHKERRQ(ierr);
  
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&Fl);CHKERRQ(ierr);
  ierr = VecZeroEntries(Fl);CHKERRQ(ierr);
  ierr = VecGetArray(Fl,&_F);CHKERRQ(ierr);
  
  ierr = DMGetCoordinatesLocal(dm,&coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coorl,&_fv_coor);CHKERRQ(ierr);
  
  ierr = DMGetLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_geometry,fv->vertex_coor_geometry,INSERT_VALUES,geometry_coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
  
  ierr = DMGetLocalVector(fv->dm_geometry,&geometry_target_coorl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_geometry,ctx->vertex_coor_geometry_target,INSERT_VALUES,geometry_target_coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(geometry_target_coorl,&_geom_target_coor);CHKERRQ(ierr);
  
  ierr = VecGetSize(Fl,&m);CHKERRQ(ierr);
  {
    if (fv->equation_type == FVDA_HYPERBOLIC || fv->equation_type == FVDA_PARABOLIC) {
      //ierr = eval_F_upwind_local(fv,_geom_coor,_X,_F);CHKERRQ(ierr);
      if (range) {
        ierr = eval_F_upwind_hr_bound_local(fv,range,_geom_coor,_fv_coor,_X,_F);CHKERRQ(ierr);
      } else {
        ierr = eval_F_upwind_hr_local(fv,_geom_coor,_fv_coor,_X,_F);CHKERRQ(ierr);
      }
      for (k=0; k<m; k++) {
        _F[k] *= -1.0;
      }
    }
    
    if (fv->equation_type == FVDA_ELLIPTIC|| fv->equation_type == FVDA_PARABOLIC) {
      ierr = eval_F_diffusion_7point_hr_local_store_MPI(fv,_geom_coor,_fv_coor,_X,_F);CHKERRQ(ierr);
    }
  }
  
  /* scale everything by dt, put in volume contributions */
  //for (k=0; k<m; k++) {
  //  _F[k] *= ctx->dt;
  //}
  
  ierr = VecRestoreArrayRead(coorl,&_fv_coor);CHKERRQ(ierr);
  ierr = VecRestoreArray(Fl,&_F);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Xl,&_X);CHKERRQ(ierr);
  
  ierr = DMLocalToGlobal(dm,Fl,ADD_VALUES,F);CHKERRQ(ierr);
  
  ierr = DMRestoreLocalVector(dm,&Fl);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Xl);CHKERRQ(ierr);
  
  
  ierr = VecRestoreArrayRead(geometry_target_coorl,&_geom_target_coor);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(fv->dm_geometry,&geometry_target_coorl);CHKERRQ(ierr);
  
  ierr = VecRestoreArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}


PetscErrorCode EnergyFV_RK2SSP(SNES snes,const PetscReal range[],PetscReal time,PetscReal dt,Vec X,Vec X_new)
{
  PetscErrorCode ierr;
  FVDA           fv;
  FVALE          ctx = NULL;
  DM             dm;
  Vec            X_1,F,vol_zero,vol_half,vol_one;
  
  PetscFunctionBegin;
  
  ierr = SNESGetApplicationContext(snes,(void*)&fv);CHKERRQ(ierr);
  if (!fv->q_dot) SETERRQ(PetscObjectComm((PetscObject)fv->dm_fv),PETSC_ERR_USER,"FVDA must be configured as time-dependent/ALE to use this residual evaluator");
  ctx = (FVALE)fv->ctx;
  dm = fv->dm_fv;
  
  ierr = VecZeroEntries(X_new);CHKERRQ(ierr);
  
  ierr = DMCreateGlobalVector(dm,&X_1);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&F);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&vol_zero);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&vol_half);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&vol_one);CHKERRQ(ierr);
  
  {
    PetscInt          c;
    PetscInt          dm_nel,dm_nen;
    const PetscInt    *dm_element,*element;
    PetscReal         cell_coor[3*DACELL3D_VERTS],dV0,dV1;
    Vec               geometry_coorl,geometry_target_coorl;
    const PetscScalar *_geom_coor,*_geom_target_coor;

    PetscScalar       *_v_zero,*_v_one;
    
    ierr = VecGetArray(vol_zero,&_v_zero);CHKERRQ(ierr);
    ierr = VecGetArray(vol_one,&_v_one);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
    ierr = DMGlobalToLocal(fv->dm_geometry,fv->vertex_coor_geometry,INSERT_VALUES,geometry_coorl);CHKERRQ(ierr);
    ierr = VecGetArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(fv->dm_geometry,&geometry_target_coorl);CHKERRQ(ierr);
    ierr = DMGlobalToLocal(fv->dm_geometry,ctx->vertex_coor_geometry_target,INSERT_VALUES,geometry_target_coorl);CHKERRQ(ierr);
    ierr = VecGetArrayRead(geometry_target_coorl,&_geom_target_coor);CHKERRQ(ierr);
    
    ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
    for (c=0; c<fv->ncells; c++) {
      element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * c];
      
      ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);
      _EvaluateCellVolume3d(cell_coor,&dV0);
      
      ierr = DACellGeometry3d_GetCoordinates(element,_geom_target_coor,cell_coor);CHKERRQ(ierr);
      _EvaluateCellVolume3d(cell_coor,&dV1);
      
      _v_zero[c] = dV0;
      _v_one[c] = dV1;
    }

    ierr = VecRestoreArrayRead(geometry_target_coorl,&_geom_target_coor);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(fv->dm_geometry,&geometry_target_coorl);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
    
    ierr = VecRestoreArray(vol_one,&_v_one);CHKERRQ(ierr);
    ierr = VecRestoreArray(vol_zero,&_v_zero);CHKERRQ(ierr);
    
    ierr = VecCopy(vol_zero,vol_half);CHKERRQ(ierr);
    ierr = VecAXPY(vol_half,1.0,vol_one);CHKERRQ(ierr);
    ierr = VecScale(vol_half,0.5);CHKERRQ(ierr);
  }
  
  ierr = fvda_highres_eval_L_forward_ale(snes,NULL,X,F);CHKERRQ(ierr);
  //ierr = fvda_highres_eval_L_forward_ale(snes,range,X,F);CHKERRQ(ierr);
  /* Perform required volume scaling */
  {
    // u^{(1)} = u^n + dt.L[ u^{n} ]

    ierr = VecCopy(X,X_1);CHKERRQ(ierr);
    ierr = VecPointwiseMult(X_1,X_1,vol_zero);CHKERRQ(ierr);
    ierr = VecAXPY(X_1,ctx->dt,F);CHKERRQ(ierr);
    ierr = VecPointwiseDivide(X_1,X_1,vol_half);CHKERRQ(ierr);
  }
  if (range) { ierr = FVEnforceBounds(fv,range,X_1);CHKERRQ(ierr); }
  
  ierr = fvda_highres_eval_L_forward_ale(snes,NULL,X_1,F);CHKERRQ(ierr);
  //ierr = fvda_highres_eval_L_forward_ale(snes,range,X_1,F);CHKERRQ(ierr);
  /* Perform required volume scaling */
  {
    // u^{n+1} = 1/2 ( u^n + u^{(1)} + dt.L[ u^{(1)} ]
    
    ierr = VecPointwiseMult(X_1,X_1,vol_half);CHKERRQ(ierr);
    
    ierr = VecCopy(X_1,X_new);CHKERRQ(ierr);
    
    ierr = VecCopy(X,X_1);CHKERRQ(ierr);
    ierr = VecPointwiseMult(X_1,X_1,vol_zero);CHKERRQ(ierr);

    ierr = VecAXPY(X_new,1.0,X_1);CHKERRQ(ierr);

    ierr = VecAXPY(X_new,ctx->dt,F);CHKERRQ(ierr);
    
    ierr = VecPointwiseDivide(X_new,X_new,vol_one);CHKERRQ(ierr);
    ierr = VecScale(X_new,0.5);CHKERRQ(ierr);

  }
  if (range) { ierr = FVEnforceBounds(fv,range,X_new);CHKERRQ(ierr); }
  
  ierr = VecDestroy(&vol_zero);CHKERRQ(ierr);
  ierr = VecDestroy(&vol_half);CHKERRQ(ierr);
  ierr = VecDestroy(&vol_one);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = VecDestroy(&X_1);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode EnergyFV_RK1(SNES snes,const PetscReal range[],PetscReal time,PetscReal dt,Vec X,Vec X_new)
{
  PetscErrorCode ierr;
  FVDA           fv;
  FVALE          ctx = NULL;
  DM             dm;
  Vec            X_1,F,vol_zero,vol_half,vol_one;
  
  PetscFunctionBegin;
  
  ierr = SNESGetApplicationContext(snes,(void*)&fv);CHKERRQ(ierr);
  if (!fv->q_dot) SETERRQ(PetscObjectComm((PetscObject)fv->dm_fv),PETSC_ERR_USER,"FVDA must be configured as time-dependent/ALE to use this residual evaluator");
  ctx = (FVALE)fv->ctx;
  dm = fv->dm_fv;
  
  ierr = VecZeroEntries(X_new);CHKERRQ(ierr);
  
  ierr = DMCreateGlobalVector(dm,&X_1);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&F);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&vol_zero);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&vol_half);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&vol_one);CHKERRQ(ierr);
  
  {
    PetscInt          c;
    PetscInt          dm_nel,dm_nen;
    const PetscInt    *dm_element,*element;
    PetscReal         cell_coor[3*DACELL3D_VERTS],dV0,dV1;
    Vec               geometry_coorl,geometry_target_coorl;
    const PetscScalar *_geom_coor,*_geom_target_coor;
    
    PetscScalar       *_v_zero,*_v_one;
    
    ierr = VecGetArray(vol_zero,&_v_zero);CHKERRQ(ierr);
    ierr = VecGetArray(vol_one,&_v_one);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
    ierr = DMGlobalToLocal(fv->dm_geometry,fv->vertex_coor_geometry,INSERT_VALUES,geometry_coorl);CHKERRQ(ierr);
    ierr = VecGetArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(fv->dm_geometry,&geometry_target_coorl);CHKERRQ(ierr);
    ierr = DMGlobalToLocal(fv->dm_geometry,ctx->vertex_coor_geometry_target,INSERT_VALUES,geometry_target_coorl);CHKERRQ(ierr);
    ierr = VecGetArrayRead(geometry_target_coorl,&_geom_target_coor);CHKERRQ(ierr);
    
    ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
    for (c=0; c<fv->ncells; c++) {
      element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * c];
      
      ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);
      _EvaluateCellVolume3d(cell_coor,&dV0);
      
      ierr = DACellGeometry3d_GetCoordinates(element,_geom_target_coor,cell_coor);CHKERRQ(ierr);
      _EvaluateCellVolume3d(cell_coor,&dV1);
      
      _v_zero[c] = dV0;
      _v_one[c] = dV1;
    }
    
    ierr = VecRestoreArrayRead(geometry_target_coorl,&_geom_target_coor);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(fv->dm_geometry,&geometry_target_coorl);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
    
    ierr = VecRestoreArray(vol_one,&_v_one);CHKERRQ(ierr);
    ierr = VecRestoreArray(vol_zero,&_v_zero);CHKERRQ(ierr);
    
    ierr = VecCopy(vol_zero,vol_half);CHKERRQ(ierr);
    ierr = VecAXPY(vol_half,1.0,vol_one);CHKERRQ(ierr);
    ierr = VecScale(vol_half,0.5);CHKERRQ(ierr);
  }
  
  ierr = fvda_highres_eval_L_forward_ale(snes,NULL,X,F);CHKERRQ(ierr);
  //ierr = fvda_highres_eval_L_forward_ale(snes,range,X,F);CHKERRQ(ierr);
  /* Perform required volume scaling */
  {
    // u^{(1)} = u^n + dt.L[ u^{n} ]
    
    ierr = VecCopy(X,X_1);CHKERRQ(ierr);
    ierr = VecPointwiseMult(X_1,X_1,vol_zero);CHKERRQ(ierr);
    ierr = VecAXPY(X_1,ctx->dt,F);CHKERRQ(ierr);
    ierr = VecPointwiseDivide(X_1,X_1,vol_one);CHKERRQ(ierr);
  }
  if (range) { ierr = FVEnforceBounds(fv,range,X_1);CHKERRQ(ierr); }
  ierr = VecCopy(X_1,X_new);CHKERRQ(ierr);
  
  
  ierr = VecDestroy(&vol_zero);CHKERRQ(ierr);
  ierr = VecDestroy(&vol_half);CHKERRQ(ierr);
  ierr = VecDestroy(&vol_one);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = VecDestroy(&X_1);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}
