
#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <fvda_impl.h>
#include <fvda.h>
#include <fvda_utils.h>


PetscErrorCode eval_F_upwind_local(FVDA fv,const PetscReal domain_geom_coor[],const PetscReal X[],PetscReal F[]);
PetscErrorCode eval_F_diffusion_7point_local(FVDA fv,const PetscReal domain_geom_coor[],const PetscReal fv_coor[],const PetscReal X[],PetscReal F[]);

PetscErrorCode eval_J_upwind_local(FVDA fv,const PetscReal domain_geom_coor[],const PetscReal X[],Mat J);
PetscErrorCode eval_J_diffusion_7point_local(FVDA fv,const PetscReal domain_geom_coor[],const PetscReal fv_coor[],const PetscReal X[],Mat J);

PetscErrorCode eval_F_upwind_hr_local(FVDA fv,const PetscReal domain_geom_coor[],const PetscReal fv_coor[],const PetscReal X[],PetscReal F[]);
PetscErrorCode eval_F_diffusion_7point_hr_local_store_MPI(FVDA fv,const PetscReal domain_geom_coor[],const PetscReal fv_coor[],const PetscReal X[],PetscReal F[]);

PetscErrorCode FVDASetup_TimeDep(FVDA fv)
{
  PetscErrorCode ierr;
  FVTD           ctx = NULL;

  
  PetscFunctionBegin;
  if (!fv->setup) SETERRQ(fv->comm,PETSC_ERR_ORDER,"Must call FVDASetup_TimeDep() after FVDASetUp()");
  if (fv->ctx) SETERRQ(PetscObjectComm((PetscObject)fv->dm_fv),PETSC_ERR_USER,"FVDA has a non-NULL context already");
  if (!fv->q_dot) SETERRQ(PetscObjectComm((PetscObject)fv->dm_fv),PETSC_ERR_USER,"FVDA solver must be configured as time-dependent to use this residual evaluator");

  ierr = PetscMalloc(sizeof(struct _p_FVTD),&ctx);CHKERRQ(ierr);
  ierr = PetscMemzero(ctx,sizeof(struct _p_FVTD));CHKERRQ(ierr);
  ctx->dt = 0;
  ierr = DMCreateGlobalVector(fv->dm_fv,&ctx->Q_k);CHKERRQ(ierr);
  fv->ctx = (void*)ctx;
  fv->ctx_destroy = FVDADestroy_TimeDep;
  PetscFunctionReturn(0);
}

PetscErrorCode FVDAAccessData_TimeDep(FVDA fv,PetscReal **dt,Vec *Qk)
{
  FVTD ctx = NULL;
  
  
  PetscFunctionBegin;
  if (!fv->setup) SETERRQ(fv->comm,PETSC_ERR_ORDER,"Must call FVDAAccessData_TimeDep() after FVDASetUp()");
  if (!fv->ctx) SETERRQ(PetscObjectComm((PetscObject)fv->dm_fv),PETSC_ERR_USER,"FVDA has a NULL context. Must call FVDASetupTimeDep() first");
  ctx = (FVTD)fv->ctx;
  if (dt) {
    *dt = &ctx->dt;
  }
  if (Qk) {
    *Qk = ctx->Q_k;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FVDADestroy_TimeDep(FVDA fv)
{
  PetscErrorCode ierr;
  FVTD           ctx = NULL;
  
  
  PetscFunctionBegin;
  if (!fv->setup) SETERRQ(fv->comm,PETSC_ERR_ORDER,"Must call FVDADestroy_TimeDep() after FVDASetUp()");
  if (!fv->ctx) SETERRQ(PetscObjectComm((PetscObject)fv->dm_fv),PETSC_ERR_USER,"FVDA has a NULL context. Must call FVDASetupTimeDep() first");
  ctx = (FVTD)fv->ctx;
  ierr = VecDestroy(&ctx->Q_k);CHKERRQ(ierr);
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  fv->ctx = NULL;
  PetscFunctionReturn(0);
}

/*
Q^k+1 - Q^k + F
*/
PetscErrorCode fvda_eval_F_timedep(SNES snes,Vec X,Vec F,void *data)
{
  PetscErrorCode    ierr;
  Vec               Xl,Fl,coorl,geometry_coorl;
  const PetscScalar *_X,*_Xk,*_fv_coor,*_geom_coor;
  PetscScalar       *_F;
  DM                dm;
  FVDA              fv = NULL;
  FVTD              ctx = NULL;
  PetscInt          k,m,c;
  PetscInt          dm_nel,dm_nen;
  const PetscInt    *dm_element,*element;
  PetscReal         cell_coor[3*DACELL3D_VERTS],dV;

  
  PetscFunctionBegin;
  ierr = SNESGetApplicationContext(snes,(void*)&fv);CHKERRQ(ierr);
  if (!fv->q_dot) SETERRQ(PetscObjectComm((PetscObject)fv->dm_fv),PETSC_ERR_USER,"FVDA must be configured as time-dependent to use this residual evaluator");
  ctx = (FVTD)fv->ctx;
  if (!ctx) SETERRQ(PetscObjectComm((PetscObject)fv->dm_fv),PETSC_ERR_USER,"FVDA has a NULL context. Must call FVDASetup_TimeDep() first");
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  
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
  
  ierr = VecGetSize(Fl,&m);CHKERRQ(ierr);
  {
    if (fv->equation_type == FVDA_HYPERBOLIC || fv->equation_type == FVDA_PARABOLIC) {
      ierr = eval_F_upwind_local(fv,_geom_coor,_X,_F);CHKERRQ(ierr);
      {
        for (k=0; k<m; k++) {
          _F[k] *= -1.0;
        }
      }
    }
    
    if (fv->equation_type == FVDA_ELLIPTIC|| fv->equation_type == FVDA_PARABOLIC) {
      ierr = eval_F_diffusion_7point_local(fv,_geom_coor,_fv_coor,_X,_F);CHKERRQ(ierr);
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
    _EvaluateCellVolume3d(cell_coor,&dV);
    
    //_F[c] = _F[c] - (_X[c] - _Xk[c]) * dV; /* both correct */
    _F[c] = -_F[c] + (_X[c] - _Xk[c]) * dV;
  }
  ierr = VecRestoreArrayRead(ctx->Q_k,&_Xk);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&_X);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&_F);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode fvda_eval_J_timedep(SNES snes,Vec X,Mat Ja,Mat Jb,void *data)
{
  PetscErrorCode    ierr;
  Vec               Xl,coorl,geometry_coorl;
  const PetscScalar *_X,*_fv_coor,*_geom_coor;
  DM                dm;
  FVDA              fv = NULL;
  FVTD              ctx = NULL;
  PetscInt          c,row,offset;
  PetscInt          dm_nel,dm_nen;
  const PetscInt    *dm_element,*element;
  PetscReal         cell_coor[3*DACELL3D_VERTS],dV;

  
  PetscFunctionBegin;
  ierr = SNESGetApplicationContext(snes,(void*)&fv);CHKERRQ(ierr);
  if (!fv->q_dot) SETERRQ(PetscObjectComm((PetscObject)fv->dm_fv),PETSC_ERR_USER,"FVDA must be configured as time-dependent to use this Jacobain evaluator");
  ctx = (FVTD)fv->ctx;
  if (!ctx) SETERRQ(PetscObjectComm((PetscObject)fv->dm_fv),PETSC_ERR_USER,"FVDA has a NULL context. Must call FVDASetup_TimeDep() first");
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  
  ierr = DMGetLocalVector(dm,&Xl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm,X,INSERT_VALUES,Xl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Xl,&_X);CHKERRQ(ierr);
  
  ierr = MatZeroEntries(Jb);CHKERRQ(ierr);
  
  ierr = DMGetCoordinatesLocal(dm,&coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coorl,&_fv_coor);CHKERRQ(ierr);
  
  ierr = DMGetLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_geometry,fv->vertex_coor_geometry,INSERT_VALUES,geometry_coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
  
  {
    if (fv->equation_type == FVDA_HYPERBOLIC || fv->equation_type == FVDA_PARABOLIC) {
      ierr = eval_J_upwind_local(fv,_geom_coor,_X,Jb);CHKERRQ(ierr);
    }
    
    if (fv->equation_type == FVDA_ELLIPTIC|| fv->equation_type == FVDA_PARABOLIC) {
      ierr = eval_J_diffusion_7point_local(fv,_geom_coor,_fv_coor,_X,Jb);CHKERRQ(ierr);
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
  // _F[c] = _F[c] - (_X[c] - _Xk[c]) * dV; // both correct //
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
  // _F[c] = -_F[c] + (_X[c] - _Xk[c]) * dV;
  ierr = MatScale(Jb,-ctx->dt);CHKERRQ(ierr);
  for (c=0; c<fv->ncells; c++) {
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * c];
    ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateCellVolume3d(cell_coor,&dV);
    
    row = offset + c;
    ierr = MatSetValue(Jb,row,row,dV,ADD_VALUES);CHKERRQ(ierr);
  }
  
  ierr = MatAssemblyBegin(Jb,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Jb,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  if (Ja != Jb) {
    ierr = MatAssemblyBegin(Ja,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Ja,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  
  ierr = DMRestoreLocalVector(dm,&Xl);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* 
X_newX1 = X + dt * flux(X,time)
*/
PetscErrorCode FVDAStep_FEuler(FVDA fv,PetscReal time,PetscReal dt,Vec X,Vec X_new)
{
  PetscErrorCode    ierr;
  Vec               Xl,Fl,coorl,geometry_coorl;
  const PetscScalar *_X,*_fv_coor,*_geom_coor;
  PetscScalar       *_F;
  DM                dm;
  PetscInt          c,dm_nel,dm_nen;
  const PetscInt    *dm_element,*element;
  PetscReal         cell_coor[3*DACELL3D_VERTS],dV;
  
  
  PetscFunctionBegin;
  dm = fv->dm_fv;
  
  ierr = DMGetLocalVector(dm,&Xl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dm,X,INSERT_VALUES,Xl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Xl,&_X);CHKERRQ(ierr);
  
  ierr = DMGetLocalVector(dm,&Fl);CHKERRQ(ierr);
  ierr = VecZeroEntries(Fl);CHKERRQ(ierr);
  ierr = VecGetArray(Fl,&_F);CHKERRQ(ierr);
  
  ierr = DMGetCoordinatesLocal(dm,&coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coorl,&_fv_coor);CHKERRQ(ierr);
  
  ierr = DMGetLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_geometry,fv->vertex_coor_geometry,INSERT_VALUES,geometry_coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
  
  if (fv->equation_type == FVDA_HYPERBOLIC) {
    ierr = eval_F_upwind_local(fv,_geom_coor,_X,_F);CHKERRQ(ierr);
    //ierr = eval_F_upwind_hr_local(fv,_geom_coor,_fv_coor,_X,_F);CHKERRQ(ierr);
  }
  
  ierr = VecRestoreArrayRead(coorl,&_fv_coor);CHKERRQ(ierr);
  ierr = VecRestoreArray(Fl,&_F);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Xl,&_X);CHKERRQ(ierr);
  
  ierr = DMLocalToGlobal(dm,Fl,INSERT_VALUES,X_new);CHKERRQ(ierr);

  ierr = VecGetArray(X_new,&_F);CHKERRQ(ierr);
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  for (c=0; c<fv->ncells; c++) {
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * c];
    ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateCellVolume3d(cell_coor,&dV);
    _F[c] = _F[c] / dV;
  }
  ierr = VecRestoreArray(X_new,&_F);CHKERRQ(ierr);
  
  ierr = VecRestoreArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);

  
  
  ierr = VecAYPX(X_new,-dt,X);CHKERRQ(ierr);
  
  ierr = DMRestoreLocalVector(dm,&Fl);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Xl);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode FVEnforceBounds(FVDA fv,const PetscReal range[],Vec Q)
{
  PetscErrorCode ierr;
  PetscReal *_Q;
  PetscInt  i;
  ierr = VecGetArray(Q,&_Q);CHKERRQ(ierr);
  for (i=0; i<fv->ncells; i++) {
    if (_Q[i] < range[0]) { _Q[i] = range[0]; }
    if (_Q[i] > range[1]) { _Q[i] = range[1]; }
  }
  ierr = VecRestoreArray(Q,&_Q);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*
 y(1) = Step_Euler(X_k,t0)

 X_k+1 = 0.5 X_k + Step_Euler(y(1),t0+dt)
 
*/
PetscErrorCode FVDAStep_RK2SSP(FVDA fv,const PetscReal range[],PetscReal time,PetscReal dt,Vec X,Vec X_new)
{
  PetscErrorCode ierr;
  DM             dm;
  Vec            y_1;
  
  PetscFunctionBegin;
  dm = fv->dm_fv;
  ierr = DMCreateGlobalVector(dm,&y_1);CHKERRQ(ierr);
  ierr = FVDAStep_FEuler(fv,time,dt,X,y_1);CHKERRQ(ierr);
  if (range) { ierr = FVEnforceBounds(fv,range,y_1);CHKERRQ(ierr); }
  ierr = FVDAStep_FEuler(fv,time+dt,dt,y_1,X_new);CHKERRQ(ierr);
  if (range) { ierr = FVEnforceBounds(fv,range,X_new);CHKERRQ(ierr); }
  ierr = VecAXPY(X_new,1.0,X);CHKERRQ(ierr);
  ierr = VecScale(X_new,0.5);CHKERRQ(ierr);
  if (range) { ierr = FVEnforceBounds(fv,range,X_new);CHKERRQ(ierr); }
  ierr = VecDestroy(&y_1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode fvda_highres_eval_F_timedep(SNES snes,Vec X,Vec F,void *data)
{
  PetscErrorCode    ierr;
  Vec               Xl,Fl,coorl,geometry_coorl;
  const PetscScalar *_X,*_Xk,*_fv_coor,*_geom_coor;
  PetscScalar       *_F;
  DM                dm;
  FVDA              fv = NULL;
  FVTD              ctx = NULL;
  PetscInt          k,m,c;
  PetscInt          dm_nel,dm_nen;
  const PetscInt    *dm_element,*element;
  PetscReal         cell_coor[3*DACELL3D_VERTS],dV;
  
  
  PetscFunctionBegin;
  ierr = SNESGetApplicationContext(snes,(void*)&fv);CHKERRQ(ierr);
  if (!fv->q_dot) SETERRQ(PetscObjectComm((PetscObject)fv->dm_fv),PETSC_ERR_USER,"FVDA must be configured as time-dependent to use this residual evaluator");
  ctx = (FVTD)fv->ctx;
  if (!ctx) SETERRQ(PetscObjectComm((PetscObject)fv->dm_fv),PETSC_ERR_USER,"FVDA has a NULL context. Must call FVDASetup_TimeDep() first");
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  
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
  
  ierr = VecGetSize(Fl,&m);CHKERRQ(ierr);
  {
    if (fv->equation_type == FVDA_HYPERBOLIC || fv->equation_type == FVDA_PARABOLIC) {
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
    _EvaluateCellVolume3d(cell_coor,&dV);
    
    //_F[c] = _F[c] - (_X[c] - _Xk[c]) * dV; /* both correct */
    _F[c] = -_F[c] + (_X[c] - _Xk[c]) * dV;
  }
  ierr = VecRestoreArrayRead(ctx->Q_k,&_Xk);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&_X);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&_F);CHKERRQ(ierr);
  
  ierr = VecRestoreArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}
