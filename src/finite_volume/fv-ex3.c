
#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <fvda_impl.h>
#include <fvda.h>
#include <fvda_utils.h>

PetscBool operator_fvspace = PETSC_TRUE;
PetscBool view = PETSC_FALSE;

PetscErrorCode bcset_west(FVDA fv,
                          DACellFace face,
                          PetscInt nfaces,
                          const PetscReal coor[],
                          const PetscReal normal[],
                          const PetscInt cell[],
                          PetscReal time,
                          FVFluxType flux[],
                          PetscReal bcvalue[],
                          void *ctx)
{
  PetscInt f;
  
  for (f=0; f<nfaces; f++) {
    flux[f] = FVFLUX_DIRICHLET_CONSTRAINT;
    bcvalue[f] = 0.3;
  }
  for (f=0; f<nfaces; f++) {
    PetscReal r2=0;
    //r2 += coor[3*f+0]*coor[3*f+0];
    r2 += coor[3*f+1]*coor[3*f+1];
    r2 += coor[3*f+2]*coor[3*f+2];
    if (r2 <= 0.4*0.4) {
      bcvalue[f] = 1.3;
    }
    
    //bcvalue[f] = 0.3;
    //if (fabs(coor[3*f+1]) < 0.4) {
    //  bcvalue[f] = 1.3;
    //}
    
  }
  PetscFunctionReturn(0);
}

PetscErrorCode bcset_default(FVDA fv,
                             DACellFace face,
                             PetscInt nfaces,
                             const PetscReal coor[],
                             const PetscReal normal[],
                             const PetscInt cell[],
                             PetscReal time,
                             FVFluxType flux[],
                             PetscReal bcvalue[],
                             void *ctx)
{
  PetscInt f;
  
  for (f=0; f<nfaces; f++) {
    flux[f] = FVFLUX_DIRICHLET_CONSTRAINT;
    bcvalue[f] = 0.3;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode t3(void)
{
  PetscErrorCode ierr;
  PetscInt       mx = 64+1;
  PetscInt       m[] = {mx,mx,mx};
  FVDA           fv;
  Vec            X,F;
  Mat            J;
  DM             dm;
  SNES           snes;
  PetscBool      found = PETSC_FALSE;
  
  ierr = PetscOptionsGetInt(NULL,NULL,"-mx",&mx,&found);CHKERRQ(ierr);
  if (found) {
    m[0] = mx;
    m[1] = mx;
    m[2] = mx;
  }
  ierr = PetscOptionsGetInt(NULL,NULL,"-my",&m[1],NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-mz",&m[2],NULL);CHKERRQ(ierr);

  ierr = FVDACreate(PETSC_COMM_WORLD,&fv);CHKERRQ(ierr);
  ierr = FVDASetDimension(fv,3);CHKERRQ(ierr);
  ierr = FVDASetSizes(fv,NULL,m);CHKERRQ(ierr);
  
  ierr = FVDASetProblemType(fv,PETSC_FALSE,FVDA_ELLIPTIC,0,0);CHKERRQ(ierr);
  
  ierr = FVDASetUp(fv);CHKERRQ(ierr);
  
  {
    Vec gcoor;
    
    ierr = DMDASetUniformCoordinates(fv->dm_geometry,-1.0,1.0,-1.0,1.0,-1.0,1.0);CHKERRQ(ierr);
    ierr = DMGetCoordinates(fv->dm_geometry,&gcoor);CHKERRQ(ierr);
    ierr = VecCopy(gcoor,fv->vertex_coor_geometry);CHKERRQ(ierr);
  }
  
  ierr = FVDAUpdateGeometry(fv);CHKERRQ(ierr);
  
  ierr = FVDARegisterFaceProperty(fv,"k",1);CHKERRQ(ierr);
  {
    PetscInt  f,nfaces;
    PetscReal *k;
    
    ierr = FVDAGetFaceInfo(fv,&nfaces,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = FVDAGetFacePropertyArray(fv,0,&k);CHKERRQ(ierr);
    for (f=0; f<nfaces; f++) {
      k[f] = 1.0;
    }
  }
  
  /* set boundary value at intitial time */
  ierr = FVDAFaceIterator(fv,DACELL_FACE_W,PETSC_FALSE,0.0,bcset_west,NULL);CHKERRQ(ierr);
  
  ierr = FVDAFaceIterator(fv,DACELL_FACE_E,PETSC_FALSE,0.0,bcset_default,NULL);CHKERRQ(ierr);
  ierr = FVDAFaceIterator(fv,DACELL_FACE_N,PETSC_FALSE,0.0,bcset_default,NULL);CHKERRQ(ierr);
  ierr = FVDAFaceIterator(fv,DACELL_FACE_S,PETSC_FALSE,0.0,bcset_default,NULL);CHKERRQ(ierr);
  ierr = FVDAFaceIterator(fv,DACELL_FACE_F,PETSC_FALSE,0.0,bcset_default,NULL);CHKERRQ(ierr);
  ierr = FVDAFaceIterator(fv,DACELL_FACE_B,PETSC_FALSE,0.0,bcset_default,NULL);CHKERRQ(ierr);
  
  dm = fv->dm_fv;

  if (operator_fvspace) {
    ierr = DMCreateMatrix(dm,&J);CHKERRQ(ierr);
  } else {
    ierr = FVDACreateMatrix(fv,DMDA_STENCIL_STAR,&J);CHKERRQ(ierr);
  }
  
  ierr = DMCreateGlobalVector(dm,&X);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&F);CHKERRQ(ierr);
  
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,dm);CHKERRQ(ierr);
  ierr = SNESSetSolution(snes,X);CHKERRQ(ierr);
  ierr = SNESSetApplicationContext(snes,(void*)fv);CHKERRQ(ierr);
  
  ierr = SNESSetFunction(snes,F,eval_F,NULL);CHKERRQ(ierr);
  //ierr = SNESSetJacobian(snes,J,J,SNESComputeJacobianDefaultColor,NULL);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,J,J,eval_J,NULL);CHKERRQ(ierr);
  
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = SNESFVDAConfigureGalerkinMG(snes,fv);CHKERRQ(ierr);
  //
  //ierr = SNESComputeJacobian(snes,X,J,J);CHKERRQ(ierr);
  //MatView(J,PETSC_VIEWER_STDOUT_WORLD);
  
  /*
   ierr = eval_J(snes,X,J,J,NULL);CHKERRQ(ierr);
   MatView(J,PETSC_VIEWER_STDOUT_WORLD);
   exit(1);
   */
  
  ierr = SNESSolve(snes,NULL,X);CHKERRQ(ierr);
  
  if (view) {
    PetscViewer viewer;
    char        fname[256];
    
    sprintf(fname,"x.vts");
    ierr = PetscViewerCreate(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer,PETSCVIEWERVTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer,fname);CHKERRQ(ierr);
    ierr = VecView(X,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    
    ierr = FVDAView_CellData(fv,X,PETSC_TRUE,"xcell");CHKERRQ(ierr);
    
    {
      DM  dmf;
      Vec Xv;
      
      ierr = FVDAFieldSetUpProjectToVertex_Q1(fv,&dmf,&Xv);CHKERRQ(ierr);
      ierr = FVDAFieldProjectToVertex_Q1(fv,X,dmf,Xv);CHKERRQ(ierr);
      
      sprintf(fname,"xv.vts");
      ierr = PetscViewerCreate(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
      ierr = PetscViewerSetType(viewer,PETSCVIEWERVTK);CHKERRQ(ierr);
      ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
      ierr = PetscViewerFileSetName(viewer,fname);CHKERRQ(ierr);
      ierr = VecView(Xv,viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

      ierr = VecDestroy(&Xv);CHKERRQ(ierr);
      ierr = DMDestroy(&dmf);CHKERRQ(ierr);
    }
  }
  
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = FVDADestroy(&fv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* autogenerated - see src/finite_volume/python/steady_diffusion_ex1.py */
PetscErrorCode mms_evaluate_Q_ex1(const PetscReal x[],PetscReal Q[])
{
  Q[0] = sin(2.1*M_PI*x[0])*sin(0.4*M_PI*x[2])*cos(1.1*M_PI*x[1]);
  PetscFunctionReturn(0);
}

PetscErrorCode mms_evaluate_gradQ_ex1(const PetscReal x[],PetscReal gradQ[])
{
  gradQ[0] = 2.1*M_PI*sin(0.4*M_PI*x[2])*cos(2.1*M_PI*x[0])*cos(1.1*M_PI*x[1]);
  gradQ[1] = -1.1*M_PI*sin(2.1*M_PI*x[0])*sin(1.1*M_PI*x[1])*sin(0.4*M_PI*x[2]);
  gradQ[2] = 0.4*M_PI*sin(2.1*M_PI*x[0])*cos(1.1*M_PI*x[1])*cos(0.4*M_PI*x[2]);
  PetscFunctionReturn(0);
}

PetscErrorCode mms_evaluate_F_ex1(const PetscReal x[],PetscReal F[])
{
  F[0] = -2.1*M_PI*sin(0.4*M_PI*x[2])*cos(2.1*M_PI*x[0])*cos(1.1*M_PI*x[1]);
  F[1] = 1.1*M_PI*sin(2.1*M_PI*x[0])*sin(1.1*M_PI*x[1])*sin(0.4*M_PI*x[2]);
  F[2] = -0.4*M_PI*sin(2.1*M_PI*x[0])*cos(1.1*M_PI*x[1])*cos(0.4*M_PI*x[2]);
  PetscFunctionReturn(0);
}

PetscErrorCode mms_evaluate_f_ex1(const PetscReal x[],PetscReal f[])
{
  f[0] = -5.78*pow(M_PI, 2)*sin(2.1*M_PI*x[0])*sin(0.4*M_PI*x[2])*cos(1.1*M_PI*x[1]);
  PetscFunctionReturn(0);
}




PetscErrorCode bcset_dirichlet_mms(
                  FVDA fv,DACellFace face,PetscInt nfaces,
                  const PetscReal coor[],const PetscReal normal[],const PetscInt cell[],PetscReal time,
                  FVFluxType flux_type[],PetscReal bcvalue[],
                  void *ctx)
{
  PetscInt f;
  PetscErrorCode ierr;
  for (f=0; f<nfaces; f++) {
    flux_type[f] = FVFLUX_DIRICHLET_CONSTRAINT;
    ierr = mms_evaluate_Q_ex1(&coor[3*f],&bcvalue[f]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* 
 Note the neagtive sign in the Neuammn condition
 
 Our FV implementation uses this
 \int_S k grad(Q) .n dS

 The weak form of -div(-k grad(Q)) = f is
 + \int_V grad(w) . (-k grad(Q)) dV - \int_S w (-k grad(Q)) . n dS = \int_V f dV
 + \int_V grad(w) . (-k grad(Q)) dV + \int_S w k grad(Q) . n dS = \int_V f dV

 Our MMS function computes the flux as -k grad(Q), but since our implementation absorbs 
 the two negatives, we need to flip the sign of the flux returned from the MMS solution.
*/
PetscErrorCode bcset_neumann_mms(
                 FVDA fv,DACellFace face,PetscInt nfaces,
                 const PetscReal coor[],const PetscReal normal[],const PetscInt cell[],PetscReal time,
                 FVFluxType flux_type[],PetscReal bcvalue[],
                 void *ctx)
{
  PetscInt f;
  PetscReal flux[3];
  PetscErrorCode ierr;
  for (f=0; f<nfaces; f++) {
    const PetscReal *x = &coor[3*f];
    const PetscReal *n = &normal[3*f];
    
    flux_type[f] = FVFLUX_NEUMANN_CONSTRAINT;
    ierr = mms_evaluate_F_ex1(x,flux);CHKERRQ(ierr);
    bcvalue[f] = -(flux[0] * n[0] + flux[1] * n[1] + flux[2] * n[2]);
  }
  PetscFunctionReturn(0);
}

/*
 Computes 
   L1(Q-Qmms)
   L2(Q-Qmms)
*/
PetscErrorCode EvaluateDiscretizationErrors(FVDA fv,PetscReal time,Vec Q,PetscReal *H,PetscReal error[])
{
  PetscErrorCode  ierr;
  PetscReal       cell_x[3],dV,Q_mms,cell_h = 0;
  const PetscInt  NSD = 3;
  PetscReal       cell_coor[3 * DACELL3D_Q1_SIZE];
  Vec             coorl;
  const PetscReal *_geom_coor,*_Q;
  PetscInt        i,d,c,dm_nel,dm_nen;
  const PetscInt  *dm_element,*element;
  
  ierr = VecGetArrayRead(Q,&_Q);CHKERRQ(ierr);
  
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(fv->dm_geometry,&coorl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_geometry,fv->vertex_coor_geometry,INSERT_VALUES,coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coorl,&_geom_coor);CHKERRQ(ierr);
  
  error[0] = 0;
  error[1] = 0;
  
  for (c=0; c<fv->ncells; c++) {
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * c];

    ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateCellVolume3d(cell_coor,&dV);

    /* average cell vertex coordinates */
    for (d=0; d<NSD; d++) {
      cell_x[d] = 0.0;
      for (i=0; i<DACELL3D_Q1_SIZE; i++) {
        cell_x[d] += cell_coor[NSD * i + d];
      }
      cell_x[d] /= (PetscReal)DACELL3D_Q1_SIZE;
    }

    {
      PetscReal cmin[] = {PETSC_MAX_REAL,PETSC_MAX_REAL,PETSC_MAX_REAL},cmax[] = {-PETSC_MAX_REAL,-PETSC_MAX_REAL,-PETSC_MAX_REAL};
      
      for (d=0; d<NSD; d++) {
        for (i=0; i<DACELL3D_Q1_SIZE; i++) {
          cmin[d] = PetscMin(cmin[d],cell_coor[NSD * i + d]);
          cmax[d] = PetscMax(cmax[d],cell_coor[NSD * i + d]);
        }
        cell_h = PetscMax(cell_h,cmax[d] - cmin[d]);
      }
    }
    
    ierr = mms_evaluate_Q_ex1(cell_x,&Q_mms);CHKERRQ(ierr);

    error[0] += PetscAbsReal(_Q[c] - Q_mms) * dV;
    error[1] += (_Q[c] - Q_mms) * (_Q[c] - Q_mms) * dV;
    
  }

  ierr = MPI_Allreduce(MPI_IN_PLACE,&cell_h,1,MPIU_REAL,MPIU_MAX,fv->comm);CHKERRQ(ierr);
  *H = cell_h;
  
  ierr = MPI_Allreduce(MPI_IN_PLACE,&error[0],1,MPIU_REAL,MPIU_SUM,fv->comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(MPI_IN_PLACE,&error[1],1,MPIU_REAL,MPIU_SUM,fv->comm);CHKERRQ(ierr);
  error[1] = PetscSqrtReal(error[1]);
  
  ierr = VecRestoreArrayRead(Q,&_Q);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(coorl,&_geom_coor);CHKERRQ(ierr);
  ierr = VecDestroy(&coorl);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode EvaluateDiscretizationGradientErrors(FVDA fv,PetscReal time,Vec gradQ,PetscReal *H,PetscReal error[])
{
  PetscErrorCode  ierr;
  PetscReal       cell_x[3],dV,gradQ_mms[3],cell_h = 0;
  const PetscInt  NSD = 3;
  PetscReal       cell_coor[3 * DACELL3D_Q1_SIZE];
  Vec             coorl;
  const PetscReal *_geom_coor,*_gradQ;
  PetscInt        i,d,c,dm_nel,dm_nen;
  const PetscInt  *dm_element,*element;
  
  ierr = VecGetArrayRead(gradQ,&_gradQ);CHKERRQ(ierr);
  
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(fv->dm_geometry,&coorl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_geometry,fv->vertex_coor_geometry,INSERT_VALUES,coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coorl,&_geom_coor);CHKERRQ(ierr);
  
  error[0] = 0;
  error[1] = 0;
  
  for (c=0; c<fv->ncells; c++) {
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * c];
    
    ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateCellVolume3d(cell_coor,&dV);
    
    /* average cell vertex coordinates */
    for (d=0; d<NSD; d++) {
      cell_x[d] = 0.0;
      for (i=0; i<DACELL3D_Q1_SIZE; i++) {
        cell_x[d] += cell_coor[NSD * i + d];
      }
      cell_x[d] /= (PetscReal)DACELL3D_Q1_SIZE;
    }
    
    {
      PetscReal cmin[] = {PETSC_MAX_REAL,PETSC_MAX_REAL,PETSC_MAX_REAL},cmax[] = {-PETSC_MAX_REAL,-PETSC_MAX_REAL,-PETSC_MAX_REAL};
      
      for (d=0; d<NSD; d++) {
        for (i=0; i<DACELL3D_Q1_SIZE; i++) {
          cmin[d] = PetscMin(cmin[d],cell_coor[NSD * i + d]);
          cmax[d] = PetscMax(cmax[d],cell_coor[NSD * i + d]);
        }
        cell_h = PetscMax(cell_h,cmax[d] - cmin[d]);
      }
    }
    
    ierr = mms_evaluate_gradQ_ex1(cell_x,gradQ_mms);CHKERRQ(ierr);
    
    for (d=0; d<NSD; d++) {
      error[0] += PetscAbsReal(_gradQ[NSD*c+d] - gradQ_mms[d]) * dV;
      
      error[1] += (_gradQ[NSD*c+d] - gradQ_mms[d]) * (_gradQ[NSD*c+d] - gradQ_mms[d]) * dV;
    }
    
  }
  
  ierr = MPI_Allreduce(MPI_IN_PLACE,&cell_h,1,MPIU_REAL,MPIU_MAX,fv->comm);CHKERRQ(ierr);
  *H = cell_h;
  
  ierr = MPI_Allreduce(MPI_IN_PLACE,&error[0],1,MPIU_REAL,MPIU_SUM,fv->comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(MPI_IN_PLACE,&error[1],1,MPIU_REAL,MPIU_SUM,fv->comm);CHKERRQ(ierr);
  error[1] = PetscSqrtReal(error[1]);
  
  ierr = VecRestoreArrayRead(gradQ,&_gradQ);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(coorl,&_geom_coor);CHKERRQ(ierr);
  ierr = VecDestroy(&coorl);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode EvaluateRHS_mms(FVDA fv,PetscReal time,Vec F)
{
  PetscErrorCode  ierr;
  PetscReal       cell_x[3],f_mms,dV;
  const PetscInt  NSD = 3;
  PetscReal       cell_coor[3 * DACELL3D_Q1_SIZE];
  Vec             coorl;
  const PetscReal *_geom_coor;
  PetscReal       *_F;
  PetscInt        i,d,c,dm_nel,dm_nen;
  const PetscInt  *dm_element,*element;
  
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  ierr = VecGetArray(F,&_F);CHKERRQ(ierr);
  
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(fv->dm_geometry,&coorl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_geometry,fv->vertex_coor_geometry,INSERT_VALUES,coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coorl,&_geom_coor);CHKERRQ(ierr);
  
  for (c=0; c<fv->ncells; c++) {
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * c];
    
    ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateCellVolume3d(cell_coor,&dV);
    
    /* average cell vertex coordinates */
    for (d=0; d<NSD; d++) {
      cell_x[d] = 0.0;
      for (i=0; i<DACELL3D_Q1_SIZE; i++) {
        cell_x[d] += cell_coor[NSD * i + d];
      }
      cell_x[d] /= (PetscReal)DACELL3D_Q1_SIZE;
    }
    
    ierr = mms_evaluate_f_ex1(cell_x,&f_mms);CHKERRQ(ierr);
    _F[c] = f_mms * dV;
  }
  ierr = VecRestoreArray(F,&_F);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(coorl,&_geom_coor);CHKERRQ(ierr);
  ierr = VecDestroy(&coorl);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/*
 Solves -div(-k grad(Q)) = f
 
 Weak form
 - \int_V w div(-k grad(Q)) dV = \int_V f dV
 + \int_V grad(w) . (-k grad(Q)) dV - \int_S w (-k grad(Q)) . n dS = \int_V f dV
 
 Neumann BC is k grad(Q)
 
*/
PetscErrorCode t3_mms(void)
{
  PetscErrorCode ierr;
  PetscInt       mx = 12,bc = 0;
  PetscInt       m[] = {mx,mx,mx};
  FVDA           fv;
  Vec            X,F,rhs;
  Mat            J;
  DM             dm;
  SNES           snes;
  PetscBool      found = PETSC_FALSE;
  PetscReal      h,error[2];
  
  ierr = PetscOptionsGetInt(NULL,NULL,"-mx",&mx,&found);CHKERRQ(ierr);
  if (found) {
    m[0] = mx;
    m[1] = mx;
    m[2] = mx;
  }
  ierr = PetscOptionsGetInt(NULL,NULL,"-bc",&bc,NULL);CHKERRQ(ierr);
  
  ierr = FVDACreate(PETSC_COMM_WORLD,&fv);CHKERRQ(ierr);
  ierr = FVDASetDimension(fv,3);CHKERRQ(ierr);
  ierr = FVDASetSizes(fv,NULL,m);CHKERRQ(ierr);
  
  ierr = FVDASetProblemType(fv,PETSC_FALSE,FVDA_ELLIPTIC,0,0);CHKERRQ(ierr);
  
  ierr = FVDASetUp(fv);CHKERRQ(ierr);
  
  {
    Vec gcoor;
    
    ierr = DMDASetUniformCoordinates(fv->dm_geometry,-1.0,1.0,-1.0,1.0,-1.0,1.0);CHKERRQ(ierr);
    ierr = DMGetCoordinates(fv->dm_geometry,&gcoor);CHKERRQ(ierr);
    ierr = VecCopy(gcoor,fv->vertex_coor_geometry);CHKERRQ(ierr);
  }
  
  ierr = FVDAUpdateGeometry(fv);CHKERRQ(ierr);
  
  ierr = FVDARegisterFaceProperty(fv,"k",1);CHKERRQ(ierr);
  {
    PetscInt  f,nfaces;
    PetscReal *k;
    
    ierr = FVDAGetFaceInfo(fv,&nfaces,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = FVDAGetFacePropertyArray(fv,0,&k);CHKERRQ(ierr);
    for (f=0; f<nfaces; f++) {
      k[f] = 1.0;
    }
  }
  
  switch (bc) {
    case 0:
      ierr = FVDAFaceIterator(fv,DACELL_FACE_W,PETSC_FALSE,0.0,bcset_dirichlet_mms,NULL);CHKERRQ(ierr);
      ierr = FVDAFaceIterator(fv,DACELL_FACE_E,PETSC_FALSE,0.0,bcset_dirichlet_mms,NULL);CHKERRQ(ierr);
      ierr = FVDAFaceIterator(fv,DACELL_FACE_N,PETSC_FALSE,0.0,bcset_dirichlet_mms,NULL);CHKERRQ(ierr);
      ierr = FVDAFaceIterator(fv,DACELL_FACE_S,PETSC_FALSE,0.0,bcset_dirichlet_mms,NULL);CHKERRQ(ierr);
      ierr = FVDAFaceIterator(fv,DACELL_FACE_F,PETSC_FALSE,0.0,bcset_dirichlet_mms,NULL);CHKERRQ(ierr);
      ierr = FVDAFaceIterator(fv,DACELL_FACE_B,PETSC_FALSE,0.0,bcset_dirichlet_mms,NULL);CHKERRQ(ierr);
      break;

    case 1:
      ierr = FVDAFaceIterator(fv,DACELL_FACE_N,PETSC_FALSE,0.0,bcset_dirichlet_mms,NULL);CHKERRQ(ierr);
      ierr = FVDAFaceIterator(fv,DACELL_FACE_S,PETSC_FALSE,0.0,bcset_dirichlet_mms,NULL);CHKERRQ(ierr);
      
      ierr = FVDAFaceIterator(fv,DACELL_FACE_W,PETSC_TRUE,0.0,bcset_neumann_mms,NULL);CHKERRQ(ierr);
      ierr = FVDAFaceIterator(fv,DACELL_FACE_E,PETSC_TRUE,0.0,bcset_neumann_mms,NULL);CHKERRQ(ierr);
      ierr = FVDAFaceIterator(fv,DACELL_FACE_F,PETSC_TRUE,0.0,bcset_neumann_mms,NULL);CHKERRQ(ierr);
      ierr = FVDAFaceIterator(fv,DACELL_FACE_B,PETSC_TRUE,0.0,bcset_neumann_mms,NULL);CHKERRQ(ierr);
      break;

    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Valid choices for -bc are {0,1}");
      break;
  }
  
  dm = fv->dm_fv;
  
  if (operator_fvspace) {
    ierr = DMCreateMatrix(dm,&J);CHKERRQ(ierr);
  } else {
    ierr = FVDACreateMatrix(fv,DMDA_STENCIL_STAR,&J);CHKERRQ(ierr);
  }
  
  ierr = DMCreateGlobalVector(dm,&X);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&F);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&rhs);CHKERRQ(ierr);
  
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,dm);CHKERRQ(ierr);
  ierr = SNESSetSolution(snes,X);CHKERRQ(ierr);
  ierr = SNESSetApplicationContext(snes,(void*)fv);CHKERRQ(ierr);
  
  ierr = EvaluateRHS_mms(fv,0.0,rhs);CHKERRQ(ierr);
  
  ierr = SNESSetFunction(snes,F,eval_F,NULL);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,J,J,eval_J,NULL);CHKERRQ(ierr);
  
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = SNESFVDAConfigureGalerkinMG(snes,fv);CHKERRQ(ierr);
  
  ierr = SNESSolve(snes,rhs,X);CHKERRQ(ierr);
  
  ierr = EvaluateDiscretizationErrors(fv,0.0,X,&h,error);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"h %1.4e  L1 %1.4e  L2 %1.4e\n",h,error[0],error[1]);
  
  
  {
    Vec gradX;
    PetscInt Xm,XM;
    
    ierr = VecGetSize(X,&XM);CHKERRQ(ierr);
    ierr = VecGetLocalSize(X,&Xm);CHKERRQ(ierr);
    ierr = VecCreate(fv->comm,&gradX);CHKERRQ(ierr);
    ierr = VecSetSizes(gradX,Xm*3,XM*3);CHKERRQ(ierr);
    ierr = VecSetFromOptions(gradX);CHKERRQ(ierr);
    ierr = VecSetUp(gradX);CHKERRQ(ierr);
    
    ierr = FVDAGradientProject(fv,X,gradX);CHKERRQ(ierr);
    ierr = EvaluateDiscretizationGradientErrors(fv,0.0,gradX,&h,error);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"h %1.4e  H1 %1.4e  H1 %1.4e\n",h,error[0],error[1]);
    
    ierr = VecDestroy(&gradX);CHKERRQ(ierr);
  }
  
  if (view) {
    //ierr = FVDAView_CellData(fv,rhs,PETSC_TRUE,"rhs");CHKERRQ(ierr);
    ierr = FVDAView_CellData(fv,X,PETSC_TRUE,"ex3_mms_xcell");CHKERRQ(ierr);
    
    ierr = FVDAView_JSON(fv,NULL,"ex3_mms");CHKERRQ(ierr);
    {
      ierr = PetscVecWriteJSON(X,0,"ex3_mms_Q");CHKERRQ(ierr);
      ierr = FVDAView_Heavy(fv,NULL,"ex3_mms");CHKERRQ(ierr);
    }
  }

  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&rhs);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = FVDADestroy(&fv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode set_field(FVDA fv,PetscInt id,Vec F)
{
  PetscErrorCode  ierr;
  PetscReal       cell_x[3],dV;
  const PetscInt  NSD = 3;
  PetscReal       cell_coor[3 * DACELL3D_Q1_SIZE];
  Vec             coorl;
  const PetscReal *_geom_coor;
  PetscReal       *_F;
  PetscInt        i,d,c,dm_nel,dm_nen;
  const PetscInt  *dm_element,*element;
  
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  ierr = VecGetArray(F,&_F);CHKERRQ(ierr);
  
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(fv->dm_geometry,&coorl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_geometry,fv->vertex_coor_geometry,INSERT_VALUES,coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coorl,&_geom_coor);CHKERRQ(ierr);
  
  for (c=0; c<fv->ncells; c++) {
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * c];
    
    ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateCellVolume3d(cell_coor,&dV);
    
    /* average cell vertex coordinates */
    for (d=0; d<NSD; d++) {
      cell_x[d] = 0.0;
      for (i=0; i<DACELL3D_Q1_SIZE; i++) {
        cell_x[d] += cell_coor[NSD * i + d];
      }
      cell_x[d] /= (PetscReal)DACELL3D_Q1_SIZE;
    }
    
    if (id == 0) { _F[c] = cell_x[0]; }
    if (id == 1) { _F[c] = cell_x[1]; }
    if (id == 2) { _F[c] = cell_x[2]; }
  }
  ierr = VecRestoreArray(F,&_F);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(coorl,&_geom_coor);CHKERRQ(ierr);
  ierr = VecDestroy(&coorl);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode t3_warp_mms(void)
{
  PetscErrorCode ierr;
  PetscInt       mx = 12,bc = 0;
  PetscInt       m[] = {mx,mx,mx};
  FVDA           fv;
  Vec            X,F,rhs;
  Mat            J;
  DM             dm;
  SNES           snes;
  PetscBool      found = PETSC_FALSE;
  PetscReal      h,error[2];
  
  ierr = PetscOptionsGetInt(NULL,NULL,"-mx",&mx,&found);CHKERRQ(ierr);
  if (found) {
    m[0] = mx;
    m[1] = mx;
    m[2] = mx;
  }
  ierr = PetscOptionsGetInt(NULL,NULL,"-bc",&bc,NULL);CHKERRQ(ierr);
  
  ierr = FVDACreate(PETSC_COMM_WORLD,&fv);CHKERRQ(ierr);
  ierr = FVDASetDimension(fv,3);CHKERRQ(ierr);
  ierr = FVDASetSizes(fv,NULL,m);CHKERRQ(ierr);
  
  ierr = FVDASetProblemType(fv,PETSC_FALSE,FVDA_ELLIPTIC,0,0);CHKERRQ(ierr);
  
  ierr = FVDASetUp(fv);CHKERRQ(ierr);
  
  {
    Vec gcoor;
    PetscReal omega[] = {1.3, 1.1};
    
    ierr = DMDASetUniformCoordinates(fv->dm_geometry,-1.0,1.0,-1.0,1.0,-1.0,1.0);CHKERRQ(ierr);
    ierr = DMDAWarpCoordinates_SinJMax(fv->dm_geometry,0.01,omega);CHKERRQ(ierr);
    
    ierr = DMGetCoordinates(fv->dm_geometry,&gcoor);CHKERRQ(ierr);
    ierr = VecCopy(gcoor,fv->vertex_coor_geometry);CHKERRQ(ierr);
  }
  
  ierr = FVDAUpdateGeometry(fv);CHKERRQ(ierr);
  
  ierr = FVDARegisterFaceProperty(fv,"k",1);CHKERRQ(ierr);
  {
    PetscInt  f,nfaces;
    PetscReal *k;
    
    ierr = FVDAGetFaceInfo(fv,&nfaces,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = FVDAGetFacePropertyArray(fv,0,&k);CHKERRQ(ierr);
    for (f=0; f<nfaces; f++) {
      k[f] = 1.0;
    }
  }
  
  switch (bc) {
    case 0:
      ierr = FVDAFaceIterator(fv,DACELL_FACE_W,PETSC_FALSE,0.0,bcset_dirichlet_mms,NULL);CHKERRQ(ierr);
      ierr = FVDAFaceIterator(fv,DACELL_FACE_E,PETSC_FALSE,0.0,bcset_dirichlet_mms,NULL);CHKERRQ(ierr);
      ierr = FVDAFaceIterator(fv,DACELL_FACE_N,PETSC_FALSE,0.0,bcset_dirichlet_mms,NULL);CHKERRQ(ierr);
      ierr = FVDAFaceIterator(fv,DACELL_FACE_S,PETSC_FALSE,0.0,bcset_dirichlet_mms,NULL);CHKERRQ(ierr);
      ierr = FVDAFaceIterator(fv,DACELL_FACE_F,PETSC_FALSE,0.0,bcset_dirichlet_mms,NULL);CHKERRQ(ierr);
      ierr = FVDAFaceIterator(fv,DACELL_FACE_B,PETSC_FALSE,0.0,bcset_dirichlet_mms,NULL);CHKERRQ(ierr);
      break;
      
    case 1:
      ierr = FVDAFaceIterator(fv,DACELL_FACE_N,PETSC_FALSE,0.0,bcset_dirichlet_mms,NULL);CHKERRQ(ierr);
      ierr = FVDAFaceIterator(fv,DACELL_FACE_S,PETSC_FALSE,0.0,bcset_dirichlet_mms,NULL);CHKERRQ(ierr);
      
      ierr = FVDAFaceIterator(fv,DACELL_FACE_W,PETSC_TRUE,0.0,bcset_neumann_mms,NULL);CHKERRQ(ierr);
      ierr = FVDAFaceIterator(fv,DACELL_FACE_E,PETSC_TRUE,0.0,bcset_neumann_mms,NULL);CHKERRQ(ierr);
      ierr = FVDAFaceIterator(fv,DACELL_FACE_F,PETSC_TRUE,0.0,bcset_neumann_mms,NULL);CHKERRQ(ierr);
      ierr = FVDAFaceIterator(fv,DACELL_FACE_B,PETSC_TRUE,0.0,bcset_neumann_mms,NULL);CHKERRQ(ierr);
      break;
      
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Valid choices for -bc are {0,1}");
      break;
  }
  
  dm = fv->dm_fv;
  
  if (operator_fvspace) {
    ierr = DMCreateMatrix(dm,&J);CHKERRQ(ierr);
  } else {
    ierr = FVDACreateMatrix(fv,DMDA_STENCIL_STAR,&J);CHKERRQ(ierr);
  }
  
  ierr = DMCreateGlobalVector(dm,&X);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&F);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&rhs);CHKERRQ(ierr);
  
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,dm);CHKERRQ(ierr);
  ierr = SNESSetSolution(snes,X);CHKERRQ(ierr);
  ierr = SNESSetApplicationContext(snes,(void*)fv);CHKERRQ(ierr);
  
  ierr = EvaluateRHS_mms(fv,0.0,rhs);CHKERRQ(ierr);
  
  ierr = SNESSetFunction(snes,F,eval_F_hr,NULL);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,J,J,eval_J,NULL);CHKERRQ(ierr);
  //ierr = SNESSetJacobian(snes,J,J,SNESComputeJacobianDefaultColor,NULL);CHKERRQ(ierr);
  
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = SNESFVDAConfigureGalerkinMG(snes,fv);CHKERRQ(ierr);

  /*
  ierr = set_field(fv,2,X);CHKERRQ(ierr);
  ierr = FVDAView_CellData(fv,X,PETSC_TRUE,"xcell");CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
  */

  ierr = SNESSolve(snes,rhs,X);CHKERRQ(ierr);
  
  ierr = EvaluateDiscretizationErrors(fv,0.0,X,&h,error);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"h %1.4e  L1 %1.4e  L2 %1.4e\n",h,error[0],error[1]);
  
  {
    Vec gradX;
    PetscInt Xm,XM;
    
    ierr = VecGetSize(X,&XM);CHKERRQ(ierr);
    ierr = VecGetLocalSize(X,&Xm);CHKERRQ(ierr);
    ierr = VecCreate(fv->comm,&gradX);CHKERRQ(ierr);
    ierr = VecSetSizes(gradX,Xm*3,XM*3);CHKERRQ(ierr);
    ierr = VecSetFromOptions(gradX);CHKERRQ(ierr);
    ierr = VecSetUp(gradX);CHKERRQ(ierr);
    
    ierr = FVDAGradientProject(fv,X,gradX);CHKERRQ(ierr);
    ierr = EvaluateDiscretizationGradientErrors(fv,0.0,gradX,&h,error);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"h %1.4e  H1 %1.4e  H1 %1.4e\n",h,error[0],error[1]);
    
    ierr = VecDestroy(&gradX);CHKERRQ(ierr);
  }

  if (view) {
    //ierr = FVDAView_CellData(fv,rhs,PETSC_TRUE,"rhs");CHKERRQ(ierr);
    ierr = FVDAView_CellData(fv,X,PETSC_TRUE,"ex3_warp_mms_xcell");CHKERRQ(ierr);
    
    ierr = FVDAView_JSON(fv,NULL,"ex3_warp_mms");CHKERRQ(ierr);
    {
      ierr = PetscVecWriteJSON(X,0,"ex3_warp_mms_Q");CHKERRQ(ierr);
      ierr = FVDAView_Heavy(fv,NULL,"ex3_warp_mms");CHKERRQ(ierr);
    }
  }

  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&rhs);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = FVDADestroy(&fv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 
 [test] -tid 0
 
 [test] -tid 1 -pc_type gamg -ksp_rtol 1.0e-10 -mx {12, 24, 48} -bc 0

 [test] -tid 1 -pc_type gamg -ksp_rtol 1.0e-10 -mx {12, 24, 48} -bc 1
 note: errors associated with gradients are not expected to converge in this case
 
 [test] -tid 2 -pc_type gamg -ksp_rtol 1.0e-10 -mx {12, 24, 48} -bc 0
 [-] -pc_type gamg -ksp_type preonly -mg_levels_ksp_type chebyshev -mg_levels_pc_type ilu -mg_levels_max_it 4 -mx 64 -snes_linesearch_type basic
 [-] -pc_type gamg -ksp_type preonly -mg_levels_ksp_type chebyshev -mg_levels_pc_type ilu -mg_levels_max_it 4 -mx 64 -snes_linesearch_type basic -snes_mf_operator
 
 [test] -tid 2 -pc_type gamg -ksp_rtol 1.0e-10 -mx {12, 24, 48} -bc 1
 [-]
 [started with this]
 -snes_monitor -tid 2 -ksp_converged_reason -pc_type gamg -ksp_type preonly -snes_view -mg_levels_ksp_type chebyshev -mg_levels_pc_type ilu -snes_linesearch_type basic

 [nice opts]
 -snes_monitor -tid 2 -ksp_converged_reason -snes_viewx -mx 64 -snes_linesearch_type basic -pc_type ksp -snes_mf_operator -ksp_ksp_type preonly  -ksp_pc_type gamg -log_view -ksp_mg_levels_ksp_type chebyshev -ksp_mg_levels_pc_type ilu
 
*/
extern PetscErrorCode PCCreate_DMDARepart(PC pc);
int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscInt       tid = 0;
  
  ierr = PetscInitialize(&argc,&args,(char*)0,NULL);if (ierr) return ierr;
  ierr = PCRegister("dmdarepart",PCCreate_DMDARepart);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-tid",&tid,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-operator_fvspace",&operator_fvspace,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-view",&view,NULL);CHKERRQ(ierr);
  switch (tid) {
    case 0:
      ierr = t3();CHKERRQ(ierr);
      break;
    case 1:
      ierr = t3_mms();CHKERRQ(ierr);
      break;
    case 2:
      ierr = t3_warp_mms();CHKERRQ(ierr);
      break;
    default: SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Valid values for -tid {0,1,2}");
      break;
  }
  ierr = PetscFinalize();
  return ierr;
}
