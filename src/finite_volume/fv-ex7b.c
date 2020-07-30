
// extension velocity bc - domain enlarges
// advance mesh coords, advance Q (backward update)

#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <fvda.h>
#include <fvda_utils.h>


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

PetscReal vel_ex1(PetscReal coor[])
{
  return(coor[0]);
  /*
  if (coor[0] > 0) {
    return(1.0);
  } else {
    return(-1.0);
  }
  */
}

/* define movement in terms of a spatially dependent velocity field */
PetscErrorCode ale_update_mesh_geometry_ex1(DM dmg,Vec xk,Vec xk1,PetscReal dt)
{
  PetscErrorCode  ierr;
  PetscInt        k,len;
  const PetscReal *_xk;
  PetscReal       *_xk1;
  
  ierr = VecGetLocalSize(xk,&len);CHKERRQ(ierr);
  len = len / 3;
  ierr = VecGetArrayRead(xk,&_xk);CHKERRQ(ierr);
  ierr = VecGetArray(xk1,&_xk1);CHKERRQ(ierr);
  for (k=0; k<len; k++) {
    PetscReal pos[3],vel[3];
    
    pos[0] = _xk[3*k+0];
    pos[1] = _xk[3*k+1];
    pos[2] = _xk[3*k+2];
    
    vel[0] = vel_ex1(pos); // pos[0];
    vel[1] = 0.0;
    vel[2] = 0.0;
    
    _xk1[3*k+0] = pos[0] + dt * vel[0];
    _xk1[3*k+1] = pos[1] + dt * vel[1];
    _xk1[3*k+2] = pos[2] + dt * vel[2];
  }
  ierr = VecRestoreArray(xk1,&_xk1);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xk,&_xk);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ale_compute_source(FVDA fv,Vec xk,Vec xk1,PetscReal dt,Vec S)
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
    ierr = VecSetValue(S,row,((dV1-dV0)/dt)/dV0,INSERT_VALUES);CHKERRQ(ierr);
  }
  
  ierr = VecRestoreArrayRead(geometry_target_coorl,&_geom_target_coor);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(fv->dm_geometry,&geometry_target_coorl);CHKERRQ(ierr);
  
  ierr = VecRestoreArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);

  ierr = VecAssemblyBegin(S);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(S);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode ale_remap_scale(FVDA fv,Vec xk,Vec xk1,Vec Q)
{
  PetscErrorCode  ierr;
  Vec               geometry_coorl,geometry_target_coorl;
  const PetscScalar *_geom_coor,*_geom_target_coor;
  PetscScalar       *_LA_Q;
  PetscInt          c,row,offset;
  PetscInt          dm_nel,dm_nen;
  const PetscInt    *dm_element,*element;
  PetscReal         cell_coor[3*DACELL3D_VERTS],dV0,dV1;
  
  
  ierr = VecGetOwnershipRange(Q,&offset,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(Q,&_LA_Q);CHKERRQ(ierr);
  
  ierr = DMGetLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_geometry,xk,INSERT_VALUES,geometry_coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
  
  ierr = DMGetLocalVector(fv->dm_geometry,&geometry_target_coorl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_geometry,xk1,INSERT_VALUES,geometry_target_coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(geometry_target_coorl,&_geom_target_coor);CHKERRQ(ierr);
  
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  for (c=0; c<fv->ncells; c++) {
    PetscReal Q_c;
    
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * c];
    
    ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateCellVolume3d(cell_coor,&dV0);
    
    ierr = DACellGeometry3d_GetCoordinates(element,_geom_target_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateCellVolume3d(cell_coor,&dV1);
    
    row = offset + c;
    
    Q_c = _LA_Q[c];
    
    _LA_Q[c] = Q_c * (dV1 / dV0);
    printf(" c %d : ratio %+1.4e\n",c,dV1 / dV0);
  }
  
  ierr = VecRestoreArrayRead(geometry_target_coorl,&_geom_target_coor);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(fv->dm_geometry,&geometry_target_coorl);CHKERRQ(ierr);
  
  ierr = VecRestoreArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
  
  ierr = VecRestoreArray(Q,&_LA_Q);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode t7(void)
{
  PetscErrorCode ierr;
  PetscInt       mx = 10;
  const PetscInt m[] = {mx,mx,mx};
  FVDA           fv;
  Vec            X,Xk,F,coortarget,source;
  Mat            J;
  DM             dm;
  SNES           snes;
  PetscInt       nt,max;
  PetscReal      *dt = NULL;
  
  
  ierr = FVDACreate(PETSC_COMM_WORLD,&fv);CHKERRQ(ierr);
  ierr = FVDASetDimension(fv,3);CHKERRQ(ierr);
  ierr = FVDASetSizes(fv,NULL,m);CHKERRQ(ierr);
  
  ierr = FVDASetProblemType(fv,PETSC_TRUE,FVDA_HYPERBOLIC,0,0);CHKERRQ(ierr);
  
  ierr = FVDASetUp(fv);CHKERRQ(ierr);
  
  ierr = FVDASetup_ALE(fv);CHKERRQ(ierr);
  
  {
    Vec gcoor;
    
    ierr = DMDASetUniformCoordinates(fv->dm_geometry,0.0,1.0,0.0,1.0,0.0,1.0);CHKERRQ(ierr);
    ierr = DMGetCoordinates(fv->dm_geometry,&gcoor);CHKERRQ(ierr);
    ierr = VecCopy(gcoor,fv->vertex_coor_geometry);CHKERRQ(ierr);
  }
  
  ierr = FVDAUpdateGeometry(fv);CHKERRQ(ierr);
  
  ierr = FVDARegisterFaceProperty(fv,"v.n",1);CHKERRQ(ierr); /* always 0 in this test */
  ierr = FVDARegisterFaceProperty(fv,"xDot.n",1);CHKERRQ(ierr);
  ierr = FVDARegisterFaceProperty(fv,"xDot",3);CHKERRQ(ierr);
  
  /* set boundary value at intitial time */
  //ierr = FVDAFaceIterator(fv,DACELL_FACE_W,PETSC_FALSE,0.0,bcset_west,NULL);CHKERRQ(ierr);
  
  dm = fv->dm_fv;
  ierr = DMCreateMatrix(dm,&J);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  ierr = DMCreateGlobalVector(dm,&source);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&X);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&F);CHKERRQ(ierr);
  
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,dm);CHKERRQ(ierr);
  ierr = SNESSetSolution(snes,X);CHKERRQ(ierr);
  ierr = SNESSetApplicationContext(snes,(void*)fv);CHKERRQ(ierr);
  
  ierr = SNESSetFunction(snes,F,fvda_eval_F_forward_ale,NULL);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,J,J,SNESComputeJacobianDefaultColor,NULL);CHKERRQ(ierr);
  //ierr = SNESSetJacobian(snes,J,J,fvda_eval_J_timedep,NULL);CHKERRQ(ierr);
  
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  
  ierr = FVDAAccessData_ALE(fv,&dt,&Xk,&coortarget);CHKERRQ(ierr);
  
  max = 20;
  *dt = 0.01;

  /* Set the initial condition to be consistent with the ambient background Dirichlet BC */
  ierr = VecSet(Xk,0.3);CHKERRQ(ierr);
  
  /* View the initial condition */
  nt = 0;
  ierr = VecCopy(Xk,X);CHKERRQ(ierr);
  {
    PetscViewer viewer;
    char        fname[256];
    
    sprintf(fname,"step%.4d-x.vts",nt);
    ierr = PetscViewerCreate(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer,PETSCVIEWERVTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer,fname);CHKERRQ(ierr);
    ierr = VecView(X,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  
  for (nt=1; nt<max; nt++) {
    PetscViewer viewer;
    char        fname[256];
    
    printf("<<<<<<<<<< step %d >>>>>>>>>>\n",nt);
    
    
    /* evaluate fluid v . n flux */

    /* evaluate mesh v . n flux */
    
    /* imposed mesh velocity field */
    {
      PetscInt        f,nfaces;
      const PetscReal *face_centroid,*face_normal;
      PetscReal       *xDot;
      
      ierr = FVDAGetFaceInfo(fv,&nfaces,NULL,NULL,&face_normal,&face_centroid);CHKERRQ(ierr);
      ierr = FVDAGetFacePropertyArray(fv,2,&xDot);CHKERRQ(ierr);
      for (f=0; f<nfaces; f++) {
        PetscReal n[3],c[3],v[3];
        
        n[0] = face_normal[3*f+0];
        n[1] = face_normal[3*f+1];
        n[2] = face_normal[3*f+2];
        
        c[0] = face_centroid[3*f+0];
        c[1] = face_centroid[3*f+1];
        c[2] = face_centroid[3*f+2];
        
        v[0] = vel_ex1(c);
        v[1] = 0;
        v[2] = 0;
        
        xDot[3*f+0] = v[0];
        xDot[3*f+1] = v[1];
        xDot[3*f+2] = v[2];
      }
    }

    
    /* imposed mesh velocity field */
    {
      PetscInt        f,nfaces;
      const PetscReal *face_centroid,*face_normal;
      PetscReal       *xDotdotn;
      
      ierr = FVDAGetFaceInfo(fv,&nfaces,NULL,NULL,&face_normal,&face_centroid);CHKERRQ(ierr);
      ierr = FVDAGetFacePropertyArray(fv,1,&xDotdotn);CHKERRQ(ierr);
      for (f=0; f<nfaces; f++) {
        PetscReal n[3],c[3],v[3];

        n[0] = face_normal[3*f+0];
        n[1] = face_normal[3*f+1];
        n[2] = face_normal[3*f+2];

        c[0] = face_centroid[3*f+0];
        c[1] = face_centroid[3*f+1];
        c[2] = face_centroid[3*f+2];
        
        v[0] = vel_ex1(c);
        v[1] = 0;
        v[2] = 0;
        
        xDotdotn[f] = v[0]*n[0] + v[1]*n[1] + v[2]*n[2];
        //xDotdotn[f] = 0;
      }
    }

    /* define mesh velocity */
    ierr = ale_update_mesh_geometry_ex1(fv->dm_geometry,fv->vertex_coor_geometry,coortarget,*dt);CHKERRQ(ierr);
    
    /* update mesh position */
    ierr = VecCopy(coortarget,fv->vertex_coor_geometry);CHKERRQ(ierr);
    {
      Vec gcoor;
      ierr = DMGetCoordinates(fv->dm_geometry,&gcoor);CHKERRQ(ierr);
      ierr = VecCopy(coortarget,gcoor);CHKERRQ(ierr);
    }
    ierr = FVDAUpdateGeometry(fv);CHKERRQ(ierr);

    ierr = ale_compute_source(fv,fv->vertex_coor_geometry,coortarget,*dt,source);CHKERRQ(ierr);
    
    ierr = FVDAPostProcessCompatibleVelocity(fv,"xDot","xDot.n",source,NULL);CHKERRQ(ierr);

    //ierr = FVDAIntegrateFlux(fv,"xDot.n",PETSC_TRUE,source);CHKERRQ(ierr);

    /* combine (v - v_mesh) . n flux */
    {
      PetscInt        f,nfaces;
      PetscReal       *vdotn,*xDotdotn;
      
      ierr = FVDAGetFaceInfo(fv,&nfaces,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
      ierr = FVDAGetFacePropertyArray(fv,0,&vdotn);CHKERRQ(ierr);
      ierr = FVDAGetFacePropertyArray(fv,1,&xDotdotn);CHKERRQ(ierr);
      for (f=0; f<nfaces; f++) {
        vdotn[f] = - xDotdotn[f];
      }
    }
    
    /* Push current state into old state */
    ierr = VecCopy(X,Xk);CHKERRQ(ierr);
  
    
    ierr = SNESSolve(snes,NULL,X);CHKERRQ(ierr);

    {
      PetscReal Xmin,Xmax;
      
      ierr = VecMin(X,NULL,&Xmin);CHKERRQ(ierr);
      ierr = VecMax(X,NULL,&Xmax);CHKERRQ(ierr);
      PetscPrintf(PETSC_COMM_WORLD,"[step %4d] min(X) %+1.4e max(X) %+1.4e\n",nt,Xmin,Xmax);
    }

    
    
    
    sprintf(fname,"step%.4d-x.vts",nt);
    ierr = PetscViewerCreate(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer,PETSCVIEWERVTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer,fname);CHKERRQ(ierr);
    ierr = VecView(X,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&source);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = FVDADestroy(&fv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  
  ierr = PetscInitialize(&argc,&args,(char*)0,NULL);if (ierr) return ierr;
  ierr = t7();CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
