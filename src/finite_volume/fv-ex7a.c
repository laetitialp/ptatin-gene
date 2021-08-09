
// shortening velocity bc - domain shirnks
// advance Q, advance mesh coords (forward update)
// KEEP

#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <fvda_impl.h>
#include <fvda.h>
#include <fvda_private.h>
#include <fvda_utils.h>

PetscErrorCode bcset_x_dir_right(FVDA fv,
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
    bcvalue[f] = 0.1;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode bcset_x_dir_NULL(FVDA fv,
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
    bcvalue[f] = 0.0;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode bcset_x_dir_NULL_n(FVDA fv,
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
    flux[f] = FVFLUX_NEUMANN_CONSTRAINT;
    bcvalue[f] = 0.0;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode bcset_x_dir(FVDA fv,
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

PetscReal vel_ex1(PetscReal coor[])
{
  return(coor[0]);
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

/*
 Computes
 v = (x1 - x0)/dt
*/
PetscErrorCode ale_compute_v_mesh(DM dmg,Vec x0,Vec x1,PetscReal dt,Vec v)
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

/*
 loop over faces
 pick a valid element
 extract all cell velocity values
 extract values associated with the face
 average face velocities
 set avg onto fv face property storage
*/
PetscErrorCode ale_compute_face_v_mesh(DM dmg,Vec x0,Vec v,FVDA fv,const char face_vec_name[])
{
  PetscErrorCode  ierr;
  Vec             vl;
  const PetscReal *_vl,*x_face;
  PetscReal       *v_face;
  DACellFace      cell_face_label;
  PetscInt        fidx[DACELL3D_FACE_VERTS];
  PetscInt        f,i,cellid;
  PetscInt        dm_nel,dm_nen;
  const PetscInt  *dm_element,*element;
  PetscReal       cell_v[3*DACELL3D_VERTS];
  
  ierr = DMCreateLocalVector(dmg,&vl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(dmg,v,INSERT_VALUES,vl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(vl,&_vl);CHKERRQ(ierr);

  ierr = FVDAGetFacePropertyByNameArray(fv,face_vec_name,&v_face);CHKERRQ(ierr);
  ierr = FVDAGetFaceInfo(fv,NULL,NULL,NULL,NULL,&x_face);CHKERRQ(ierr);
  
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  
  for (f=0; f<fv->nfaces; f++) {
    PetscReal avg_v[] = {0,0,0};
    
    cell_face_label = fv->face_type[f];
    
    ierr = FVDAGetValidElement(fv,f,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];

    for (i=0; i<DACELL3D_VERTS; i++) {
      cell_v[3*i+0] = _vl[3*element[i]+0];
      cell_v[3*i+1] = _vl[3*element[i]+1];
      cell_v[3*i+2] = _vl[3*element[i]+2];
    }
    
    ierr = DACellGeometry3d_GetFaceIndices(NULL,cell_face_label,fidx);CHKERRQ(ierr);
    
    for (i=0; i<DACELL3D_FACE_VERTS; i++) {
      avg_v[0] += cell_v[3*fidx[i]  ];
      avg_v[1] += cell_v[3*fidx[i]+1];
      avg_v[2] += cell_v[3*fidx[i]+2];
    }
    avg_v[0] = avg_v[0] * 0.25;
    avg_v[1] = avg_v[1] * 0.25;
    avg_v[2] = avg_v[2] * 0.25;
    
    v_face[3*f+0] = avg_v[0];
    v_face[3*f+1] = avg_v[1];
    v_face[3*f+2] = avg_v[2];
    //printf("[face f %d] x: %+1.4e %+1.4e %+1.4e avg: %+1.4e %+1.4e %+1.4e\n",f,x_face[3*f+0],x_face[3*f+1],x_face[3*f+2],avg_v[0],avg_v[1],avg_v[2]);
  }
  
  ierr = VecRestoreArrayRead(vl,&_vl);CHKERRQ(ierr);
  ierr = VecDestroy(&vl);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}


PetscErrorCode ale_forward_compute_source(FVDA fv,Vec xk,Vec xk1,PetscReal dt,Vec S,PetscBool forward)
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

PetscErrorCode t7_forward(void)
{
  PetscErrorCode ierr;
  PetscInt       mx = 100;
  const PetscInt m[] = {mx,4,4};
  FVDA           fv;
  Vec            X,Xk,F,coortarget,source,v_mesh;
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
    
    ierr = DMDASetUniformCoordinates(fv->dm_geometry,0.0,1.0,-1.0,1.0,-1.0,1.0);CHKERRQ(ierr);
    ierr = DMGetCoordinates(fv->dm_geometry,&gcoor);CHKERRQ(ierr);
    ierr = VecCopy(gcoor,fv->vertex_coor_geometry);CHKERRQ(ierr);
  }
  
  ierr = FVDAUpdateGeometry(fv);CHKERRQ(ierr);
  
  ierr = FVDARegisterFaceProperty(fv,"v.n",1);CHKERRQ(ierr); /* always 0 in this test */
  ierr = FVDARegisterFaceProperty(fv,"xDot.n",1);CHKERRQ(ierr);
  ierr = FVDARegisterFaceProperty(fv,"xDot",3);CHKERRQ(ierr);
  
  dm = fv->dm_fv;
  ierr = DMCreateMatrix(dm,&J);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  ierr = DMCreateGlobalVector(dm,&source);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&X);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&F);CHKERRQ(ierr);
  
  ierr = DMCreateGlobalVector(fv->dm_geometry,&v_mesh);CHKERRQ(ierr);
  
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,dm);CHKERRQ(ierr);
  ierr = SNESSetSolution(snes,X);CHKERRQ(ierr);
  ierr = SNESSetApplicationContext(snes,(void*)fv);CHKERRQ(ierr);
  
  ierr = SNESSetFunction(snes,F,fvda_eval_F_forward_ale,NULL);CHKERRQ(ierr);
  //ierr = SNESSetJacobian(snes,J,J,SNESComputeJacobianDefaultColor,NULL);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,J,J,fvda_eval_J_forward_ale,NULL);CHKERRQ(ierr);
  
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  
  ierr = FVDAAccessData_ALE(fv,&dt,&Xk,&coortarget);CHKERRQ(ierr);
  
  max = 100;
  *dt = 0.005;

  //ierr = FVDAFaceIterator(fv,DACELL_FACE_E,PETSC_FALSE,0.0,bcset_x_dir_right,NULL);CHKERRQ(ierr);
  ierr = FVDAFaceIterator(fv,DACELL_FACE_E,PETSC_FALSE,0.0,bcset_x_dir_NULL_n,NULL);CHKERRQ(ierr);
  ierr = FVDAFaceIterator(fv,DACELL_FACE_W,PETSC_FALSE,0.0,bcset_x_dir,NULL);CHKERRQ(ierr);

  
  /* Set the initial condition to be consistent with the ambient background Dirichlet BC */
  ierr = VecSet(Xk,0.3);CHKERRQ(ierr);
  {
    Vec cellcoor;
    PetscReal *_cellcoor,*_Xk;
    PetscInt c;
    
    DMGetCoordinates(fv->dm_fv,&cellcoor);
    VecGetArray(cellcoor,&_cellcoor);
    VecGetArray(Xk,&_Xk);
    
    for (c=0; c<fv->ncells; c++) {
      if (_cellcoor[3*c+0] > 0.5) {
        _Xk[c] = 0.1;
      }
    }
    VecRestoreArray(cellcoor,&_cellcoor);
    VecRestoreArray(Xk,&_Xk);
  }

  /* View the initial condition */
  nt = 0;
  ierr = VecCopy(Xk,X);CHKERRQ(ierr);
  {
    PetscViewer viewer;
    char        fname[256];
    
    sprintf(fname,"step%.4d-x.vts",(int)nt);
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
    
    PetscPrintf(PETSC_COMM_WORLD,"<<<<<<<<<< step %d >>>>>>>>>>\n",nt);
    
    ierr = ale_update_mesh_geometry_ex1(fv->dm_geometry,fv->vertex_coor_geometry,coortarget,*dt);CHKERRQ(ierr);

    ierr = ale_compute_v_mesh(fv->dm_geometry,fv->vertex_coor_geometry,coortarget,*dt,v_mesh);CHKERRQ(ierr);
    
    ierr = ale_compute_face_v_mesh(fv->dm_geometry,fv->vertex_coor_geometry,v_mesh,fv,"xDot");CHKERRQ(ierr);
    
    ierr = ale_forward_compute_source(fv,fv->vertex_coor_geometry,coortarget,*dt,source,PETSC_TRUE);CHKERRQ(ierr);
    
    ierr = FVDAPostProcessCompatibleVelocity(fv,"xDot","xDot.n",source,NULL);CHKERRQ(ierr);

    /* compute V1 - dt \int_S0 v.n dS and compare with V0 */
    {
      PetscInt c;
      Vec      coor0_l,coor1_l;
      const PetscReal *_coor0,*_coor1;
      Vec x0,x1;
      PetscReal dV0,dV1;
      PetscInt          dm_nel,dm_nen;
      const PetscInt    *dm_element,*element;
      PetscReal         cell_coor[3*DACELL3D_VERTS];
      const PetscReal *_source;
      PetscReal delta;
      
      ierr = FVDAIntegrateFlux(fv,"xDot.n",PETSC_TRUE,source);CHKERRQ(ierr);
      //printf("[source \\int_s]\n");
      //VecView(source,PETSC_VIEWER_STDOUT_WORLD);
      
      ierr = VecGetArrayRead(source,&_source);CHKERRQ(ierr);

      x0 = fv->vertex_coor_geometry;
      x1 = coortarget;
      
      ierr = DMGetLocalVector(fv->dm_geometry,&coor0_l);CHKERRQ(ierr);
      ierr = DMGlobalToLocal(fv->dm_geometry,x0,INSERT_VALUES,coor0_l);CHKERRQ(ierr);
      ierr = VecGetArrayRead(coor0_l,&_coor0);CHKERRQ(ierr);
      
      ierr = DMGetLocalVector(fv->dm_geometry,&coor1_l);CHKERRQ(ierr);
      ierr = DMGlobalToLocal(fv->dm_geometry,x1,INSERT_VALUES,coor1_l);CHKERRQ(ierr);
      ierr = VecGetArrayRead(coor1_l,&_coor1);CHKERRQ(ierr);
      
      ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
      for (c=0; c<fv->ncells; c++) {
        element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * c];
        
        ierr = DACellGeometry3d_GetCoordinates(element,_coor0,cell_coor);CHKERRQ(ierr);
        _EvaluateCellVolume3d(cell_coor,&dV0);
        
        ierr = DACellGeometry3d_GetCoordinates(element,_coor1,cell_coor);CHKERRQ(ierr);
        _EvaluateCellVolume3d(cell_coor,&dV1);
        
/*
          printf("[cell %d] v.n dS %+1.6e dV0 %+1.6e dV1 %+1.6e\n",c,_source[c],dV0,dV1);
          printf("          V1-dt.source %+1.6e || V0 %+1.6e\n",dV1 - (*dt) *_source[c],dV0);
*/
        
        delta = (dV1 - (*dt) *_source[c]) - dV0;
        delta = fabs(delta) / dV0;
        if (delta > 1.0e-12) {
          printf("[cell %d] v.n dS %+1.6e dV0 %+1.6e dV1 %+1.6e\n",(int)c,_source[c],dV0,dV1);
          printf("          V1-dt.source %+1.6e || V0 %+1.6e\n",dV1 - (*dt) *_source[c],dV0);
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"GCL violated");
        }
      }
      
      ierr = VecRestoreArrayRead(coor1_l,&_coor1);CHKERRQ(ierr);
      ierr = DMRestoreLocalVector(fv->dm_geometry,&coor1_l);CHKERRQ(ierr);

      ierr = VecRestoreArrayRead(coor0_l,&_coor0);CHKERRQ(ierr);
      ierr = DMRestoreLocalVector(fv->dm_geometry,&coor0_l);CHKERRQ(ierr);

      ierr = VecRestoreArrayRead(source,&_source);CHKERRQ(ierr);
    }
    
    /* combine (v - v_mesh) . n flux */
    {
      PetscInt        f,nfaces;
      const PetscReal *face_centroid,*face_normal;
      PetscReal       *vdotn,*xDotdotn;
      
      ierr = FVDAGetFaceInfo(fv,&nfaces,NULL,NULL,&face_normal,&face_centroid);CHKERRQ(ierr);
      ierr = FVDAGetFacePropertyArray(fv,0,&vdotn);CHKERRQ(ierr);
      ierr = FVDAGetFacePropertyArray(fv,1,&xDotdotn);CHKERRQ(ierr);
      for (f=0; f<nfaces; f++) {
        vdotn[f] = -xDotdotn[f];
        
        //vdotn[f] = 1.0*face_normal[3*f+0] + 0*face_normal[3*f+1] + 0*face_normal[3*f+2];
        //vdotn[f] = face_centroid[3*f+0]*face_normal[3*f+0] + 0*face_normal[3*f+1] + 0*face_normal[3*f+2];
        //vdotn[f] = -1.0*face_normal[3*f+0] + 0*face_normal[3*f+1] + 0*face_normal[3*f+2];

        //if (fabs(face_normal[3*f+0]) > 0.99) {
        //  printf(" vmesh.n: %+1.6e , x: %+1.6e %+1.6e %+1.6e, n: %+1.6e %+1.6e %+1.6e\n",vdotn[f],face_centroid[3*f+0],face_centroid[3*f+1],face_centroid[3*f+2],face_normal[3*f+0],face_normal[3*f+1],face_normal[3*f+2]);
        //}
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
    
    /* update mesh position */
    ierr = VecCopy(coortarget,fv->vertex_coor_geometry);CHKERRQ(ierr);
    {
      Vec gcoor;
      ierr = DMGetCoordinates(fv->dm_geometry,&gcoor);CHKERRQ(ierr);
      ierr = VecCopy(coortarget,gcoor);CHKERRQ(ierr);
    }
    ierr = FVDAUpdateGeometry(fv);CHKERRQ(ierr);

    
    sprintf(fname,"step%.4d-x.vts",(int)nt);
    ierr = PetscViewerCreate(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer,PETSCVIEWERVTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer,fname);CHKERRQ(ierr);
    ierr = VecView(X,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  }
  
  
  ierr = VecDestroy(&v_mesh);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&source);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = FVDADestroy(&fv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode t7_backward(void)
{
  PetscErrorCode ierr;
  PetscInt       mx = 10;
  const PetscInt m[] = {mx,mx,mx};
  FVDA           fv;
  Vec            X,Xk,F,coortarget,source,v_mesh,coorprev;
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
    
    ierr = DMDASetUniformCoordinates(fv->dm_geometry,0.0,1.0,-1.0,1.0,-1.0,1.0);CHKERRQ(ierr);
    ierr = DMGetCoordinates(fv->dm_geometry,&gcoor);CHKERRQ(ierr);
    ierr = VecCopy(gcoor,fv->vertex_coor_geometry);CHKERRQ(ierr);
  }
  
  ierr = FVDAUpdateGeometry(fv);CHKERRQ(ierr);
  
  ierr = FVDARegisterFaceProperty(fv,"v.n",1);CHKERRQ(ierr); /* always 0 in this test */
  ierr = FVDARegisterFaceProperty(fv,"xDot.n",1);CHKERRQ(ierr);
  ierr = FVDARegisterFaceProperty(fv,"xDot",3);CHKERRQ(ierr);
  
  dm = fv->dm_fv;
  ierr = DMCreateMatrix(dm,&J);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  ierr = DMCreateGlobalVector(dm,&source);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&X);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&F);CHKERRQ(ierr);
  
  ierr = DMCreateGlobalVector(fv->dm_geometry,&v_mesh);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(fv->dm_geometry,&coorprev);CHKERRQ(ierr);
  
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,dm);CHKERRQ(ierr);
  ierr = SNESSetSolution(snes,X);CHKERRQ(ierr);
  ierr = SNESSetApplicationContext(snes,(void*)fv);CHKERRQ(ierr);
  
  ierr = SNESSetFunction(snes,F,fvda_eval_F_backward_ale,NULL);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,J,J,SNESComputeJacobianDefaultColor,NULL);CHKERRQ(ierr);
  //ierr = SNESSetJacobian(snes,J,J,fvda_eval_J_timedep,NULL);CHKERRQ(ierr);
  
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  
  ierr = FVDAAccessData_ALE(fv,&dt,&Xk,&coortarget);CHKERRQ(ierr);
  
  max = 30;
  *dt = 0.05;
  
  ierr = FVDAFaceIterator(fv,DACELL_FACE_E,PETSC_FALSE,0.0,bcset_x_dir,NULL);CHKERRQ(ierr);
  ierr = FVDAFaceIterator(fv,DACELL_FACE_W,PETSC_FALSE,0.0,bcset_x_dir,NULL);CHKERRQ(ierr);
  
  
  /* Set the initial condition to be consistent with the ambient background Dirichlet BC */
  ierr = VecSet(Xk,0.3);CHKERRQ(ierr);
  
  /* View the initial condition */
  nt = 0;
  ierr = VecCopy(Xk,X);CHKERRQ(ierr);
  {
    PetscViewer viewer;
    char        fname[256];
    
    sprintf(fname,"step%.4d-x.vts",(int)nt);
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
    
    PetscPrintf(PETSC_COMM_WORLD,"<<<<<<<<<< step %d >>>>>>>>>>\n",nt);
    
    ierr = ale_update_mesh_geometry_ex1(fv->dm_geometry,fv->vertex_coor_geometry,coortarget,*dt);CHKERRQ(ierr);
    
    ierr = ale_compute_v_mesh(fv->dm_geometry,fv->vertex_coor_geometry,coortarget,*dt,v_mesh);CHKERRQ(ierr);
    
    ierr = ale_compute_face_v_mesh(fv->dm_geometry,fv->vertex_coor_geometry,v_mesh,fv,"xDot");CHKERRQ(ierr);

    ierr = ale_forward_compute_source(fv,fv->vertex_coor_geometry,coortarget,*dt,source,PETSC_FALSE);CHKERRQ(ierr);
    
    
    /* update mesh position */
    ierr = VecCopy(fv->vertex_coor_geometry,coorprev);CHKERRQ(ierr);
    
    ierr = VecCopy(coortarget,fv->vertex_coor_geometry);CHKERRQ(ierr);
    {
      Vec gcoor;
      ierr = DMGetCoordinates(fv->dm_geometry,&gcoor);CHKERRQ(ierr);
      ierr = VecCopy(coortarget,gcoor);CHKERRQ(ierr);
    }
    ierr = FVDAUpdateGeometry(fv);CHKERRQ(ierr);
    
    ierr = FVDAPostProcessCompatibleVelocity(fv,"xDot","xDot.n",source,NULL);CHKERRQ(ierr);
    
    /* compute V1 - dt \int_S0 v.n dS and compare with V1 */
    {
      PetscInt c;
      Vec      coor0_l,coor1_l;
      const PetscReal *_coor0,*_coor1;
      Vec x0,x1;
      PetscReal dV0,dV1;
      PetscInt          dm_nel,dm_nen;
      const PetscInt    *dm_element,*element;
      PetscReal         cell_coor[3*DACELL3D_VERTS];
      const PetscReal *_source;
      PetscReal delta;
      
      ierr = FVDAIntegrateFlux(fv,"xDot.n",PETSC_TRUE,source);CHKERRQ(ierr);
      //printf("[source \\int_s]\n");
      //VecView(source,PETSC_VIEWER_STDOUT_WORLD);
      
      ierr = VecGetArrayRead(source,&_source);CHKERRQ(ierr);
      
      x0 = coorprev;
      x1 = coortarget;
      
      ierr = DMGetLocalVector(fv->dm_geometry,&coor0_l);CHKERRQ(ierr);
      ierr = DMGlobalToLocal(fv->dm_geometry,x0,INSERT_VALUES,coor0_l);CHKERRQ(ierr);
      ierr = VecGetArrayRead(coor0_l,&_coor0);CHKERRQ(ierr);
      
      ierr = DMGetLocalVector(fv->dm_geometry,&coor1_l);CHKERRQ(ierr);
      ierr = DMGlobalToLocal(fv->dm_geometry,x1,INSERT_VALUES,coor1_l);CHKERRQ(ierr);
      ierr = VecGetArrayRead(coor1_l,&_coor1);CHKERRQ(ierr);
      
      ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
      for (c=0; c<fv->ncells; c++) {
        element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * c];
        
        ierr = DACellGeometry3d_GetCoordinates(element,_coor0,cell_coor);CHKERRQ(ierr);
        _EvaluateCellVolume3d(cell_coor,&dV0);
        
        ierr = DACellGeometry3d_GetCoordinates(element,_coor1,cell_coor);CHKERRQ(ierr);
        _EvaluateCellVolume3d(cell_coor,&dV1);
        
/*
          printf("[cell %d] v.n dS %+1.6e dV0 %+1.6e dV1 %+1.6e\n",c,_source[c],dV0,dV1);
          printf("          V1-dt.source %+1.6e || V1 %+1.6e\n",dV0 - (*dt) *_source[c],dV0);
*/
        
        delta = (dV1 - (*dt) *_source[c]) - dV0;
        delta = fabs(delta) / dV1;
        if (delta > 1.0e-12) {
          printf("[cell %d] v.n dS %+1.6e dV0 %+1.6e dV1 %+1.6e\n",(int)c,_source[c],dV0,dV1);
          printf("          V1-dt.source %+1.6e || V0 %+1.6e\n",dV1 - (*dt) *_source[c],dV0);
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"GCL violated");
        }
      }
      
      ierr = VecRestoreArrayRead(coor1_l,&_coor1);CHKERRQ(ierr);
      ierr = DMRestoreLocalVector(fv->dm_geometry,&coor1_l);CHKERRQ(ierr);
      
      ierr = VecRestoreArrayRead(coor0_l,&_coor0);CHKERRQ(ierr);
      ierr = DMRestoreLocalVector(fv->dm_geometry,&coor0_l);CHKERRQ(ierr);
      
      ierr = VecRestoreArrayRead(source,&_source);CHKERRQ(ierr);
    }
    
    /* combine (v - v_mesh) . n flux */
    {
      PetscInt        f,nfaces;
      const PetscReal *face_centroid,*face_normal;
      PetscReal       *vdotn,*xDotdotn;
      
      ierr = FVDAGetFaceInfo(fv,&nfaces,NULL,NULL,&face_normal,&face_centroid);CHKERRQ(ierr);
      ierr = FVDAGetFacePropertyArray(fv,0,&vdotn);CHKERRQ(ierr);
      ierr = FVDAGetFacePropertyArray(fv,1,&xDotdotn);CHKERRQ(ierr);
      for (f=0; f<nfaces; f++) {
        vdotn[f] = - xDotdotn[f];
      }
    }
    
    /* Push current state into old state */
    ierr = VecCopy(X,Xk);CHKERRQ(ierr);
    
    ierr = VecCopy(coorprev,fv->vertex_coor_geometry);CHKERRQ(ierr);
    
    ierr = SNESSolve(snes,NULL,X);CHKERRQ(ierr);

    ierr = VecCopy(fv->vertex_coor_geometry,coortarget);CHKERRQ(ierr);

    {
      PetscReal Xmin,Xmax;
      
      ierr = VecMin(X,NULL,&Xmin);CHKERRQ(ierr);
      ierr = VecMax(X,NULL,&Xmax);CHKERRQ(ierr);
      PetscPrintf(PETSC_COMM_WORLD,"[step %4d] min(X) %+1.4e max(X) %+1.4e\n",nt,Xmin,Xmax);
    }
    
    
    
    sprintf(fname,"step%.4d-x.vts",(int)nt);
    ierr = PetscViewerCreate(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer,PETSCVIEWERVTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer,fname);CHKERRQ(ierr);
    ierr = VecView(X,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    
  }
  
  
  ierr = VecDestroy(&coorprev);CHKERRQ(ierr);
  ierr = VecDestroy(&v_mesh);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&source);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = FVDADestroy(&fv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 ./arch-darwin-c-debug/bin/fv-ex7a.app -ksp_monitor -pc_type bjacobi -ksp_type fgmres -fvpp_monitor -fvpp_pc_type lu -ksp_converged_reason -pc_type lu -pc_factor_mat_solver_type umfpack
*/
int main(int argc,char **args)
{
  PetscErrorCode ierr;
  
  ierr = PetscInitialize(&argc,&args,(char*)0,NULL);if (ierr) return ierr;
  ierr = t7_forward();CHKERRQ(ierr);
  //ierr = t7_backward();CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
