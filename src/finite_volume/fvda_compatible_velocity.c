
#include <petsc.h>
#include <petscvec.h>
#include <petscmat.h>
#include <petscksp.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <fvda_impl.h>
#include <fvda.h>
#include <fvda_utils.h>

//#define FVPP_LOG

PetscErrorCode _FVPostProcessCompatibleVelocity_SEQ(FVDA fv,const char name_v[],const char name_v_dot_n[],PetscReal w[],Vec source);
PetscErrorCode _FVPostProcessCompatibleVelocity_v2_SEQ(FVDA fv,const char name_v[],const char name_v_dot_n[],Vec source,KSP ksp);
PetscErrorCode _FVPostProcessCompatibleVelocity_v2_MPI(FVDA fv,const char name_v[],const char name_v_dot_n[],Vec source,KSP ksp);

/*
 Performs the post-processing described in this paper
 
 "Postprocessing of non-conservative flux for compatibility with transport in heterogeneous media"
 Lars H. Odsaetera, Mary F. Wheeler, Trond Kvamsdala, Mats G. Larson
 Comput. Methods Appl. Mech. Engrg. 315 (2017) 799–830
*/

/*
 - Assume user has registered the field for the output. This is to ensure that the order of the fields
 is controlled / known to the user
 - The input source defines the cell-wise constant (e.g. average) value associated with div(u)
 - The corrected value for u.n will be stored in the array named "name_v_dot_n"
 */
PetscErrorCode _FVPostProcessCompatibleVelocity_SEQ(FVDA fv,const char name_v[],const char name_v_dot_n[],PetscReal w[],Vec source)
{
  PetscErrorCode    ierr;
  Vec               R,y;
  Mat               M;
  KSP               ksp;
  PC                pc;
  const PetscScalar *LA_s,*LA_y;
  const PetscReal   *edge_velocity;
  PetscReal         *_un,*_vn;
  PetscInt          *edge_table;
  PetscInt          c,ncells,e,nedges_interior,nedges;
  PetscReal         *weights;
#ifdef FVPP_LOG
  FILE              *fpi = NULL;
#endif
  const PetscInt    nsd = 3;
  PetscReal         *dS,*dV;
  PetscInt          dir,cellid;
  Vec               geometry_coorl;
  const PetscScalar *_geom_coor;
  PetscReal         cell_coor[3*DACELL3D_VERTS];
  PetscInt          dm_nel,dm_nen;
  const PetscInt    *dm_element,*element;
  PetscBool         monitor = PETSC_FALSE;
  
  
  PetscFunctionBegin;
  ierr = PetscOptionsGetBool(NULL,NULL,"-fvpp_monitor",&monitor,NULL);CHKERRQ(ierr);
  ncells          = fv->ncells;
  nedges_interior = fv->nfaces_interior;
  nedges          = fv->nfaces;
  edge_table      = fv->face_fv_map;
  
#ifdef FVPP_LOG
  fpi = fopen("fvpp-compat.dat","w");
  if (!fpi) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Failed to open file: fvpp-compat.dat");
  fprintf(fpi,"#edge <v.n (input)> <correction> <v.n (output)>\n");
#endif
  
  ierr = FVDAGetFacePropertyByNameArrayRead(fv,name_v,&edge_velocity);CHKERRQ(ierr);
  ierr = FVDAGetFacePropertyByNameArray(fv,name_v_dot_n,&_vn);CHKERRQ(ierr);
  
  ierr = PetscMalloc1(nedges,&_un);CHKERRQ(ierr);
  ierr = PetscMalloc1(nedges,&dS);CHKERRQ(ierr);
  ierr = PetscMalloc1(ncells,&dV);CHKERRQ(ierr);
  ierr = PetscMalloc1(nedges,&weights);CHKERRQ(ierr);
  
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);

  ierr = DMGetLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_geometry,fv->vertex_coor_geometry,INSERT_VALUES,geometry_coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);

  ierr = VecCreate(PetscObjectComm((PetscObject)fv->dm_fv),&R);CHKERRQ(ierr);
  ierr = VecSetSizes(R,ncells,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(R);CHKERRQ(ierr);

  for (e=0; e<nedges; e++) {
    ierr = FVDAGetValidElement(fv,e,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
    ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateFaceArea3d(fv->face_type[e],cell_coor,&dS[e]);
  }
  
  for (c=0; c<ncells; c++) {
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * c];
    ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateCellVolume3d(cell_coor,&dV[c]);
  }
  
  /* 1: compute u.n along each facet */
  for (e=0; e<nedges; e++) {
    PetscReal Fn = 0;
    for (dir=0; dir<nsd; dir++) {
      Fn += edge_velocity[nsd*e+dir] * fv->face_normal[nsd*e+dir];
    }
    _un[e] = Fn;
  }
  
  /* 2: assemble residual */
  ierr = VecZeroEntries(R);CHKERRQ(ierr);
  
  /* volume terms */
  ierr = VecGetArrayRead(source,&LA_s);CHKERRQ(ierr);
  for (c=0; c<ncells; c++) {
    PetscReal r;
    
    r = LA_s[c] * dV[c];
    ierr = VecSetValue(R,c,r,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(source,&LA_s);CHKERRQ(ierr);
  
  /* face terms */
  for (e=0; e<nedges; e++) {
    PetscReal Fn,i_Fn,r;
    PetscInt elow,ehigh;
    
    elow  = fv->face_fv_map[2*e+0];
    ehigh = fv->face_fv_map[2*e+1];
    
    Fn = _un[e];
    i_Fn = Fn * dS[e];
    r = -i_Fn;
    ierr = VecSetValue(R,elow,r,ADD_VALUES);CHKERRQ(ierr);
    if (ehigh >= 0) {
      r = i_Fn;
      ierr = VecSetValue(R,ehigh,r,ADD_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(R);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(R);CHKERRQ(ierr);
  
  if (monitor) {
    PetscReal div_min,div_max;
    
    ierr = VecMin(R,NULL,&div_min);CHKERRQ(ierr);
    ierr = VecMax(R,NULL,&div_max);CHKERRQ(ierr);
    PetscPrintf(PetscObjectComm((PetscObject)fv->dm_fv),"[input] \\int_v q - \\int_e Vn.n_E dv : min/max %+1.6e / %+1.6e\n",div_min,div_max);
  }
  
  /* 3: assemble the projection operator */
  
  ierr = MatCreate(PetscObjectComm((PetscObject)fv->dm_fv),&M);CHKERRQ(ierr);
  ierr = MatSetSizes(M,ncells,ncells,ncells,ncells);CHKERRQ(ierr);
  ierr = MatSetType(M,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetUp(M);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(M,8,NULL);CHKERRQ(ierr);
  
  for (e=0; e<nedges; e++) {
    PetscInt elow,ehigh;
    
    elow  = fv->face_fv_map[2*e+0];
    ehigh = fv->face_fv_map[2*e+1];

    ierr = MatSetValue(M,elow,elow,0.0,INSERT_VALUES);CHKERRQ(ierr);
    if (ehigh >= 0) {
      ierr = MatSetValue(M,elow,ehigh,0.0,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValue(M,ehigh,elow,0.0,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValue(M,ehigh,ehigh,0.0,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  for (e=0; e<nedges; e++) {
    weights[e] = 1.0;
  }
  if (w) {
    for (e=0; e<nedges; e++) {
      weights[e] = 1.0 / w[e];
    }
  }
  
  for (e=0; e<nedges; e++) {
    PetscReal d;
    PetscInt  elow,ehigh;
    
    elow  = fv->face_fv_map[2*e+0];
    ehigh = fv->face_fv_map[2*e+1];
    
    d = weights[e] * dS[e];
    ierr = MatSetValue(M,elow,elow,d,ADD_VALUES);CHKERRQ(ierr);
    if (ehigh >= 0) {
      ierr = MatSetValue(M,ehigh,ehigh,d,ADD_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  for (e=0; e<nedges; e++) {
    PetscReal d;
    PetscInt  elow,ehigh;
    
    elow  = fv->face_fv_map[2*e+0];
    ehigh = fv->face_fv_map[2*e+1];
    
    if (ehigh >= 0) {
      d = -weights[e] * dS[e];
      
      ierr = MatSetValue(M,elow,ehigh,d,INSERT_VALUES);CHKERRQ(ierr); // same as INSERT, use ADD for i-i coupling
      ierr = MatSetValue(M,ehigh,elow,d,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  /* 4: solve projection */
  ierr = MatCreateVecs(M,&y,NULL);CHKERRQ(ierr);
  
  ierr = KSPCreate(PetscObjectComm((PetscObject)fv->dm_fv),&ksp);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(ksp,"fvpp_");CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,M,M);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  
  ierr = KSPSolve(ksp,R,y);CHKERRQ(ierr);
  
  /* 5: make correction */
  ierr = VecGetArrayRead(y,&LA_y);CHKERRQ(ierr);
  for (e=0; e<nedges; e++) {
    PetscReal Un,Vn,correction;
    PetscInt  elow,ehigh;
    
    elow  = fv->face_fv_map[2*e+0];
    ehigh = fv->face_fv_map[2*e+1];
    
    correction = 0.0;
    Un = _un[e];
    
    if (ehigh >= 0) {
      PetscReal jump_y = LA_y[ elow ] - LA_y[ ehigh ]; /* see definition (2.19) */
      
      correction = weights[e] * jump_y;
    } else {
      correction = weights[e] * (LA_y[ elow ]);
    }
    Vn = Un + correction;
    _vn[e] = Vn;
#ifdef FVPP_LOG
    fprintf(fpi,"%d %+1.6e  %+1.6e  %+1.6e\n",e,Un,correction,Vn);
#endif
  }
  ierr = VecRestoreArrayRead(y,&LA_y);CHKERRQ(ierr);
  
  /* diagnostic */
  if (monitor) {
    PetscReal div_min,div_max;
    
    ierr = VecZeroEntries(R);CHKERRQ(ierr);
    
    /* volume terms */
    ierr = VecGetArrayRead(source,&LA_s);CHKERRQ(ierr);
    for (c=0; c<ncells; c++) {
      PetscReal r;
      
      r = LA_s[c] * dV[c];
      ierr = VecSetValue(R,c,r,ADD_VALUES);CHKERRQ(ierr);
    }
    ierr = VecRestoreArrayRead(source,&LA_s);CHKERRQ(ierr);
    
    /* face terms */
    for (e=0; e<nedges; e++) {
      PetscReal Fn,i_Fn,r;
      PetscInt  elow,ehigh;
      
      elow  = fv->face_fv_map[2*e+0];
      ehigh = fv->face_fv_map[2*e+1];
      
      Fn = _vn[e];
      i_Fn = Fn * dS[e];
      r = -i_Fn;
      ierr = VecSetValue(R,elow,r,ADD_VALUES);CHKERRQ(ierr);
      if (ehigh >= 0) {
        r = i_Fn;
        ierr = VecSetValue(R,ehigh,r,ADD_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = VecAssemblyBegin(R);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(R);CHKERRQ(ierr);
    
    ierr = VecMin(R,NULL,&div_min);CHKERRQ(ierr);
    ierr = VecMax(R,NULL,&div_max);CHKERRQ(ierr);
    PetscPrintf(PetscObjectComm((PetscObject)fv->dm_fv),"[corrected] \\int_v q - \\int_e Vn.n_E dv : min/max %+1.6e / %+1.6e\n",div_min,div_max);
  }
  
  ierr = VecRestoreArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);

#ifdef FVPP_LOG
  fclose(fpi);
#endif
  ierr = PetscFree(dV);CHKERRQ(ierr);
  ierr = PetscFree(dS);CHKERRQ(ierr);
  ierr = PetscFree(_un);CHKERRQ(ierr);
  ierr = PetscFree(weights);CHKERRQ(ierr);
  ierr = VecDestroy(&R);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = MatDestroy(&M);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode _FVPostProcessCompatibleVelocity_v2_SEQ(FVDA fv,const char name_v[],const char name_v_dot_n[],Vec source,KSP ksp)
{
  PetscErrorCode    ierr;
  Vec               R,y;
  Mat               M;
  const PetscScalar *LA_s,*LA_y;
  const PetscReal   *edge_velocity;
  PetscReal         *_un,*_vn;
  PetscInt          *edge_table;
  PetscInt          c,ncells,e,nedges_interior,nedges;
  PetscReal         weights_e = 1.0;
#ifdef FVPP_LOG
  FILE              *fpi = NULL;
#endif
  const PetscInt    nsd = 3;
  PetscReal         *dS,*dV;
  PetscInt          dir,cellid;
  Vec               geometry_coorl;
  const PetscScalar *_geom_coor;
  PetscReal         cell_coor[3*DACELL3D_VERTS];
  PetscInt          dm_nel,dm_nen;
  const PetscInt    *dm_element,*element;
  PetscBool         monitor = PETSC_FALSE;

  
  PetscFunctionBegin;
  ierr = PetscOptionsGetBool(NULL,NULL,"-fvpp_monitor",&monitor,NULL);CHKERRQ(ierr);
  ncells          = fv->ncells;
  nedges_interior = fv->nfaces_interior;
  nedges          = fv->nfaces;
  edge_table      = fv->face_fv_map;
  
#ifdef FVPP_LOG
  fpi = fopen("fvpp-compat.dat","w");
  if (!fpi) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Failed to open file: fvpp-compat.dat");
  fprintf(fpi,"#edge <coor> <v.n (input)> <correction> <v.n (output)>\n");
#endif
  
  ierr = KSPGetOperators(ksp,&M,NULL);CHKERRQ(ierr);
  
  ierr = FVDAGetFacePropertyByNameArrayRead(fv,name_v,&edge_velocity);CHKERRQ(ierr);
  ierr = FVDAGetFacePropertyByNameArray(fv,name_v_dot_n,&_vn);CHKERRQ(ierr);
  
  ierr = PetscMalloc1(nedges,&_un);CHKERRQ(ierr);
  ierr = PetscMalloc1(nedges,&dS);CHKERRQ(ierr);
  ierr = PetscMalloc1(ncells,&dV);CHKERRQ(ierr);
  
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  
  ierr = DMGetLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_geometry,fv->vertex_coor_geometry,INSERT_VALUES,geometry_coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
  
  ierr = MatCreateVecs(M,&R,&y);CHKERRQ(ierr);
  
  for (e=0; e<nedges; e++) {
    ierr = FVDAGetValidElement(fv,e,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
    ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateFaceArea3d(fv->face_type[e],cell_coor,&dS[e]);
  }
  
  for (c=0; c<ncells; c++) {
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * c];
    ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateCellVolume3d(cell_coor,&dV[c]);
  }
  
  /* 1: compute u.n along each facet */
  for (e=0; e<nedges; e++) {
    PetscReal Fn = 0;
    for (dir=0; dir<nsd; dir++) {
      Fn += edge_velocity[nsd*e+dir] * fv->face_normal[nsd*e+dir];
    }
    _un[e] = Fn;
  }
  
  /* 2: assemble residual */
  ierr = VecZeroEntries(R);CHKERRQ(ierr);
  
  /* volume terms */
  ierr = VecGetArrayRead(source,&LA_s);CHKERRQ(ierr);
  for (c=0; c<ncells; c++) {
    PetscReal r;
    
    r = LA_s[c] * dV[c];
    ierr = VecSetValue(R,c,r,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(source,&LA_s);CHKERRQ(ierr);
  
  /* face terms */
  for (e=0; e<nedges; e++) {
    PetscReal Fn,i_Fn,r;
    PetscInt  elow,ehigh;
    
    elow  = fv->face_fv_map[2*e+0];
    ehigh = fv->face_fv_map[2*e+1];
    
    Fn = _un[e];
    i_Fn = Fn * dS[e];
    r = -i_Fn;
    ierr = VecSetValue(R,elow,r,ADD_VALUES);CHKERRQ(ierr);
    if (ehigh >= 0) {
      r = i_Fn;
      ierr = VecSetValue(R,ehigh,r,ADD_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecAssemblyBegin(R);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(R);CHKERRQ(ierr);
  
  if (monitor) {
    PetscReal div_min,div_max;
    
    ierr = VecMin(R,NULL,&div_min);CHKERRQ(ierr);
    ierr = VecMax(R,NULL,&div_max);CHKERRQ(ierr);
    PetscPrintf(PetscObjectComm((PetscObject)fv->dm_fv),"[input] \\int_v q - \\int_e Vn.n_E dv : min/max %+1.6e / %+1.6e\n",div_min,div_max);
  }
  
  /* 3: assemble the projection operator */
  ierr = MatZeroEntries(M);CHKERRQ(ierr);
  
  for (e=0; e<nedges; e++) {
    PetscReal d;
    PetscInt  elow,ehigh;
    
    elow  = fv->face_fv_map[2*e+0];
    ehigh = fv->face_fv_map[2*e+1];
    
    d = weights_e * dS[e];
    ierr = MatSetValueLocal(M,elow,elow,d,ADD_VALUES);CHKERRQ(ierr);
    if (ehigh >= 0) {
      ierr = MatSetValueLocal(M,ehigh,ehigh,d,ADD_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(M,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
  
  for (e=0; e<nedges; e++) {
    PetscReal d;
    PetscInt  elow,ehigh;
    
    elow  = fv->face_fv_map[2*e+0];
    ehigh = fv->face_fv_map[2*e+1];
    
    if (ehigh >= 0) {
      d = -weights_e * dS[e];
      
      ierr = MatSetValueLocal(M,elow,ehigh,d,INSERT_VALUES);CHKERRQ(ierr); // same as INSERT, use ADD for i-i coupling
      ierr = MatSetValueLocal(M,ehigh,elow,d,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  /* 4: solve projection */
  ierr = KSPSetOperators(ksp,M,M);CHKERRQ(ierr);
  
  ierr = KSPSolve(ksp,R,y);CHKERRQ(ierr);
  
  /* 5: make correction */
  ierr = VecGetArrayRead(y,&LA_y);CHKERRQ(ierr);
  for (e=0; e<nedges; e++) {
    PetscReal Un,Vn,correction;
    PetscInt  elow,ehigh;
    
    elow  = fv->face_fv_map[2*e+0];
    ehigh = fv->face_fv_map[2*e+1];
    
    correction = 0.0;
    Un = _un[e];
    
    if (ehigh >= 0) {
      PetscReal jump_y = LA_y[ elow ] - LA_y[ ehigh ]; /* see definition (2.19) */
      
      correction = weights_e * jump_y;
    } else {
      correction = weights_e * (LA_y[ elow ]);
    }
    Vn = Un + correction;
    _vn[e] = Vn;
#ifdef FVPP_LOG
    {
      const PetscReal *face_centroid;
      
      ierr = FVDAGetFaceInfo(fv,NULL,NULL,NULL,NULL,&face_centroid);CHKERRQ(ierr);
      fprintf(fpi,"%d %+1.6e %+1.6e %+1.6e  %+1.6e  %+1.6e  %+1.6e\n",e,face_centroid[3*e+0],face_centroid[3*e+1],face_centroid[3*e+2],Un,correction,Vn);
    }
#endif
  }
  ierr = VecRestoreArrayRead(y,&LA_y);CHKERRQ(ierr);
  
  /* diagnostic */
  if (monitor) {
    PetscReal div_min,div_max;
    
    ierr = VecZeroEntries(R);CHKERRQ(ierr);
    
    /* volume terms */
    ierr = VecGetArrayRead(source,&LA_s);CHKERRQ(ierr);
    for (c=0; c<ncells; c++) {
      PetscReal r;
      
      r = LA_s[c] * dV[c];
      ierr = VecSetValue(R,c,r,ADD_VALUES);CHKERRQ(ierr);
    }
    ierr = VecRestoreArrayRead(source,&LA_s);CHKERRQ(ierr);
    
    /* face terms */
    for (e=0; e<nedges; e++) {
      PetscReal Fn,i_Fn,r;
      PetscInt  elow,ehigh;
      
      elow  = fv->face_fv_map[2*e+0];
      ehigh = fv->face_fv_map[2*e+1];
      
      Fn = _vn[e];
      i_Fn = Fn * dS[e];
      r = -i_Fn;
      ierr = VecSetValue(R,elow,r,ADD_VALUES);CHKERRQ(ierr);
      if (ehigh >= 0) {
        r = i_Fn;
        ierr = VecSetValue(R,ehigh,r,ADD_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = VecAssemblyBegin(R);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(R);CHKERRQ(ierr);
    
    ierr = VecMin(R,NULL,&div_min);CHKERRQ(ierr);
    ierr = VecMax(R,NULL,&div_max);CHKERRQ(ierr);
    PetscPrintf(PetscObjectComm((PetscObject)fv->dm_fv),"[corrected] \\int_v q - \\int_e Vn.n_E dv : min/max %+1.6e / %+1.6e\n",div_min,div_max);
  }
  
  ierr = VecRestoreArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
  
#ifdef FVPP_LOG
  fclose(fpi);
#endif
  ierr = PetscFree(dV);CHKERRQ(ierr);
  ierr = PetscFree(dS);CHKERRQ(ierr);
  ierr = PetscFree(_un);CHKERRQ(ierr);
  ierr = VecDestroy(&R);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode _FVPostProcessCompatibleVelocity_v2_MPI(FVDA fv,const char name_v[],const char name_v_dot_n[],Vec source,KSP ksp)
{
  PetscErrorCode    ierr;
  Vec               R,y,yl;
  Mat               M;
  const PetscScalar *LA_s,*LA_y;
  const PetscReal   *edge_velocity;
  PetscReal         *_un,*_vn;
  PetscInt          *edge_table;
  PetscInt          c,ncells,e,nedges_interior,nedges;
  PetscReal         weights_e = 1.0;
#ifdef FVPP_LOG
  FILE              *fpi = NULL;
#endif
  const PetscInt    nsd = 3;
  PetscReal         *dS,*dV,*LA_R;
  PetscInt          dir,cellid;
  PetscReal         cell_coor[3*DACELL3D_VERTS];
  PetscInt          dm_nel,dm_nen;
  const PetscInt    *dm_element,*element;
  PetscBool         monitor = PETSC_FALSE;

  
  PetscFunctionBegin;
  ierr = PetscOptionsGetBool(NULL,NULL,"-fvpp_monitor",&monitor,NULL);CHKERRQ(ierr);
  ncells          = fv->ncells;
  nedges_interior = fv->nfaces_interior;
  nedges          = fv->nfaces;
  edge_table      = fv->face_fv_map;
  
#ifdef FVPP_LOG
  {
    char        fname[PETSC_MAX_PATH_LEN];
    PetscMPIInt commrank;
    
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)fv->dm_fv),&commrank);CHKERRQ(ierr);
    ierr = PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"fvpp-compat-r%d.dat",(int)commrank);CHKERRQ(ierr);
    fpi = fopen(fname,"w");
    if (!fpi) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Failed to open file: %s",fname);
  }
  fprintf(fpi,"#edge <coor> <v.n (input)> <correction> <v.n (output)>\n");
#endif
  
  ierr = KSPGetOperators(ksp,&M,NULL);CHKERRQ(ierr);
  
  ierr = FVDAGetFacePropertyByNameArrayRead(fv,name_v,&edge_velocity);CHKERRQ(ierr);
  ierr = FVDAGetFacePropertyByNameArray(fv,name_v_dot_n,&_vn);CHKERRQ(ierr);
  
  ierr = PetscMalloc1(nedges,&_un);CHKERRQ(ierr);
  ierr = PetscMalloc1(nedges,&dS);CHKERRQ(ierr);
  ierr = PetscMalloc1(ncells,&dV);CHKERRQ(ierr);
  
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  
  ierr = DMGetGlobalVector(fv->dm_fv,&R);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(fv->dm_fv,&y);CHKERRQ(ierr);
  ierr = DMGetLocalVector(fv->dm_fv,&yl);CHKERRQ(ierr);
  
  {
    Vec               geometry_coorl;
    const PetscScalar *_geom_coor;
    
    ierr = DMGetLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
    ierr = DMGlobalToLocal(fv->dm_geometry,fv->vertex_coor_geometry,INSERT_VALUES,geometry_coorl);CHKERRQ(ierr);
    ierr = VecGetArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
    
    
    for (e=0; e<nedges; e++) {
      ierr = FVDAGetValidElement(fv,e,&cellid);CHKERRQ(ierr);
      element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
      ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);
      _EvaluateFaceArea3d(fv->face_type[e],cell_coor,&dS[e]);
    }
    
    for (c=0; c<ncells; c++) {
      element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * c];
      ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);
      _EvaluateCellVolume3d(cell_coor,&dV[c]);
    }

    ierr = VecRestoreArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
  }
  
  /* 1: compute u.n along each facet */
  for (e=0; e<nedges; e++) {
    PetscReal Fn = 0;
    for (dir=0; dir<nsd; dir++) {
      Fn += edge_velocity[nsd*e+dir] * fv->face_normal[nsd*e+dir];
    }
    _un[e] = Fn;
  }
  
  /* 2: assemble residual */
  ierr = VecZeroEntries(R);CHKERRQ(ierr);
  
  /* face terms */
  /*
  for (e=0; e<nedges; e++) {
    PetscReal Fn,i_Fn,r;
    PetscInt elow,ehigh;
    
    elow  = fv->face_fv_map[2*e+0];
    ehigh = fv->face_fv_map[2*e+1];
    
    Fn = _un[e];
    i_Fn = Fn * dS[e];
    r = -i_Fn;
    ierr = VecSetValueLocal(R,elow,r,ADD_VALUES);CHKERRQ(ierr);
    if (ehigh >= 0) {
      r = i_Fn;
      ierr = VecSetValueLocal(R,ehigh,r,ADD_VALUES);CHKERRQ(ierr);
    }
  }
  */
  
  ierr = FVDAIntegrateFlux(fv,name_v,PETSC_FALSE,R);CHKERRQ(ierr);
  ierr = VecScale(R,-1.0);CHKERRQ(ierr);
  
  /* volume terms */
  ierr = VecGetArrayRead(source,&LA_s);CHKERRQ(ierr);
  ierr = VecGetArray(R,&LA_R);CHKERRQ(ierr);
  for (c=0; c<ncells; c++) {
    PetscReal r;
    
    r = LA_s[c] * dV[c];
    LA_R[c] += r;
  }
  ierr = VecRestoreArray(R,&LA_R);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(source,&LA_s);CHKERRQ(ierr);

  ierr = VecAssemblyBegin(R);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(R);CHKERRQ(ierr);

  if (monitor) {
    PetscReal div_min,div_max;
    
    ierr = VecMin(R,NULL,&div_min);CHKERRQ(ierr);
    ierr = VecMax(R,NULL,&div_max);CHKERRQ(ierr);
    PetscPrintf(PetscObjectComm((PetscObject)fv->dm_fv),"[input] \\int_v q - \\int_e Vn.n_E dv : min/max %+1.6e / %+1.6e\n",div_min,div_max);
  }
  
  /* 3: assemble the projection operator */
  ierr = MatZeroEntries(M);CHKERRQ(ierr);
  
  for (e=0; e<nedges; e++) {
    PetscReal d;
    PetscInt  elow,ehigh;

    if (fv->face_element_map[2*e+0] == E_MINUS_OFF_RANK) continue;

    elow  = fv->face_fv_map[2*e+0];
    ehigh = fv->face_fv_map[2*e+1];
    
    d = weights_e * dS[e];
    ierr = MatSetValueLocal(M,elow,elow,d,ADD_VALUES);CHKERRQ(ierr);
    if (ehigh >= 0) {
      ierr = MatSetValueLocal(M,ehigh,ehigh,d,ADD_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(M,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
  
  for (e=0; e<nedges; e++) {
    PetscReal d;
    PetscInt  elow,ehigh;
    
    elow  = fv->face_fv_map[2*e+0];
    ehigh = fv->face_fv_map[2*e+1];
    
    if (ehigh >= 0) {
      d = -weights_e * dS[e];
      
      ierr = MatSetValueLocal(M,elow,ehigh,d,INSERT_VALUES);CHKERRQ(ierr); // same as INSERT, use ADD for i-i coupling
      ierr = MatSetValueLocal(M,ehigh,elow,d,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  /* 4: solve projection */
  ierr = KSPSetOperators(ksp,M,M);CHKERRQ(ierr);
  
  ierr = KSPSolve(ksp,R,y);CHKERRQ(ierr);
  
  ierr = DMGlobalToLocal(fv->dm_fv,y,INSERT_VALUES,yl);CHKERRQ(ierr);
  
  /* 5: make correction */
  ierr = VecGetArrayRead(yl,&LA_y);CHKERRQ(ierr);
  for (e=0; e<nedges; e++) {
    PetscReal Un,Vn,correction;
    PetscInt  elow,ehigh;
    
    elow  = fv->face_fv_map[2*e+0];
    ehigh = fv->face_fv_map[2*e+1];
    
    correction = 0.0;
    Un = _un[e];
    
    if (ehigh >= 0) {
      PetscReal jump_y = LA_y[ elow ] - LA_y[ ehigh ]; /* see definition (2.19) */
      
      correction = weights_e * jump_y;
    } else {
      correction = weights_e * (LA_y[ elow ]);
    }
    Vn = Un + correction;
    _vn[e] = Vn;
#ifdef FVPP_LOG
    {
      const PetscReal *face_centroid;
      
      ierr = FVDAGetFaceInfo(fv,NULL,NULL,NULL,NULL,&face_centroid);CHKERRQ(ierr);
      fprintf(fpi,"%d %+1.6e %+1.6e %+1.6e  %+1.6e  %+1.6e  %+1.6e\n",e,face_centroid[3*e+0],face_centroid[3*e+1],face_centroid[3*e+2],Un,correction,Vn);
    }
#endif
  }
  ierr = VecRestoreArrayRead(yl,&LA_y);CHKERRQ(ierr);
  
  /* diagnostic */
  if (monitor) {
    PetscReal div_min,div_max;
    
    ierr = VecZeroEntries(R);CHKERRQ(ierr);
    
    /* face terms */
    /*
    for (e=0; e<nedges; e++) {
      PetscReal Fn,i_Fn,r;
      PetscInt elow,ehigh;
      
      elow  = fv->face_fv_map[2*e+0];
      ehigh = fv->face_fv_map[2*e+1];
      
      Fn = _vn[e];
      i_Fn = Fn * dS[e];
      r = -i_Fn;
      ierr = VecSetValueLocal(R,elow,r,ADD_VALUES);CHKERRQ(ierr);
      if (ehigh >= 0) {
        r = i_Fn;
        ierr = VecSetValueLocal(R,ehigh,r,ADD_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = VecAssemblyBegin(R);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(R);CHKERRQ(ierr);
    */
    ierr = FVDAIntegrateFlux(fv,name_v_dot_n,PETSC_TRUE,R);CHKERRQ(ierr);
    ierr = VecScale(R,-1.0);CHKERRQ(ierr);

    /* volume terms */
    ierr = VecGetArrayRead(source,&LA_s);CHKERRQ(ierr);
    ierr = VecGetArray(R,&LA_R);CHKERRQ(ierr);
    for (c=0; c<ncells; c++) {
      PetscReal r;
      
      r = LA_s[c] * dV[c];
      LA_R[c] += r;
    }
    ierr = VecRestoreArray(R,&LA_R);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(source,&LA_s);CHKERRQ(ierr);
    
    ierr = VecMin(R,NULL,&div_min);CHKERRQ(ierr);
    ierr = VecMax(R,NULL,&div_max);CHKERRQ(ierr);
    PetscPrintf(PetscObjectComm((PetscObject)fv->dm_fv),"[corrected] \\int_v q - \\int_e Vn.n_E dv : min/max %+1.6e / %+1.6e\n",div_min,div_max);
  }
  
  ierr = DMRestoreLocalVector(fv->dm_fv,&yl);CHKERRQ(ierr);

#ifdef FVPP_LOG
  fclose(fpi);
#endif
  ierr = PetscFree(dV);CHKERRQ(ierr);
  ierr = PetscFree(dS);CHKERRQ(ierr);
  ierr = PetscFree(_un);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(fv->dm_fv,&R);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(fv->dm_fv,&y);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode FVDAPostProcessCompatibleVelocity(FVDA fv,const char name_v[],const char name_v_dot_n[],Vec source,KSP _ksp)
{
  PetscErrorCode ierr;
  PetscMPIInt    commsize;
  KSP            ksp = NULL;
  
  
  PetscFunctionBegin;
  ksp = _ksp;
  if (ksp) {
    ierr = PetscObjectReference((PetscObject)ksp);CHKERRQ(ierr);
  } else {
    ierr = FVDAPPCompatibleVelocityCreate(fv,&ksp);CHKERRQ(ierr);
  }
  
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)fv->dm_fv),&commsize);CHKERRQ(ierr);
  if (commsize == 1) {
    ierr = _FVPostProcessCompatibleVelocity_v2_SEQ(fv,name_v,name_v_dot_n,source,ksp);CHKERRQ(ierr);
  } else {
    ierr = _FVPostProcessCompatibleVelocity_v2_MPI(fv,name_v,name_v_dot_n,source,ksp);CHKERRQ(ierr);
  }

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#include <petsc/private/dmdaimpl.h>
PetscErrorCode FVDAPPCompatibleVelocityCreate(FVDA fv,KSP *ksp)
{
  PetscErrorCode ierr;
  Mat            A;
  PetscBool      use_fv_space = PETSC_FALSE;
  
  PetscFunctionBegin;
  /* 
   Creating the operator with the FV DMDA over allocates the matrix.
   FV space uses stencil type BOX, post-proc only requires stencil type STAR.
   This changes nnz from 27 to 7, which results in ~ >2x faster solve times with MG.
  */
  ierr = PetscOptionsGetBool(NULL,NULL,"-fvpp_operator_fvspace",&use_fv_space,NULL);CHKERRQ(ierr);
  if (use_fv_space) {
    ierr = DMCreateMatrix(fv->dm_fv,&A);CHKERRQ(ierr);
  } else {
    //DM              dm;
    DM_DA           *da = (DM_DA*)fv->dm_fv->data;
    DMDAStencilType stype = da->stencil_type;
    
    da->stencil_type = DMDA_STENCIL_STAR;
    //ierr = DMClone(fv->dm_fv,&dm);CHKERRQ(ierr);
    //ierr = DMCreateMatrix(dm,&A);CHKERRQ(ierr);
    ierr = DMCreateMatrix(fv->dm_fv,&A);CHKERRQ(ierr);
    da->stencil_type = stype;
    //ierr = DMDestroy(&dm);CHKERRQ(ierr);
  }
  
  ierr = KSPCreate(PetscObjectComm((PetscObject)fv->dm_fv),ksp);CHKERRQ(ierr);
  //ierr = KSPSetDM(*ksp,fv->dm_fv);CHKERRQ(ierr);
  //ierr = KSPSetDMActive(*ksp,PETSC_FALSE);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(*ksp,"fvpp_");CHKERRQ(ierr);
  ierr = KSPSetTolerances(*ksp,1.0e-10,1.0e-20,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(*ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(*ksp,A,A);CHKERRQ(ierr); /* we set A here so that KSP owns A */
  ierr = MatDestroy(&A);CHKERRQ(ierr);

  /*
  {
    DM dmc;
    Mat interp;
    
    ierr = DMDASetInterpolationType(fv->dm_fv, DMDA_Q0);CHKERRQ(ierr);
    ierr = DMDACreate3d(fv->comm,
                        DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,
                        DMDA_STENCIL_BOX,
                        fv->Mi[0]/2,fv->Mi[1]/2,fv->Mi[2]/2,
                        1,1,1,
                        1,
                        1,
                        NULL,NULL,NULL,&dmc);CHKERRQ(ierr);
    ierr = DMSetUp(dmc);CHKERRQ(ierr);
    ierr = DMDASetInterpolationType(dmc, DMDA_Q0);CHKERRQ(ierr);
    
    PC pc;
    
    DMCreateInterpolation(dmc,fv->dm_fv,&interp,NULL);
    KSPGetPC(ksp,&pc);
    PCSetType(pc,PCMG);
    PCMGSetLevels(pc,2,NULL);
    PCMGSetInterpolation(pc,1,interp);
  }
  */
  {
    PC        pc;
    PetscBool ismg;
    DM        dm,*dml;
    Mat       interp;
    PetscInt  nlevels,k;
    
    ierr = KSPGetPC(*ksp,&pc);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)pc,PCMG,&ismg);CHKERRQ(ierr);
    if (ismg) {
      ierr = DMClone(fv->dm_fv,&dm);CHKERRQ(ierr);
      ierr = DMDASetInterpolationType(dm,DMDA_Q0);CHKERRQ(ierr);
      ierr = PCMGGetLevels(pc,&nlevels);CHKERRQ(ierr);
      ierr = PetscCalloc1(nlevels,&dml);CHKERRQ(ierr);
      dml[0] = dm;
      for (k=1; k<nlevels; k++) {
        ierr = DMCoarsen(dml[k-1],fv->comm,&dml[k]);CHKERRQ(ierr);
        ierr = DMDASetInterpolationType(dml[k],DMDA_Q0);CHKERRQ(ierr);
      }
      for (k=1; k<nlevels; k++) {
        ierr = DMCreateInterpolation(dml[k],dml[k-1],&interp,NULL);CHKERRQ(ierr);
        ierr = PCMGSetInterpolation(pc,nlevels-k,interp);CHKERRQ(ierr);
        ierr = MatDestroy(&interp);CHKERRQ(ierr);
      }
      ierr = PCMGSetGalerkin(pc,PC_MG_GALERKIN_BOTH);CHKERRQ(ierr);
      for (k=0; k<nlevels; k++) {
        ierr = DMDestroy(&dml[k]);CHKERRQ(ierr);
      }
      ierr = PetscFree(dml);CHKERRQ(ierr);
    }
  }
  
  PetscFunctionReturn(0);
}

/*
 Evaluates \int \vec F \cdot \vec n dS over each finite volume cell and stores the result in a vector.
 
 [input]
 fv: the finite volume context
 field_name: textual name associated with the face quantity (F) to integrate
 f_dot_n: flag indicating whether the face quanity is a vector or scalar
 R: vector to store the result of the cell-wise integrals
 
 [Notes]
 f_dot_n = PETSC_TRUE implies the aux field called "field_name" is a scalar, namely \vec F \cdot \vec n
 f_dot_n = PETSC_FALSE implies the aux field called "field_name" is a vector, namely \vec F
*/
PetscErrorCode FVDAIntegrateFlux(FVDA fv,const char field_name[],PetscBool f_dot_n,Vec R)
{
  PetscErrorCode    ierr;
  const PetscReal   *edge_flux;
  PetscInt          *edge_table;
  PetscInt          e,nedges;
  Vec               Rl;
  PetscReal         *_r;
  Vec               geometry_coorl;
  const PetscScalar *_geom_coor;
  PetscReal         cell_coor[3*DACELL3D_VERTS];
  PetscInt          dm_nel,dm_nen,cellid;
  const PetscInt    *dm_element,*element;
  
  
  PetscFunctionBegin;
  nedges     = fv->nfaces;
  edge_table = fv->face_fv_map;

  ierr = FVDAGetFacePropertyByNameArrayRead(fv,field_name,&edge_flux);CHKERRQ(ierr);
  
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);

  ierr = DMGetLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_geometry,fv->vertex_coor_geometry,INSERT_VALUES,geometry_coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);

  ierr = VecZeroEntries(R);CHKERRQ(ierr);
  ierr = DMGetLocalVector(fv->dm_fv,&Rl);CHKERRQ(ierr);
  ierr = VecZeroEntries(Rl);CHKERRQ(ierr);
  ierr = VecGetArray(Rl,&_r);CHKERRQ(ierr);
  
  for (e=0; e<nedges; e++) {
    PetscReal Fn,i_Fn,dS;
    PetscInt  elow,ehigh;
    
    elow  = fv->face_fv_map[2*e+0];
    ehigh = fv->face_fv_map[2*e+1];
   
    ierr = FVDAGetValidElement(fv,e,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
    ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateFaceArea3d(fv->face_type[e],cell_coor,&dS);
    
    if (f_dot_n) {
      Fn = edge_flux[e];
    } else {
      Fn = edge_flux[3*e+0] * fv->face_normal[3*e+0]
         + edge_flux[3*e+1] * fv->face_normal[3*e+1]
         + edge_flux[3*e+2] * fv->face_normal[3*e+2];
    }
    i_Fn = Fn * dS;
    
    _r[elow]  += i_Fn; // cell[-]
    if (ehigh >= 0) {
      _r[ehigh] -= i_Fn; // cell[+]
    }
  }
  ierr = VecRestoreArray(Rl,&_r);CHKERRQ(ierr);

  ierr = DMLocalToGlobal(fv->dm_fv,Rl,INSERT_VALUES,R);CHKERRQ(ierr);
  
  ierr = DMRestoreLocalVector(fv->dm_fv,&Rl);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode FVDAIntegrateFlux_Local(FVDA fv,const char field_name[],PetscBool f_dot_n,PetscReal factor,
                                     const PetscScalar _geom_coor[],PetscReal _r[])
{
  PetscErrorCode    ierr;
  const PetscReal   *edge_flux;
  PetscInt          *edge_table;
  PetscInt          e,nedges;
  PetscReal         cell_coor[3*DACELL3D_VERTS];
  PetscInt          dm_nel,dm_nen,cellid;
  const PetscInt    *dm_element,*element;
  
  
  PetscFunctionBegin;
  nedges     = fv->nfaces;
  edge_table = fv->face_fv_map;
  
  ierr = FVDAGetFacePropertyByNameArrayRead(fv,field_name,&edge_flux);CHKERRQ(ierr);
  
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  
  for (e=0; e<nedges; e++) {
    PetscReal Fn,i_Fn,dS;
    PetscInt  elow,ehigh;
    
    elow  = fv->face_fv_map[2*e+0];
    ehigh = fv->face_fv_map[2*e+1];
    
    ierr = FVDAGetValidElement(fv,e,&cellid);CHKERRQ(ierr);
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * cellid];
    ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateFaceArea3d(fv->face_type[e],cell_coor,&dS);
    
    if (f_dot_n) {
      Fn = edge_flux[e];
    } else {
      Fn = edge_flux[3*e+0] * fv->face_normal[3*e+0]
         + edge_flux[3*e+1] * fv->face_normal[3*e+1]
         + edge_flux[3*e+2] * fv->face_normal[3*e+2];
    }
    i_Fn = factor * Fn * dS;
    
    _r[elow]  += i_Fn; // cell[-]
    if (ehigh >= 0) {
      _r[ehigh] -= i_Fn; // cell[+]
    }
  }
  
  PetscFunctionReturn(0);
}

