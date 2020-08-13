
#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <fvda_impl.h>
#include <fvda.h>
#include <fvda_utils.h>


PetscErrorCode t5_iflux(void)
{
  PetscErrorCode ierr;
  PetscInt       mx = 3;
  PetscInt       m[] = {0,0,0};
  FVDA           fv;
  Vec            F;
  DM             dm;
  PetscReal      sum;
  PetscBool      found = PETSC_FALSE;
  
  
  m[0] = m[1] = m[2] = mx;
  ierr = PetscOptionsGetInt(NULL,NULL,"-mx",&mx,&found);CHKERRQ(ierr);
  if (found) { m[0] = m[1] = m[2] = mx; }
  
  ierr = FVDACreate(PETSC_COMM_WORLD,&fv);CHKERRQ(ierr);
  ierr = FVDASetDimension(fv,3);CHKERRQ(ierr);
  ierr = FVDASetSizes(fv,NULL,m);CHKERRQ(ierr);
  
  ierr = FVDASetProblemType(fv,PETSC_FALSE,FVDA_HYPERBOLIC,0,0);CHKERRQ(ierr);
  
  ierr = FVDASetUp(fv);CHKERRQ(ierr);
  
  {
    Vec gcoor;
    
    ierr = DMDASetUniformCoordinates(fv->dm_geometry,0.0,1.0,0.0,1.0,0.0,1.0);CHKERRQ(ierr);
    ierr = DMGetCoordinates(fv->dm_geometry,&gcoor);CHKERRQ(ierr);
    ierr = VecCopy(gcoor,fv->vertex_coor_geometry);CHKERRQ(ierr);
  }
  
  ierr = FVDAUpdateGeometry(fv);CHKERRQ(ierr);
  
  ierr = FVDARegisterFaceProperty(fv,"v",3);CHKERRQ(ierr);
  ierr = FVDARegisterFaceProperty(fv,"v.n",1);CHKERRQ(ierr);
  
  /*
   v = 3x*iHat + 2y*jHat + 1z*kHat
   div(v) = 3 + 2 + 1 = 6
  */
  {
    PetscInt        f,nfaces;
    const PetscReal *face_normal,*face_centroid;
    PetscReal       *field,n[3],c[3];
    PetscReal       v[3]; /* imposed velocity field */
    
    ierr = FVDAGetFaceInfo(fv,&nfaces,NULL,NULL,&face_normal,&face_centroid);CHKERRQ(ierr);
    
    ierr = FVDAGetFacePropertyArray(fv,0,&field);CHKERRQ(ierr);
    for (f=0; f<nfaces; f++) {
      n[0] = face_normal[3*f+0];
      n[1] = face_normal[3*f+1];
      n[2] = face_normal[3*f+2];

      c[0] = face_centroid[3*f+0];
      c[1] = face_centroid[3*f+1];
      c[2] = face_centroid[3*f+2];
      
      v[0] = 3.0 * c[0];
      v[1] = 2.0 * c[1];
      v[2] = 1.0 * c[2];
      
      field[3*f+0] = v[0];
      field[3*f+1] = v[1];
      field[3*f+2] = v[2];
    }
    
    ierr = FVDAGetFacePropertyArray(fv,1,&field);CHKERRQ(ierr);
    for (f=0; f<nfaces; f++) {
      n[0] = face_normal[3*f+0];
      n[1] = face_normal[3*f+1];
      n[2] = face_normal[3*f+2];
      
      c[0] = face_centroid[3*f+0];
      c[1] = face_centroid[3*f+1];
      c[2] = face_centroid[3*f+2];

      v[0] = 3.0 * c[0];
      v[1] = 2.0 * c[1];
      v[2] = 1.0 * c[2];

      field[f] = v[0]*n[0] + v[1]*n[1] + v[2]*n[2];
    }
  }
  
  dm = fv->dm_fv;
  ierr = DMCreateGlobalVector(dm,&F);CHKERRQ(ierr);
  
  ierr = FVDAIntegrateFlux(fv,"v",PETSC_FALSE,F);CHKERRQ(ierr);
  ierr = VecSum(F,&sum);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"\\int v.(n dS) = %+1.12e (exact = 6)\n",sum);
  
  ierr = FVDAIntegrateFlux(fv,"v.n",PETSC_TRUE,F);CHKERRQ(ierr);
  ierr = VecSum(F,&sum);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"\\int (v.n) dS = %+1.12e (exact = 6)\n",sum);
  
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = FVDADestroy(&fv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 
 Use:
   PetscInt mx = 64;
 with the options:
   -fvpp_ksp_monitor -fvpp_pc_type mg -fvpp_ksp_rtol 1.0e-10 -fvpp_ksp_type fgmres -fvpp_pc_mg_levels 5 -fvpp_pc_mg_galerkin -fvpp_mg_levels_ksp_max_it 2

   -fvpp_pc_type mg -fvpp_ksp_rtol 1.0e-10 -fvpp_ksp_type cg -fvpp_pc_mg_levels 5 -fvpp_pc_mg_galerkin -fvpp_mg_levels_ksp_max_it 1 -fvpp_ksp_view -log_view -fvpp_mg_coarse_pc_factor_mat_solver_type umfpack -fvpp_mg_levels_pc_type jacobi | grep KSPSolve
 
   -mx 64 -fvpp_pc_type mg -fvpp_pc_mg_levels 5 -fvpp_ksp_monitor -fvpp_ksp_type cg -fvpp_mg_levels_ksp_max_it 4 -fvpp_mg_levels_ksp_type chebyshev -fvpp_mg_levels_pc_type jacobi -fvpp_mg_levels_ksp_chebyshev_esteig 0,0.01,0,1.1 -fvpp_ksp_pc_side left -fvpp_ksp_type cg -log_view -fvpp_mg_levels_ksp_norm_type none -fvpp_mg_levels_esteig_ksp_norm_type none  -fvpp_ksp_view -fvpp_mg_levels_esteig_ksp_type cg -fvpp_pc_mg_log
*/
PetscErrorCode t5_pp(void)
{
  PetscErrorCode ierr;
  PetscInt       mx = 64;
  PetscInt       m[] = {0,0,0};
  FVDA           fv;
  Vec            F;
  DM             dm;
  PetscBool      found = PETSC_FALSE;
  
  m[0] = m[1] = m[2] = mx;
  found = PETSC_FALSE; ierr = PetscOptionsGetInt(NULL,NULL,"-mx",&mx,&found);CHKERRQ(ierr);
  if (found) { m[0] = m[1] = m[2] = mx; }
  ierr = PetscOptionsGetInt(NULL,NULL,"-my",&m[1],&found);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-mz",&m[2],&found);CHKERRQ(ierr);
  
  ierr = FVDACreate(PETSC_COMM_WORLD,&fv);CHKERRQ(ierr);
  ierr = FVDASetDimension(fv,3);CHKERRQ(ierr);
  ierr = FVDASetSizes(fv,NULL,m);CHKERRQ(ierr);
  
  ierr = FVDASetProblemType(fv,PETSC_FALSE,FVDA_HYPERBOLIC,0,0);CHKERRQ(ierr);
  
  ierr = FVDASetUp(fv);CHKERRQ(ierr);
  
  {
    Vec gcoor;
    
    ierr = DMDASetUniformCoordinates(fv->dm_geometry,-1.0,1.0,-1.0,1.0,-1.0,1.0);CHKERRQ(ierr);
    ierr = DMGetCoordinates(fv->dm_geometry,&gcoor);CHKERRQ(ierr);
    ierr = VecCopy(gcoor,fv->vertex_coor_geometry);CHKERRQ(ierr);
  }
  
  ierr = FVDAUpdateGeometry(fv);CHKERRQ(ierr);

  ierr = FVDARegisterFaceProperty(fv,"v",3);CHKERRQ(ierr);
  {
    PetscInt        f,nfaces;
    const PetscReal *face_centroid,*face_normal;
    PetscReal       *vf;
    const PetscReal velocity[] = { 1.0, 0.0, 0.0 }; /* imposed velocity field */
    
    ierr = FVDAGetFaceInfo(fv,&nfaces,NULL,NULL,&face_normal,&face_centroid);CHKERRQ(ierr);
    ierr = FVDAGetFacePropertyArray(fv,0,&vf);CHKERRQ(ierr);
    for (f=0; f<nfaces; f++) {
      vf[3*f+0] = velocity[0];
      vf[3*f+1] = velocity[1];
      vf[3*f+2] = velocity[2];
    }
  }
  
  ierr = FVDARegisterFaceProperty(fv,"v.n",1);CHKERRQ(ierr);
  {
    PetscInt  f,nfaces;
    PetscReal *vdotn;
    
    ierr = FVDAGetFaceInfo(fv,&nfaces,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = FVDAGetFacePropertyArray(fv,1,&vdotn);CHKERRQ(ierr);
    for (f=0; f<nfaces; f++) {
      vdotn[f] = 0.0;
    }
  }
  
  dm = fv->dm_fv;
  ierr = DMCreateGlobalVector(dm,&F);CHKERRQ(ierr);
  ierr = VecSet(F,1.0e-2);CHKERRQ(ierr);
  
  /*
  ierr = _FVPostProcessCompatibleVelocity_SEQ(fv,"v","v.n",NULL,F);CHKERRQ(ierr);
  */
  
  {
    KSP ksp;
    PetscLogDouble t0,t1;
    
    ierr = FVDAPPCompatibleVelocityCreate(fv,&ksp);CHKERRQ(ierr);
    PetscTime(&t0);
    ierr = FVDAPostProcessCompatibleVelocity(fv,"v","v.n",F,ksp);CHKERRQ(ierr);
    PetscTime(&t1);
    PetscPrintf(PETSC_COMM_WORLD,"FVDAPostProcessCompatibleVelocity: %1.4e (sec)\n",t1-t0);
    ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  }
  /*
  {
   ierr = FVDAPostProcessCompatibleVelocity(fv,"v","v.n",F,NULL);CHKERRQ(ierr);
  }
  */
  
  ierr = FVDAViewStatistics(fv,PETSC_TRUE);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = FVDADestroy(&fv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  
  ierr = PetscInitialize(&argc,&args,(char*)0,NULL);if (ierr) return ierr;
  //ierr = t5_iflux();CHKERRQ(ierr);
  ierr = t5_pp();CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
