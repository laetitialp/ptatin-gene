#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <fvda_impl.h>
#include <fvda_utils.h>

#include "petsc/private/dmdaimpl.h"

#include "ptatin3d.h"
#include "private/ptatin_impl.h"
#include "ptatin_init.h"
#include "ptatin_log.h"

#include "material_point_utils.h"
#include "material_point_std_utils.h"
#include "material_point_popcontrol.h"
#include "ptatin_models.h"
#include "ptatin_utils.h"
#include "stokes_form_function.h"
#include "stokes_operators.h"
#include "stokes_operators_mf.h"
#include "stokes_assembly.h"
#include "dmda_element_q2p1.h"
#include "dmda_duplicate.h"
#include "dmda_project_coords.h"
#include "dmda_update_coords.h"
#include "monitors.h"
#include "mp_advection.h"
#include "mesh_update.h"

#include <phys_comp_energy.h>
#include "ptatin3d_energy.h"
#include "energy_assembly.h"
#include <ptatin3d_energyfv.h>
#include <fvda.h>
#include <ptatin3d_energyfv_impl.h>

PetscErrorCode EvalRHS_HeatProd(FVDA fv,Vec F)
{
  PetscErrorCode  ierr;
  PetscReal       dV;
  const PetscReal *H;
  PetscReal       cell_coor[3 * DACELL3D_Q1_SIZE];
  Vec             coorl;
  const PetscReal *_geom_coor;
  PetscReal       *_F;
  PetscInt        c,dm_nel,dm_nen;
  const PetscInt  *dm_element,*element;
  
  PetscFunctionBegin;
  
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  ierr = VecGetArray(F,&_F);CHKERRQ(ierr);
  
  ierr = DMDAGetElements(fv->dm_geometry,&dm_nel,&dm_nen,&dm_element);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(fv->dm_geometry,&coorl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(fv->dm_geometry,fv->vertex_coor_geometry,INSERT_VALUES,coorl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coorl,&_geom_coor);CHKERRQ(ierr);
  
  ierr = FVDAGetCellPropertyByNameArrayRead(fv,"H",&H);CHKERRQ(ierr);
  
  for (c=0; c<fv->ncells; c++) {
    element = (const PetscInt*)&dm_element[DACELL3D_Q1_SIZE * c];
    
    ierr = DACellGeometry3d_GetCoordinates(element,_geom_coor,cell_coor);CHKERRQ(ierr);
    _EvaluateCellVolume3d(cell_coor,&dV);
    
    _F[c] = -H[c] * dV;
  }
  ierr = VecRestoreArray(F,&_F);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(coorl,&_geom_coor);CHKERRQ(ierr);
  ierr = VecDestroy(&coorl);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode eval_F_diffusion_SteadyState(SNES snes,Vec X,Vec F,void *data)
{
  PetscErrorCode    ierr;
  Vec               Xl,Fl,coorl,geometry_coorl;
  const PetscScalar *_X,*_fv_coor,*_geom_coor;
  PetscScalar       *_F;
  DM                dm;
  FVDA              fv = NULL;
  
  
  PetscFunctionBegin;
  ierr = SNESGetApplicationContext(snes,(void*)&fv);CHKERRQ(ierr);
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
  
  {
    if (fv->equation_type == FVDA_ELLIPTIC|| fv->equation_type == FVDA_PARABOLIC) {
      //ierr = eval_F_diffusion_7point_hr_local(fv,_geom_coor,_fv_coor,_X,_F);CHKERRQ(ierr);
      //ierr = eval_F_diffusion_7point_hr_local_store(fv,_geom_coor,_fv_coor,_X,_F);CHKERRQ(ierr);
      ierr = eval_F_diffusion_7point_hr_local_store_MPI(fv,_geom_coor,_fv_coor,_X,_F);CHKERRQ(ierr);
    }
  }
  
  ierr = VecRestoreArrayRead(coorl,&_fv_coor);CHKERRQ(ierr);
  ierr = VecRestoreArray(Fl,&_F);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Xl,&_X);CHKERRQ(ierr);
  
  ierr = DMLocalToGlobal(dm,Fl,ADD_VALUES,F);CHKERRQ(ierr);
  
  ierr = DMRestoreLocalVector(dm,&Fl);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Xl);CHKERRQ(ierr);
  
  ierr = VecRestoreArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode eval_J_diffusion_SteadyState(SNES snes,Vec X,Mat Ja,Mat Jb,void *data)
{
  PetscErrorCode    ierr;
  Vec               Xl,coorl,geometry_coorl;
  const PetscScalar *_X,*_fv_coor,*_geom_coor;
  DM                dm;
  FVDA              fv = NULL;
  
  ierr = SNESGetApplicationContext(snes,(void*)&fv);CHKERRQ(ierr);
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

  {
    if (fv->equation_type == FVDA_ELLIPTIC|| fv->equation_type == FVDA_PARABOLIC) {
      ierr = eval_J_diffusion_7point_local(fv,_geom_coor,_fv_coor,_X,Jb);CHKERRQ(ierr);
    }
  }
  
  ierr = VecRestoreArrayRead(geometry_coorl,&_geom_coor);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(fv->dm_geometry,&geometry_coorl);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(coorl,&_fv_coor);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Xl,&_X);CHKERRQ(ierr);
  
  ierr = MatAssemblyBegin(Jb,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Jb,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (Ja != Jb) {
    ierr = MatAssemblyBegin(Ja,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Ja,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  
  ierr = DMRestoreLocalVector(dm,&Xl);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PhysCompEnergyFVSetUp_SteadyState(PhysCompEnergyFV energy,pTatinCtx pctx)
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
  ierr = PetscObjectSetName((PetscObject)energy->T,"T");CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(energy->fv->dm_fv,&energy->Told);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(energy->fv->dm_fv,&energy->F);CHKERRQ(ierr);
  /*ierr = DMCreateMatrix(energy->fv->dm_fv,&energy->J);CHKERRQ(ierr);*/
  ierr = FVDACreateMatrix(energy->fv,DMDA_STENCIL_STAR,&energy->J);CHKERRQ(ierr);
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

  // High-resolution diffusion
  ierr = SNESSetFunction(energy->snes,energy->F,          eval_F_diffusion_SteadyState,NULL);CHKERRQ(ierr);
  ierr = SNESSetJacobian(energy->snes,energy->J,energy->J,eval_J_diffusion_SteadyState,NULL);CHKERRQ(ierr);

  ierr = SNESSetFromOptions(energy->snes);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/* ptatin-PhysCompEnergyFV functionality */
PetscErrorCode pTatinPhysCompActivate_EnergyFV_SteadyState(pTatinCtx user,PetscBool load)
{
  PetscErrorCode   ierr;
  PhysCompEnergyFV energy;
  
  PetscFunctionBegin;
  if (load && (user->energyfv_ctx == NULL)) {
    PetscInt nsub[] = {3,3,3};
    
    ierr = PhysCompEnergyFVCreate(PETSC_COMM_WORLD,&energy);CHKERRQ(ierr);
    ierr = PhysCompEnergyFVSetParams(energy,0,0,nsub);CHKERRQ(ierr);
    ierr = PhysCompEnergyFVSetFromOptions(energy);CHKERRQ(ierr);
    ierr = PhysCompEnergyFVSetUp_SteadyState(energy,user);CHKERRQ(ierr);
    
    if (user->restart_from_file) {
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"pTatinPhysCompActivate_EnergyFV should not be called during restart");
    } else {
      ierr = PhysCompAddMaterialPointCoefficients_Energy(user->materialpoint_db);CHKERRQ(ierr);
    }
    
    ierr = PhysCompEnergyFVUpdateGeometry(energy,user->stokes_ctx);CHKERRQ(ierr);
    //ierr = FVDAView_CellData(energy->fv,energy->T,PETSC_TRUE,"xcell");CHKERRQ(ierr);
    user->energyfv_ctx = energy;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode pTatin3d_SteadyStateDiffusion_TFV_driver(int argc,char **argv)
{
  pTatinCtx         user;
  pTatinModel       model;
  PhysCompStokes    stokes;
  PhysCompEnergyFV  energyfv = NULL;
  DM                multipys_pack,dav,dap;
  Vec               F,rhs;
  PetscBool         active_energy;
  PetscBool         monitor_stages = PETSC_FALSE,write_icbc = PETSC_FALSE;
  DataBucket        materialpoint_db;
  PetscLogDouble    time[2];
  PetscErrorCode    ierr;

  PetscFunctionBegin;

  ierr = pTatin3dCreateContext(&user);CHKERRQ(ierr);
  ierr = pTatin3dSetFromOptions(user);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-monitor_stages",&monitor_stages,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-ptatin_driver_write_icbc",&write_icbc,NULL);CHKERRQ(ierr);

  /* Register all models */
  ierr = pTatinModelRegisterAll();CHKERRQ(ierr);
  /* Load model, call an initialization routines */
  ierr = pTatinModelLoad(user);CHKERRQ(ierr);
  ierr = pTatinGetModel(user,&model);CHKERRQ(ierr);

  ierr = pTatinModel_Initialize(model,user);CHKERRQ(ierr);

  /* Generate physics modules */
  ierr = pTatin3d_PhysCompStokesCreate(user);CHKERRQ(ierr);
  ierr = pTatinGetStokesContext(user,&stokes);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"Generated vel/pressure mesh --> ");
  pTatinGetRangeCurrentMemoryUsage(NULL);

  /* Pack all physics together */
  /* Here it's simple, we don't need a DM for this, just assign the pack DM to be equal to the stokes DM */
  ierr = PetscObjectReference((PetscObject)stokes->stokes_pack);CHKERRQ(ierr);
  user->pack = stokes->stokes_pack;

  /* fetch some local variables */
  multipys_pack = user->pack;
  dav           = stokes->dav;
  dap           = stokes->dap;

  ierr = pTatin3dCreateMaterialPoints(user,dav);CHKERRQ(ierr);
  ierr = pTatinGetMaterialPoints(user,&materialpoint_db,NULL);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"Generated material points --> ");
  pTatinGetRangeCurrentMemoryUsage(NULL);

  /* mesh geometry */
  ierr = pTatinModel_ApplyInitialMeshGeometry(model,user);CHKERRQ(ierr);

  ierr = pTatinLogBasicDMDA(user,"Velocity",dav);CHKERRQ(ierr);
  ierr = pTatinLogBasicDMDA(user,"Pressure",dap);CHKERRQ(ierr);

  /* generate energy solver */
  /* NOTE - Generating the thermal solver here will ensure that the initial geometry on the mechanical model is copied */
  /* NOTE - Calling pTatinPhysCompActivate_Energy() after pTatin3dCreateMaterialPoints() is essential */
  {
    PetscBool load_energy = PETSC_FALSE;
    
    PetscOptionsGetBool(NULL,NULL,"-activate_energyfv",&load_energy,NULL);
    ierr = pTatinPhysCompActivate_EnergyFV_SteadyState(user,load_energy);CHKERRQ(ierr);
    ierr = pTatinContextValid_EnergyFV(user,&active_energy);CHKERRQ(ierr);
  }
  
  if (active_energy) {
    DM dmfv;
    
    ierr = pTatinGetContext_EnergyFV(user,&energyfv);CHKERRQ(ierr);

    ierr = FVDAGetDM(energyfv->fv,&dmfv);CHKERRQ(ierr);
    ierr = pTatinLogBasicDMDA(user,"EnergyFV",dmfv);CHKERRQ(ierr);

    ierr = pTatinCtxAttachModelData(user,"PhysCompEnergy_T",(void*)energyfv->T);CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(dmfv,&rhs);CHKERRQ(ierr);

    pTatinGetRangeCurrentMemoryUsage(NULL);
  }

  /* interpolate material point coordinates (needed if mesh was modified) */
  ierr = MaterialPointCoordinateSetUp(user,dav);CHKERRQ(ierr);

  /* material geometry */
  ierr = pTatinModel_ApplyInitialMaterialGeometry(model,user);CHKERRQ(ierr);
  if (active_energy) {
    //SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Requires update for FV support");
    PetscPrintf(PETSC_COMM_WORLD,"********* <FV SUPPORT NOTE> IS THIS REQUIRED?? pTatinPhysCompEnergy_MPProjectionQ1 ****************\n");
    
    ierr = EnergyFVEvaluateCoefficients(user,0.0,energyfv,NULL,NULL);CHKERRQ(ierr);
    
    ierr = pTatinPhysCompEnergyFV_MPProjection(energyfv,user);CHKERRQ(ierr);
    
    ierr = FVDACellPropertyProjectToFace_HarmonicMean(energyfv->fv,"k","k");CHKERRQ(ierr);
    
    /* Evaluate \int_V -H dV */
    ierr = EvalRHS_HeatProd(energyfv->fv,rhs);CHKERRQ(ierr);
  }
  DataBucketView(PetscObjectComm((PetscObject)multipys_pack), materialpoint_db,"MaterialPoints StokesCoefficients",DATABUCKET_VIEW_STDOUT);

  /* boundary conditions */
  ierr = pTatinModel_ApplyBoundaryCondition(model,user);CHKERRQ(ierr);

  if (active_energy) {
    ierr = pTatinPhysCompEnergyFV_Initialise(energyfv,energyfv->T);CHKERRQ(ierr);
  }

  ierr = pTatinLogBasic(user);CHKERRQ(ierr);

  /* solve energy equation */
  if (active_energy) {
    {
      PetscTime(&time[0]);
      ierr = SNESSolve(energyfv->snes,rhs,energyfv->T);CHKERRQ(ierr);
      PetscTime(&time[1]);
      
      ierr = pTatinLogBasicSNES(user,"EnergyFV_StadyState",energyfv->snes);CHKERRQ(ierr);
      ierr = pTatinLogBasicCPUtime(user,"EnergyFV_SteadyState-Solve",time[1]-time[0]);CHKERRQ(ierr);
    } 
  }

    /* output */
    {
    PetscViewer viewer;
    char        fname[256];
    
    sprintf(fname,"T_steady.vts");
    ierr = PetscViewerCreate(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer,PETSCVIEWERVTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer,fname);CHKERRQ(ierr);
    ierr = VecView(energyfv->T,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    
    sprintf(fname,"rhs_steady.vts");
    ierr = PetscViewerCreate(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer,PETSCVIEWERVTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer,fname);CHKERRQ(ierr);
    ierr = VecView(rhs,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&rhs);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = pTatin3dDestroyContext(&user);

  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscMPIInt rank;

  ierr = pTatinInitialize(&argc,&argv,0,NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  ierr = PetscMemorySetGetMaximumUsage();CHKERRQ(ierr);

  ierr = pTatin3d_SteadyStateDiffusion_TFV_driver(argc,argv);CHKERRQ(ierr);

  ierr = pTatinGetRangeMaximumMemoryUsage(NULL);CHKERRQ(ierr);

  ierr = pTatinFinalize();CHKERRQ(ierr);
  return 0;
}
