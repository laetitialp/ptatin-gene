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
#include "dmda_redundant.h"
#include "dmda_project_coords.h"
#include "dmda_update_coords.h"
#include "dmda_checkpoint.h"
#include "monitors.h"
#include "mp_advection.h"
#include "mesh_update.h"

#include <cjson_utils.h>
#include "litho_pressure_PDESolve.h"

static PetscErrorCode ProjectStokesVariablesOnQuadraturePoints_PressurePoisson(pTatinCtx ptatin)
{
  int               npoints;
  DataField         PField_std;
  DataField         PField_stokes;
  MPntStd           *mp_std;
  MPntPStokes       *mp_stokes;
  PhysCompStokes    stokes;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  
  /* Marker -> quadrature point projection */
  DataBucketGetDataFieldByName(ptatin->materialpoint_db, MPntStd_classname     , &PField_std);
  DataBucketGetDataFieldByName(ptatin->materialpoint_db, MPntPStokes_classname , &PField_stokes);

  DataBucketGetSizes(ptatin->materialpoint_db,&npoints,NULL,NULL);
  DataFieldGetEntries(PField_std,(void**)&mp_std);
  DataFieldGetEntries(PField_stokes,(void**)&mp_stokes);
  
  ierr = pTatinGetStokesContext(ptatin,&stokes);CHKERRQ(ierr);

  switch (ptatin->coefficient_projection_type) {

    case -1:      /* Perform null projection use the values currently defined on the quadrature points */
      break;

    case 0:     /* Perform P0 projection over Q2 element directly onto quadrature points */
      //SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"P0 [arithmetic avg] marker->quadrature projection not supported");
            ierr = MPntPStokesProj_P0(CoefAvgARITHMETIC,npoints,mp_std,mp_stokes,stokes->dav,stokes->volQ);CHKERRQ(ierr);
            ierr = QPntSurfCoefStokes_ProjectP0_Surface(stokes->mfi,stokes->volQ,stokes->surfQ);CHKERRQ(ierr);
      break;
    case 10:
      //SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"P0 [harmonic avg] marker->quadrature projection not supported");
            ierr = MPntPStokesProj_P0(CoefAvgHARMONIC,npoints,mp_std,mp_stokes,stokes->dav,stokes->volQ);CHKERRQ(ierr);
            ierr = QPntSurfCoefStokes_ProjectP0_Surface(stokes->mfi,stokes->volQ,stokes->surfQ);CHKERRQ(ierr);
      break;
    case 20:
      //SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"P0 [geometric avg] marker->quadrature projection not supported");
            ierr = MPntPStokesProj_P0(CoefAvgGEOMETRIC,npoints,mp_std,mp_stokes,stokes->dav,stokes->volQ);CHKERRQ(ierr);
            ierr = QPntSurfCoefStokes_ProjectP0_Surface(stokes->mfi,stokes->volQ,stokes->surfQ);CHKERRQ(ierr);
      break;
    case 30:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"P0 [dominant phase] marker->quadrature projection not supported");
      break;

    case 1:     /* Perform Q1 projection over Q2 element and interpolate back to quadrature points */
      ierr = SwarmUpdateGaussPropertiesLocalL2Projection_Q1_MPntPStokes(npoints,mp_std,mp_stokes,stokes->dav,stokes->volQ,stokes->surfQ,stokes->mfi);CHKERRQ(ierr);
      break;

    case 2:       /* Perform Q2 projection and interpolate back to quadrature points */
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Q2 marker->quadrature projection not supported");
      break;

    case 3:       /* Perform P1 projection and interpolate back to quadrature points */
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"P1 marker->quadrature projection not supported");
      break;
        case 4:
            ierr = SwarmUpdateGaussPropertiesOne2OneMap_MPntPStokes(npoints,mp_std,mp_stokes,stokes->volQ);CHKERRQ(ierr);
            break;

    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Viscosity projection type is not defined");
      break;
  }
  
  DataFieldRestoreEntries(PField_stokes,(void**)&mp_stokes);
  DataFieldRestoreEntries(PField_std,(void**)&mp_std);
  
  PetscFunctionReturn(0);
}

static PetscErrorCode pTatin3d_Solve_PoissonPressure(pTatinCtx ptatin, PDESolveLithoP poisson_pressure)
{
  Mat            J = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = DMSetMatType(poisson_pressure->da,MATAIJ);CHKERRQ(ierr);
  ierr = DMCreateMatrix(poisson_pressure->da,&J);CHKERRQ(ierr);
  ierr = MatSetFromOptions(J);CHKERRQ(ierr);
  
  ierr = SNESSolve_LithoPressure(poisson_pressure,J,poisson_pressure->X,poisson_pressure->F,ptatin);CHKERRQ(ierr);

  ierr = MatDestroy(&J);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode OutputPoissonPressure(pTatinCtx ptatin, PDESolveLithoP poisson_pressure, char stepname[], PetscBool vts)
{
  PetscViewer    viewer;
  char           fname[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  
  if (vts) {
    ierr = PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s/%sPoissonPressure.vts",ptatin->outputpath,stepname);CHKERRQ(ierr);
    ierr = PetscViewerCreate(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer,PETSCVIEWERVTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer,fname);CHKERRQ(ierr);
    ierr = VecView(poisson_pressure->X,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  } else {
    ierr = PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s/%sPoissonPressure.pbvec",ptatin->outputpath,stepname);CHKERRQ(ierr);
    ierr = PetscViewerCreate(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer,PETSCVIEWERBINARY);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer,fname);CHKERRQ(ierr);
    ierr = VecView(poisson_pressure->X,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode pTatin3d_PoissonPressure_FromModelICState(int argc,char **argv)
{
  pTatinCtx         ptatin;
  pTatinModel       model;
  PhysCompStokes    stokes;
  PDESolveLithoP    poisson_pressure;
  DM                multipys_pack,dav,dap;
  Vec               X_stokes;
  DataBucket        materialpoint_db;
  char              stepname[PETSC_MAX_PATH_LEN];
  PetscBool         output_vts=PETSC_FALSE;
  PetscErrorCode    ierr;

  PetscFunctionBegin;

  ierr = PetscOptionsGetBool(NULL,NULL,"-output_vts",&output_vts,NULL);CHKERRQ(ierr);

  ierr = pTatin3dCreateContext(&ptatin);CHKERRQ(ierr);
  ierr = pTatin3dSetFromOptions(ptatin);CHKERRQ(ierr);

  /* Load model, call an initialization routines */
  ierr = pTatinModelLoad(ptatin);CHKERRQ(ierr);
  ierr = pTatinGetModel(ptatin,&model);CHKERRQ(ierr);

  ierr = pTatinModel_Initialize(model,ptatin);CHKERRQ(ierr);

  /* Generate physics modules */
  ierr = pTatin3d_PhysCompStokesCreate(ptatin);CHKERRQ(ierr);
  ierr = pTatinGetStokesContext(ptatin,&stokes);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"Generated vel/pressure mesh --> ");
  pTatinGetRangeCurrentMemoryUsage(NULL);

   /* Here it's simple, we don't need a DM for this, just assign the pack DM to be equal to the stokes DM */
  ierr = PetscObjectReference((PetscObject)stokes->stokes_pack);CHKERRQ(ierr);
  ptatin->pack = stokes->stokes_pack;

  /* fetch some local variables */
  multipys_pack = ptatin->pack;
  dav           = stokes->dav;
  dap           = stokes->dap;

  /* IF I DON'T DO THIS, THE IS's OBTAINED FROM DMCompositeGetGlobalISs() are wrong !! */
  {
    Vec X;

    ierr = DMGetGlobalVector(multipys_pack,&X);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(multipys_pack,&X);CHKERRQ(ierr);
  }

  ierr = pTatin3dCreateMaterialPoints(ptatin,dav);CHKERRQ(ierr);
  ierr = pTatinGetMaterialPoints(ptatin,&materialpoint_db,NULL);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"Generated material points --> ");
  pTatinGetRangeCurrentMemoryUsage(NULL);

  /* mesh geometry */
  ierr = pTatinModel_ApplyInitialMeshGeometry(model,ptatin);CHKERRQ(ierr);

  ierr = pTatinLogBasicDMDA(ptatin,"Velocity",dav);CHKERRQ(ierr);
  ierr = pTatinLogBasicDMDA(ptatin,"Pressure",dap);CHKERRQ(ierr);

  /* work vector for solution and residual (Not used in this driver but necessary for the BC function) */
  ierr = DMCreateGlobalVector(multipys_pack,&X_stokes);CHKERRQ(ierr);
  ierr = pTatinPhysCompAttachData_Stokes(ptatin,X_stokes);CHKERRQ(ierr);

  /* Create Poisson Pressure struct */
  ierr = pTatinPhysCompActivate_LithoP(ptatin,PETSC_TRUE);CHKERRQ(ierr);
  ierr = pTatinGetContext_LithoP(ptatin,&poisson_pressure);CHKERRQ(ierr);

  /* interpolate material point coordinates (needed if mesh was modified) */
  ierr = MaterialPointCoordinateSetUp(ptatin,dav);CHKERRQ(ierr);

  /* material geometry */
  ierr = pTatinModel_ApplyInitialMaterialGeometry(model,ptatin);CHKERRQ(ierr);
  DataBucketView(PetscObjectComm((PetscObject)multipys_pack), materialpoint_db,"MaterialPoints StokesCoefficients",DATABUCKET_VIEW_STDOUT);

  /* gravity */
  ierr = pTatinModel_ApplyGravity(model,ptatin);CHKERRQ(ierr);

  /* initial condition */
  ierr = pTatinModel_ApplyInitialSolution(model,ptatin,X_stokes);CHKERRQ(ierr);

  /* initial density */
  ierr = pTatinModel_ApplyInitialStokesVariableMarkers(model,ptatin,X_stokes);CHKERRQ(ierr);

  /* boundary conditions */
  ierr = pTatinModel_ApplyBoundaryCondition(model,ptatin);CHKERRQ(ierr);

  ierr = pTatin3d_Solve_PoissonPressure(ptatin,poisson_pressure);CHKERRQ(ierr);

  /* Output the pressure */
  if (ptatin->step) {
    ierr = PetscSNPrintf(stepname,PETSC_MAX_PATH_LEN-1,"step%1.6D",ptatin->step);CHKERRQ(ierr);
  } else {
    ierr = PetscSNPrintf(stepname,PETSC_MAX_PATH_LEN-1,"step0");CHKERRQ(ierr);
  }
  ierr = OutputPoissonPressure(ptatin,poisson_pressure,stepname,output_vts);CHKERRQ(ierr);
  /* Output material points variables and stokes */
  ierr = pTatinModel_Output(model,ptatin,X_stokes,stepname);CHKERRQ(ierr); 

  ierr = PhysCompDestroy_LithoP(&poisson_pressure);CHKERRQ(ierr);
  ptatin->litho_p_ctx = NULL;
  ierr = VecDestroy(&X_stokes);CHKERRQ(ierr);
  ierr = pTatin3dDestroyContext(&ptatin);

  PetscFunctionReturn(0);
}

static PetscErrorCode pTatin3d_LoadModelDefinition_FromFile(pTatinCtx *pctx, Vec *v1)
{
  pTatinCtx         ptatin;
  pTatinModel       model = NULL;
  PhysCompStokes    stokes = NULL;
  DM                dmstokes,dmv,dmp;
  Vec               X_stokes;
  DataBucket        materialpoint_db;
  PetscLogDouble    time[2];
  PetscErrorCode    ierr;

  PetscTime(&time[0]);
  ierr = pTatin3dLoadContext_FromFile(&ptatin);CHKERRQ(ierr);
  PetscTime(&time[1]);
  ierr = pTatin3dSetFromOptions(ptatin);CHKERRQ(ierr);
  ierr = pTatinLogNote(ptatin,"  [ptatin_driver.Load]");CHKERRQ(ierr);
  ierr = pTatinLogBasicCPUtime(ptatin,"Checkpoint.read()",time[1]-time[0]);CHKERRQ(ierr);

  /* driver specific options parsed here */

  /* Register all models */
  ierr = pTatinModelLoad(ptatin);CHKERRQ(ierr);
  ierr = pTatinGetModel(ptatin,&model);CHKERRQ(ierr);

  ierr = pTatinModel_Initialize(model,ptatin);CHKERRQ(ierr);

  /* Create Stokes context */
  ierr = pTatin3d_PhysCompStokesLoad_FromFile(ptatin);CHKERRQ(ierr);
  ierr = pTatinGetStokesContext(ptatin,&stokes);CHKERRQ(ierr);
  ierr = PhysCompStokesGetDMComposite(stokes,&dmstokes);CHKERRQ(ierr);
  ierr = PhysCompStokesGetDMs(stokes,&dmv,&dmp);CHKERRQ(ierr);

  { /* IF I DON'T DO THIS, THE IS's OBTAINED FROM DMCompositeGetGlobalISs() are wrong !! */
    Vec X;

    ierr = DMGetGlobalVector(dmstokes,&X);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dmstokes,&X);CHKERRQ(ierr);
  }
  /* Pack all physics together */
  ierr = PetscObjectReference((PetscObject)dmstokes);CHKERRQ(ierr);
  ptatin->pack = dmstokes;

  ierr = pTatin3dLoadMaterialPoints_FromFile(ptatin,dmv);CHKERRQ(ierr);
  ierr = pTatinGetMaterialPoints(ptatin,&materialpoint_db,NULL);CHKERRQ(ierr);
  
  /* work vector for solution */
  ierr = DMCreateGlobalVector(dmstokes,&X_stokes);CHKERRQ(ierr);
  ierr = pTatinPhysCompAttachData_Stokes(ptatin,X_stokes);CHKERRQ(ierr);

  /* initial condition - call ptatin method, then clobber */
  ierr = pTatinModel_ApplyInitialSolution(model,ptatin,X_stokes);CHKERRQ(ierr);
  ierr = pTatin3dLoadState_FromFile(ptatin,dmstokes,NULL,X_stokes,NULL);CHKERRQ(ierr);

  ierr = ProjectStokesVariablesOnQuadraturePoints_PressurePoisson(ptatin);CHKERRQ(ierr);
  
  if (v1) { *v1 = X_stokes; }
  else    { ierr = VecDestroy(&X_stokes);CHKERRQ(ierr); }

  *pctx = ptatin;
  PetscFunctionReturn(0);
}

static PetscErrorCode pTatin3d_PoissonPressure_FromFile(pTatinCtx ptatin, Vec v1)
{
  pTatinModel    model = NULL;
  PhysCompStokes stokes = NULL;
  PDESolveLithoP poisson_pressure;
  DM             dmstokes,dav,dap;
  Vec            X_stokes = NULL;
  char           stepname[PETSC_MAX_PATH_LEN];
  PetscMPIInt    rank;
  PetscBool      output_vts=PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = PetscOptionsGetBool(NULL,NULL,"-output_vts",&output_vts,NULL);CHKERRQ(ierr);

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = pTatinGetModel(ptatin,&model);CHKERRQ(ierr);
  ierr = pTatinGetStokesContext(ptatin,&stokes);CHKERRQ(ierr);
  ierr = PhysCompStokesGetDMComposite(stokes,&dmstokes);CHKERRQ(ierr);
  ierr = PhysCompStokesGetDMs(stokes,&dav,&dap);CHKERRQ(ierr);

  /* Pack all physics together */
  /* Here it's simple, we don't need a DM for this, just assign the pack DM to be equal to the stokes DM */
  ierr = PetscObjectReference((PetscObject)stokes->stokes_pack);CHKERRQ(ierr);
  ptatin->pack = stokes->stokes_pack;

  if (v1) {
    X_stokes = v1;
  } else {
    ierr = DMCreateGlobalVector(dmstokes,&X_stokes);CHKERRQ(ierr);
    ierr = pTatinPhysCompAttachData_Stokes(ptatin,X_stokes);CHKERRQ(ierr);
  }

  /* Create Poisson Pressure struct */
  ierr = pTatinPhysCompActivate_LithoP(ptatin,PETSC_TRUE);CHKERRQ(ierr);
  ierr = pTatinGetContext_LithoP(ptatin,&poisson_pressure);CHKERRQ(ierr);
  
  ierr = pTatinModel_ApplyBoundaryCondition(model,ptatin);CHKERRQ(ierr);

  ierr = pTatin3d_Solve_PoissonPressure(ptatin,poisson_pressure);CHKERRQ(ierr);

  /* Output the pressure */
  if (ptatin->step) {
    ierr = PetscSNPrintf(stepname,PETSC_MAX_PATH_LEN-1,"step%1.6D",ptatin->step);CHKERRQ(ierr);
  } else {
    ierr = PetscSNPrintf(stepname,PETSC_MAX_PATH_LEN-1,"step0");CHKERRQ(ierr);
  }
  ierr = OutputPoissonPressure(ptatin,poisson_pressure,stepname,output_vts);CHKERRQ(ierr);
  /* Output material points variables and stokes */
  ierr = pTatinModel_Output(model,ptatin,X_stokes,stepname);CHKERRQ(ierr); 

  ierr = PhysCompDestroy_LithoP(&poisson_pressure);CHKERRQ(ierr);
  ptatin->litho_p_ctx = NULL;

  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  pTatinCtx      ptatin = NULL;
  PetscBool      run = PETSC_FALSE;
  PetscBool      load = PETSC_FALSE;
  PetscErrorCode ierr;
  PetscMPIInt rank;

  ierr = pTatinInitialize(&argc,&argv,0,NULL);CHKERRQ(ierr);
  /* Register all models */
  ierr = pTatinModelRegisterAll();CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  ierr = PetscMemorySetGetMaximumUsage();CHKERRQ(ierr);

  ierr = PetscOptionsGetBool(NULL,NULL,"-run",&run,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-load",&load,NULL);CHKERRQ(ierr);

  if (run && load) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Both -run and -load have been provided, only one of the two can be used");
  }

  if (run || !load) {
    ierr = pTatin3d_PoissonPressure_FromModelICState(argc,argv);CHKERRQ(ierr);
  }

  if (load || !run) {
    Vec       X_stokes = NULL;
    PetscBool restart_string_found = PETSC_FALSE,flg = PETSC_FALSE;
    char      outputpath[PETSC_MAX_PATH_LEN];
    /* look for a default restart file */
    ierr = PetscOptionsGetString(NULL,NULL,"-output_path",outputpath,PETSC_MAX_PATH_LEN-1,&flg);CHKERRQ(ierr);
    if (flg) {
      char fname[PETSC_MAX_PATH_LEN];

      ierr = PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s/restart.default",outputpath);CHKERRQ(ierr);
      ierr = pTatinTestFile(fname,'r',&restart_string_found);CHKERRQ(ierr);
      if (restart_string_found) {
        PetscPrintf(PETSC_COMM_WORLD,"[pTatin] Detected default restart option file helper: %s\n",fname);
        //ierr = PetscOptionsInsertFile(PETSC_COMM_WORLD,NULL,fname,PETSC_TRUE);CHKERRQ(ierr);
        ierr = PetscOptionsInsert(NULL,&argc,&argv,fname);CHKERRQ(ierr);
      }
    }

    ierr = pTatin3d_LoadModelDefinition_FromFile(&ptatin,&X_stokes);CHKERRQ(ierr);
    ierr = pTatin3d_PoissonPressure_FromFile(ptatin,X_stokes);CHKERRQ(ierr);

    if (X_stokes) { ierr = VecDestroy(&X_stokes);CHKERRQ(ierr); }
    if (ptatin) { ierr = pTatin3dDestroyContext(&ptatin); }
    ptatin = NULL;
  }

  ierr = pTatinGetRangeMaximumMemoryUsage(NULL);CHKERRQ(ierr);

  ierr = pTatinFinalize();CHKERRQ(ierr);
  return 0;
}