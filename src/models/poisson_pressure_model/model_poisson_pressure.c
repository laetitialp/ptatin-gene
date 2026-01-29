#include "petsc.h"
#include "ptatin3d.h"
#include "private/ptatin_impl.h"
#include "private/quadrature_impl.h"

#include "dmda_bcs.h"
#include "data_bucket.h"
#include "MPntStd_def.h"
#include "MPntPStokes_def.h"
#include "MPntPStokesPl_def.h"
#include "MPntPEnergy_def.h"
#include "stokes_form_function.h"
#include "ptatin_std_dirichlet_boundary_conditions.h"
#include "dmda_iterator.h"
#include "mesh_deformation.h"
#include "mesh_update.h"
#include "dmda_remesh.h"
#include "dmda_element_q2p1.h"
#include "material_point_std_utils.h"
#include "material_point_popcontrol.h"
#include "model_utils.h"
#include "ptatin_utils.h"

#include "output_material_points.h"
#include "output_material_points_p0.h"
#include "energy_output.h"
#include "xdmf_writer.h"
#include "output_paraview.h"

#include "ptatin3d_energy.h"
#include <ptatin3d_energyfv.h>
#include <ptatin3d_energyfv_impl.h>
#include <material_constants_energy.h>

#include "quadrature.h"
#include "element_utils_q1.h"
#include "element_type_Q2.h"
#include "dmda_element_q2p1.h"

#include "litho_pressure_PDESolve.h"

#include "poisson_pressure_ctx.h"

static const char MODEL_NAME_R[] = "model_poisson_pressure_";

static PetscErrorCode ModelInitialGeometryPoissonPressure(ModelPoissonPressureCtx *data)
{
  PetscInt       nn;
  PetscBool      found;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* box geometry, [m] */
  data->Lx = 1000.0e3; 
  data->Ly = 0.0e3;
  data->Lz = 600.0e3;
  data->Ox = 0.0e3;
  data->Oy = -250.0e3;
  data->Oz = 0.0e3;

  data->y_continent[0] = -25.0e3;  // Conrad
  data->y_continent[1] = -35.0e3;  // Moho
  data->y_continent[2] = -120.0e3; // LAB

  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R,"-Lx",&data->Lx,&found);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R,"-Ly",&data->Ly,&found);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R,"-Lz",&data->Lz,&found);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R,"-Ox",&data->Ox,&found);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R,"-Oy",&data->Oy,&found);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R,"-Oz",&data->Oz,&found);CHKERRQ(ierr);

  nn = 3;
  ierr = PetscOptionsGetRealArray(NULL,MODEL_NAME_R,"-y_continent",data->y_continent,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != 3) {
      SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_USER,"Expected 3 values for -y_continent. Found %d",nn);
    }
  }

  /* reports before scaling */
  PetscPrintf(PETSC_COMM_WORLD,"************ Box Geometry ************\n",NULL);
  PetscPrintf(PETSC_COMM_WORLD,"[Ox,Lx] = [%+1.4e [m], %+1.4e [m]]\n", data->Ox ,data->Lx );
  PetscPrintf(PETSC_COMM_WORLD,"[Oy,Ly] = [%+1.4e [m], %+1.4e [m]]\n", data->Oy ,data->Ly );
  PetscPrintf(PETSC_COMM_WORLD,"[Oz,Lz] = [%+1.4e [m], %+1.4e [m]]\n", data->Oz ,data->Lz );
  PetscPrintf(PETSC_COMM_WORLD,"********** Initial layering **********\n",NULL);
  PetscPrintf(PETSC_COMM_WORLD,"Conrad: %+1.4e [m]\n", data->y_continent[0]);
  PetscPrintf(PETSC_COMM_WORLD,"Moho:   %+1.4e [m]\n", data->y_continent[1]);
  PetscPrintf(PETSC_COMM_WORLD,"LAB:    %+1.4e [m]\n", data->y_continent[2]);

  data->geometry_type = 0;
  ierr = PetscOptionsGetInt(NULL,MODEL_NAME_R,"-geometry_type",&data->geometry_type,&found);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetMaterialParametersVISCOUSPoissonPressure(pTatinCtx ptatin,DataBucket materialconstants, ModelPoissonPressureCtx *data)
{
  PetscInt       region_idx;
  PetscReal      density,viscosity;
  char           *option_name;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  for (region_idx=0; region_idx<data->n_phases; region_idx++) {
    MaterialConstantsSetValues_MaterialType(materialconstants,region_idx,VISCOUS_CONSTANT,PLASTIC_NONE,SOFTENING_NONE,DENSITY_CONSTANT);
    /* Set region viscosity */
    viscosity = 1.0e+23;
    if (asprintf (&option_name, "-eta0_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R, option_name,&viscosity,NULL);CHKERRQ(ierr); 
    free (option_name);
    ierr = MaterialConstantsSetValues_ViscosityConst(materialconstants,region_idx,viscosity);CHKERRQ(ierr);
    
    /* Set region density */
    density = 2700.0;
    if (asprintf (&option_name, "-rho0_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R, option_name,&density,NULL);CHKERRQ(ierr);
    free (option_name);
    ierr = MaterialConstantsSetValues_DensityConst(materialconstants,region_idx,density);CHKERRQ(ierr);
  }

  for (region_idx=0; region_idx<data->n_phases; region_idx++) {
    MaterialConstantsPrintAll(materialconstants,region_idx);
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetBoundaryValues(ModelPoissonPressureCtx *data)
{
  PetscBool      found;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  data->pressure_jmin = 0.0;
  data->pressure_jmax = 0.0;

  data->dirichlet_jmin = PETSC_FALSE;
  data->dirichlet_jmax = PETSC_FALSE;

  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R,"-pressure_jmin",&data->pressure_jmin,&found);CHKERRQ(ierr);
  if (!found) {
    PetscPrintf(PETSC_COMM_WORLD,"No value provided for JMIN Boundary condition. Using default %+1.4e. Use -pressure_jmin to set.\n", data->pressure_jmin);
  }
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R,"-pressure_jmax",&data->pressure_jmax,&found);CHKERRQ(ierr);
  if (!found) {
    PetscPrintf(PETSC_COMM_WORLD,"No value provided for JMAX Boundary condition. Using default %+1.4e. Use -pressure_jmax to set.\n", data->pressure_jmax);
  }
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_R,"-dirichlet_jmin",&data->dirichlet_jmin,&found);CHKERRQ(ierr);
  if (!found) {
    PetscPrintf(PETSC_COMM_WORLD,"No Dirichlet provided for JMIN Boundary condition. Using Neumann. Use -dirichlet_jmin to set.\n");
  }
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_R,"-dirichlet_jmax",&data->dirichlet_jmax,&found);CHKERRQ(ierr);
  if (!found) {
    PetscPrintf(PETSC_COMM_WORLD,"No Dirichlet provided for JMAX Boundary condition. Using Neumann. Use -dirichlet_jmax to set.\n");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetIsostaticParameters(ModelPoissonPressureCtx *data)
{
  PetscReal      isostatic_density_ref,isostatic_depth;
  PetscBool      found;
  char           option_value[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R,"-isostatic_density_ref",&isostatic_density_ref,&found);CHKERRQ(ierr);
  if (found) {
    isostatic_density_ref = isostatic_density_ref / data->density_bar;
    PetscSNPrintf(option_value,PETSC_MAX_PATH_LEN-1,"%1.5e",isostatic_density_ref);
    ierr = PetscOptionsSetValue(NULL,"-isostatic_density_ref_adim",option_value);CHKERRQ(ierr);
  }
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R,"-isostatic_depth",&isostatic_depth,&found);CHKERRQ(ierr);
  if (found) {
    isostatic_depth = isostatic_depth / data->length_bar;
    PetscSNPrintf(option_value,PETSC_MAX_PATH_LEN-1,"%1.5e",isostatic_depth);
    ierr = PetscOptionsSetValue(NULL,"-isostatic_compensation_depth_adim",option_value);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelScaleParametersPoissonPressure(DataBucket materialconstants, ModelPoissonPressureCtx *data)
{
  PetscInt       region_idx;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* scaling values */
  data->length_bar     = 1.0e5;
  data->viscosity_bar  = 1.0e22;
  data->velocity_bar   = 1.0e-10;
  /* Compute additional scaling parameters */
  data->time_bar         = data->length_bar / data->velocity_bar;
  data->pressure_bar     = data->viscosity_bar/data->time_bar;
  data->density_bar      = data->pressure_bar * (data->time_bar*data->time_bar)/(data->length_bar*data->length_bar); // kg.m^-3
  data->acceleration_bar = data->length_bar / (data->time_bar*data->time_bar);
  
  /* Scale length */
  data->Lx = data->Lx / data->length_bar;
  data->Ly = data->Ly / data->length_bar;
  data->Lz = data->Lz / data->length_bar;
  data->Ox = data->Ox / data->length_bar;
  data->Oy = data->Oy / data->length_bar;
  data->Oz = data->Oz / data->length_bar;

  data->y_continent[0] = data->y_continent[0] / data->length_bar;
  data->y_continent[1] = data->y_continent[1] / data->length_bar;
  data->y_continent[2] = data->y_continent[2] / data->length_bar;

  /* Scale BCs */
  data->pressure_jmin = data->pressure_jmin / data->pressure_bar;
  data->pressure_jmax = data->pressure_jmax / data->pressure_bar;

  /* scale material properties */
  for (region_idx=0; region_idx<data->n_phases; region_idx++) {
    ierr = MaterialConstantsScaleAll(materialconstants,region_idx,data->length_bar,data->velocity_bar,data->time_bar,data->viscosity_bar,data->density_bar,data->pressure_bar);CHKERRQ(ierr);
    ierr = MaterialConstantsEnergyScaleAll(materialconstants,region_idx,data->length_bar,data->time_bar,data->pressure_bar);CHKERRQ(ierr);
  }

  PetscPrintf(PETSC_COMM_WORLD,"[Rift Nitsche Model]:  during the solve scaling is done using \n");
  PetscPrintf(PETSC_COMM_WORLD,"  L*    : %1.4e [m]\n",       data->length_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  U*    : %1.4e [m.s^-1]\n",  data->velocity_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  t*    : %1.4e [s]\n",       data->time_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  eta*  : %1.4e [Pa.s]\n",    data->viscosity_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  rho*  : %1.4e [kg.m^-3]\n", data->density_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  P*    : %1.4e [Pa]\n",      data->pressure_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  a*    : %1.4e [m.s^-2]\n",  data->acceleration_bar );

  ierr = ModelSetIsostaticParameters(data);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode ModelInitializePoissonPressure(pTatinCtx ptatin,void *ctx)
{
  ModelPoissonPressureCtx *data;
  RheologyConstants       *rheology;
  DataBucket              materialconstants;
  PetscErrorCode          ierr;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  data = (ModelPoissonPressureCtx*)ctx;
  
  ierr = pTatinGetRheology(ptatin,&rheology);CHKERRQ(ierr);
  ierr = pTatinGetMaterialConstants(ptatin,&materialconstants);CHKERRQ(ierr); 

  /* Set the rheology to viscous */
  rheology->rheology_type = RHEOLOGY_VP_STD;
  
  /* Number of materials */
  data->n_phases = 4;
  rheology->nphases_active = data->n_phases;
  /* Initial geometry */
  ierr = ModelInitialGeometryPoissonPressure(data);CHKERRQ(ierr);
  /* BCs data */
  ierr = ModelSetBoundaryValues(data);CHKERRQ(ierr);
  /* Materials parameters */
  ierr = ModelSetMaterialParametersVISCOUSPoissonPressure(ptatin,materialconstants,data);CHKERRQ(ierr);
  /* Scale parameters */
  ierr = ModelScaleParametersPoissonPressure(materialconstants,data);CHKERRQ(ierr);

  data->output_markers = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_R,"-output_markers",&data->output_markers,NULL);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetMeshRefinementPoissonPressure(DM dav)
{
  PetscInt       dir,npoints;
  PetscReal      *xref,*xnat;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  dir = 1; // 0 = x; 1 = y; 2 = z;
  npoints = 4;

  ierr = PetscMalloc1(npoints,&xref);CHKERRQ(ierr); 
  ierr = PetscMalloc1(npoints,&xnat);CHKERRQ(ierr); 

  xref[0] = 0.0;
  xref[1] = 0.28; //0.375;
  xref[2] = 0.65; //0.75;
  xref[3] = 1.0;

  xnat[0] = 0.0;
  xnat[1] = 0.8;
  xnat[2] = 0.935;//0.95;
  xnat[3] = 1.0;

  ierr = DMDACoordinateRefinementTransferFunction(dav,dir,PETSC_TRUE,npoints,xref,xnat);CHKERRQ(ierr);

  ierr = PetscFree(xref);CHKERRQ(ierr);
  ierr = PetscFree(xnat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyInitialMeshGeometryPoissonPressure(pTatinCtx ptatin,void *ctx)
{
  ModelPoissonPressureCtx *data = (ModelPoissonPressureCtx*)ctx;
  PhysCompStokes stokes;
  DM             stokes_pack,dav,dap;
  PetscReal      gvec[] = { 0.0, -9.8, 0.0 };
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  ierr = pTatinGetStokesContext(ptatin,&stokes);CHKERRQ(ierr);
  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(dav,data->Ox,data->Lx,data->Oy,data->Ly,data->Oz,data->Lz);CHKERRQ(ierr);
  /* Mesh Refinement */
  ierr = ModelSetMeshRefinementPoissonPressure(dav);CHKERRQ(ierr);
  /* Gravity */
  ierr = PhysCompStokesSetGravityVector(ptatin->stokes_ctx,gvec);CHKERRQ(ierr);
  ierr = PhysCompStokesScaleGravityVector(ptatin->stokes_ctx,1.0/data->acceleration_bar);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#if 0
PetscErrorCode ModelApplyGravityPoissonPressure(pTatinCtx ptatin, void *ctx)
{
  ModelPoissonPressureCtx *data = (ModelPoissonPressureCtx*)ctx;
  Gravity                 gravity;
  PetscReal               gvec[] = { 0.0, -9.8, 0.0 };
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  ierr = pTatinCreateGravity(ptatin,GRAVITY_CONSTANT);CHKERRQ(ierr);
  ierr = pTatinGetGravityCtx(ptatin,&gravity);CHKERRQ(ierr);
  ierr = GravitySet_Constant(gravity,gvec);CHKERRQ(ierr);
  ierr = GravityScale(gravity,1.0/data->acceleration_bar);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
#endif

static PetscErrorCode ModelSetInitialMaterialLayeringPoissonPressure_Horizontal(MPntStd *material_point, double *position, ModelPoissonPressureCtx *data)
{
  int region_idx;
  PetscFunctionBegin;

  /* Set default region index to 0 */
  region_idx = 0;
  /* Attribute region index to layers */
  if (position[1] <= data->y_continent[0]) { region_idx = 1; }
  if (position[1] <= data->y_continent[1]) { region_idx = 2; }
  if (position[1] <= data->y_continent[2]) { region_idx = 3; }
  /* Set value */
  MPntStdSetField_phase_index(material_point,region_idx);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetInitialMaterialLayeringPoissonPressure_LateralVariations(MPntStd *material_point, double *position, ModelPoissonPressureCtx *data)
{
  PetscReal centre[3];
  int       region_idx;
  PetscFunctionBegin;

  centre[0] = 0.5*(data->Ox + data->Lx);
  centre[1] = 0.5*(data->Oy + data->Ly);
  centre[2] = 0.5*(data->Oz + data->Lz);

  if (position[1] > data->y_continent[2]) {
    if (position[0] < centre[0] && position[2] < centre[2]) {
      region_idx = 0;
    } else if (position[0] < centre[0] && position[2] >= centre[2]) {
      region_idx = 1;
    } else {
      region_idx = 2;
    }
  } else { 
    region_idx = 3; 
  }
  MPntStdSetField_phase_index(material_point,region_idx);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetInitialMaterialLayeringPoissonPressure(MPntStd *material_point, double *position, ModelPoissonPressureCtx *data)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  switch (data->geometry_type) {
    case 0:
      ierr = ModelSetInitialMaterialLayeringPoissonPressure_Horizontal(material_point,position,data);CHKERRQ(ierr);
      break;

    case 1:
      ierr = ModelSetInitialMaterialLayeringPoissonPressure_LateralVariations(material_point,position,data);CHKERRQ(ierr);
      break; 

    default:
      SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_USER,"Unreconnized value for geometry_type. Only 1 and 2 are available, you provided %d",data->geometry_type);
      break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyInitialMaterialParametersPoissonPressure(pTatinCtx ptatin,void *ctx)
{
  ModelPoissonPressureCtx *data = (ModelPoissonPressureCtx*)ctx;
  DataBucket              db;
  DataField               PField_std;
  DM                      dau,dap;
  int                     n_mp_points,p;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  ierr = pTatinGetMaterialPoints(ptatin,&db,NULL);CHKERRQ(ierr);
  /* std variables */
  DataBucketGetDataFieldByName(db,MPntStd_classname,&PField_std);
  DataFieldGetAccess(PField_std);
  DataFieldVerifyAccess(PField_std,sizeof(MPntStd));

  DataBucketGetSizes(db,&n_mp_points,0,0);
  for (p=0; p<n_mp_points; p++) {
    MPntStd       *material_point;
    double        *position;

    DataFieldAccessPoint(PField_std,p,(void**)&material_point);

    /* Access coordinates of the marker */
    MPntStdGetField_global_coord(material_point,&position);

    /* Layering geometry */
    ierr = ModelSetInitialMaterialLayeringPoissonPressure(material_point,position,data);CHKERRQ(ierr);
  }
  DataFieldRestoreAccess(PField_std);

  /* Generate the initial isostatic topography from lithostatic pressure */
  ierr = LagrangianAdvectionFromIsostaticDisplacementVector(ptatin);CHKERRQ(ierr);
  /* Refine mesh */
  ierr = DMCompositeGetEntries(ptatin->stokes_ctx->stokes_pack,&dau,&dap);CHKERRQ(ierr);
  ierr = ModelSetMeshRefinementPoissonPressure(dau);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyInitialStokesVariableMarkersPoissonPressure(pTatinCtx ptatin,Vec X,void *ctx)
{
  DM             stokes_pack,dau,dap;
  PhysCompStokes stokes;
  Vec            Uloc,Ploc;
  PetscScalar    *LA_Uloc,*LA_Ploc;
  DataField      PField;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);
  
  DataBucketGetDataFieldByName(ptatin->material_constants,MaterialConst_MaterialType_classname,&PField);
  ierr = pTatinGetStokesContext(ptatin,&stokes);CHKERRQ(ierr);
  stokes_pack = stokes->stokes_pack;

  ierr = DMCompositeGetEntries(stokes_pack,&dau,&dap);CHKERRQ(ierr);
  ierr = DMCompositeGetLocalVectors(stokes_pack,&Uloc,&Ploc);CHKERRQ(ierr);

  ierr = DMCompositeScatter(stokes_pack,X,Uloc,Ploc);CHKERRQ(ierr);
  ierr = VecGetArray(Uloc,&LA_Uloc);CHKERRQ(ierr);
  ierr = VecGetArray(Ploc,&LA_Ploc);CHKERRQ(ierr);
  ierr = pTatin_EvaluateRheologyNonlinearities(ptatin,dau,LA_Uloc,dap,LA_Ploc);CHKERRQ(ierr);
  ierr = VecRestoreArray(Uloc,&LA_Uloc);CHKERRQ(ierr);
  ierr = VecRestoreArray(Ploc,&LA_Ploc);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyInitialVelocityFieldPoissonPressure(DM dau, Vec velocity)
{
  PetscReal      zero = 0.0;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  /* Initialize to zero the velocity vector */
  ierr = VecZeroEntries(velocity);CHKERRQ(ierr);

  /* x component */
  ierr = DMDAVecTraverse3d(dau,velocity,0,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  /* y component */
  ierr = DMDAVecTraverse3d(dau,velocity,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  /* z component */
  ierr = DMDAVecTraverse3d(dau,velocity,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyInitialHydrostaticPressureFieldPoissonPressure(pTatinCtx c, DM dau, DM dap, Vec pressure, ModelPoissonPressureCtx *data)
{
  PetscReal                                    MeshMin[3],MeshMax[3],domain_height;
  DMDAVecTraverse3d_HydrostaticPressureCalcCtx HPctx;
  PetscErrorCode                               ierr;

  PetscFunctionBegin;

  /* Initialize pressure vector to zero */
  ierr = VecZeroEntries(pressure);CHKERRQ(ierr);
  ierr = DMGetBoundingBox(dau,MeshMin,MeshMax);CHKERRQ(ierr);
  domain_height = MeshMax[1] - MeshMin[1];

  /* Values for hydrostatic pressure computing */
  HPctx.surface_pressure = 0.0;
  HPctx.ref_height = domain_height;
  HPctx.ref_N      = c->stokes_ctx->my-1;
  HPctx.grav       = 9.8 / data->acceleration_bar;
  HPctx.rho        = 3300.0 / data->density_bar;

  ierr = DMDAVecTraverseIJK(dap,pressure,0,DMDAVecTraverseIJK_HydroStaticPressure_v2,     (void*)&HPctx);CHKERRQ(ierr);
  ierr = DMDAVecTraverseIJK(dap,pressure,2,DMDAVecTraverseIJK_HydroStaticPressure_dpdy_v2,(void*)&HPctx);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyInitialSolutionPoissonPressure(pTatinCtx c,Vec X,void *ctx)
{
  ModelPoissonPressureCtx *data;
  DM                      stokes_pack,dau,dap;
  Vec                     velocity,pressure;
  PetscErrorCode          ierr;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  data = (ModelPoissonPressureCtx*)ctx;
  
  /* Access velocity and pressure vectors */
  stokes_pack = c->stokes_ctx->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dau,&dap);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);
  /* Velocity IC */
  ierr = ModelApplyInitialVelocityFieldPoissonPressure(dau,velocity);CHKERRQ(ierr);
  /* Pressure IC */
  ierr = ModelApplyInitialHydrostaticPressureFieldPoissonPressure(c,dau,dap,pressure,data);CHKERRQ(ierr);
  /* Restore velocity and pressure vectors */
  ierr = DMCompositeRestoreAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyBoundaryConditionsVelocity(DM dav, BCList bclist)
{
  PetscReal      zero=0.0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  
  /* Free-slip */
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,0,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,0,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMAX_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  /* Free-surface */

  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyBoundaryConditionsPoissonPressure(pTatinCtx ptatin,void *ctx)
{
  ModelPoissonPressureCtx *data = (ModelPoissonPressureCtx*)ctx;
  PhysCompStokes          stokes;
  PDESolveLithoP          poisson_pressure;
  DM                      dav,dap;
  PetscBool               active_poisson;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  /* Velocity */
  ierr = pTatinGetStokesContext(ptatin,&stokes);CHKERRQ(ierr);
  ierr = DMCompositeGetEntries(stokes->stokes_pack,&dav,&dap);CHKERRQ(ierr);
  ierr = ModelApplyBoundaryConditionsVelocity(dav,stokes->u_bclist);CHKERRQ(ierr);

  /* Poisson pressure */
  ierr = pTatinContextValid_LithoP(ptatin,&active_poisson);CHKERRQ(ierr);
  if (active_poisson) {
    ierr = pTatinGetContext_LithoP(ptatin,&poisson_pressure);CHKERRQ(ierr);

    if (data->dirichlet_jmax) {
      ierr = DMDABCListTraverse3d(poisson_pressure->bclist,poisson_pressure->da,DMDABCList_JMAX_LOC,0,BCListEvaluator_constant,(void*)&data->pressure_jmax);CHKERRQ(ierr);
    }
    if (data->dirichlet_jmin) {
      ierr = DMDABCListTraverse3d(poisson_pressure->bclist,poisson_pressure->da,DMDABCList_JMIN_LOC,0,BCListEvaluator_constant,(void*)&data->pressure_jmin);CHKERRQ(ierr);
    }
    if (!data->dirichlet_jmax && !data->dirichlet_jmin) {
      PetscPrintf(PETSC_COMM_WORLD,"No Dirichlet BC provided, at least 1 is required, assuming JMAX BC value = %+1.4e\n", data->pressure_jmax);
      ierr = DMDABCListTraverse3d(poisson_pressure->bclist,poisson_pressure->da,DMDABCList_JMAX_LOC,0,BCListEvaluator_constant,(void*)&data->pressure_jmax);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyBoundaryConditionMGPoissonPressure(PetscInt nl,BCList bclist[],SurfBCList surf_bclist[],DM dav[],pTatinCtx c,void *ctx)
{
  PetscInt         n;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  /* Define velocity boundary conditions on each level within the MG hierarchy */
  for (n=0; n<nl; n++) {
    ierr = ModelApplyBoundaryConditionsVelocity(dav[n],bclist[n]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelOutputMarkerFieldsPoissonPressure(pTatinCtx ptatin,const char prefix[])
{
  DataBucket               materialpoint_db;
  int                      nf;
  const MaterialPointField mp_prop_list[] = { MPField_Std, MPField_Stokes };
  char                     mp_file_prefix[256];
  PetscErrorCode           ierr;

  PetscFunctionBegin;

  nf = sizeof(mp_prop_list)/sizeof(mp_prop_list[0]);

  ierr = pTatinGetMaterialPoints(ptatin,&materialpoint_db,NULL);CHKERRQ(ierr);
  sprintf(mp_file_prefix,"%s_mpoints",prefix);
  ierr = SwarmViewGeneric_ParaView(materialpoint_db,nf,mp_prop_list,ptatin->outputpath,mp_file_prefix);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode ModelOutputPoissonPressure(pTatinCtx ptatin,Vec X,const char prefix[],void *ctx)
{
  ModelPoissonPressureCtx     *data;
  const MaterialPointVariable mp_prop_list[] = { MPV_region, MPV_viscosity, MPV_density };
  PetscErrorCode              ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  data = (ModelPoissonPressureCtx*)ctx;
  
  /* Output Velocity and pressure */
  ierr = pTatin3d_ModelOutputPetscVec_VelocityPressure_Stokes(ptatin,X,prefix);CHKERRQ(ierr);
  
  /* Output markers cell fields (for production runs) */
  ierr = pTatin3dModelOutput_MarkerCellFieldsP0_PetscVec(ptatin,PETSC_FALSE,sizeof(mp_prop_list)/sizeof(MaterialPointVariable),mp_prop_list,prefix);CHKERRQ(ierr);
  
  /* Output raw markers and vtu velocity and pressure (for testing and debugging) */
  if (data->output_markers) {
    ierr = pTatin3d_ModelOutput_VelocityPressure_Stokes(ptatin,X,prefix);CHKERRQ(ierr);
    ierr = ModelOutputMarkerFieldsPoissonPressure(ptatin,prefix);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ModelDestroyPoissonPressure(pTatinCtx c,void *ctx)
{
  ModelPoissonPressureCtx *data;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);
  data = (ModelPoissonPressureCtx*)ctx;

  /* Free contents of structure */
  /* Free structure */
  ierr = PetscFree(data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode pTatinModelRegister_PoissonPressure(void)
{
  ModelPoissonPressureCtx *data;
  pTatinModel             model;
  PetscErrorCode          ierr;

  PetscFunctionBegin;

  /* Allocate memory for the data structure for this model */
  ierr = PetscMalloc(sizeof(ModelPoissonPressureCtx),&data);CHKERRQ(ierr);
  ierr = PetscMemzero(data,sizeof(ModelPoissonPressureCtx));CHKERRQ(ierr);

  /* register user model */
  ierr = pTatinModelCreate(&model);CHKERRQ(ierr);

  /* Set name, model select via -ptatin_model NAME */
  ierr = pTatinModelSetName(model,"poisson_pressure");CHKERRQ(ierr);

  /* Set model data */
  ierr = pTatinModelSetUserData(model,data);CHKERRQ(ierr);

  /* Set function pointers */
  ierr = pTatinModelSetFunctionPointer(model,PTATIN_MODEL_INIT,                  (void (*)(void))ModelInitializePoissonPressure);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(model,PTATIN_MODEL_APPLY_INIT_MESH_GEOM,  (void (*)(void))ModelApplyInitialMeshGeometryPoissonPressure);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(model,PTATIN_MODEL_APPLY_INIT_MAT_GEOM,   (void (*)(void))ModelApplyInitialMaterialParametersPoissonPressure);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(model,PTATIN_MODEL_APPLY_INIT_STOKES_VARIABLE_MARKERS,(void (*)(void))ModelApplyInitialStokesVariableMarkersPoissonPressure);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(model,PTATIN_MODEL_APPLY_INIT_SOLUTION,   (void (*)(void))ModelApplyInitialSolutionPoissonPressure);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(model,PTATIN_MODEL_APPLY_BC,              (void (*)(void))ModelApplyBoundaryConditionsPoissonPressure);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(model,PTATIN_MODEL_APPLY_BCMG,            (void (*)(void))ModelApplyBoundaryConditionMGPoissonPressure);CHKERRQ(ierr);
  //ierr = pTatinModelSetFunctionPointer(model,PTATIN_MODEL_ADAPT_MP_RESOLUTION,   (void (*)(void))NULL);CHKERRQ(ierr);
  //ierr = pTatinModelSetFunctionPointer(model,PTATIN_MODEL_APPLY_UPDATE_MESH_GEOM,(void (*)(void))NULL);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(model,PTATIN_MODEL_OUTPUT,                (void (*)(void))ModelOutputPoissonPressure);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(model,PTATIN_MODEL_DESTROY,               (void (*)(void))ModelDestroyPoissonPressure);CHKERRQ(ierr);
  //ierr = pTatinModelSetFunctionPointer(model,PTATIN_MODEL_APPLY_GRAVITY,         (void (*)(void))ModelApplyGravityPoissonPressure);CHKERRQ(ierr);

  /* Insert model into list */
  ierr = pTatinModelRegister(model);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}