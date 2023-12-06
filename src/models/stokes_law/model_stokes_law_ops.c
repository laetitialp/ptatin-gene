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
#include "QPntSurfCoefStokes_def.h"
#include "mesh_entity.h"
#include "surface_constraint.h"
#include "surfbclist.h"
#include "element_utils_q1.h"
#include "element_type_Q2.h"
#include "dmda_element_q2p1.h"

#include "stokes_law_ctx.h"

static const char MODEL_NAME_R[] = "model_stokes_law_";

static PetscErrorCode ModelInitialGeometry_StokesLaw(ModelStokesLawCtx *data)
{
  PetscInt       nn,d;
  PetscBool      found;
  PetscReal      min_L;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* box geometry, [m] */
  data->O[0] = 0.0;
  data->O[1] = 0.0;
  data->O[2] = 0.0;
  data->L[0] = 1.0; 
  data->L[1] = 10.0;
  data->L[2] = 1.0;

  nn = 3;
  ierr = PetscOptionsGetRealArray(NULL,MODEL_NAME_R,"-O",data->O,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != 3) {
      SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_USER,"Expected 3 values for -O. Found %d",nn);
    }
  }

  nn = 3;
  ierr = PetscOptionsGetRealArray(NULL,MODEL_NAME_R,"-L",data->L,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != 3) {
      SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_USER,"Expected 3 values for -L. Found %d",nn);
    }
  }

  /* sphere radius, by default set to 1/10 of the domain min length */
  min_L = 1.0e32;
  for (d=0; d<3; d++) {
    min_L = PetscMin(min_L, (data->L[d] - data->O[d]) );  
  }
  data->r_s = 0.1 * min_L;
  /* use option to erase it */
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R,"-radius_sphere",&data->r_s,NULL);CHKERRQ(ierr);

  /* sphere centre initial coordinates */
  data->sphere_centre[0] = 0.5*(data->O[0] + data->L[0]);
  data->sphere_centre[1] = 0.8*(data->O[1] + data->L[1]);
  data->sphere_centre[2] = 0.5*(data->O[2] + data->L[2]);
  nn = 3;
  ierr = PetscOptionsGetRealArray(NULL,MODEL_NAME_R,"-sphere_centre",data->sphere_centre,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != 3) {
      SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_USER,"Expected 3 values for -sphere_centre. Found %d",nn);
    }
  }


  /* reports before scaling */
  PetscPrintf(PETSC_COMM_WORLD,"************ Box Geometry ************\n",NULL);
  for (d=0; d<3; d++) {
    PetscPrintf(PETSC_COMM_WORLD,"[O[%d],L[%d]] = [%+1.4e [m], %+1.4e [m]]\n", data->O[d] ,data->L[d] );
  }
  PetscPrintf(PETSC_COMM_WORLD,"Sphere radius = %1.4e [m]\n",data->r_s);
  PetscPrintf(PETSC_COMM_WORLD,"Sphere centre = [ %1.4e, %1.4e, %1.4e ] [m]\n",data->sphere_centre[0], data->sphere_centre[1], data->sphere_centre[2]);


  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetMaterialParameters_StokesLaw(DataBucket materialconstants, ModelStokesLawCtx *data)
{
  DataField                 PField,PField_k;
  EnergyMaterialConstants   *matconstants_e;
  EnergyConductivityConst   *data_k;
  PetscInt                  region_idx;
  int                       source_type[7] = {0, 0, 0, 0, 0, 0, 0};
  PetscReal                 density,viscosity;
  PetscReal                 beta,alpha,conductivity,Cp;
  char                      *option_name;
  PetscErrorCode            ierr;

  PetscFunctionBegin;

  /* Energy material constants */
  DataBucketGetDataFieldByName(materialconstants,EnergyMaterialConstants_classname,&PField);
  DataFieldGetEntries(PField,(void**)&matconstants_e);
  
  /* Conductivity */
  DataBucketGetDataFieldByName(materialconstants,EnergyConductivityConst_classname,&PField_k);
  DataFieldGetEntries(PField_k,(void**)&data_k);

  Cp = 1000.0;
  for (region_idx=0; region_idx<data->nregions; region_idx++) {
    MaterialConstantsSetValues_MaterialType(materialconstants,region_idx,VISCOUS_CONSTANT,PLASTIC_NONE,SOFTENING_NONE,DENSITY_CONSTANT);
    /* Set region viscosity */
    viscosity = 1.0e+23;
    if (asprintf (&option_name, "-viscosity_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R, option_name,&viscosity,NULL);CHKERRQ(ierr); 
    free (option_name);
    ierr = MaterialConstantsSetValues_ViscosityConst(materialconstants,region_idx,viscosity);CHKERRQ(ierr);
    
    /* Set region density */
    density = 1000.0;
    if (asprintf (&option_name, "-density_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R, option_name,&density,NULL);CHKERRQ(ierr);
    free (option_name);
    ierr = MaterialConstantsSetValues_DensityConst(materialconstants,region_idx,density);CHKERRQ(ierr);

    /* ENERGY PARAMETERS */
    if (asprintf (&option_name, "-conductivity_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R, option_name,&conductivity,NULL);CHKERRQ(ierr);
    free (option_name);

    alpha = 0.0;
    beta  = 0.0;
    /* Set energy params for region_idx */
    MaterialConstantsSetValues_EnergyMaterialConstants(region_idx,matconstants_e,alpha,beta,density,Cp,ENERGYDENSITY_CONSTANT,ENERGYCONDUCTIVITY_CONSTANT,source_type);
    EnergyConductivityConstSetField_k0(&data_k[region_idx],conductivity);
  }

  for (region_idx=0; region_idx<data->nregions; region_idx++) {
    MaterialConstantsPrintAll(materialconstants,region_idx);
    MaterialConstantsEnergyPrintAll(materialconstants,region_idx);
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetGravityVector(ModelStokesLawCtx *data)
{
  PetscInt       nn;
  PetscBool      found;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  data->gravity[0] =  0.0;
  data->gravity[1] = -1.0;
  data->gravity[2] =  0.0;

  nn = 3;
  ierr = PetscOptionsGetRealArray(NULL,MODEL_NAME_R,"-gravity_vector",data->gravity,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != 3) {
      SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_USER,"Expected 3 values for -gravity_vector. Found %d",nn);
    }
  }
  
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetStokesTerminalVelocity(ModelStokesLawCtx *data)
{
  PetscInt d;
  PetscFunctionBegin;
  /* 
  This is the velocity obtained after the all forces balance i.e.,
  \lim_{t \rightarrow \infty} \vec u (t) = frac{2 r^2 \vec g }{9 \eta} (\rho_s - \rho_f)
  */
  for (d=0; d<3; d++) {
    data->u_T[d] = (2.0 * data->r_s*data->r_s * data->gravity[d]) / (9.0 * data->eta_f) * (data->rho_s - data->rho_f);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelScaleParameters_StokesLaw(DataBucket materialconstants, ModelStokesLawCtx *data)
{
  PetscInt       d,region_idx;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  /* scaling values */
  data->length_bar      = 1.0;
  data->viscosity_bar   = 1.0e6;
  data->velocity_bar    = 1.0e-3;
  /* Compute additional scaling parameters */
  data->time_bar         = data->length_bar / data->velocity_bar;
  data->pressure_bar     = data->viscosity_bar/data->time_bar;
  data->density_bar      = data->pressure_bar * (data->time_bar*data->time_bar)/(data->length_bar*data->length_bar); // kg.m^-3
  data->acceleration_bar = data->length_bar / (data->time_bar*data->time_bar);
  
  /* Scale length */
  for (d=0; d<3; d++) {
    data->O[d] /= data->length_bar;
    data->L[d] /= data->length_bar;
    data->sphere_centre[d] /= data->length_bar;
  }
  data->r_s /= data->length_bar;

  /* scale gravity */
  for (d=0; d<3; d++) {
    data->gravity[d] /= data->acceleration_bar;
  }

  /* Scale material parameters */
  for (region_idx=0; region_idx<data->nregions; region_idx++) {
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

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelGetMaterialParameters_StokesLaw(DataBucket materialconstants, ModelStokesLawCtx *data)
{
  DataField                    PField_rho,PField_eta;
  MaterialConst_DensityConst   *data_rho;
  MaterialConst_ViscosityConst *data_eta;
  PetscInt                     region_idx;

  PetscFunctionBegin;

  /* Attach scaled viscosity and density values to model data structure */

  DataBucketGetDataFieldByName(materialconstants,MaterialConst_DensityConst_classname,&PField_rho);
  DataFieldGetAccess(PField_rho);

  DataBucketGetDataFieldByName(materialconstants,MaterialConst_ViscosityConst_classname,&PField_eta);
  DataFieldGetAccess(PField_eta);

  /* Fluid */
  region_idx = 0;
  DataFieldAccessPoint(PField_rho,region_idx,(void**)&data_rho);
  DataFieldAccessPoint(PField_eta,region_idx,(void**)&data_eta);

  MaterialConst_DensityConstGetField_density(data_rho,&data->rho_f);
  MaterialConst_ViscosityConstGetField_eta0(data_eta, &data->eta_f);
  PetscPrintf(PETSC_COMM_WORLD,"Scaled fluid parameters: rho_f = %1.4e, eta_f = %1.4e\n",data->rho_f,data->eta_f);

  /* Sphere */
  region_idx = 1;
  DataFieldAccessPoint(PField_rho,region_idx,(void**)&data_rho);
  DataFieldAccessPoint(PField_eta,region_idx,(void**)&data_eta);

  MaterialConst_DensityConstGetField_density(data_rho,&data->rho_s);
  MaterialConst_ViscosityConstGetField_eta0(data_eta, &data->eta_s);
  PetscPrintf(PETSC_COMM_WORLD,"Scaled sphere parameters: rho_s = %1.4e, eta_s = %1.4e\n",data->rho_s,data->eta_s);

  DataFieldRestoreAccess(PField_rho);
  DataFieldRestoreAccess(PField_eta);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetTemperatureParameters_StokesLaw(ModelStokesLawCtx *data)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  data->Ttop    = 1.0; // Top temperature BC
  data->Tbottom = 1.0; // Bottom temperature BC
  data->T_f     = 1.0; // temperature of the fluid
  data->T_s     = 5.0; // temperature of the sphere

  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R,"-Ttop",   &data->Ttop,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R,"-Tbottom",&data->Tbottom,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R,"-Tfluid", &data->T_f,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R,"-Tsphere",&data->T_s,NULL);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetBCsType(ModelStokesLawCtx *data)
{
  PetscBool      bc_free_surface;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  bc_free_surface = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_R,"-bc_free_surface",&bc_free_surface,NULL);CHKERRQ(ierr);

  data->bc_type = 0;
  if (bc_free_surface) { data->bc_type = 1; }

  PetscFunctionReturn(0);
}

PetscErrorCode ModelInitialize_StokesLaw(pTatinCtx ptatin, void *ctx)
{
  ModelStokesLawCtx  *data;
  RheologyConstants  *rheology;
  DataBucket         materialconstants;
  PetscErrorCode     ierr;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  data = (ModelStokesLawCtx*)ctx;
  
  ierr = pTatinGetRheology(ptatin,&rheology);CHKERRQ(ierr);
  ierr = pTatinGetMaterialConstants(ptatin,&materialconstants);CHKERRQ(ierr); 

  /* Set the rheology to visco-plastic temperature dependant */
  rheology->rheology_type = RHEOLOGY_VP_STD;
  /* force energy equation to be introduced */
  ierr = PetscOptionsInsertString(NULL,"-activate_energyfv true");CHKERRQ(ierr);
  
  /* Number of materials */
  data->nregions = 2;
  rheology->nphases_active = data->nregions;
  /* Initial geometry */
  ierr = ModelInitialGeometry_StokesLaw(data);CHKERRQ(ierr);
  /* Gravity */
  ierr = ModelSetGravityVector(data);CHKERRQ(ierr);
  /* BCs data */
  ierr = ModelSetBCsType(data);CHKERRQ(ierr);
  ierr = ModelSetTemperatureParameters_StokesLaw(data);CHKERRQ(ierr);
  /* Materials parameters */
  ierr = ModelSetMaterialParameters_StokesLaw(materialconstants,data);CHKERRQ(ierr);
  /* Scale parameters */
  ierr = ModelScaleParameters_StokesLaw(materialconstants,data);CHKERRQ(ierr);
  /* Attach scaled material parameters to model data structure */
  ierr = ModelGetMaterialParameters_StokesLaw(materialconstants,data);CHKERRQ(ierr);
  /* Terminal velocity */
  ierr = ModelSetStokesTerminalVelocity(data);CHKERRQ(ierr);
  
  data->output_markers  = PETSC_FALSE;
  data->output_petscvec = PETSC_FALSE;
  data->refine_mesh     = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_R,"-output_markers",  &data->output_markers,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_R,"-output_petscvec", &data->output_petscvec,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_R,"-refine_mesh",     &data->refine_mesh,NULL);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyMeshRefinement(DM dav)
{
  PetscInt       npoints;
  PetscReal      *xref,*xnat;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  npoints = 4;

  ierr = PetscMalloc1(npoints,&xref);CHKERRQ(ierr); 
  ierr = PetscMalloc1(npoints,&xnat);CHKERRQ(ierr);

  xref[0] = 0.0;
  xref[1] = 0.1;
  xref[2] = 0.9;
  xref[3] = 1.0;

  xnat[0] = 0.0;
  xnat[1] = 0.25;
  xnat[2] = 0.75;
  xnat[3] = 1.0;

  /* x dir refinement */
  ierr = DMDACoordinateRefinementTransferFunction(dav,0,PETSC_TRUE,npoints,xref,xnat);CHKERRQ(ierr);
  /* y dir refinement */
  ierr = DMDACoordinateRefinementTransferFunction(dav,1,PETSC_TRUE,npoints,xref,xnat);CHKERRQ(ierr);
  /* z dir refinement */
  ierr = DMDACoordinateRefinementTransferFunction(dav,2,PETSC_TRUE,npoints,xref,xnat);CHKERRQ(ierr);

  ierr = PetscFree(xref);CHKERRQ(ierr);
  ierr = PetscFree(xnat);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyInitialMeshGeometry_StokesLaw(pTatinCtx ptatin, void *ctx)
{
  ModelStokesLawCtx *data = (ModelStokesLawCtx*)ctx;
  PhysCompStokes    stokes;
  DM                stokes_pack,dav,dap;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  ierr = pTatinGetStokesContext(ptatin,&stokes);CHKERRQ(ierr);
  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(dav,data->O[0],data->L[0],data->O[1],data->L[1],data->O[2],data->L[2]);CHKERRQ(ierr);
  
  if (data->refine_mesh) {
    ierr = ModelApplyMeshRefinement(dav);CHKERRQ(ierr);
  }
  /* Gravity */
  ierr = PhysCompStokesSetGravityVector(ptatin->stokes_ctx,data->gravity);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyInitialMaterialParameters_StokesLaw(pTatinCtx ptatin,void *ctx)
{
  ModelStokesLawCtx *data = (ModelStokesLawCtx*)ctx;
  DataField         PField_std,PField_stokes;
  DataBucket        db;
  int               n_mp_points,p,d;
  PetscErrorCode    ierr;

  PetscFunctionBegin;

  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  ierr = pTatinGetMaterialPoints(ptatin,&db,NULL);CHKERRQ(ierr);
  /* std variables */
  DataBucketGetDataFieldByName(db,MPntStd_classname,&PField_std);
  DataFieldGetAccess(PField_std);
  DataFieldVerifyAccess(PField_std,sizeof(MPntStd));

  /* Stokes variables */
  DataBucketGetDataFieldByName(db,MPntPStokes_classname,&PField_stokes);
  DataFieldGetAccess(PField_stokes);
  DataFieldVerifyAccess(PField_stokes,sizeof(MPntPStokes));
  
  DataBucketGetSizes(db,&n_mp_points,0,0);
  for (p=0; p<n_mp_points; p++) {
    MPntStd     *mpp_std;
    MPntPStokes *mpp_stokes;
    double      *position,sep;

    DataFieldAccessPoint(PField_std,   p,(void**)&mpp_std);
    DataFieldAccessPoint(PField_stokes,p,(void**)&mpp_stokes);    

    /* Access coordinates of the marker */
    MPntStdGetField_global_coord(mpp_std,&position);

    /* Background fluid */
    MPntStdSetField_phase_index(mpp_std,0);
    MPntPStokesSetField_eta_effective(mpp_stokes,data->eta_f);
    MPntPStokesSetField_density(mpp_stokes,data->rho_f);

    /* Sphere */
    sep = 0.0;
    for (d=0; d<3; d++) {
      sep += (position[d] - data->sphere_centre[d])*(position[d] - data->sphere_centre[d]);
    }
    /* check if inside the sphere */
    if (sep <= data->r_s*data->r_s) {
      MPntStdSetField_phase_index(mpp_std,1);
      MPntPStokesSetField_eta_effective(mpp_stokes,data->eta_s);
      MPntPStokesSetField_density(mpp_stokes,data->rho_s);
    }
  }

  DataFieldRestoreAccess(PField_std);
  DataFieldRestoreAccess(PField_stokes);

  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyInitialStokesVariableMarkers_StokesLaw(pTatinCtx ptatin, Vec X, void *ctx)
{
  DM                         stokes_pack,dau,dap;
  PhysCompStokes             stokes;
  Vec                        Uloc,Ploc;
  PetscScalar                *LA_Uloc,*LA_Ploc;
  DataField                  PField;
  PetscErrorCode             ierr;

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

static PetscBool AnalyticalStokesLaw(PetscScalar position[], PetscScalar *value, void *ctx)
{
  ModelStokesLawCtx *data = (ModelStokesLawCtx*)ctx;
  PetscBool         impose=PETSC_TRUE;

  PetscFunctionBegin;

  *value = data->u_T[ data->component ];// * ( 1.0 - PetscExpReal(-9.0 * data->eta_f * data->time / (2.0 * data->r_s*data->r_s * data->rho_s) ) );
  
  PetscFunctionReturn(impose);
}

static PetscErrorCode ModelApplyInitialVelocityField_StokesLaw(pTatinCtx ptatin, DM dau, Vec velocity, ModelStokesLawCtx *data)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  /* Initialize to zero the velocity vector */
  ierr = VecZeroEntries(velocity);CHKERRQ(ierr);
#if 0
  ierr = pTatinGetTime(ptatin,&data->time);CHKERRQ(ierr);
  /* x component */
  data->component = 0;
  ierr = DMDAVecTraverse3d(dau,velocity,data->component,AnalyticalStokesLaw,(void*)data);CHKERRQ(ierr);
  /* y component */
  data->component = 1;
  ierr = DMDAVecTraverse3d(dau,velocity,data->component,AnalyticalStokesLaw,(void*)data);CHKERRQ(ierr);
  /* z component */
  data->component = 2;
  ierr = DMDAVecTraverse3d(dau,velocity,data->component,AnalyticalStokesLaw,(void*)data);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyInitialHydrostaticPressureField_StokesLaw(pTatinCtx ptatin, DM dau, DM dap, Vec pressure, ModelStokesLawCtx *data)
{
  DMDAVecTraverse3d_HydrostaticPressureCalcCtx HPctx;
  PetscInt                                     d;
  PetscReal                                    MeshMin[3],MeshMax[3],domain_height,norm_g;
  PetscErrorCode                               ierr;

  PetscFunctionBegin;

  /* Initialize pressure vector to zero */
  ierr = VecZeroEntries(pressure);CHKERRQ(ierr);
  ierr = DMGetBoundingBox(dau,MeshMin,MeshMax);CHKERRQ(ierr);
  domain_height = MeshMax[1] - MeshMin[1];

  norm_g = 0.0;
  for (d=0; d<3; d++) {
    norm_g += data->gravity[d]*data->gravity[d];
  }
  norm_g = PetscSqrtReal(norm_g);

  /* Values for hydrostatic pressure computing */
  HPctx.surface_pressure = 0.0;
  HPctx.ref_height = domain_height;
  HPctx.ref_N      = ptatin->stokes_ctx->my-1;
  HPctx.grav       = norm_g;
  HPctx.rho        = data->rho_f;

  ierr = DMDAVecTraverseIJK(dap,pressure,0,DMDAVecTraverseIJK_HydroStaticPressure_v2,     (void*)&HPctx);CHKERRQ(ierr);
  ierr = DMDAVecTraverseIJK(dap,pressure,2,DMDAVecTraverseIJK_HydroStaticPressure_dpdy_v2,(void*)&HPctx);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

static PetscBool InitialTemperatureField(PetscScalar position[], PetscScalar *value, void *ctx)
{
  ModelStokesLawCtx *data = (ModelStokesLawCtx*)ctx;
  PetscInt          d;
  PetscBool         impose = PETSC_TRUE;
  PetscReal         sep;
  PetscFunctionBegin;

  *value = data->T_f;

  sep = 0.0;
  for (d=0; d<3; d++) {
    sep += (position[d] - data->sphere_centre[d])*(position[d] - data->sphere_centre[d]);
  }
  if (sep <= data->r_s*data->r_s) {
    *value = data->T_s;
  }

  PetscFunctionReturn(impose);
}

static PetscErrorCode ModelSetTemperatureInitialSolution_StokesLaw(pTatinCtx ptatin, ModelStokesLawCtx *data)
{
  PhysCompEnergyFV energy;
  PetscErrorCode   ierr;
  PetscFunctionBegin;

  ierr = pTatinGetContext_EnergyFV(ptatin,&energy);CHKERRQ(ierr);
  ierr = FVDAVecTraverse(energy->fv,energy->T,0.0,0,InitialTemperatureField,(void*)data);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyInitialSolution_StokesLaw(pTatinCtx ptatin,Vec X,void *ctx)
{
  ModelStokesLawCtx *data;
  DM                stokes_pack,dau,dap;
  Vec               velocity,pressure;
  PetscBool         active_energy;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  data = (ModelStokesLawCtx*)ctx;
  
  /* Access velocity and pressure vectors */
  stokes_pack = ptatin->stokes_ctx->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dau,&dap);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);
  /* Velocity IC */
  ierr = ModelApplyInitialVelocityField_StokesLaw(ptatin,dau,velocity,data);CHKERRQ(ierr);
  /* Pressure IC */
  ierr = ModelApplyInitialHydrostaticPressureField_StokesLaw(ptatin,dau,dap,pressure,data);CHKERRQ(ierr);
  /* Restore velocity and pressure vectors */
  ierr = DMCompositeRestoreAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);
  /* Temperature IC */
  ierr = pTatinContextValid_EnergyFV(ptatin,&active_energy);CHKERRQ(ierr);
  if (active_energy) {
    ierr = ModelSetTemperatureInitialSolution_StokesLaw(ptatin,data);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode FreeSlipBCs(pTatinCtx ptatin, DM dav, BCList bclist, ModelStokesLawCtx *data)
{
  PetscReal      zero = 0.0;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  /* normal x */
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,0,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,0,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  /* normal y */
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMAX_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  /* normal z */
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMAX_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode FreeSlip_FreeSurface(pTatinCtx ptatin, DM dav, BCList bclist, ModelStokesLawCtx *data)
{
  PetscReal      zero = 0.0;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  /* normal x */
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,0,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,0,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  /* normal y */
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  /* normal z */
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMAX_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyBoundaryConditionsVelocity_StokesLaw(pTatinCtx ptatin, 
                                                                     DM dav, 
                                                                     BCList bclist,
                                                                     ModelStokesLawCtx *data)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  switch (data->bc_type) {
    case 0:
      ierr = FreeSlipBCs(ptatin,dav,bclist,data);CHKERRQ(ierr);
      break;

    case 1:
      ierr = FreeSlip_FreeSurface(ptatin,dav,bclist,data);CHKERRQ(ierr);
      break;

    default:
      ierr = FreeSlipBCs(ptatin,dav,bclist,data);CHKERRQ(ierr);
      break;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyTemperatureBCs_StokesLaw(pTatinCtx ptatin, ModelStokesLawCtx *data)
{
  PhysCompEnergyFV energy;
  PetscReal        val_T;
  PetscErrorCode   ierr;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  ierr = pTatinGetContext_EnergyFV(ptatin,&energy);CHKERRQ(ierr);

  val_T = data->Tbottom;
  ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_S,PETSC_FALSE,0.0,FVDABCMethod_SetDirichlet,(void*)&val_T);CHKERRQ(ierr);
  
  val_T = data->Ttop;
  ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_N,PETSC_FALSE,0.0,FVDABCMethod_SetDirichlet,(void*)&val_T);CHKERRQ(ierr);

  ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_E,PETSC_FALSE,0.0,FVDABCMethod_SetNatural,NULL);CHKERRQ(ierr);
  ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_W,PETSC_FALSE,0.0,FVDABCMethod_SetNatural,NULL);CHKERRQ(ierr);
  ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_F,PETSC_FALSE,0.0,FVDABCMethod_SetNatural,NULL);CHKERRQ(ierr);
  ierr = FVDAFaceIterator(energy->fv,DACELL_FACE_B,PETSC_FALSE,0.0,FVDABCMethod_SetNatural,NULL);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyBoundaryConditions_StokesLaw(pTatinCtx ptatin, void *ctx)
{
  ModelStokesLawCtx *data = (ModelStokesLawCtx*)ctx;
  PhysCompStokes    stokes;
  DM                stokes_pack,dav,dap;
  PetscBool         active_energy;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  /* Define velocity boundary conditions */
  ierr = pTatinGetStokesContext(ptatin,&stokes);CHKERRQ(ierr);
  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);

  ierr = ModelApplyBoundaryConditionsVelocity_StokesLaw(ptatin,dav,stokes->u_bclist,data);CHKERRQ(ierr);

  /* Define boundary conditions for any other physics */
  /* Temperature */
  ierr = pTatinContextValid_EnergyFV(ptatin,&active_energy);CHKERRQ(ierr);
  if (active_energy) {
    ierr = ModelApplyTemperatureBCs_StokesLaw(ptatin,data);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyBoundaryConditionMG_StokesLaw(PetscInt nl,
                                                       BCList bclist[],
                                                       SurfBCList surf_bclist[],
                                                       DM dav[],
                                                       pTatinCtx ptatin,
                                                       void *ctx)
{
  ModelStokesLawCtx *data;
  PetscInt          n;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  data = (ModelStokesLawCtx*)ctx;
  /* Define velocity boundary conditions on each level within the MG hierarchy */
  for (n=0; n<nl; n++) {
    ierr = ModelApplyBoundaryConditionsVelocity_StokesLaw(ptatin,dav[n],bclist[n],data);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeBallVelocity(pTatinCtx ptatin, Vec X, ModelStokesLawCtx *data)
{
  PhysCompStokes    stokes;
  DM                stokes_pack,dau,dap;
  Vec               Uloc,Ploc;
  PetscScalar       *LA_Uloc;
  PetscReal         elu[3*Q2_NODES_PER_EL_3D];
  PetscReal         u_point[3],u_avg[3],u_diff[3];
  PetscReal         Ni[Q2_NODES_PER_EL_3D];
  DataField         PField_std;
  DataBucket        db;
  PetscInt          nel,nen_u,vel_el_lidx[3*U_BASIS_FUNCTIONS];
  const PetscInt    *elnidx_u;
  int               n_mp_points,p,k,d,cnt_ball_point;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;

  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  /* get stokes data structure */
  ierr = pTatinGetStokesContext(ptatin,&stokes);CHKERRQ(ierr);
  stokes_pack = stokes->stokes_pack;
  /* get velocity and pressure */
  ierr = DMCompositeGetEntries(stokes_pack,&dau,&dap);CHKERRQ(ierr);
  ierr = DMCompositeGetLocalVectors(stokes_pack,&Uloc,&Ploc);CHKERRQ(ierr);

  /* get the local (ghosted) entries for each physics */
  ierr = DMCompositeScatter(stokes_pack,X,Uloc,Ploc);CHKERRQ(ierr);
  ierr = VecGetArray(Uloc,&LA_Uloc);CHKERRQ(ierr);

  /* get u element information */
  ierr = DMDAGetElements_pTatinQ2P1(dau,&nel,&nen_u,&elnidx_u);CHKERRQ(ierr);

  ierr = pTatinGetMaterialPoints(ptatin,&db,NULL);CHKERRQ(ierr);
  /* std variables */
  DataBucketGetDataFieldByName(db,MPntStd_classname,&PField_std);
  DataFieldGetAccess(PField_std);
  DataFieldVerifyAccess(PField_std,sizeof(MPntStd));
  
  DataBucketGetSizes(db,&n_mp_points,0,0);

  cnt_ball_point = 0;
  u_avg[0] = u_avg[1] = u_avg[2] = 0.0;
  for (p=0; p<n_mp_points; p++) {
    MPntStd  *mpp_std;
    int      region,eidx;
    double   *xi;

    DataFieldAccessPoint(PField_std,   p,(void**)&mpp_std);   

    /* Access region of the marker */
    MPntStdGetField_phase_index(mpp_std,&region);

    /* move to next point if not in the sphere */
    if (region != 1) { continue; }

    /* Access local coordinates of the marker */
    MPntStdGetField_local_coord(mpp_std,&xi);
    /* Access element index of the marker */
    MPntStdGetField_local_element_index(mpp_std,&eidx);

    /* Get element indices */
    ierr = StokesVelocity_GetElementLocalIndices(vel_el_lidx,(PetscInt*)&elnidx_u[nen_u*eidx]);CHKERRQ(ierr);
    /* Get element velocity */
    ierr = DMDAGetVectorElementFieldQ2_3D(elu,(PetscInt*)&elnidx_u[nen_u*eidx],(PetscScalar*)LA_Uloc);CHKERRQ(ierr);

    pTatin_ConstructNi_Q2_3D( xi, Ni );

    /* interpolate velocity at point */
    u_point[0] = u_point[1] = u_point[2] = 0.0;
    for (k=0; k<Q2_NODES_PER_EL_3D; k++) {
      for (d=0; d<3; d++) {
        u_point[d] += Ni[k] * elu[3*k + d];
      }
    }

    /* Increment */
    for (d=0; d<3; d++) {
      u_avg[d] += u_point[d];
    }
    cnt_ball_point++; 
  }
  PetscPrintf(PETSC_COMM_WORLD,"Points in ball: %d\n",cnt_ball_point);
  /* average */
  for (d=0; d<3; d++) {
    u_avg[d] /= cnt_ball_point;
  }

  /* Compare with the velocity predicted by the analytical solution */
  PetscPrintf(PETSC_COMM_WORLD,"****************** Analytical Solution ******************\n");
  for (d=0; d<3; d++) {
    u_diff[d] = PetscAbsReal(u_avg[d] - data->u_T[d]);
    PetscPrintf(PETSC_COMM_WORLD,"Analytical velocity: u_ana[%d] = %1.4e\n",d,data->u_T[d]);
    PetscPrintf(PETSC_COMM_WORLD,"Numerical average:   u_avg[%d] = %1.4e\n",d,u_avg[d]);
    PetscPrintf(PETSC_COMM_WORLD,"Difference:          u_dif[%d] = %1.4e\n",d,u_diff[d]);
  }
  
  DataFieldRestoreAccess(PField_std);
  ierr = VecRestoreArray(Uloc,&LA_Uloc);CHKERRQ(ierr);
  ierr = DMCompositeRestoreLocalVectors(stokes_pack,&Uloc,&Ploc);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelOutputMarkerFields_StokesLaw(pTatinCtx ptatin,const char prefix[])
{
  DataBucket               materialpoint_db;
  int                      nf;
  const MaterialPointField mp_prop_list[] = { MPField_Std, MPField_Stokes};//, MPField_Energy };
  char                     mp_file_prefix[256];
  PetscErrorCode           ierr;

  PetscFunctionBegin;

  nf = sizeof(mp_prop_list)/sizeof(mp_prop_list[0]);

  ierr = pTatinGetMaterialPoints(ptatin,&materialpoint_db,NULL);CHKERRQ(ierr);
  sprintf(mp_file_prefix,"%s_mpoints",prefix);
  ierr = SwarmViewGeneric_ParaView(materialpoint_db,nf,mp_prop_list,ptatin->outputpath,mp_file_prefix);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelOutputEnergyFV_StokesLaw(pTatinCtx ptatin, const char prefix[], PetscBool been_here, ModelStokesLawCtx *data)
{
  PhysCompEnergyFV energy;
  char             root[PETSC_MAX_PATH_LEN],pvoutputdir[PETSC_MAX_PATH_LEN],fname[PETSC_MAX_PATH_LEN];
  char             pvdfilename[PETSC_MAX_PATH_LEN],vtkfilename[PETSC_MAX_PATH_LEN];
  char             stepprefix[PETSC_MAX_PATH_LEN];
  PetscErrorCode   ierr;

  PetscFunctionBegin;

  ierr = pTatinGetContext_EnergyFV(ptatin,&energy);CHKERRQ(ierr);
  // PVD
  PetscSNPrintf(pvdfilename,PETSC_MAX_PATH_LEN-1,"%s/timeseries_T_fv.pvd",ptatin->outputpath);
  if (prefix) { PetscSNPrintf(vtkfilename, PETSC_MAX_PATH_LEN-1, "%s_T_fv.pvtu",prefix);
  } else {      PetscSNPrintf(vtkfilename, PETSC_MAX_PATH_LEN-1, "T_fv.pvtu");           }
  
  PetscSNPrintf(stepprefix,PETSC_MAX_PATH_LEN-1,"step%D",ptatin->step);
  if (!been_here) { /* new file */
    ierr = ParaviewPVDOpen(pvdfilename);CHKERRQ(ierr);
    ierr = ParaviewPVDAppend(pvdfilename,ptatin->time,vtkfilename,stepprefix);CHKERRQ(ierr);
  } else {
    ierr = ParaviewPVDAppend(pvdfilename,ptatin->time,vtkfilename,stepprefix);CHKERRQ(ierr);
  }
  
  ierr = PetscSNPrintf(root,PETSC_MAX_PATH_LEN-1,"%s",ptatin->outputpath);CHKERRQ(ierr);
  ierr = PetscSNPrintf(pvoutputdir,PETSC_MAX_PATH_LEN-1,"%s/step%D",root,ptatin->step);CHKERRQ(ierr);
  
  /* PetscVec */
  if (data->output_petscvec) {
    ierr = PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s_energy",prefix);CHKERRQ(ierr);
    ierr = FVDAView_JSON(energy->fv,pvoutputdir,fname);CHKERRQ(ierr); /* write meta data abour fv mesh, its DMDA and the coords */
    ierr = FVDAView_Heavy(energy->fv,pvoutputdir,fname);CHKERRQ(ierr);  /* write cell fields */
    ierr = PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s/%s_energy_T",pvoutputdir,prefix);CHKERRQ(ierr);
    ierr = PetscVecWriteJSON(energy->T,0,fname);CHKERRQ(ierr); /* write cell temperature */
  }
  if (data->output_markers) {
    PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s/%s_T_fv",pvoutputdir,prefix);
    ierr = FVDAView_CellData(energy->fv,energy->T,PETSC_TRUE,fname);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelOutputPetscVec_StokesLaw(pTatinCtx ptatin, Vec X, const char prefix[])
{
  const MaterialPointVariable mp_prop_list[] = { MPV_region, MPV_viscosity, MPV_density };
  PetscInt                    nfields;
  PetscErrorCode              ierr;
  PetscFunctionBegin;

  nfields = sizeof(mp_prop_list)/sizeof(MaterialPointVariable);
  /* Output Velocity and pressure */
  ierr = pTatin3d_ModelOutputPetscVec_VelocityPressure_Stokes(ptatin,X,prefix);CHKERRQ(ierr);
  /* Output markers cell fields */
  ierr = pTatin3dModelOutput_MarkerCellFieldsP0_PetscVec(ptatin,PETSC_FALSE,nfields,mp_prop_list,prefix);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode ModelOutput_StokesLaw(pTatinCtx ptatin, Vec X, const char prefix[], void *ctx)
{
  ModelStokesLawCtx *data;
  PetscBool         active_energy;
  PetscErrorCode    ierr;
  static PetscBool  been_here = PETSC_FALSE;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  data = (ModelStokesLawCtx*)ctx;

  /* Get the velocity of the ball and compare with analytical solution */
  ierr = ComputeBallVelocity(ptatin,X,data);CHKERRQ(ierr);
  
  /* Output raw markers and vtu velocity and pressure (for testing and debugging) */
  if (data->output_markers) {
    ierr = pTatin3d_ModelOutput_VelocityPressure_Stokes(ptatin,X,prefix);CHKERRQ(ierr);
    ierr = ModelOutputMarkerFields_StokesLaw(ptatin,prefix);CHKERRQ(ierr);
  }

  if (data->output_petscvec) {
    ierr = ModelOutputPetscVec_StokesLaw(ptatin,X,prefix);CHKERRQ(ierr);
  }

  /* Output temperature (FV) */
  ierr = pTatinContextValid_EnergyFV(ptatin,&active_energy);CHKERRQ(ierr);
  if (active_energy) {
    ierr = ModelOutputEnergyFV_StokesLaw(ptatin,prefix,been_here,data);CHKERRQ(ierr);
  }
  been_here = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode ModelAdaptMaterialPointResolution_StokesLaw(pTatinCtx ptatin, void *ctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);
  PetscPrintf(PETSC_COMM_WORLD,"  NO MARKER INJECTION ON FACES \n", PETSC_FUNCTION_NAME);
  /* Perform injection and cleanup of markers */
  ierr = MaterialPointPopulationControl_v1(ptatin);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode ModelDestroy_StokesLaw(pTatinCtx ptatin, void *ctx)
{
  ModelStokesLawCtx *data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);
  data = (ModelStokesLawCtx*)ctx;

  /* Free contents of structure */
  /* Free structure */
  ierr = PetscFree(data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode pTatinModelRegister_StokesLaw(void)
{
  ModelStokesLawCtx *data;
  pTatinModel       m;
  PetscErrorCode    ierr;

  PetscFunctionBegin;

  /* Allocate memory for the data structure for this model */
  ierr = PetscMalloc(sizeof(ModelStokesLawCtx),&data);CHKERRQ(ierr);
  ierr = PetscMemzero(data,sizeof(ModelStokesLawCtx));CHKERRQ(ierr);

  /* register user model */
  ierr = pTatinModelCreate(&m);CHKERRQ(ierr);

  /* Set name, model select via -ptatin_model NAME */
  ierr = pTatinModelSetName(m,"stokes_law");CHKERRQ(ierr);

  /* Set model data */
  ierr = pTatinModelSetUserData(m,data);CHKERRQ(ierr);

  /* Set function pointers */
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_INIT,                  (void (*)(void))ModelInitialize_StokesLaw);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_MESH_GEOM,  (void (*)(void))ModelApplyInitialMeshGeometry_StokesLaw);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_MAT_GEOM,   (void (*)(void))ModelApplyInitialMaterialParameters_StokesLaw);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_STOKES_VARIABLE_MARKERS,(void (*)(void))ModelApplyInitialStokesVariableMarkers_StokesLaw);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_SOLUTION,   (void (*)(void))ModelApplyInitialSolution_StokesLaw);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_BC,              (void (*)(void))ModelApplyBoundaryConditions_StokesLaw);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_BCMG,            (void (*)(void))ModelApplyBoundaryConditionMG_StokesLaw);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_ADAPT_MP_RESOLUTION,   (void (*)(void))ModelAdaptMaterialPointResolution_StokesLaw);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_UPDATE_MESH_GEOM,(void (*)(void))NULL);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_OUTPUT,                (void (*)(void))ModelOutput_StokesLaw);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_DESTROY,               (void (*)(void))ModelDestroy_StokesLaw);CHKERRQ(ierr);

  /* Insert model into list */
  ierr = pTatinModelRegister(m);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
