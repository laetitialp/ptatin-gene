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
#include "gravity.h"
#include "stokes_output.c"

#include "spherical_sinker.h"

static const char MODEL_NAME_R[] = "model_spherical_sinker_";
PetscLogEvent   PTATIN_MaterialPointPopulationControlRemove;

static PetscErrorCode ModelSetGeometry(ModelSphericalCtx *data)
{
  PetscInt       nn;
  PetscBool      found;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  data->O[0] = M_PI / 3.0;      // Theta  min
  data->O[1] = 6375.0e3-1200e3; // Radius min
  data->O[2] = -M_PI / 6.0;     // Phi    min

  data->L[0] = M_PI / 1.5;      // Theta  max
  data->L[1] = 6375.0e3;        // Radius max
  data->L[2] = M_PI / 6.0;      // Phi    max

  nn = 3;
  ierr = PetscOptionsGetRealArray(NULL,MODEL_NAME_R,"-Origin",data->O,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != 3) {
      SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_USER,"Expected 3 values for -Origin. Found %d",nn);
    }
  }
  nn = 3;
  ierr = PetscOptionsGetRealArray(NULL,MODEL_NAME_R,"-Length",data->L,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != 3) {
      SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_USER,"Expected 3 values for -Length. Found %d",nn);
    }
  }

  /* reports before scaling */
  PetscPrintf(PETSC_COMM_WORLD,"************ Box Geometry ************\n",NULL);
  PetscPrintf(PETSC_COMM_WORLD,"Theta  [min, max] = [ %+1.4e [rad], %+1.4e [rad] ]\n", data->O[0] ,data->L[0] );
  PetscPrintf(PETSC_COMM_WORLD,"Radius [min, max] = [ %+1.4e [m],   %+1.4e [m]   ]\n", data->O[1] ,data->L[1] );
  PetscPrintf(PETSC_COMM_WORLD,"Phi    [min, max] = [ %+1.4e [rad], %+1.4e [rad] ]\n", data->O[2] ,data->L[2] );

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetGravityParameters(ModelSphericalCtx *data)
{
  PetscInt       nn;
  PetscBool      found,gravity_constant,gravity_radial_constant,gravity_none;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  data->gravity_vector[0] = 0.0;
  data->gravity_vector[1] = -9.8;
  data->gravity_vector[2] = 0.0;

  data->gravity_magnitude = -9.8;

  nn = 3;
  ierr = PetscOptionsGetRealArray(NULL,MODEL_NAME_R,"-gravity_vector",data->gravity_vector,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != 3) {
      SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_USER,"Expected 3 values for -gravity_vector. Found %d",nn);
    }
  }
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R,"-gravity_magnitude",&data->gravity_magnitude,NULL);CHKERRQ(ierr);

  gravity_constant        = PETSC_FALSE;
  gravity_radial_constant = PETSC_FALSE;
  gravity_none            = PETSC_FALSE;

  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_R,"-gravity_constant",&gravity_constant,&found);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_R,"-gravity_radial_constant",&gravity_radial_constant,&found);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_R,"-gravity_none",&gravity_none,&found);CHKERRQ(ierr);

  data->gravity_type = -1;
  if (gravity_constant) {
    data->gravity_type = 0;
  } else if (gravity_radial_constant) {
    data->gravity_type = 1;
  } else if (gravity_none) {
    data->gravity_type = 2;
  }
  
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetInclusionParameters(ModelSphericalCtx *data)
{
  PetscReal Ox,Oy,Oz,Lx,Ly,Lz,model_Lx,model_Ly,model_Lz;
  PetscFunctionBegin;

  Ox = data->O[1] * PetscCosReal(data->O[0]);
  Oy = data->O[1] * PetscSinReal(data->O[0]) * PetscCosReal(data->O[2]);
  Oz = data->O[1] * PetscSinReal(data->O[0]) * PetscSinReal(data->O[2]);

  Lx = data->L[1] * PetscCosReal(data->L[0]);
  Ly = data->L[1] * PetscSinReal(data->L[0]) * PetscCosReal(data->L[2]);
  Lz = data->L[1] * PetscSinReal(data->L[0]) * PetscSinReal(data->L[2]);

  model_Lx = Lx - Ox;
  model_Ly = Ly - Oy;
  model_Lz = Lz - Oz;

  data->inclusion_origin[0] = 18.0e5;  //0.2 * model_Lx;
  data->inclusion_origin[1] = 58.0e5;  //0.7 * data->L[1];
  data->inclusion_origin[2] = 0.0;  //0.2 * model_Lz;

  data->inclusion_radius[0] = 3.0e5;//0.2 * model_Ly;
  data->inclusion_radius[1] = 3.0e5;//0.2 * model_Ly;
  data->inclusion_radius[2] = 3.0e5;//0.2 * model_Ly;

  PetscPrintf(PETSC_COMM_WORLD,"************ Inclusion ************\n",NULL);
  PetscPrintf(PETSC_COMM_WORLD,"Origin = [ %+1.4e, %+1.4e, %+1.4e [m] ]\n", data->inclusion_origin[0] ,data->inclusion_origin[1], data->inclusion_origin[2] );
  PetscPrintf(PETSC_COMM_WORLD,"Radius = [ %+1.4e, %+1.4e, %+1.4e [m] ]\n", data->inclusion_radius[0] ,data->inclusion_radius[1], data->inclusion_radius[2] );

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelScaleParameters(ModelSphericalCtx *data)
{
  PetscInt d;
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

  data->O[1] /= data->length_bar;
  data->L[1] /= data->length_bar;

  for (d=0; d<3; d++) {
    data->inclusion_origin[d] /= data->length_bar;
    data->inclusion_radius[d] /= data->length_bar;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ModelInitialize_Spherical(pTatinCtx ptatin, void *ctx)
{
  ModelSphericalCtx *data;
  RheologyConstants *rheology;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  data = (ModelSphericalCtx*)ctx;
  ierr = pTatinGetRheology(ptatin,&rheology);CHKERRQ(ierr);

  rheology->rheology_type  = RHEOLOGY_VISCOUS;
  data->n_phases = 2;
  rheology->nphases_active = data->n_phases;

  ierr = ModelSetGeometry(data);CHKERRQ(ierr);
  ierr = ModelSetGravityParameters(data);CHKERRQ(ierr);
  ierr = ModelSetInclusionParameters(data);CHKERRQ(ierr);
  ierr = ModelScaleParameters(data);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetGravity_None(pTatinCtx ptatin)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = pTatinCreateGravity(ptatin,GRAVITY_NONE);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetGravity_Constant(pTatinCtx ptatin, ModelSphericalCtx *data)
{
  Gravity        gravity;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  ierr = pTatinCreateGravity(ptatin,GRAVITY_CONSTANT);CHKERRQ(ierr);
  ierr = pTatinGetGravityCtx(ptatin,&gravity);CHKERRQ(ierr);
  ierr = GravitySet_Constant(gravity,data->gravity_vector);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetGravity_RadialConstant(pTatinCtx ptatin, ModelSphericalCtx *data)
{
  Gravity        gravity;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  ierr = pTatinCreateGravity(ptatin,GRAVITY_RADIAL_CONSTANT);CHKERRQ(ierr);
  ierr = pTatinGetGravityCtx(ptatin,&gravity);CHKERRQ(ierr);
  ierr = GravitySet_RadialConstant(gravity,data->gravity_magnitude);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetGravity(pTatinCtx ptatin, ModelSphericalCtx *data)
{
  Gravity        gravity;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  switch (data->gravity_type) {
    case 0:
      ierr = ModelSetGravity_Constant(ptatin,data);CHKERRQ(ierr);
      break;

    case 1:
      ierr = ModelSetGravity_RadialConstant(ptatin,data);CHKERRQ(ierr);
      break;

    case 2:
      ierr = ModelSetGravity_None(ptatin);CHKERRQ(ierr);
      break;

    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Unknown gravity type");
      break;
  }
  ierr = pTatinGetGravityCtx(ptatin,&gravity);CHKERRQ(ierr);
  ierr = GravityScale(gravity,1.0/data->acceleration_bar);CHKERRQ(ierr);
  ierr = pTatinQuadratureSetGravity(ptatin);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyGravity_Spherical(pTatinCtx ptatin, void *ctx)
{
  ModelSphericalCtx *data;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);
  
  data = (ModelSphericalCtx*)ctx;
  
  ierr = ModelSetGravity(ptatin,data);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyInitialMeshGeometry_Spherical(pTatinCtx ptatin, void *ctx)
{
  ModelSphericalCtx *data;
  PhysCompStokes    stokes;
  DM                stokes_pack,dav,dap;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);
  data = (ModelSphericalCtx*)ctx;

  /* Get stokes DMs */
  ierr = pTatinGetStokesContext(ptatin,&stokes);CHKERRQ(ierr);
  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);
  /* Mesh */
  ierr = DMDASetUniformSphericalToCartesianCoordinates(dav,data->O[0],data->L[0],data->O[1],data->L[1],data->O[2],data->L[2]);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyInitialMaterialParameters_Spherical(pTatinCtx ptatin, void *ctx)
{
  ModelSphericalCtx *data;
  int               p,d,n_mp_points;
  DataBucket        db;
  DataField         PField_std,PField_stokes;
  PetscReal         eta_0,eta_1,rho_0,rho_1;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);
  data = (ModelSphericalCtx*)ctx;

  rho_0 = 2700.0;
  rho_1 = 5400.0;

  eta_0 = 1.0e+20;
  eta_1 = 1.0e+23;

  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R,"-rho_0",&rho_0,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R,"-rho_1",&rho_1,NULL);CHKERRQ(ierr);

  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R,"-eta_0",&eta_0,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R,"-eta_1",&eta_1,NULL);CHKERRQ(ierr);

  eta_0 /= data->viscosity_bar;
  eta_1 /= data->viscosity_bar;
  rho_0 /= data->density_bar;
  rho_1 /= data->density_bar;

  ierr = pTatinGetMaterialPoints(ptatin,&db,NULL);CHKERRQ(ierr);
  /* std variables */
  DataBucketGetDataFieldByName(db,MPntStd_classname,&PField_std);
  DataFieldGetAccess(PField_std);
  DataFieldVerifyAccess(PField_std,sizeof(MPntStd));
  /* stokes variables */
  DataBucketGetDataFieldByName(db,MPntPStokes_classname,&PField_stokes);
  DataFieldGetAccess(PField_stokes);
  DataFieldVerifyAccess(PField_stokes,sizeof(MPntPStokes));

  DataBucketGetSizes(db,&n_mp_points,0,0);

  for (p=0; p<n_mp_points; p++) {
    MPntStd     *mp_std;
    MPntPStokes *mp_stokes;
    double      *position,viscosity,density;
    double      r[3],dist;
    int         phase; 

    /* Access material point */
    DataFieldAccessPoint(PField_std,p,   (void**)&mp_std);
    DataFieldAccessPoint(PField_stokes,p,(void**)&mp_stokes);

    /* Get coords */
    MPntStdGetField_global_coord(mp_std,&position);

    phase     = 0;
    viscosity = eta_0;
    density   = rho_0;

    dist = 0.0;
    for (d=0; d<3; d++) {
      r[d] = (position[d] - data->inclusion_origin[d]) / (data->inclusion_radius[d]);
      dist += r[d]*r[d];
    }

    if ( dist < 1.0 ) {
      phase = 1;
      viscosity = eta_1;
      density   = rho_1;
    }
    MPntStdSetField_phase_index(mp_std,phase);
    MPntPStokesSetField_eta_effective(mp_stokes,viscosity);
    MPntPStokesSetField_density(mp_stokes,density);
  }
  DataFieldRestoreAccess(PField_std);
  DataFieldRestoreAccess(PField_stokes);
  
  PetscFunctionReturn(0);
}

/* This function is not called in linear driver */
PetscErrorCode ModelApplyInitialStokesVariableMarkers_Spherical(pTatinCtx ptatin, Vec X, void *ctx)
{
  ModelSphericalCtx          *data;
  DM                         stokes_pack,dau,dap;
  PhysCompStokes             stokes;
  Vec                        Uloc,Ploc;
  PetscScalar                *LA_Uloc,*LA_Ploc;
  DataField                  PField;
  PetscErrorCode             ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);
  
  data = (ModelSphericalCtx*)ctx;

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

static PetscErrorCode ModelApplyInitialHydrostaticPressureField(pTatinCtx ptatin, DM dau, DM dap, Vec pressure, ModelSphericalCtx *data)
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
  HPctx.ref_N      = ptatin->stokes_ctx->my-1;
  HPctx.grav       = 9.8 / data->acceleration_bar;
  HPctx.rho        = 2700.0 / data->density_bar;

  ierr = DMDAVecTraverseIJK(dap,pressure,0,DMDAVecTraverseIJK_HydroStaticPressure_v2,     (void*)&HPctx);CHKERRQ(ierr);
  ierr = DMDAVecTraverseIJK(dap,pressure,2,DMDAVecTraverseIJK_HydroStaticPressure_dpdy_v2,(void*)&HPctx);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyInitialSolution_Spherical(pTatinCtx ptatin, Vec X, void *ctx)
{
  ModelSphericalCtx                            *data;
  DM                                           stokes_pack,dau,dap;
  Vec                                          velocity,pressure;
  PetscBool                                    active_energy;
  PetscErrorCode                               ierr;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  data = (ModelSphericalCtx*)ctx;
  
  /* Access velocity and pressure vectors */
  stokes_pack = ptatin->stokes_ctx->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dau,&dap);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);
  /* Velocity IC */
  ierr = VecZeroEntries(velocity);CHKERRQ(ierr);
  /* Pressure IC */
  ierr = ModelApplyInitialHydrostaticPressureField(ptatin,dau,dap,pressure,data);CHKERRQ(ierr);
  /* Restore velocity and pressure vectors */
  ierr = DMCompositeRestoreAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ConstantUdotN_NavierSlip(Facet F,const PetscReal qp_coor[],PetscReal udotn[],void *data)
{
  PetscReal *input = (PetscReal*)data;
  udotn[0] = input[0];
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyNormalNavierSlip(SurfBCList surflist,PetscBool insert_if_not_found,ModelSphericalCtx *data)
{
  SurfaceConstraint sc;
  MeshEntity        facets;
  PetscInt          nsides;
  HexElementFace    sides[] = { HEX_FACE_Nxi, HEX_FACE_Pxi, HEX_FACE_Neta, HEX_FACE_Peta, HEX_FACE_Nzeta, HEX_FACE_Pzeta };
  PetscReal         uD_c[] = {0.0};
  PetscErrorCode    ierr;
  
  ierr = SurfBCListGetConstraint(surflist,"boundary",&sc);CHKERRQ(ierr);
  if (!sc) {
    if (insert_if_not_found) {
      ierr = SurfBCListAddConstraint(surflist,"boundary",&sc);CHKERRQ(ierr);
      ierr = SurfaceConstraintSetType(sc,SC_NITSCHE_NAVIER_SLIP);CHKERRQ(ierr);
      ierr = SurfaceConstraintNitscheNavierSlip_SetPenalty(sc,1.0e3);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Surface constraint not found"); 
    }
  }
  ierr = SurfaceConstraintGetFacets(sc,&facets);CHKERRQ(ierr);
  
  nsides = sizeof(sides) / sizeof(HexElementFace);
  ierr = MeshFacetMarkDomainFaces(facets,sc->fi,nsides,sides);CHKERRQ(ierr);
  
  SURFC_CHKSETVALS(SC_NITSCHE_NAVIER_SLIP,ConstantUdotN_NavierSlip);
  ierr = SurfaceConstraintSetValues(sc,(SurfCSetValuesGeneric)ConstantUdotN_NavierSlip,(void*)uD_c);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyVelocityBoundaryConditions(DM dav, BCList bclist, SurfBCList surflist, PetscBool insert_if_not_found, ModelSphericalCtx *data)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* Navier slip u.n = 0 on faces xmin, xmax, zmin, zmax, ymin */
  ierr = ModelApplyNormalNavierSlip(surflist,insert_if_not_found,data);CHKERRQ(ierr);
  /* Free surface top */
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyPoissonPressureBoundaryConditions(pTatinCtx ptatin)
{
  PDESolveLithoP poisson_pressure;
  PetscReal      zero=0.0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = pTatinGetContext_LithoP(ptatin,&poisson_pressure);CHKERRQ(ierr);
  /* P = 0 at surface */
  ierr = DMDABCListTraverse3d(poisson_pressure->bclist,poisson_pressure->da,DMDABCList_JMAX_LOC,0,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyBoundaryConditions_Spherical(pTatinCtx ptatin, void *ctx)
{
  ModelSphericalCtx *data;
  PhysCompStokes    stokes;
  DM                stokes_pack,dav,dap;
  PetscBool         active_poisson=PETSC_FALSE;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  data = (ModelSphericalCtx*)ctx;

  /* Define velocity boundary conditions */
  ierr = pTatinGetStokesContext(ptatin,&stokes);CHKERRQ(ierr);
  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);
  /* Apply BCs */
  ierr = ModelApplyVelocityBoundaryConditions(dav,stokes->u_bclist,stokes->surf_bclist,PETSC_TRUE,data);CHKERRQ(ierr);

  /* Poisson Pressure */
  ierr = pTatinContextValid_LithoP(ptatin,&active_poisson);CHKERRQ(ierr);
  if (active_poisson) {
    ierr = ModelApplyPoissonPressureBoundaryConditions(ptatin);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyBoundaryConditionMG_Spherical(PetscInt nl,BCList bclist[],SurfBCList surf_bclist[],DM dav[],pTatinCtx ptatin,void *ctx)
{
  ModelSphericalCtx *data;
  PetscInt          n;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  data = (ModelSphericalCtx*)ctx;
  /* Define velocity boundary conditions on each level within the MG hierarchy */
  for (n=0; n<nl; n++) {
    ierr = ModelApplyVelocityBoundaryConditions(dav[n],bclist[n],surf_bclist[n],PETSC_FALSE,data);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ModelAdaptMaterialPointResolution_Spherical(pTatinCtx ptatin, void *ctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);
  PetscPrintf(PETSC_COMM_WORLD,"  NO MARKER INJECTION ON FACES \n", PETSC_FUNCTION_NAME);
  /* Perform injection and cleanup of markers */
  ierr = MaterialPointPopulationControl_v1(ptatin);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelOutputMarkerFields(pTatinCtx ptatin,const char prefix[])
{
  DataBucket               materialpoint_db;
  int                      nf;
  const MaterialPointField mp_prop_list[] = { MPField_Std, MPField_Stokes};
  char                     mp_file_prefix[256];
  PetscErrorCode           ierr;

  PetscFunctionBegin;

  nf = sizeof(mp_prop_list)/sizeof(mp_prop_list[0]);

  ierr = pTatinGetMaterialPoints(ptatin,&materialpoint_db,NULL);CHKERRQ(ierr);
  sprintf(mp_file_prefix,"%s_mpoints",prefix);
  ierr = SwarmViewGeneric_ParaView(materialpoint_db,nf,mp_prop_list,ptatin->outputpath,mp_file_prefix);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode ModelOutput_Spherical(pTatinCtx ptatin,Vec X,const char prefix[],void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);
  /* Output Velocity and pressure */
  ierr = pTatin3d_ModelOutputPetscVec_VelocityPressure_Stokes(ptatin,X,prefix);CHKERRQ(ierr);
  ierr = pTatin3d_ModelOutput_VelocityPressure_Stokes(ptatin,X,prefix);CHKERRQ(ierr);
  /* Output markers */
  ierr = ModelOutputMarkerFields(ptatin,prefix);CHKERRQ(ierr);

  /* Output volume quadrature points stokes fields */
  ierr = VolumeQuadratureViewParaview_Stokes(ptatin->stokes_ctx,ptatin->outputpath,"qp_stokes");CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode ModelDestroy_Spherical(pTatinCtx ptatin, void *ctx)
{
  ModelSphericalCtx *data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);
  data = (ModelSphericalCtx*)ctx;

  /* Free contents of structure */
  /* Free structure */
  ierr = PetscFree(data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode pTatinModelRegister_Spherical(void)
{
  ModelSphericalCtx *data;
  pTatinModel       m;
  PetscErrorCode    ierr;

  PetscFunctionBegin;

  /* Allocate memory for the data structure for this model */
  ierr = PetscMalloc(sizeof(ModelSphericalCtx),&data);CHKERRQ(ierr);
  ierr = PetscMemzero(data,sizeof(ModelSphericalCtx));CHKERRQ(ierr);

  /* register user model */
  ierr = pTatinModelCreate(&m);CHKERRQ(ierr);

  /* Set name, model select via -ptatin_model NAME */
  ierr = pTatinModelSetName(m,"spherical_sinker");CHKERRQ(ierr);

  /* Set model data */
  ierr = pTatinModelSetUserData(m,data);CHKERRQ(ierr);

  /* Set function pointers */
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_INIT,                  (void (*)(void))ModelInitialize_Spherical);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_MESH_GEOM,  (void (*)(void))ModelApplyInitialMeshGeometry_Spherical);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_MAT_GEOM,   (void (*)(void))ModelApplyInitialMaterialParameters_Spherical);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_STOKES_VARIABLE_MARKERS,(void (*)(void))ModelApplyInitialStokesVariableMarkers_Spherical);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_SOLUTION,   (void (*)(void))ModelApplyInitialSolution_Spherical);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_BC,              (void (*)(void))ModelApplyBoundaryConditions_Spherical);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_BCMG,            (void (*)(void))ModelApplyBoundaryConditionMG_Spherical);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_ADAPT_MP_RESOLUTION,   (void (*)(void))ModelAdaptMaterialPointResolution_Spherical);CHKERRQ(ierr);
  //ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_UPDATE_MESH_GEOM,(void (*)(void))ModelApplyUpdateMeshGeometry_Spherical);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_OUTPUT,                (void (*)(void))ModelOutput_Spherical);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_DESTROY,               (void (*)(void))ModelDestroy_Spherical);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_GRAVITY,         (void (*)(void))ModelApplyGravity_Spherical);CHKERRQ(ierr);

  /* Insert model into list */
  ierr = pTatinModelRegister(m);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}