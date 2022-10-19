#include "petsc.h"
#include "ptatin3d.h"
#include "private/ptatin_impl.h"

#include "dmda_bcs.h"
#include "data_bucket.h"
#include "MPntStd_def.h"
#include "MPntPStokes_def.h"
#include "ptatin_std_dirichlet_boundary_conditions.h"
#include "dmda_iterator.h"
#include "mesh_deformation.h"
#include "mesh_update.h"
#include "dmda_remesh.h"
#include "dmda_element_q2p1.h"
#include "material_point_std_utils.h"
#include "material_point_popcontrol.h"
#include "output_material_points.h"
#include "xdmf_writer.h"
#include "output_material_points_p0.h"
#include "private/quadrature_impl.h"
#include "quadrature.h"
#include "QPntSurfCoefStokes_def.h"

#include "mesh_entity.h"
#include "surface_constraint.h"
#include "surfbclist.h"
#include "sctest_ctx.h"

static const char M_NAME[] = "surf_constraint_model_";

static PetscErrorCode Model_SetParameters_SCTest(RheologyConstants *rheology, DataBucket materialconstants, ModelSCTestCtx *data);

static PetscErrorCode InitialVelocityBoundaryValues(ModelSCTestCtx *data)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  data->norm_u = 1.0;
  data->alpha  = 45.0;
  data->theta  = 0.0;
  ierr = PetscOptionsGetReal(NULL,M_NAME,"-norm_u",&data->norm_u,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,M_NAME,"-alpha_u",&data->alpha,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,M_NAME,"-theta_u",&data->theta,NULL);CHKERRQ(ierr);
  data->alpha  = data->alpha * M_PI/180.0;
  data->theta  = data->theta * M_PI/180.0;
  data->uz0    = data->norm_u * cos(data->alpha);
  data->ux0    = sqrt( pow(data->norm_u,2) - pow(data->uz0,2) );

  PetscFunctionReturn(0);
}
#if 1
static inline void RotationMatrix(double theta,double r[3],double R[3][3])
{
  
  R[0][0] = cos(theta) + (1.0-cos(theta))*pow(r[0],2);
  R[0][1] = r[0]*r[1]*(1.0-cos(theta)) - r[2]*sin(theta);
  R[0][2] = r[0]*r[2]*(1.0-cos(theta)) + r[1]*sin(theta);

  R[1][0] = r[0]*r[1]*(1.0-cos(theta)) + r[2]*sin(theta);
  R[1][1] = cos(theta)+(1.0-cos(theta))*pow(r[1],2);
  R[1][2] = r[1]*r[2]*(1.0-cos(theta)) - r[0]*sin(theta);

  R[2][0] = r[0]*r[2]*(1.0-cos(theta))-r[1]*sin(theta);
  R[2][1] = r[1]*r[2]*(1.0-cos(theta))+r[0]*sin(theta);
  R[2][2] = cos(theta)+(1.0-cos(theta))*pow(r[2],2);

}

static PetscErrorCode Rotate_u(PetscReal theta, PetscReal r[], PetscReal u[], PetscReal w[])
{
  PetscInt i,j;
  PetscReal R[3][3];
  PetscFunctionBegin;

  RotationMatrix(theta,r,R);

  w[0] = w[1] = w[2] = 0.0;
  for (i=0;i<3;i++) {
    for (j=0;j<3;j++) {
      w[i] += R[i][j] * u[j];
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode RotateReferential(PetscReal position[], PetscReal coords_rt[] ,ModelSCTestCtx *data)
{
  PetscReal coords[] = {0.0,0.0,0.0};
  PetscReal r[] = {0.0,1.0,0.0};
  PetscReal coords_r[3];
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* Translate to (0,0,0) */
  coords[0] = position[0] - 0.5*(data->Lx + data->Ox);
  coords[1] = position[1] - 0.5*(data->Ly + data->Oy);
  coords[2] = position[2] - 0.5*(data->Lz + data->Oz);

  /* Rotate */
  ierr = Rotate_u(-data->theta,r,coords,coords_r);CHKERRQ(ierr);

  /* Translate back */
  coords_rt[0] = coords_r[0] + 0.5*(data->Lx + data->Ox);
  coords_rt[1] = coords_r[1] + 0.5*(data->Ly + data->Oy);
  coords_rt[2] = coords_r[2] + 0.5*(data->Lz + data->Oz);
  PetscFunctionReturn(0);
}

static PetscErrorCode VelocityAnalyticalFunction(PetscReal position[], PetscReal u[], ModelSCTestCtx *data)
{
  PetscFunctionBegin;
  u[0] = 2.0/(data->Lz - data->Oz) * data->ux0*position[2] - data->ux0;
  u[1] = 0;
  u[2] = 2.0/(data->Lz - data->Oz) * data->uz0*position[2] - data->uz0;
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeNewBasisVectors(ModelSCTestCtx *data)
{
  PetscReal u0[] = {0.0,0.0,0.0};
  PetscReal r[] = {0.0,1.0,0.0};
  PetscErrorCode ierr;

  PetscFunctionBegin;

  u0[0] = data->ux0;
  u0[1] = 0.0;
  u0[2] = data->uz0;

  ierr = Rotate_u(data->theta,r,u0,data->t1_hat);CHKERRQ(ierr);
  ierr = Rotate_u(-0.5*M_PI,r,data->t1_hat,data->n_hat);CHKERRQ(ierr);

  for (int i=0; i<3; i++) {
    //PetscPrintf(PETSC_COMM_WORLD,"n_hat[%d] = %1.4f, t1_hat[%d] = %1.4f \n",i,data->n_hat[i],i,data->t1_hat[i]);
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult3x3(PetscReal A[][3], PetscReal B[][3], PetscReal C[][3])
{
  PetscInt i,j,k;

  for (i=0;i<3;i++) {
    for (j=0;j<3;j++) {
      C[i][j] = 0.0;
    }
  }

  for (i=0;i<3;i++) {
    for (j=0;j<3;j++) {
      for (k=0;k<3;k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeStrainRateBoundaryValue(ModelSCTestCtx *data)
{
  PetscInt       i,j;
  PetscReal      r[] = {0.0,1.0,0.0};
  PetscReal      R[3][3],R_transpose[3][3],E[3][3],ERT[3][3],E_R[3][3];
  const PetscInt indices_voigt[][3] = { {0, 3, 4}, {3, 1, 5}, {4, 5, 2} };
  PetscErrorCode ierr;

  PetscFunctionBegin;

  E[0][0] = 0.0; 
  E[1][1] = 0.0;
  E[2][2] = 2.0 / (data->Lz - data->Oz) * data->uz0;

  E[0][1] = 0.0;
  E[0][2] = 1.0 / (data->Lz - data->Oz) * data->ux0;;
  E[1][2] = 0.0;

  E[1][0] = E[0][1];
  E[2][0] = E[0][2];
  E[2][1] = E[1][2];

  RotationMatrix(data->theta,r,R);

  /* Transpose R */
  for (i=0;i<3;i++) {
    for (j=0;j<3;j++) {
      R_transpose[i][j] = R[j][i];
    }
  }

  /* Compute E*R^T */
  ierr = MatMult3x3(E,R_transpose,ERT);CHKERRQ(ierr);
  /* Compute R*E*R^T */
  ierr = MatMult3x3(R,ERT,E_R);CHKERRQ(ierr);
  for (i=0;i<3;i++) {
    for (j=0;j<3;j++) {
      data->epsilon_s[ indices_voigt[i][j] ] = E_R[i][j];
      //PetscPrintf(PETSC_COMM_WORLD,"E_R[%d][%d] = %+1.2e, E_s[%d] = %+1.2e\n",i,j,E_R[i][j],indices_voigt[i][j],data->epsilon_s[ indices_voigt[i][j] ]);
    }
  }
  
  PetscFunctionReturn(0);
}

PetscBool BCListEvaluator_RotatedVelocityField(PetscScalar position[], PetscScalar *value, void *ctx)
{
  ModelSCTestCtx *data = (ModelSCTestCtx*)ctx;
  PetscReal      coords_rt[] = {0.0,0.0,0.0};
  PetscReal      u_xr[] = {0.0,0.0,0.0};
  PetscReal      r[] = {0.0,1.0,0.0};
  PetscReal      u_R[] = {0.0,0.0,0.0};
  PetscBool      impose=PETSC_TRUE;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = RotateReferential(position,coords_rt,data);CHKERRQ(ierr);
  ierr = VelocityAnalyticalFunction(coords_rt,u_xr,data);CHKERRQ(ierr);
  Rotate_u(data->theta,r,u_xr,u_R);
  *value = u_R[ data->direction_BC ];

  PetscFunctionReturn(impose);
}

static PetscErrorCode ModelSetValues_RotatedExtensionGeneralNavierSlip(ModelSCTestCtx *data)
{
  PetscInt       nn;
  PetscBool      found;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = InitialVelocityBoundaryValues(data);CHKERRQ(ierr);
  ierr = ComputeNewBasisVectors(data);CHKERRQ(ierr); 
  ierr = ComputeStrainRateBoundaryValue(data);CHKERRQ(ierr);

  data->H[0] = 0; // H_00
  data->H[1] = 1; // H_11
  data->H[2] = 0; // H_22
  data->H[3] = 1; // H_01 = H_10
  data->H[4] = 1; // H_02 = H_20
  data->H[5] = 0; // H_12 = H_21
  nn = 6;
  ierr = PetscOptionsGetRealArray(NULL,M_NAME,"-mathcalH",data->H,&nn,&found);CHKERRQ(ierr);
  if (found) {if (nn != 6) { SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_USER,"Wrong number of entries for -mathcalH, expected 6, passed %d",nn); } }

  PetscFunctionReturn(0);
}
#endif
static PetscErrorCode ModelSetValues_ObliqueExtensionGeneralNavierSlip(ModelSCTestCtx *data)
{
  PetscInt       nn;
  PetscBool      found;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = InitialVelocityBoundaryValues(data);

  data->epsilon_s[0] = 0.0;                                     // E_xx
  data->epsilon_s[1] = 0.0;                                     // E_yy
  data->epsilon_s[2] = 2.0 / (data->Lz - data->Oz) * data->uz0; // E_zz
  data->epsilon_s[3] = 0.0;                                     // E_xy
  data->epsilon_s[4] = 1.0 / (data->Lz - data->Oz) * data->ux0; // E_xz
  data->epsilon_s[5] = 0.0;                                     // E_yz

  //PetscPrintf(PETSC_COMM_WORLD,"E_xx = %+1.4e, E_yy = %+1.4e, E_zz = %+1.4e, E_xy = %+1.4e, E_xz = %+1.4e, Eyz = %+1.4e\n",data->epsilon_s[0],data->epsilon_s[1],data->epsilon_s[2],data->epsilon_s[3],data->epsilon_s[4],data->epsilon_s[5]);

  data->t1_hat[0] = data->ux0;
  data->t1_hat[1] = 0.0;
  data->t1_hat[2] = data->uz0;

  data->n_hat[0] = -data->uz0;
  data->n_hat[1] = 0.0;
  data->n_hat[2] = data->ux0;

  data->H[0] = 0; // H_00
  data->H[1] = 1; // H_11
  data->H[2] = 0; // H_22
  data->H[3] = 1; // H_01 = H_10
  data->H[4] = 1; // H_02 = H_20
  data->H[5] = 0; // H_12 = H_21
  nn = 6;
  ierr = PetscOptionsGetRealArray(NULL,M_NAME,"-mathcalH",data->H,&nn,&found);CHKERRQ(ierr);
  if (found) {if (nn != 6) { SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_USER,"Wrong number of entries for -mathcalH, expected 6, passed %d",nn); } }

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelInitialize_SCTest(pTatinCtx c,void *ctx)
{
  ModelSCTestCtx    *data;
  RheologyConstants *rheology;
  DataBucket         materialconstants;
  PetscBool          flg;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  data = (ModelSCTestCtx*)ctx;
  ierr = pTatinGetRheology(c,&rheology);CHKERRQ(ierr);

  /* Number of regions */
  rheology->nphases_active = 3;

  data->Ox = 0.0;
  data->Oy = -250.0e3;
  data->Oz = 0.0;
  data->Lx = 1000.0e3;
  data->Ly = 0.0;
  data->Lz = 600.0e3;

  ierr = PetscOptionsGetReal(NULL,M_NAME,"-Lx",&data->Lx,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,M_NAME,"-Ly",&data->Ly,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,M_NAME,"-Lz",&data->Lz,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,M_NAME,"-Ox",&data->Ox,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,M_NAME,"-Oy",&data->Oy,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,M_NAME,"-Oz",&data->Oz,&flg);CHKERRQ(ierr);

  data->PolarMesh = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,M_NAME,"-PolarMesh",&data->PolarMesh,NULL);CHKERRQ(ierr);

  /* reports before scaling */
  PetscPrintf(PETSC_COMM_WORLD,"********** Box Geometry **********\n",NULL);
  PetscPrintf(PETSC_COMM_WORLD,"[Ox,Lx] = [%+1.4e [m], %+1.4e [m]]\n", data->Ox ,data->Lx );
  PetscPrintf(PETSC_COMM_WORLD,"[Oy,Ly] = [%+1.4e [m], %+1.4e [m]]\n", data->Oy ,data->Ly );
  PetscPrintf(PETSC_COMM_WORLD,"[Oz,Lz] = [%+1.4e [m], %+1.4e [m]]\n", data->Oz ,data->Lz );

  /* Layering */
  data->layer1 = -40.0e3;
  data->layer2 = -100.0e3;
  ierr = PetscOptionsGetReal(NULL,M_NAME,"-layer1",&data->layer1,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,M_NAME,"-layer2",&data->layer2,&flg);CHKERRQ(ierr);

  /* Material constants */
  ierr = pTatinGetMaterialConstants(c,&materialconstants);CHKERRQ(ierr);
  ierr = MaterialConstantsSetDefaults(materialconstants);CHKERRQ(ierr);

  ierr = Model_SetParameters_SCTest(rheology,materialconstants,data);CHKERRQ(ierr);
  //ierr = ModelSetValues_ObliqueExtensionGeneralNavierSlip(data);CHKERRQ(ierr);
  ierr = ModelSetValues_RotatedExtensionGeneralNavierSlip(data);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode Model_SetRheology_VISCOUS(RheologyConstants *rheology, DataBucket materialconstants, ModelSCTestCtx *data)
{
  PetscInt       region_idx;
  PetscReal      density,viscosity;
  char           *option_name;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  rheology->rheology_type = RHEOLOGY_VP_STD;

  for (region_idx=0; region_idx<rheology->nphases_active;region_idx++) {
    MaterialConstantsSetValues_MaterialType(materialconstants,region_idx,VISCOUS_CONSTANT,PLASTIC_NONE,SOFTENING_NONE,DENSITY_CONSTANT);
    /* Set region viscosity */
    viscosity = 1.0e+23;
    if (asprintf (&option_name, "-eta0_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,M_NAME, option_name,&viscosity,NULL);CHKERRQ(ierr); 
    free (option_name);
    ierr = MaterialConstantsSetValues_ViscosityConst(materialconstants,region_idx,viscosity);CHKERRQ(ierr);
    
    /* Set region density */
    density = 2700.0;
    if (asprintf (&option_name, "-rho0_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,M_NAME, option_name,&density,NULL);CHKERRQ(ierr);
    free (option_name);
    ierr = MaterialConstantsSetValues_DensityConst(materialconstants,region_idx,density);CHKERRQ(ierr);
  }

  for (region_idx=0; region_idx<rheology->nphases_active;region_idx++) {
    MaterialConstantsPrintAll(materialconstants,region_idx);
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode Model_SetParameters_SCTest(RheologyConstants *rheology, DataBucket materialconstants, ModelSCTestCtx *data)
{
  PetscInt       region_idx;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* scaling */
  data->length_bar       = 1.0e5;
  data->viscosity_bar    = 1.0e22;
  data->velocity_bar     = 1.0e-10;
  data->time_bar         = data->length_bar / data->velocity_bar;
  data->pressure_bar     = data->viscosity_bar/data->time_bar;
  data->density_bar      = data->pressure_bar * (data->time_bar*data->time_bar)/(data->length_bar*data->length_bar); // kg.m^-3
  data->acceleration_bar = data->length_bar / (data->time_bar*data->time_bar);

  PetscPrintf(PETSC_COMM_WORLD,"[surface_constraint_test]:  during the solve scaling will be done using \n");
  PetscPrintf(PETSC_COMM_WORLD,"  L*    : %1.4e [m]\n", data->length_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  U*    : %1.4e [m.s^-1]\n", data->velocity_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  t*    : %1.4e [s]\n", data->time_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  eta*  : %1.4e [Pa.s]\n", data->viscosity_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  rho*  : %1.4e [kg.m^-3]\n", data->density_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  P*    : %1.4e [Pa]\n", data->pressure_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  a*    : %1.4e [m.s^-2]\n", data->acceleration_bar );

  /* Scale viscosity cutoff */
  rheology->eta_lower_cutoff_global = rheology->eta_lower_cutoff_global / data->viscosity_bar;
  rheology->eta_upper_cutoff_global = rheology->eta_upper_cutoff_global / data->viscosity_bar;
  /* Scale length */
  data->Lx = data->Lx / data->length_bar;
  data->Ly = data->Ly / data->length_bar;
  if (data->PolarMesh) {
    data->Lz = data->Lz * M_PI/180.0;
    data->Oz = data->Oz * M_PI/180.0;
  } else {
    data->Lz = data->Lz / data->length_bar;
    data->Oz = data->Oz / data->length_bar;
  }
  data->Ox = data->Ox / data->length_bar;
  data->Oy = data->Oy / data->length_bar;
  data->layer1 = data->layer1 / data->length_bar;
  data->layer2 = data->layer2 / data->length_bar;

  ierr = Model_SetRheology_VISCOUS(rheology,materialconstants,data);CHKERRQ(ierr);
  for (region_idx=0; region_idx<rheology->nphases_active;region_idx++) {
    MaterialConstantsScaleAll(materialconstants,region_idx,data->length_bar,data->velocity_bar,data->time_bar,data->viscosity_bar,data->density_bar,data->pressure_bar);
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelCreatePolarMeshGeometry(DM da)
{
  DM               cda;
  Vec              coord;
  PetscScalar      *LA_coords;
  PetscInt         i,j,k,M,N,P,si,sj,sk,nx,ny,nz;
  PetscErrorCode   ierr;

  PetscFunctionBegin;

  /* Get DM info */
  ierr = DMDAGetInfo(da,0,&M,&N,&P,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&si,&sj,&sk,&nx,&ny,&nz);CHKERRQ(ierr);
  /* Get DM coords */
  ierr = DMGetCoordinateDM(da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinates(da,&coord);CHKERRQ(ierr);
  ierr = VecGetArray(coord,&LA_coords);CHKERRQ(ierr);

  for (k=0;k<nz;k++) {
    for (j=0;j<ny;j++) {
      for (i=0;i<nx;i++) {
        PetscInt    nidx;
        PetscScalar r,phi;

        nidx = i + j*nx + k*nx*ny;
        r   = LA_coords[3*nidx + 0];
        phi = LA_coords[3*nidx + 2];

        LA_coords[3*nidx + 0] = r*cos(phi); 
        LA_coords[3*nidx + 2] = r*sin(phi);
      }
    }
  }
  ierr = VecRestoreArray(coord,&LA_coords);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyInitialMeshGeometry_SCTest(pTatinCtx c,void *ctx)
{
  ModelSCTestCtx   *data = (ModelSCTestCtx*)ctx;
  PhysCompStokes    stokes;
  DM                stokes_pack,dav,dap;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  ierr = pTatinGetStokesContext(c,&stokes);CHKERRQ(ierr);
  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(dav,data->Ox,data->Lx,data->Oy,data->Ly,data->Oz,data->Lz);CHKERRQ(ierr);
  if (data->PolarMesh) {
    ierr = ModelCreatePolarMeshGeometry(dav);CHKERRQ(ierr);
  }

  ierr = DMDABilinearizeQ2Elements(dav);CHKERRQ(ierr);
  
  PetscReal gvec[] = { 0.0, -9.8, 0.0 };
  ierr = PhysCompStokesSetGravityVector(c->stokes_ctx,gvec);CHKERRQ(ierr);
  ierr = PhysCompStokesScaleGravityVector(c->stokes_ctx,1.0/data->acceleration_bar);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyInitialMaterialGeometry_VISCOUS(pTatinCtx c,void *ctx)
{
  ModelSCTestCtx *data = (ModelSCTestCtx*)ctx;
  DataBucket      db;
  DataField       PField_std;
  PetscInt        p;
  int             n_mp_points;
  
  PetscFunctionBegin;
  
  /* define properties on material points */
  db = c->materialpoint_db;
  DataBucketGetDataFieldByName(db,MPntStd_classname,&PField_std);
  DataFieldGetAccess(PField_std);
  DataFieldVerifyAccess(PField_std,sizeof(MPntStd));

  DataBucketGetSizes(db,&n_mp_points,0,0);
  for (p=0; p<n_mp_points; p++) {
    MPntStd       *material_point;
    int           region_idx;
    double        *position;

    DataFieldAccessPoint(PField_std,p,(void**)&material_point);

    /* Access using the getter function */
    MPntStdGetField_global_coord(material_point,&position);

    region_idx = 0;
    if (position[1] < data->layer1) { region_idx = 1; }
    if (position[1] < data->layer2) { region_idx = 2; }
        
    MPntStdSetField_phase_index(material_point,region_idx);
  }
  DataFieldRestoreAccess(PField_std);
  
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyInitialMaterialGeometry_SCTest(pTatinCtx c,void *ctx)
{
  ModelSCTestCtx *data = (ModelSCTestCtx*)ctx;
  PetscErrorCode  ierr;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  
  ierr = ModelApplyInitialMaterialGeometry_VISCOUS(c,data);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyInitialStokesVariableMarkers_SCTest(pTatinCtx user,Vec X,void *ctx)
{
  DM                         stokes_pack,dau,dap;
  PhysCompStokes             stokes;
  Vec                        Uloc,Ploc;
  PetscScalar                *LA_Uloc,*LA_Ploc;
  DataField                  PField;
  MaterialConst_MaterialType *truc;
  PetscInt                   regionidx;
  PetscErrorCode             ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  DataBucketGetDataFieldByName(user->material_constants,MaterialConst_MaterialType_classname,&PField);

  ierr = pTatinGetStokesContext(user,&stokes);CHKERRQ(ierr);
  stokes_pack = stokes->stokes_pack;

  ierr = DMCompositeGetEntries(stokes_pack,&dau,&dap);CHKERRQ(ierr);
  ierr = DMCompositeGetLocalVectors(stokes_pack,&Uloc,&Ploc);CHKERRQ(ierr);

  ierr = DMCompositeScatter(stokes_pack,X,Uloc,Ploc);CHKERRQ(ierr);
  ierr = VecGetArray(Uloc,&LA_Uloc);CHKERRQ(ierr);
  ierr = VecGetArray(Ploc,&LA_Ploc);CHKERRQ(ierr);

  ierr = pTatin_EvaluateRheologyNonlinearities(user,dau,LA_Uloc,dap,LA_Ploc);CHKERRQ(ierr);

  ierr = VecRestoreArray(Uloc,&LA_Uloc);CHKERRQ(ierr);
  ierr = VecRestoreArray(Ploc,&LA_Ploc);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelOutput_SCTest(pTatinCtx c,Vec X,const char prefix[],void *ctx)
{
  char           pvoutputdir[PETSC_MAX_PATH_LEN];
  DataBucket     materialpoint_db;
  const int      nf = 3;
  const          MaterialPointField mp_prop_list[] = { MPField_Std, MPField_Stokes, MPField_StokesPl};//, MPField_Energy };
  char           mp_file_prefix[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  ierr = PetscSNPrintf(pvoutputdir,PETSC_MAX_PATH_LEN-1,"%s/step%D",c->outputpath,c->step);CHKERRQ(ierr);
  SurfaceQuadratureViewParaview_Stokes(c->stokes_ctx,pvoutputdir,"surfq");

  ierr = pTatin3d_ModelOutput_VelocityPressure_Stokes(c,X,prefix);CHKERRQ(ierr);
  // testing
  //ierr = pTatin3d_ModelOutputLite_Velocity_Stokes(c,X,prefix);CHKERRQ(ierr);
  //ierr = pTatinOutputLiteMeshVelocitySlicedPVTS(c->stokes_ctx->stokes_pack,c->outputpath,prefix);CHKERRQ(ierr);
  //ierr = ptatin3d_StokesOutput_VelocityXDMF(c,X,prefix);CHKERRQ(ierr);
  // testing
  //ierr = pTatin3d_ModelOutputPetscVec_VelocityPressure_Stokes(c,X,prefix);CHKERRQ(ierr);
  //ierr = pTatin3d_ModelOutput_MPntStd(c,prefix);CHKERRQ(ierr);
  
  ierr = pTatinGetMaterialPoints(c,&materialpoint_db,NULL);CHKERRQ(ierr);
  sprintf(mp_file_prefix,"%s_mpoints",prefix);
  ierr = SwarmViewGeneric_ParaView(materialpoint_db,nf,mp_prop_list,c->outputpath,mp_file_prefix);CHKERRQ(ierr);
  /*
  {
    PhysCompStokes    stokes;
    SurfaceConstraint sc;
    ierr = pTatinGetStokesContext(c,&stokes);CHKERRQ(ierr);
    ierr = SurfBCListGetConstraint(stokes->surf_bclist,"boundary",&sc);CHKERRQ(ierr);
    ierr = SurfaceConstraintViewParaview(sc, pvoutputdir, "boundary");CHKERRQ(ierr);
    ierr = SurfBCListGetConstraint(stokes->surf_bclist,"bc_traction",&sc);CHKERRQ(ierr);
    ierr = SurfaceConstraintViewParaview(sc, pvoutputdir, "bc_traction");CHKERRQ(ierr); 
  }
  */
  // tests for alternate (output/load)ing of "single file marker" formats
  //ierr = SwarmDataWriteToPetscVec(c->materialpoint_db,prefix);CHKERRQ(ierr);
  //ierr = SwarmDataLoadFromPetscVec(c->materialpoint_db,prefix);CHKERRQ(ierr);
  /*
     {
     MaterialPointVariable vars[] = { MPV_viscosity, MPV_density, MPV_region };
  //ierr = pTatin3dModelOutput_MarkerCellFieldsP0_ParaView(c,sizeof(vars)/sizeof(MaterialPointVariable),vars,PETSC_TRUE,prefix);CHKERRQ(ierr);
  ierr = pTatin3dModelOutput_MarkerCellFieldsP0_PetscVec(c,PETSC_TRUE,sizeof(vars)/sizeof(MaterialPointVariable),vars,prefix);CHKERRQ(ierr);
  }
  */
  PetscFunctionReturn(0);
}

static PetscErrorCode const_udotn(Facet F,const PetscReal qp_coor[],PetscReal udotn[],void *data)
{
  PetscReal *input = (PetscReal*)data;
  udotn[0] = input[0];
  PetscFunctionReturn(0);
}

static PetscErrorCode const_ud(Facet F,const PetscReal qp_coor[],PetscReal uD[],void *data)
{
  PetscReal *input = (PetscReal*)data;
  uD[0] = input[0]*F->centroid_normal[0];
  uD[1] = input[1]*F->centroid_normal[1];
  uD[2] = input[2]*F->centroid_normal[2];
  PetscFunctionReturn(0);
}

static PetscErrorCode angular_udotn(Facet F,const PetscReal qp_coor[],PetscReal udotn[],void *data)
{
  PetscReal *input = (PetscReal*)data;
  /* Use relation v = \omega * r with \omega the angular velocity 
  and r the distance from the pole, here the x coord */
  udotn[0] = input[0]*qp_coor[0];
  PetscFunctionReturn(0);
}

static PetscErrorCode const_traction(Facet F,const PetscReal qp_coor[],PetscReal traction[],void *data)
{
  PetscReal *input = (PetscReal*)data;
  traction[0] = input[0];
  traction[1] = input[1];
  traction[2] = input[2];
  PetscFunctionReturn(0);
}

static PetscErrorCode general_navier_slip(Facet F,const PetscReal qp_coor[],
                                          PetscReal n_hat[],
                                          PetscReal t1_hat[],
                                          PetscReal tauS[],
                                          PetscReal H[],
                                          void *data)
{
  ModelSCTestCtx *model_data = (ModelSCTestCtx*)data;
  PetscInt i,j;

  for (i=0;i<5;i++) {
    /* WARNING MINUS SIGN HERE TO CHECK THE POTENTIAL MISTAKE IN FORMS */
    tauS[i] = - 2.0 * 10.0 * model_data->epsilon_s[i]; // eta hard coded as 10.0 here ==> 1e+23 Pa.s
    H[i] = model_data->H[i];
  }
  
  for (j=0;j<3;j++) {
    t1_hat[j] = model_data->t1_hat[j];
    n_hat[j] = model_data->n_hat[j];
  }

  PetscFunctionReturn(0);
}

PetscBool MarkFacetsHalfFace(Facet facet,void *data)
{
  PetscBool selected_facet = PETSC_FALSE;

  if (facet->centroid[1] > -1.0 && facet->label != HEX_FACE_Peta){
    selected_facet = PETSC_TRUE;
  }

  return selected_facet;
}

static PetscErrorCode BCTypeSlipNitscheFreeSurface(SurfBCList surflist,PetscBool insert_if_not_found)
{
  SurfaceConstraint sc;
  MeshEntity        facets;
  PetscErrorCode    ierr;
  
  
  ierr = SurfBCListGetConstraint(surflist,"boundary",&sc);CHKERRQ(ierr);
  if (!sc) {
    if (insert_if_not_found) {
      ierr = SurfBCListAddConstraint(surflist,"boundary",&sc);CHKERRQ(ierr);
      ierr = SurfaceConstraintSetType(sc,SC_NITSCHE_NAVIER_SLIP);CHKERRQ(ierr);
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Surface constraint not found");
  }
  ierr = SurfaceConstraintGetFacets(sc,&facets);CHKERRQ(ierr);
  
  {
    PetscInt       nsides;
    HexElementFace sides[] = { HEX_FACE_Nxi, HEX_FACE_Pxi, HEX_FACE_Neta, HEX_FACE_Nzeta, HEX_FACE_Pzeta };
    nsides = sizeof(sides) / sizeof(HexElementFace);
    ierr = MeshFacetMarkDomainFaces(facets,sc->fi,nsides,sides);CHKERRQ(ierr);
  }
  
  SURFC_CHKSETVALS(SC_NITSCHE_NAVIER_SLIP,const_udotn);
  {
    PetscReal uD_c[] = {0.0};
    ierr = SurfaceConstraintSetValues(sc,(SurfCSetValuesGeneric)const_udotn,(void*)uD_c);CHKERRQ(ierr);
  }

  ierr = SurfBCListGetConstraint(surflist,"bc_traction",&sc);CHKERRQ(ierr);
  if (!sc) {
    if (insert_if_not_found) {
      ierr = SurfBCListAddConstraint(surflist,"bc_traction",&sc);CHKERRQ(ierr);
      ierr = SurfaceConstraintSetType(sc,SC_TRACTION);CHKERRQ(ierr);
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Surface constraint not found");
  }
  ierr = SurfaceConstraintGetFacets(sc,&facets);CHKERRQ(ierr);
  
  /*
  {
    PetscInt       nsides;
    HexElementFace sides[] = { HEX_FACE_Nxi, HEX_FACE_Pxi, HEX_FACE_Nzeta, HEX_FACE_Pzeta };
    nsides = sizeof(sides) / sizeof(HexElementFace);
    ierr = MeshFacetMarkDomainFaces(facets,sc->fi,nsides,sides);CHKERRQ(ierr);
  }
  */

  {
    ierr = MeshFacetMark(facets,sc->fi,MarkFacetsHalfFace,NULL);CHKERRQ(ierr);
  }
  
  SURFC_CHKSETVALS(SC_TRACTION,const_traction);
  {
    PetscReal traction[] = {0.0, -20.0, 0.0};
    ierr = SurfaceConstraintSetValues(sc,(SurfCSetValuesGeneric)const_traction,(void*)traction);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ApplyNitscheFreeslip(SurfBCList surflist,PetscBool insert_if_not_found)
{
  SurfaceConstraint sc;
  MeshEntity        facets;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = SurfBCListGetConstraint(surflist,"NitscheFreeSlip",&sc);CHKERRQ(ierr);
  if (!sc) {
    if (insert_if_not_found) {
      ierr = SurfBCListAddConstraint(surflist,"NitscheFreeSlip",&sc);CHKERRQ(ierr);
      ierr = SurfaceConstraintSetType(sc,SC_NITSCHE_NAVIER_SLIP);CHKERRQ(ierr);
      ierr = SurfaceConstraintNitscheNavierSlip_SetPenalty(sc,1.0e5);CHKERRQ(ierr);
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Surface constraint not found");
  }
  ierr = SurfaceConstraintGetFacets(sc,&facets);CHKERRQ(ierr);

  {
    PetscInt       nsides;
    HexElementFace sides[] = { HEX_FACE_Nxi, HEX_FACE_Pxi };
    nsides = sizeof(sides) / sizeof(HexElementFace);
    ierr = MeshFacetMarkDomainFaces(facets,sc->fi,nsides,sides);CHKERRQ(ierr);
  }
  SURFC_CHKSETVALS(SC_NITSCHE_NAVIER_SLIP,const_udotn);
  {
    PetscReal uD_c[] = {0.0};
    ierr = SurfaceConstraintSetValues(sc,(SurfCSetValuesGeneric)const_udotn,(void*)uD_c);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ApplyNitsche_AngularVelocityNormal(SurfBCList surflist,PetscBool insert_if_not_found,ModelSCTestCtx *data)
{
  SurfaceConstraint sc;
  MeshEntity        facets;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = SurfBCListGetConstraint(surflist,"NitscheAngularZmax",&sc);CHKERRQ(ierr);
  if (!sc) {
    if (insert_if_not_found) {
      ierr = SurfBCListAddConstraint(surflist,"NitscheAngularZmax",&sc);CHKERRQ(ierr);
      ierr = SurfaceConstraintSetType(sc,SC_NITSCHE_NAVIER_SLIP);CHKERRQ(ierr);
      ierr = SurfaceConstraintNitscheNavierSlip_SetPenalty(sc,1.0e5);CHKERRQ(ierr);
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Surface constraint not found");
  }
  ierr = SurfaceConstraintGetFacets(sc,&facets);CHKERRQ(ierr);
  {
    PetscInt       nsides;
    HexElementFace sides[] = { HEX_FACE_Pzeta, HEX_FACE_Nzeta };
    nsides = sizeof(sides) / sizeof(HexElementFace);
    ierr = MeshFacetMarkDomainFaces(facets,sc->fi,nsides,sides);CHKERRQ(ierr);
  }
  SURFC_CHKSETVALS(SC_NITSCHE_NAVIER_SLIP,angular_udotn);
  {
    PetscReal uD_c[] = {0.0};
    uD_c[0] = 1.0; //* (M_PI/180.0) / (3600.0*24.0*365.0*1.0e6 / data->time_bar);
    ierr = SurfaceConstraintSetValues(sc,(SurfCSetValuesGeneric)const_udotn,(void*)uD_c);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ApplyWeakDirichket(SurfBCList surflist,PetscBool insert_if_not_found,ModelSCTestCtx *data)
{
  SurfaceConstraint sc;
  MeshEntity        facets;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = SurfBCListGetConstraint(surflist,"NitscheDirichlet",&sc);CHKERRQ(ierr);
  if (!sc) {
    if (insert_if_not_found) {
      ierr = SurfBCListAddConstraint(surflist,"NitscheDirichlet",&sc);CHKERRQ(ierr);
      ierr = SurfaceConstraintSetType(sc,SC_NITSCHE_DIRICHLET);CHKERRQ(ierr);
      ierr = SurfaceConstraintNitscheDirichlet_SetPenalty(sc,1.0e3);CHKERRQ(ierr);
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Surface constraint not found");
  }
  ierr = SurfaceConstraintGetFacets(sc,&facets);CHKERRQ(ierr);
  {
    PetscInt       nsides;
    HexElementFace sides[] = { HEX_FACE_Pzeta, HEX_FACE_Nzeta };
    nsides = sizeof(sides) / sizeof(HexElementFace);
    ierr = MeshFacetMarkDomainFaces(facets,sc->fi,nsides,sides);CHKERRQ(ierr);
  }
  SURFC_CHKSETVALS(SC_NITSCHE_DIRICHLET,const_ud);
  {
    PetscReal uD_c[] = {0.0, 0.0, 1.0};
    ierr = SurfaceConstraintSetValues(sc,(SurfCSetValuesGeneric)const_ud,(void*)uD_c);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ApplyStrongDirichlet(DM dav, BCList bclist)
{
  PetscErrorCode ierr;
  PetscScalar zero = 0.0, one=1.0, mone=-1.0;
  
  //ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,0,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  //ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,0,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  /*
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,0,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,2,BCListEvaluator_constant,(void*)&mone);CHKERRQ(ierr);
  
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMAX_LOC,0,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMAX_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMAX_LOC,2,BCListEvaluator_constant,(void*)&one);CHKERRQ(ierr);
  */
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ApplyGeneralNavierSlip(SurfBCList surflist,PetscBool insert_if_not_found,ModelSCTestCtx *data)
{
  SurfaceConstraint sc;
  MeshEntity        facets;
  PetscErrorCode    ierr;

  PetscFunctionBegin;

  ierr = SurfBCListGetConstraint(surflist,"boundary",&sc);CHKERRQ(ierr);
  if (!sc) {
    if (insert_if_not_found) {
      ierr = SurfBCListAddConstraint(surflist,"boundary",&sc);CHKERRQ(ierr);
      ierr = SurfaceConstraintSetType(sc,SC_NITSCHE_GENERAL_SLIP);CHKERRQ(ierr);
      ierr = SurfaceConstraintNitscheGeneralSlip_SetPenalty(sc,1.0e3);CHKERRQ(ierr);
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Surface constraint not found");
  }
  ierr = SurfaceConstraintGetFacets(sc,&facets);CHKERRQ(ierr);
  
  {
    PetscInt       nsides;
    HexElementFace sides[] = {HEX_FACE_Nxi, HEX_FACE_Pxi}; //{ HEX_FACE_Nzeta, HEX_FACE_Pzeta };
    nsides = sizeof(sides) / sizeof(HexElementFace);
    ierr = MeshFacetMarkDomainFaces(facets,sc->fi,nsides,sides);CHKERRQ(ierr);
  }

  SURFC_CHKSETVALS(SC_NITSCHE_GENERAL_SLIP,general_navier_slip);
  {
    ierr = SurfaceConstraintSetValues(sc,(SurfCSetValuesGeneric)general_navier_slip,(void*)data);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyObliqueExtensionPullApart_Nitsche(DM dav, BCList bclist,SurfBCList surflist,PetscBool insert_if_not_found,ModelSCTestCtx *data)
{
  PetscReal      ux,uz,u_bot,Szy,Sxz;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* Apply Oblique extension on IMAX and IMIN faces */
  ux = -data->ux0;
  uz = -data->uz0;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,0,BCListEvaluator_constant,(void*)&ux);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,2,BCListEvaluator_constant,(void*)&uz);CHKERRQ(ierr);

  ux = data->ux0;
  uz = data->uz0;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMAX_LOC,0,BCListEvaluator_constant,(void*)&ux);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMAX_LOC,2,BCListEvaluator_constant,(void*)&uz);CHKERRQ(ierr);

  /* Apply General Navier Slip BC on IMAX and IMIN faces */
  ierr = ApplyGeneralNavierSlip(surflist,insert_if_not_found,data);CHKERRQ(ierr);

  /* Apply inflow base */
  Szy = (data->Lz - data->Oz)*(data->Ly - data->Oy);
  Sxz = (data->Lx - data->Ox)*(data->Lz - data->Oz);
  u_bot = (-4.0/M_PI * data->alpha + 2.0) *  data->norm_u * Szy/Sxz;
  PetscPrintf(PETSC_COMM_WORLD,"u_bot = %1.4e \n",u_bot);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,1,BCListEvaluator_constant,(void*)&u_bot);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyRotatedExtension_Nitsche(DM dav, BCList bclist,SurfBCList surflist,PetscBool insert_if_not_found,ModelSCTestCtx *data)
{
  PetscReal      ux,uz,u_bot,Szy,Sxz;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* Apply Oblique extension on IMAX and IMIN faces */
  data->direction_BC = 0;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,0,BCListEvaluator_RotatedVelocityField,(void*)data);CHKERRQ(ierr);
  data->direction_BC = 2;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,2,BCListEvaluator_RotatedVelocityField,(void*)data);CHKERRQ(ierr);

  data->direction_BC = 0;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMAX_LOC,0,BCListEvaluator_RotatedVelocityField,(void*)data);CHKERRQ(ierr);
  data->direction_BC = 2;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMAX_LOC,2,BCListEvaluator_RotatedVelocityField,(void*)data);CHKERRQ(ierr);

  /* Apply General Navier Slip BC on IMAX and IMIN faces */
  ierr = ApplyGeneralNavierSlip(surflist,insert_if_not_found,data);CHKERRQ(ierr);

  /* Apply inflow base */
  Szy = (data->Lz - data->Oz)*(data->Ly - data->Oy);
  Sxz = (data->Lx - data->Ox)*(data->Lz - data->Oz);
  u_bot = (-4.0/M_PI * data->alpha + 2.0) *  data->norm_u * Szy/Sxz;
  PetscPrintf(PETSC_COMM_WORLD,"u_bot = %1.4e \n",u_bot);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,1,BCListEvaluator_constant,(void*)&u_bot);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyBoundaryCondition_SCTest(pTatinCtx user,void *ctx)
{
  ModelSCTestCtx *data = (ModelSCTestCtx*)ctx;
  PhysCompStokes stokes;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  ierr = pTatinGetStokesContext(user,&stokes);CHKERRQ(ierr);
  //ierr = ModelApplyObliqueExtensionPullApart_Nitsche(stokes->dav,stokes->u_bclist,stokes->surf_bclist,PETSC_TRUE,data);CHKERRQ(ierr);
  ierr = ModelApplyRotatedExtension_Nitsche(stokes->dav,stokes->u_bclist,stokes->surf_bclist,PETSC_TRUE,data);CHKERRQ(ierr);
  //if (data->PolarMesh) {
    //ierr = ApplyNitscheFreeslip(stokes->surf_bclist,PETSC_TRUE);CHKERRQ(ierr);
    //ierr = ApplyNitsche_AngularVelocityNormal(stokes->surf_bclist,PETSC_TRUE,data);CHKERRQ(ierr);
    //ierr = ApplyWeakDirichket(stokes->surf_bclist,PETSC_TRUE,data);CHKERRQ(ierr);
    //ierr = ApplyStrongDirichlet(stokes->dav,stokes->u_bclist);CHKERRQ(ierr);
  //} else {
  //  ierr = BCTypeSlipNitscheFreeSurface(stokes->surf_bclist,PETSC_TRUE);CHKERRQ(ierr);
  //}

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyBoundaryConditionMG(PetscInt nl,BCList bclist[],SurfBCList surf_bclist[],DM dav[],pTatinCtx user,void *ctx)
{
  ModelSCTestCtx *data = (ModelSCTestCtx*)ctx;
  PetscInt       n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);

  for (n=0; n<nl; n++) {
    //ierr = ModelApplyObliqueExtensionPullApart_Nitsche(dav[n],bclist[n],surf_bclist[n],PETSC_FALSE,data);CHKERRQ(ierr);
    ierr = ModelApplyRotatedExtension_Nitsche(dav[n],bclist[n],surf_bclist[n],PETSC_FALSE,data);CHKERRQ(ierr);
  }
#if 0
  //if (data->PolarMesh) {
    for (n=0; n<nl; n++) {
      ierr = ApplyNitscheFreeslip(surf_bclist[n],PETSC_FALSE);CHKERRQ(ierr);
      ierr = ApplyNitsche_AngularVelocityNormal(surf_bclist[n],PETSC_FALSE,data);CHKERRQ(ierr);
      //ierr = ApplyWeakDirichket(surf_bclist[n],PETSC_FALSE,data);CHKERRQ(ierr);
    }
    for (n=0; n<nl; n++) {
      ierr = ApplyStrongDirichlet(dav[n],bclist[n]);CHKERRQ(ierr);
    }
  //} else {
  //  for (n=0; n<nl; n++) {
  //    ierr = BCTypeSlipNitscheFreeSurface(surf_bclist[n],PETSC_TRUE);CHKERRQ(ierr);
  //  }
  //}
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode ModelDestroy_SCTest(pTatinCtx c,void *ctx)
{
  ModelSCTestCtx  *data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);
  data = (ModelSCTestCtx*)ctx;

  /* Free contents of structure */

  /* Free structure */
  ierr = PetscFree(data);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode pTatinModelRegister_SCTest(void)
{
  ModelSCTestCtx *data;
  pTatinModel      m;
  PetscErrorCode   ierr;

  PetscFunctionBegin;

  /* Allocate memory for the data structure for this model */
  ierr = PetscMalloc(sizeof(ModelSCTestCtx),&data);CHKERRQ(ierr);
  ierr = PetscMemzero(data,sizeof(ModelSCTestCtx));CHKERRQ(ierr);

  /* register user model */
  ierr = pTatinModelCreate(&m);CHKERRQ(ierr);

  /* Set name, model select via -ptatin_model NAME */
  ierr = pTatinModelSetName(m,"surf_constraint_model");CHKERRQ(ierr);

  /* Set model data */
  ierr = pTatinModelSetUserData(m,data);CHKERRQ(ierr);

  /* Set function pointers */
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_INIT,                  (void (*)(void))ModelInitialize_SCTest);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_MESH_GEOM,  (void (*)(void))ModelApplyInitialMeshGeometry_SCTest);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_MAT_GEOM,   (void (*)(void))ModelApplyInitialMaterialGeometry_SCTest);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_STOKES_VARIABLE_MARKERS,(void (*)(void))ModelApplyInitialStokesVariableMarkers_SCTest);CHKERRQ(ierr);
  //ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_SOLUTION,   (void (*)(void))ModelApplyInitialSolution_Debug);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_BC,              (void (*)(void))ModelApplyBoundaryCondition_SCTest);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_BCMG,            (void (*)(void))ModelApplyBoundaryConditionMG);CHKERRQ(ierr);
  //ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_UPDATE_MESH_GEOM,(void (*)(void))ModelApplyUpdateMeshGeometry_Debug);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_OUTPUT,                (void (*)(void))ModelOutput_SCTest);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_DESTROY,               (void (*)(void))ModelDestroy_SCTest);CHKERRQ(ierr);

  /* Insert model into list */
  ierr = pTatinModelRegister(m);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}