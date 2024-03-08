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

#include "pswarm.h"

#include "oblique_rift_nitsche_ctx.h"

static const char MODEL_NAME_R[] = "model_rift_nitsche_";
static PetscLogEvent   PTATIN_MaterialPointPopulationControlRemove;

static PetscErrorCode ModelInitialGeometry_RiftNitsche(ModelRiftNitscheCtx *data)
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

  PetscFunctionReturn(0);
}

/* Dirichlet boundary velocity vector data */
static PetscErrorCode ModelInitialBoundaryVelocity_RiftNitsche(ModelRiftNitscheCtx *data)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* Norm in cm/yr */
  data->norm_u  = 1.0;
  /* Angle in degree */
  data->alpha_u = 45.0;
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R,"-norm_u",&data->norm_u,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R,"-alpha_u",&data->alpha_u,NULL);CHKERRQ(ierr);
  data->alpha_u = data->alpha_u * M_PI/180.0;
  /* z component */
  data->u_bc[2] = data->norm_u * cos(data->alpha_u);
  /* default y component */
  data->u_bc[1] = 0.0;
  /* x component */
  data->u_bc[0] = sqrt( pow(data->norm_u,2.0) - pow(data->u_bc[2],2.0) );

  /* Rotation angle for the vector field rotation */
  data->alpha_r = 0.0;
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R,"-alpha_r",&data->alpha_r,NULL);CHKERRQ(ierr);
  data->alpha_r = data->alpha_r * M_PI/180.0;

  data->atan_sharpness = 5.0e-4;
  data->atan_offset = 300.0e3;
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R,"-bc_atan_sharpness",&data->atan_sharpness,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R,"-bc_atan_offset",&data->atan_offset,NULL);CHKERRQ(ierr);

  data->time_full_velocity = 1.0;
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R,"-bc_time_full_velocity",&data->time_full_velocity,NULL);CHKERRQ(ierr);

  /* reports before scaling */
  PetscPrintf(PETSC_COMM_WORLD,"************ Velocity Dirichlet BCs ************\n",NULL);
  PetscPrintf(PETSC_COMM_WORLD,"||u||= %1.4e, ux = %+1.4e, uy = %+1.4e, uz = %+1.4e [cm/yr]\n",data->norm_u,data->u_bc[0],data->u_bc[1],data->u_bc[2]);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelAnalyticalVelocityFunction_Linear(PetscReal position[], PetscReal u[], ModelRiftNitscheCtx *data)
{
  PetscFunctionBegin;
  u[0] = 2.0/(data->Lz - data->Oz) * position[2] * data->u_bc[0] - data->u_bc[0];
  u[1] = 2.0*(position[1] - data->Ly)*data->u_bc[2]*(data->Ly - data->Oy)/((data->Lz - data->Oz)*(data->Oy - data->Ly));
  u[2] = 2.0/(data->Lz - data->Oz) * position[2] * data->u_bc[2] - data->u_bc[2];
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelAnalyticalVelocityFunction_Arctan(PetscReal position[], PetscReal u[], ModelRiftNitscheCtx *data)
{
  PetscFunctionBegin;
  u[0] = 2.0/M_PI * data->u_bc[0] * PetscAtanReal(data->atan_sharpness*(position[2]-data->atan_offset));
  u[1] = 2.0*(position[1] - data->Ly)*data->u_bc[2]*(data->Ly - data->Oy)/((data->Lz - data->Oz)*(data->Oy - data->Ly));
  u[2] = 2.0/M_PI * data->u_bc[2] * PetscAtanReal(data->atan_sharpness*(position[2]-data->atan_offset));
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelAnalyticalVelocityFunction(PetscReal position[], PetscReal u[], ModelRiftNitscheCtx *data)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch(data->u_func_type) {
    case 0:
      ierr = ModelAnalyticalVelocityFunction_Linear(position,u,data);CHKERRQ(ierr);
      break;

    case 1:
      ierr = ModelAnalyticalVelocityFunction_Arctan(position,u,data);CHKERRQ(ierr);
      break;

    case 2:
      ierr = ModelAnalyticalVelocityFunction_Arctan(position,u,data);CHKERRQ(ierr);
      break;

    default:
      ierr = ModelAnalyticalVelocityFunction_Linear(position,u,data);CHKERRQ(ierr);
      break;
  }

  PetscFunctionReturn(0);
}

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

static PetscErrorCode MatMult3x3(PetscReal A[][3], PetscReal B[][3], PetscReal C[][3])
{
  PetscInt i,j,k;

  for (i=0; i<3; i++) {
    for (j=0; j<3; j++) {
      C[i][j] = 0.0;
    }
  }

  for (i=0; i<3; i++) {
    for (j=0; j<3; j++) {
      for (k=0; k<3; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode Rotate_u(PetscReal theta, PetscReal r[], PetscReal u[], PetscReal w[])
{
  PetscInt  i,j;
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

static PetscErrorCode RotateReferential(PetscReal position[], PetscReal coords_rt[], PetscBool ccw, ModelRiftNitscheCtx *data)
{
  PetscReal      coords[] = {0.0,0.0,0.0};
  PetscReal      r[] = {0.0,1.0,0.0};
  PetscReal      coords_r[3];
  PetscReal      angle;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* Translate to (0,0,0) */
  coords[0] = position[0] - 0.5*(data->Lx + data->Ox);
  coords[1] = position[1] - 0.5*(data->Ly + data->Oy);
  coords[2] = position[2] - 0.5*(data->Lz + data->Oz);

  /* Rotate */
  if (ccw) {
    angle = -data->alpha_r;
  } else {
    angle = data->alpha_r;
  }
  ierr = Rotate_u(angle,r,coords,coords_r);CHKERRQ(ierr);

  /* Translate back */
  coords_rt[0] = coords_r[0] + 0.5*(data->Lx + data->Ox);
  coords_rt[1] = coords_r[1] + 0.5*(data->Ly + data->Oy);
  coords_rt[2] = coords_r[2] + 0.5*(data->Lz + data->Oz);
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeNewBasisVectors(ModelRiftNitscheCtx *data)
{
  PetscReal      u0[] = {0.0,0.0,0.0};
  PetscReal      r[] = {0.0,1.0,0.0};
  PetscErrorCode ierr;

  PetscFunctionBegin;

  u0[0] = data->u_bc[0]/data->norm_u;
  u0[1] = 0.0;
  u0[2] = data->u_bc[2]/data->norm_u;

  ierr = Rotate_u(data->alpha_r,r,u0,data->t1_hat);CHKERRQ(ierr);
  ierr = Rotate_u(-0.5*M_PI,r,data->t1_hat,data->n_hat);CHKERRQ(ierr);

  PetscPrintf(PETSC_COMM_WORLD,"************ Rotated basis vectors ************\n",NULL);
  for (int i=0; i<3; i++) {
    PetscPrintf(PETSC_COMM_WORLD,"n_hat[%d] = %1.4f, t1_hat[%d] = %1.4f \n",i,data->n_hat[i],i,data->t1_hat[i]);
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetGeneralSlipBoundaryValues_ObliqueExtensionZ(ModelRiftNitscheCtx *data)
{
  PetscInt       nn;
  PetscBool      found;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  data->epsilon_s[0] = 0.0;                                         // E_xx
  data->epsilon_s[1] = 0.0;                                         // E_yy
  data->epsilon_s[2] = 2.0 / (data->Lz - data->Oz) * data->u_bc[2]; // E_zz
  data->epsilon_s[3] = 0.0;                                         // E_xy
  data->epsilon_s[4] = 1.0 / (data->Lz - data->Oz) * data->u_bc[0]; // E_xz
  data->epsilon_s[5] = 0.0;                                         // E_yz

  /* Do not worry if the norm of these vectors is not 1, it is handled internally */
  /* Tangent vector 1 */
  data->t1_hat[0] = data->u_bc[0];
  data->t1_hat[1] = 0.0;
  data->t1_hat[2] = data->u_bc[2];
  /* Normal vector */
  data->n_hat[0] = -data->u_bc[2];
  data->n_hat[1] = 0.0;
  data->n_hat[2] = data->u_bc[0];

  /* 
  Set which component of the rotated stress tensor:
    will be constrained (1) 
    will be treated as unknown (0) 
  */
  data->H[0] = 0; // H_00
  data->H[1] = 1; // H_11
  data->H[2] = 0; // H_22
  data->H[3] = 1; // H_01 = H_10
  data->H[4] = 0; // H_02 = H_20
  data->H[5] = 0; // H_12 = H_21
  nn = 6;
  ierr = PetscOptionsGetRealArray(NULL,MODEL_NAME_R,"-mathcalH",data->H,&nn,&found);CHKERRQ(ierr);
  if (found) {if (nn != 6) { SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_USER,"Wrong number of entries for -mathcalH, expected 6, passed %d",nn); } }
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelRotateStrainRateBoundaryValue(ModelRiftNitscheCtx *data)
{
  PetscInt       i,j;
  PetscReal      r[] = {0.0,1.0,0.0};
  PetscReal      R[3][3],R_transpose[3][3],E[3][3],ERT[3][3],E_R[3][3];
  const PetscInt indices_voigt[][3] = { {0, 3, 4}, {3, 1, 5}, {4, 5, 2} };
  PetscErrorCode ierr;

  PetscFunctionBegin;

  E[0][0] = 0.0; 
  E[1][1] = 0.0;
  E[2][2] = 2.0 / (data->Lz - data->Oz) * data->u_bc[2];

  E[0][1] = 0.0;
  E[0][2] = 1.0 / (data->Lz - data->Oz) * data->u_bc[0];
  E[1][2] = 0.0;

  E[1][0] = E[0][1];
  E[2][0] = E[0][2];
  E[2][1] = E[1][2];

  RotationMatrix(data->alpha_r,r,R);

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

static PetscErrorCode ModelSetGeneralSlipBoundaryValues_RotatedVelocityField(ModelRiftNitscheCtx *data)
{
  PetscInt       nn;
  PetscBool      found;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = ComputeNewBasisVectors(data);CHKERRQ(ierr);
  ierr = ModelRotateStrainRateBoundaryValue(data);CHKERRQ(ierr);
  /* 
  Set which component of the rotated stress tensor:
    will be constrained (1) 
    will be treated as unknown (0) 
  */
  data->H[0] = 0; // H_00
  data->H[1] = 1; // H_11
  data->H[2] = 0; // H_22
  data->H[3] = 1; // H_01 = H_10
  data->H[4] = 1; // H_02 = H_20
  data->H[5] = 0; // H_12 = H_21
  nn = 6;
  ierr = PetscOptionsGetRealArray(NULL,MODEL_NAME_R,"-mathcalH",data->H,&nn,&found);CHKERRQ(ierr);
  if (found) {if (nn != 6) { SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_USER,"Wrong number of entries for -mathcalH, expected 6, passed %d",nn); } }
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetGeneralSlipBoundaryValues_RiftNitsche(ModelRiftNitscheCtx *data)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (data->bc_type)
  {
    case 0:
      ierr = ModelSetGeneralSlipBoundaryValues_ObliqueExtensionZ(data);CHKERRQ(ierr);
      break;

    case 1:
      ierr = ModelSetGeneralSlipBoundaryValues_ObliqueExtensionZ(data);CHKERRQ(ierr);
      break;

    case 2:
      ierr = ModelSetGeneralSlipBoundaryValues_ObliqueExtensionZ(data);CHKERRQ(ierr);
      break;

    case 3:
      ierr = ModelSetGeneralSlipBoundaryValues_RotatedVelocityField(data);CHKERRQ(ierr);
      break;

    case 4:
      ierr = ModelSetGeneralSlipBoundaryValues_RotatedVelocityField(data);CHKERRQ(ierr);
      break;

    case 5:
      ierr = ModelSetGeneralSlipBoundaryValues_RotatedVelocityField(data);CHKERRQ(ierr);
      break;

    case 6:
      ierr = ModelSetGeneralSlipBoundaryValues_RotatedVelocityField(data);CHKERRQ(ierr);
      break;

    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"bc_type must be set with one of the bc options\n");
      break;
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetMaterialParametersVISCOUS_RiftNitsche(pTatinCtx c,DataBucket materialconstants, ModelRiftNitscheCtx *data)
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

static PetscErrorCode ModelSetMaterialParametersConstantVPT_RiftNitsche(pTatinCtx c,DataBucket materialconstants, ModelRiftNitscheCtx *data)
{
  DataField                 PField,PField_k,PField_Q;
  EnergyConductivityConst   *data_k;
  EnergySourceConst         *data_Q;
  EnergyMaterialConstants   *matconstants_e;
  PetscInt                  region_idx;
  int                       source_type[7] = {0, 0, 0, 0, 0, 0, 0};
  PetscReal                 viscosity;
  PetscReal                 phi,phi_inf,Co,Co_inf,Tens_cutoff,Hst_cutoff,eps_min,eps_max;
  PetscReal                 beta,alpha,rho,heat_source,conductivity,Cp;
  char                      *option_name;
  PetscErrorCode            ierr;

  PetscFunctionBegin;

  /* Material constants */
  ierr = MaterialConstantsSetDefaults(materialconstants);CHKERRQ(ierr);
  
  /* Energy material constants */
  DataBucketGetDataFieldByName(materialconstants,EnergyMaterialConstants_classname,&PField);
  DataFieldGetEntries(PField,(void**)&matconstants_e);
  
  /* Conductivity */
  DataBucketGetDataFieldByName(materialconstants,EnergyConductivityConst_classname,&PField_k);
  DataFieldGetEntries(PField_k,(void**)&data_k);
  
  /* Heat source */
  DataBucketGetDataFieldByName(materialconstants,EnergySourceConst_classname,&PField_Q);
  DataFieldGetEntries(PField_Q,(void**)&data_Q);

  /* Set default values for parameters */
  source_type[0] = ENERGYSOURCE_CONSTANT;
  source_type[1] = ENERGYSOURCE_SHEAR_HEATING;
  Cp             = 800.0;
  /* Set material parameters from options file */
  for (region_idx=0; region_idx<data->n_phases; region_idx++) {
    /* Set material constitutive laws */
    ierr = MaterialConstantsSetValues_MaterialType(materialconstants,region_idx,VISCOUS_CONSTANT,PLASTIC_DP,SOFTENING_LINEAR,DENSITY_BOUSSINESQ);CHKERRQ(ierr);

    /* VISCOUS PARAMETERS */
    viscosity = 1.0e+23;
    if (asprintf (&option_name, "-eta0_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R, option_name,&viscosity,NULL);CHKERRQ(ierr); 
    free (option_name);
    ierr = MaterialConstantsSetValues_ViscosityConst(materialconstants,region_idx,viscosity);CHKERRQ(ierr);

    /* PLASTIC PARAMETERS */
    if (asprintf (&option_name, "-phi_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R, option_name,&phi,NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-phi_inf_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R, option_name,&phi_inf,NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-Co_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R, option_name,&Co,NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-Co_inf_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R, option_name,&Co_inf,NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-Tens_cutoff_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R, option_name,&Tens_cutoff,NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-Hst_cutoff_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R, option_name,&Hst_cutoff,NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-eps_min_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R, option_name,&eps_min,NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-eps_max_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R, option_name,&eps_max,NULL);CHKERRQ(ierr);
    free (option_name);

    phi     = M_PI * phi/180.0;
    phi_inf = M_PI * phi_inf/180.0;
    /* Set plastic params for region_idx */
    MaterialConstantsSetValues_PlasticDP(materialconstants,region_idx,phi,phi_inf,Co,Co_inf,Tens_cutoff,Hst_cutoff);
    MaterialConstantsSetValues_SoftLin(materialconstants,region_idx,eps_min,eps_max);

    /* ENERGY PARAMETERS */
    if (asprintf (&option_name, "-alpha_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R, option_name,&alpha,NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-beta_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R, option_name,&beta,NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-rho_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R, option_name,&rho,NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-heat_source_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R, option_name,&heat_source,NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-conductivity_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R, option_name,&conductivity,NULL);CHKERRQ(ierr);
    free (option_name);
    
    /* Set energy params for region_idx */
    MaterialConstantsSetValues_EnergyMaterialConstants(region_idx,matconstants_e,alpha,beta,rho,Cp,ENERGYDENSITY_CONSTANT,ENERGYCONDUCTIVITY_CONSTANT,source_type);
    MaterialConstantsSetValues_DensityBoussinesq(materialconstants,region_idx,rho,alpha,beta);
    EnergySourceConstSetField_HeatSource(&data_Q[region_idx],heat_source);
    EnergyConductivityConstSetField_k0(&data_k[region_idx],conductivity);
  }

  /* Report all material parameters values */
  for (region_idx=0; region_idx<data->n_phases; region_idx++) {
    ierr = MaterialConstantsPrintAll(materialconstants,region_idx);CHKERRQ(ierr);
    ierr = MaterialConstantsEnergyPrintAll(materialconstants,region_idx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetMaterialParameters_MantleDiffusionLaw(pTatinCtx c,DataBucket materialconstants, ModelRiftNitscheCtx *data)
{
  PetscInt       region_mantle;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  
  region_mantle = 3;
  /* Set material constitutive laws */
  ierr = MaterialConstantsSetValues_MaterialType(materialconstants,region_mantle,VISCOUS_ARRHENIUS_DISLDIFF,PLASTIC_DP,SOFTENING_LINEAR,DENSITY_BOUSSINESQ);CHKERRQ(ierr);
  /* VISCOUS PARAMETERS */
  ierr = MaterialConstantsSetFromOptions_ViscosityArrh_DislDiff(materialconstants,MODEL_NAME_R,region_mantle,PETSC_TRUE);CHKERRQ(ierr);
  /* PLASTIC PARAMETERS */
  ierr = MaterialConstantsSetFromOptions_PlasticDP(materialconstants,MODEL_NAME_R,region_mantle,PETSC_TRUE);CHKERRQ(ierr);
  /* SOFTENING */
  ierr = MaterialConstantsSetFromOptions_SoftLin(materialconstants,MODEL_NAME_R,region_mantle,PETSC_TRUE);CHKERRQ(ierr);
  /* DENSITY PARAMETERS */
  ierr = MaterialConstantsSetFromOptions_DensityBoussinesq(materialconstants,MODEL_NAME_R,region_mantle,PETSC_TRUE);CHKERRQ(ierr);

  /* Report */
  ierr = MaterialConstantsPrintAll(materialconstants,region_mantle);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetMaterialParametersVPT_RiftNitsche(pTatinCtx c,DataBucket materialconstants, ModelRiftNitscheCtx *data)
{
  DataField                 PField,PField_k,PField_Q;
  EnergyConductivityConst   *data_k;
  EnergySourceConst         *data_Q;
  EnergyMaterialConstants   *matconstants_e;
  PetscInt                  region_idx;
  int                       source_type[7] = {0, 0, 0, 0, 0, 0, 0};
  PetscReal                 preexpA,Ascale,entalpy,Vmol,nexp,Tref;
  PetscReal                 phi,phi_inf,Co,Co_inf,Tens_cutoff,Hst_cutoff,eps_min,eps_max;
  PetscReal                 beta,alpha,rho,heat_source,conductivity,Cp;
  char                      *option_name;
  PetscErrorCode            ierr;

  PetscFunctionBegin;

  /* Material constants */
  ierr = MaterialConstantsSetDefaults(materialconstants);CHKERRQ(ierr);
  
  /* Energy material constants */
  DataBucketGetDataFieldByName(materialconstants,EnergyMaterialConstants_classname,&PField);
  DataFieldGetEntries(PField,(void**)&matconstants_e);
  
  /* Conductivity */
  DataBucketGetDataFieldByName(materialconstants,EnergyConductivityConst_classname,&PField_k);
  DataFieldGetEntries(PField_k,(void**)&data_k);
  
  /* Heat source */
  DataBucketGetDataFieldByName(materialconstants,EnergySourceConst_classname,&PField_Q);
  DataFieldGetEntries(PField_Q,(void**)&data_Q);

  /* Set default values for parameters */
  source_type[0] = ENERGYSOURCE_CONSTANT;
  source_type[1] = ENERGYSOURCE_SHEAR_HEATING;
  Cp             = 800.0;
  /* Set material parameters from options file */
  for (region_idx=0; region_idx<data->n_phases; region_idx++) {
    /* Set material constitutive laws */
    MaterialConstantsSetValues_MaterialType(materialconstants,region_idx,VISCOUS_ARRHENIUS_2,PLASTIC_DP,SOFTENING_LINEAR,DENSITY_BOUSSINESQ);

    /* VISCOUS PARAMETERS */
    if (asprintf (&option_name, "-preexpA_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R, option_name,&preexpA,NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-Ascale_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R, option_name,&Ascale,NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-entalpy_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R, option_name,&entalpy,NULL);CHKERRQ(ierr);
    free (option_name); 
    if (asprintf (&option_name, "-Vmol_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R, option_name,&Vmol,NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-nexp_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R, option_name,&nexp,NULL);CHKERRQ(ierr);
    free (option_name); 
    if (asprintf (&option_name, "-Tref_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R, option_name,&Tref,NULL);CHKERRQ(ierr);
    free (option_name);
    /* Set viscous params for region_idx */
    MaterialConstantsSetValues_ViscosityArrh(materialconstants,region_idx,preexpA,Ascale,entalpy,Vmol,nexp,Tref);  

    /* PLASTIC PARAMETERS */
    if (asprintf (&option_name, "-phi_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R, option_name,&phi,NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-phi_inf_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R, option_name,&phi_inf,NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-Co_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R, option_name,&Co,NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-Co_inf_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R, option_name,&Co_inf,NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-Tens_cutoff_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R, option_name,&Tens_cutoff,NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-Hst_cutoff_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R, option_name,&Hst_cutoff,NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-eps_min_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R, option_name,&eps_min,NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-eps_max_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R, option_name,&eps_max,NULL);CHKERRQ(ierr);
    free (option_name);

    phi     = M_PI * phi/180.0;
    phi_inf = M_PI * phi_inf/180.0;
    /* Set plastic params for region_idx */
    MaterialConstantsSetValues_PlasticDP(materialconstants,region_idx,phi,phi_inf,Co,Co_inf,Tens_cutoff,Hst_cutoff);
    MaterialConstantsSetValues_SoftLin(materialconstants,region_idx,eps_min,eps_max);

    /* ENERGY PARAMETERS */
    if (asprintf (&option_name, "-alpha_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R, option_name,&alpha,NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-beta_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R, option_name,&beta,NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-rho_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R, option_name,&rho,NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-heat_source_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R, option_name,&heat_source,NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-conductivity_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R, option_name,&conductivity,NULL);CHKERRQ(ierr);
    free (option_name);
    
    /* Set energy params for region_idx */
    MaterialConstantsSetValues_EnergyMaterialConstants(region_idx,matconstants_e,alpha,beta,rho,Cp,ENERGYDENSITY_CONSTANT,ENERGYCONDUCTIVITY_CONSTANT,source_type);
    MaterialConstantsSetValues_DensityBoussinesq(materialconstants,region_idx,rho,alpha,beta);
    EnergySourceConstSetField_HeatSource(&data_Q[region_idx],heat_source);
    EnergyConductivityConstSetField_k0(&data_k[region_idx],conductivity);
  }

  /* Report all material parameters values */
  for (region_idx=0; region_idx<data->n_phases; region_idx++) {
    ierr = MaterialConstantsPrintAll(materialconstants,region_idx);CHKERRQ(ierr);
    ierr = MaterialConstantsEnergyPrintAll(materialconstants,region_idx);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetViscousType(ModelRiftNitscheCtx *data)
{
  PetscBool      viscous_const=PETSC_FALSE;
  PetscBool      viscous_vpt_const=PETSC_FALSE;
  PetscBool      viscous_vpt_disl_diff=PETSC_FALSE;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_R,"-viscous_constant",&viscous_const,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_R,"-viscous_vpt_constant",&viscous_vpt_const,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_R,"-viscous_vpt_disl_diff",&viscous_vpt_disl_diff,NULL);CHKERRQ(ierr);

  data->viscous_type = 0;
  if (viscous_const) {
    data->viscous_type = 1;
  } else if (viscous_vpt_const) {
    data->viscous_type = 2;
  } else if (viscous_vpt_disl_diff) {
    data->viscous_type = 3;
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetMaterialParameters_RiftNitsche(pTatinCtx c,DataBucket materialconstants, ModelRiftNitscheCtx *data)
{
  PetscBool      viscous_const=PETSC_FALSE;
  PetscBool      viscous_vpt_const=PETSC_FALSE;
  PetscErrorCode ierr;
  PetscFunctionBegin;

#if 0
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_R,"-viscous_constant",&viscous_const,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_R,"-viscous_vpt_constant",&viscous_vpt_const,NULL);CHKERRQ(ierr);

  if (viscous_const) {
    ierr = ModelSetMaterialParametersVISCOUS_RiftNitsche(c,materialconstants,data);CHKERRQ(ierr);
  } else if (viscous_vpt_const) {
    ierr = ModelSetMaterialParametersConstantVPT_RiftNitsche(c,materialconstants,data);CHKERRQ(ierr);
  } else {
    ierr = ModelSetMaterialParametersVPT_RiftNitsche(c,materialconstants,data);CHKERRQ(ierr);
  }
#endif

  ierr = ModelSetViscousType(data);CHKERRQ(ierr);
  switch (data->viscous_type)
  {
    case 0:
      ierr = ModelSetMaterialParametersVPT_RiftNitsche(c,materialconstants,data);CHKERRQ(ierr);
      break;

    case 1:
      ierr = ModelSetMaterialParametersVISCOUS_RiftNitsche(c,materialconstants,data);CHKERRQ(ierr);
      break;
    
    case 2:
      ierr = ModelSetMaterialParametersConstantVPT_RiftNitsche(c,materialconstants,data);CHKERRQ(ierr);
      break;

    case 3:
      ierr = ModelSetMaterialParametersVPT_RiftNitsche(c,materialconstants,data);CHKERRQ(ierr);
      ierr = ModelSetMaterialParameters_MantleDiffusionLaw(c,materialconstants,data);CHKERRQ(ierr);
      break;

    default:
      PetscPrintf(PETSC_COMM_WORLD,"[[ WARNING ]] No viscous type provided. Using default Arrhenius + Drucker-Prager + Boussinesq\n");
      ierr = ModelSetMaterialParametersVPT_RiftNitsche(c,materialconstants,data);CHKERRQ(ierr);
      break;
  }


  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetViscosityCutoff_RiftNitsche(ModelRiftNitscheCtx *data)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  data->eta_cutoff = PETSC_TRUE;
  data->eta_max = 1.0e+25;
  data->eta_min = 1.0e+19;
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_R,"-apply_viscosity_cutoff",&data->eta_cutoff,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R,"-eta_lower_cutoff",      &data->eta_min,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R,"-eta_upper_cutoff",      &data->eta_max,NULL);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetSPMParameters_RiftNitsche(ModelRiftNitscheCtx *data)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  data->diffusivity_spm = 1.0e-6;
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R,"-diffusivity_spm",&data->diffusivity_spm,NULL);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetTemperatureBCs_RiftNitsche(ModelRiftNitscheCtx *data)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  data->Ttop = 0.0; // Top temperature BC
  data->Tbottom = 1450.0; // Bottom temperature BC
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R,"-Ttop",&data->Ttop,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R,"-Tbottom",&data->Tbottom,NULL);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetWeakZoneParameters_RiftNitsche(ModelRiftNitscheCtx *data)
{
  PetscInt       nn;
  PetscBool      found,wz_notch,wz_gauss,wz_oblique,wz_double;
  PetscBool      wz_oblique_gauss,wz_straight_gauss;
  PetscBool      wz_centre_equal,wz_centre_opts,wz_centre_angle;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  wz_notch          = PETSC_FALSE;
  wz_gauss          = PETSC_FALSE;
  wz_oblique        = PETSC_FALSE;
  wz_double         = PETSC_FALSE;
  wz_oblique_gauss  = PETSC_FALSE;
  wz_straight_gauss = PETSC_FALSE;
  wz_centre_equal   = PETSC_FALSE;
  wz_centre_opts    = PETSC_FALSE;
  wz_centre_angle   = PETSC_FALSE;

  data->n_notches = 3;
  data->wz_width = 100.0e3;
  data->wz_sigma[0] = 3.0e+5;
  data->wz_sigma[1] = 3.0e+5;
  data->wz_angle = 45.0;
  data->wz_origin = 300.0e3; // metres
  data->wz_offset = 0.0; // metres

  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_R,"-wz_notch",         &wz_notch,         &found);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_R,"-wz_gauss",         &wz_gauss,         &found);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_R,"-wz_oblique",       &wz_oblique,       &found);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_R,"-wz_double",        &wz_double,        &found);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_R,"-wz_oblique_gauss", &wz_oblique_gauss, &found);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_R,"-wz_straight_gauss",&wz_straight_gauss,&found);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_R,"-wz_centre_equal",  &wz_centre_equal,  &found);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_R,"-wz_centre_opts",   &wz_centre_opts,   &found);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_R,"-wz_centre_angle",  &wz_centre_angle,  &found);CHKERRQ(ierr);

  ierr = PetscOptionsGetInt(NULL,MODEL_NAME_R,"-wz_n_notches",&data->n_notches,&found);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R,"-wz_width",   &data->wz_width, &found);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R,"-wz_angle",   &data->wz_angle, &found);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R,"-wz_origin",  &data->wz_origin,&found);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_R,"-wz_offset",  &data->wz_offset,&found);CHKERRQ(ierr);

  nn = 2;
  ierr = PetscOptionsGetRealArray(NULL,MODEL_NAME_R,"-wz_sigma",data->wz_sigma,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != 2) {
      SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_USER,"Expected 2 values for -wz_sigma. Found %d",nn);
    }
  }

  data->wz_angle = data->wz_angle * M_PI/180.0;

  PetscPrintf(PETSC_COMM_WORLD,"************ Weak Zones Type ************\n",NULL);
  data->wz_type = -1;
  if (wz_notch) {
    data->wz_type = 0;
    PetscPrintf(PETSC_COMM_WORLD,"Type %d: [[ CUBES ]]\n",data->wz_type);
  } else if (wz_gauss) {
    data->wz_type = 1;
    PetscPrintf(PETSC_COMM_WORLD,"Type %d: [[ GAUSSIANS ]]\n",data->wz_type);
  } else if (wz_oblique) {
    data->wz_type = 2;
    PetscPrintf(PETSC_COMM_WORLD,"Type %d: [[ OBLIQUE ]]\n",data->wz_type);
  } else if (wz_double) {
    data->wz_type = 3;
    PetscPrintf(PETSC_COMM_WORLD,"Type %d: [[ DOUBLE OFFSET RECTANGLES ]]\n",data->wz_type);
  } else if (wz_oblique_gauss) {
    data->wz_type = 4;
    PetscPrintf(PETSC_COMM_WORLD,"Type %d: [[ OBLIQUE GAUSSIAN ]]\n",data->wz_type);
  } else if (wz_straight_gauss) {
    data->wz_type = 5;
    PetscPrintf(PETSC_COMM_WORLD,"Type %d: [[ STRAIGHT GAUSSIAN ]]\n",data->wz_type);
  } else {
    PetscPrintf(PETSC_COMM_WORLD,"[[ WARNING ]]  No weak zone type chosen, setting initial plastic strain to 0\n");
  }

  data->wz_centre_type = -1;
  if (wz_centre_equal) {
    data->wz_centre_type = 0;
    PetscPrintf(PETSC_COMM_WORLD,"Weak Zone Centre Type %d: [[ EQUALLY SPACED IN X DIRECTION AND OFFSET FROM GIVEN DISTANCE ]]\n",data->wz_centre_type);
  } else if (wz_centre_opts) {
    data->wz_centre_type = 1;
    PetscPrintf(PETSC_COMM_WORLD,"Weak Zone Centre Type %d: [[ ARBITRARY PLACED FROM OPTIONS ]]\n",data->wz_centre_type);
  } else if (wz_centre_angle) {
    data->wz_centre_type = 2;
    PetscPrintf(PETSC_COMM_WORLD,"Weak Zone Centre Type %d: [[ EQUALLY SPACED IN X DIRECTION AND OFFSET FROM GIVEN ANGLE ]]\n",data->wz_centre_type);
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetPassiveMarkersSwarmParameters(pTatinCtx c, ModelRiftNitscheCtx *data)
{
  PSwarm         pswarm;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = PSwarmCreate(PETSC_COMM_WORLD,&pswarm);CHKERRQ(ierr);
  ierr = PSwarmSetOptionsPrefix(pswarm,"passive_");CHKERRQ(ierr);
  ierr = PSwarmSetPtatinCtx(pswarm,c);CHKERRQ(ierr);
  ierr = PSwarmSetTransportModeType(pswarm,PSWARM_TM_LAGRANGIAN);CHKERRQ(ierr);

  ierr = PSwarmSetFromOptions(pswarm);CHKERRQ(ierr);

  /* Copy reference into model data for later use in different functions */
  data->pswarm = pswarm;

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelScaleParameters_RiftNitsche(DataBucket materialconstants, ModelRiftNitscheCtx *data)
{
  PetscInt  region_idx,i;
  PetscReal cm_per_year2m_per_sec,Myr2sec;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* scaling values */
  cm_per_year2m_per_sec = 1.0e-2 / ( 365.0 * 24.0 * 60.0 * 60.0 );
  Myr2sec               = 1.0e6 * ( 365.0 * 24.0 * 3600.0 );
  data->length_bar     = 1.0e5;
  data->viscosity_bar  = 1.0e22;
  data->velocity_bar   = 1.0e-10;
  /* Compute additional scaling parameters */
  data->time_bar         = data->length_bar / data->velocity_bar;
  data->pressure_bar     = data->viscosity_bar/data->time_bar;
  data->density_bar      = data->pressure_bar * (data->time_bar*data->time_bar)/(data->length_bar*data->length_bar); // kg.m^-3
  data->acceleration_bar = data->length_bar / (data->time_bar*data->time_bar);
  
  /* Scale viscosity cutoff */
  data->eta_max = data->eta_max / data->viscosity_bar;
  data->eta_min = data->eta_min / data->viscosity_bar;
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

  data->wz_origin = data->wz_origin / data->length_bar;
  data->wz_offset = data->wz_offset / data->length_bar;

  data->atan_sharpness = data->atan_sharpness * data->length_bar;
  data->atan_offset    = data->atan_offset / data->length_bar;

  data->wz_width = data->wz_width / data->length_bar;
  for (i=0; i<2; i++) { data->wz_sigma[i] = data->wz_sigma[i] / data->length_bar; }
  
  for (i=0; i<2; i++) { 
    data->split_face_max[i] = data->split_face_max[i] / data->length_bar;
    data->split_face_min[i] = data->split_face_min[i] / data->length_bar; 
  }

  data->time_full_velocity = data->time_full_velocity*Myr2sec / data->time_bar;

  /* Scale velocity */
  data->norm_u = data->norm_u*cm_per_year2m_per_sec / data->velocity_bar;
  for (i=0; i<3; i++) { data->u_bc[i] = data->u_bc[i]*cm_per_year2m_per_sec / data->velocity_bar; }

  data->diffusivity_spm = data->diffusivity_spm / (data->length_bar*data->length_bar/data->time_bar);

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

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetBCType_RiftNitsche(ModelRiftNitscheCtx *data)
{
  PetscBool      bc_nitsche,bc_dirichlet,bc_freeslip_nitsche,found;
  PetscBool      bc_strikeslip,bc_u_func_atan,bc_strike_analogue;
  PetscBool      bc_strike_analogue_nitsche,bc_u_func_mixte,bc_diri_neumann;
  PetscInt       nn;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* Get BC type from options */
  bc_nitsche                 = PETSC_FALSE;
  bc_dirichlet               = PETSC_FALSE;
  bc_freeslip_nitsche        = PETSC_FALSE;
  bc_strikeslip              = PETSC_FALSE;
  bc_strike_analogue         = PETSC_FALSE;
  bc_strike_analogue_nitsche = PETSC_FALSE;
  bc_u_func_atan             = PETSC_FALSE;
  bc_u_func_mixte            = PETSC_FALSE;
  bc_diri_neumann            = PETSC_FALSE;

  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_R,"-bc_nitsche",                &bc_nitsche,&found);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_R,"-bc_dirichlet",              &bc_dirichlet,&found);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_R,"-bc_freeslip_nitsche",       &bc_freeslip_nitsche,&found);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_R,"-bc_strike_slip",            &bc_strikeslip,&found);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_R,"-bc_strike_analogue",        &bc_strike_analogue,&found);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_R,"-bc_strike_analogue_nitsche",&bc_strike_analogue_nitsche,&found);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_R,"-bc_u_func_atan",            &bc_u_func_atan,&found);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_R,"-bc_u_func_mixte",           &bc_u_func_mixte,&found);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_R,"-bc_diri_neumann",           &bc_diri_neumann,&found);CHKERRQ(ierr);

  /* Split face location for the analogue BCs */
  data->split_face_min[0] = 290.0e3;
  data->split_face_min[1] = 310.0e3;

  data->split_face_max[0] = 290.0e3;
  data->split_face_max[1] = 310.0e3;
  nn = 2;
  ierr = PetscOptionsGetRealArray(NULL, MODEL_NAME_R,"-bc_split_face_min",data->split_face_min,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != 2) {
      SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_USER,"Expected 2 values for -split_face_min. Found %d",nn);
    }
  }
  nn = 2;
  ierr = PetscOptionsGetRealArray(NULL, MODEL_NAME_R,"-bc_split_face_max",data->split_face_max,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != 2) {
      SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_USER,"Expected 2 values for -split_face_max. Found %d",nn);
    }
  }


  /* Replace boolean by int for switch */
  data->bc_type = -1;
  if (bc_nitsche) {
    data->bc_type = 0;
    PetscPrintf(PETSC_COMM_WORLD,"[[ BC TYPE %d: Oblique Navier-slip ACTIVATED ]]\n",data->bc_type);
  } else if (bc_dirichlet) {
    data->bc_type = 1;
    PetscPrintf(PETSC_COMM_WORLD,"[[ BC TYPE %d: Oblique Dirichlet ACTIVATED ]]\n",data->bc_type);
  } else if (bc_freeslip_nitsche) {
    data->bc_type = 2;
    PetscPrintf(PETSC_COMM_WORLD,"[[ BC TYPE %d: Orthogonal Normal Navier-slip ACTIVATED ]]\n",data->bc_type);
  } else if (bc_strikeslip) {
    data->bc_type = 3;
    PetscPrintf(PETSC_COMM_WORLD,"[[ BC TYPE %d: Rotated Strike-Slip Navier-slip ACTIVATED ]]\n",data->bc_type);
  } else if (bc_strike_analogue) {
    data->bc_type = 4;
    PetscPrintf(PETSC_COMM_WORLD,"[[ BC TYPE %d: Analogue model BCs (base driven) ACTIVATED ]]\n",data->bc_type);
  } else if (bc_strike_analogue_nitsche) {
    data->bc_type = 5;
    PetscPrintf(PETSC_COMM_WORLD,"[[ BC TYPE %d: Analogue model BCs (base driven) + Nitsche on vertical sides ACTIVATED ]]\n",data->bc_type);
  } else if (bc_diri_neumann) {
    data->bc_type = 6;
    PetscPrintf(PETSC_COMM_WORLD,"[[ BC TYPE %d: Dirichlet Facets + Neumann ACTIVATED ]]\n",data->bc_type);
  }

  data->u_func_type = 0;
  if (bc_u_func_atan) {
    data->u_func_type = 1;
    PetscPrintf(PETSC_COMM_WORLD,"[[ Using arctangent function for the boundary velocity ]]\n");
  } else if (bc_u_func_mixte) {
    data->u_func_type = 2;
    PetscPrintf(PETSC_COMM_WORLD,"[[ Using arctangent on Dirichlet sides and linear derivative on Navier-slip sides ]]\n");
  }

  PetscFunctionReturn(0);
}

PetscErrorCode ModelInitialize_RiftNitsche(pTatinCtx c,void *ctx)
{
  ModelRiftNitscheCtx  *data;
  RheologyConstants    *rheology;
  DataBucket           materialconstants;
  PetscErrorCode       ierr;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  data = (ModelRiftNitscheCtx*)ctx;
  
  ierr = pTatinGetRheology(c,&rheology);CHKERRQ(ierr);
  ierr = pTatinGetMaterialConstants(c,&materialconstants);CHKERRQ(ierr); 

  /* Set the rheology to visco-plastic temperature dependant */
  rheology->rheology_type = RHEOLOGY_VP_STD;
  /* force energy equation to be introduced */
  ierr = PetscOptionsInsertString(NULL,"-activate_energyfv true");CHKERRQ(ierr);
  
  /* Number of materials */
  data->n_phases = 4;
  rheology->nphases_active = data->n_phases;
  /* Viscosity cutoff */
  ierr = ModelSetViscosityCutoff_RiftNitsche(data);CHKERRQ(ierr);
  /* Initial geometry */
  ierr = ModelInitialGeometry_RiftNitsche(data);CHKERRQ(ierr);
  ierr = ModelSetWeakZoneParameters_RiftNitsche(data);CHKERRQ(ierr);
  /* BCs data */
  ierr = ModelInitialBoundaryVelocity_RiftNitsche(data);CHKERRQ(ierr);
  ierr = ModelSetTemperatureBCs_RiftNitsche(data);CHKERRQ(ierr);
  ierr = ModelSetBCType_RiftNitsche(data);CHKERRQ(ierr);
  /* Materials parameters */
  ierr = ModelSetMaterialParameters_RiftNitsche(c,materialconstants,data);CHKERRQ(ierr);
  ierr = ModelSetSPMParameters_RiftNitsche(data);CHKERRQ(ierr);
  /* Scale parameters */
  ierr = ModelScaleParameters_RiftNitsche(materialconstants,data);CHKERRQ(ierr);
  /* Use scaled values to compute background strain rate for Navier slip BCs */
  ierr = ModelSetGeneralSlipBoundaryValues_RiftNitsche(data);CHKERRQ(ierr);

  /* Fetch scaled values for the viscosity cutoff */
  rheology->apply_viscosity_cutoff_global = data->eta_cutoff;
  rheology->eta_upper_cutoff_global = data->eta_max;
  rheology->eta_lower_cutoff_global = data->eta_min;
  
  data->output_markers       = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_R,"-output_markers",       &data->output_markers,NULL);CHKERRQ(ierr);

  /* passive markers */
  ierr = ModelSetPassiveMarkersSwarmParameters(c,data);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetMeshRefinement_Horizontal(DM dav, ModelRiftNitscheCtx *data)
{
  PetscInt       npoints,n;
  PetscReal      *xref,*xnat_x,*xnat_z,*dx,*dz;
  PetscReal      Lx,Lz;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  npoints = 6;

  ierr = PetscMalloc1(npoints,&xref);CHKERRQ(ierr); 
  ierr = PetscMalloc1(npoints,&xnat_x);CHKERRQ(ierr);
  ierr = PetscMalloc1(npoints,&xnat_z);CHKERRQ(ierr);
  ierr = PetscMalloc1(npoints,&dx);CHKERRQ(ierr);
  ierr = PetscMalloc1(npoints,&dz);CHKERRQ(ierr);

  /* x direction interval in meters */
  dx[0] = 0.0e3; 
  dx[1] = 50.0e3; 
  dx[2] = 75.0e3; 
  dx[3] = 225.0e3; 
  dx[4] = 250.0e3; 
  dx[5] = 300.0e3;
  /* z direction interval in meters */
  dz[0] = 0.0e3; 
  dz[1] = 90.0e3; 
  dz[2] = 120.0e3; 
  dz[3] = 180.0e3; 
  dz[4] = 210.0e3; 
  dz[5] = 300.0e3;
  /* Scale */
  for (n=0; n<npoints; n++) {
    dx[n] = dx[n] / data->length_bar;
    dz[n] = dz[n] / data->length_bar;
  }

  /* Get model length in the x and z directions */
  Lx = data->Lx - data->Ox;
  Lz = data->Lz - data->Oz;
  /* normalize dx */
  for (n=0; n<npoints; n++) {
    xnat_x[n] = dx[n] / Lx;
    xnat_z[n] = dz[n] / Lz;
  }
  /* Refinement in the centre */
  xref[0] = 0.0;
  xref[1] = 0.02;
  xref[2] = 0.1;
  xref[3] = 0.9;
  xref[4] = 0.98;
  xref[5] = 1.0;

  /* x dir refinement */
  ierr = DMDACoordinateRefinementTransferFunction(dav,0,PETSC_TRUE,npoints,xref,xnat_x);CHKERRQ(ierr);
  /* z dir refinement */
  ierr = DMDACoordinateRefinementTransferFunction(dav,2,PETSC_TRUE,npoints,xref,xnat_z);CHKERRQ(ierr);

  ierr = PetscFree(xref);CHKERRQ(ierr);
  ierr = PetscFree(xnat_x);CHKERRQ(ierr);
  ierr = PetscFree(xnat_z);CHKERRQ(ierr);
  ierr = PetscFree(dx);CHKERRQ(ierr);
  ierr = PetscFree(dz);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetMeshRefinement_Vertical(DM dav)
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

static PetscErrorCode ModelSetMeshRefinement_RiftNitsche(DM dav, ModelRiftNitscheCtx *data)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = ModelSetMeshRefinement_Vertical(dav);CHKERRQ(ierr);
  if (data->bc_type == 4) {
    ierr = ModelSetMeshRefinement_Horizontal(dav,data);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyInitialMeshGeometry_RiftNitsche(pTatinCtx c,void *ctx)
{
  ModelRiftNitscheCtx *data = (ModelRiftNitscheCtx*)ctx;
  PhysCompStokes      stokes;
  DM                  stokes_pack,dav,dap;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  ierr = pTatinGetStokesContext(c,&stokes);CHKERRQ(ierr);
  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(dav,data->Ox,data->Lx,data->Oy,data->Ly,data->Oz,data->Lz);CHKERRQ(ierr);
  /* Mesh Refinement */
  ierr = ModelSetMeshRefinement_RiftNitsche(dav,data);CHKERRQ(ierr);
  ierr = DMDABilinearizeQ2Elements(dav);CHKERRQ(ierr);
  /* Gravity */
  PetscReal gvec[] = { 0.0, -9.8, 0.0 };
  ierr = PhysCompStokesSetGravityVector(c->stokes_ctx,gvec);CHKERRQ(ierr);
  ierr = PhysCompStokesScaleGravityVector(c->stokes_ctx,1.0/data->acceleration_bar);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetInitialMaterialLayering_RiftNitsche(MPntStd *material_point, double *position, ModelRiftNitscheCtx *data)
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

static PetscErrorCode ComputeWeakZonesCentreEquallySpaced(PetscReal *notch_centre, PetscInt dim, ModelRiftNitscheCtx *data)
{
  PetscInt i;
  PetscReal L,O;

  PetscFunctionBegin;

  L = 0.0;
  O = 0.0;
  if (dim == 0) {
    L = data->Lx - data->Ox;
    O = data->Ox;
  } else if (dim == 1) {
    L = data->Lz - data->Oz;
    O = data->Oz;
  } else {
    SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_USER,"[[%s]] dim must be 0, or 1 dim = %d was given\n",PETSC_FUNCTION_NAME,dim);
  }
  
  for (i=0; i<data->n_notches; i++) {
    notch_centre[2*i + dim] = 0.5*L/data->n_notches + i*L/data->n_notches + O;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode WeakZonesCentreOffset(PetscReal *notch_centre, PetscInt dim, ModelRiftNitscheCtx *data)
{
  PetscInt i;
  PetscFunctionBegin;

  for (i=0; i<data->n_notches; i++) {
    notch_centre[2*i + dim] = data->wz_origin + data->wz_offset*(-2.0*i/(data->n_notches - 1.0) + 1.0);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode WeakZonesCentreOffsetFromAngle(PetscReal *notch_centre, PetscInt dim, ModelRiftNitscheCtx *data)
{
  PetscInt  i,d;
  PetscReal centre[2];

  PetscFunctionBegin;

  if (dim == 0) {
    d = 1;
    centre[0] = 0.5 * (data->Oz + data->Lz);
    centre[1] = 0.5 * (data->Ox + data->Lx);
  } else if (dim == 1) {
    d = 0;
    centre[0] = 0.5 * (data->Ox + data->Lx);
    centre[1] = 0.5 * (data->Oz + data->Lz);
  } else {
    SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_USER,"dim can only be 0 or 1. Given %d",dim);
  }

  for (i=0; i<data->n_notches; i++) {
    notch_centre[2*i + dim] = PetscTanReal(data->wz_angle) * (notch_centre[2*i + d] - centre[0]) + centre[1];
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode WeakZoneCentreFromOptions(PetscReal *notch_centre, ModelRiftNitscheCtx *data)
{
  PetscInt       i,d,nn;
  PetscBool      found=PETSC_FALSE;
  PetscReal      centre[] = {0.0, 0.0};
  char           *option_name;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  for (i=0; i<data->n_notches; i++) {
    /* Search for option -wz_centrei with i \in n_notches */
    if (asprintf (&option_name, "-wz_centre%d", (int)i) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    nn = 2;
    ierr = PetscOptionsGetRealArray(NULL, MODEL_NAME_R, option_name, centre, &nn, &found);CHKERRQ(ierr);
    if (found) {
      if (nn != 2) {
        SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_USER,"Expected 2 values for -wz_centre%d. Found %d",i,nn);
      }
    } else {
      SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_USER,"In function [[%s]] No options found for weak zone centre coordinates. Use -wz_centre%d x,z to set it.",PETSC_FUNCTION_NAME,i);
    }
    free (option_name);
    /* Set centre coordinates, don't forget the length scalling */
    for (d=0; d<2; d++) { notch_centre[2*i + d] = centre[d] / data->length_bar; }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SetWeakZonesCentreCoordinates(PetscReal *notch_centre, ModelRiftNitscheCtx *data)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  switch(data->wz_centre_type) {
    case 0:
      ierr = ComputeWeakZonesCentreEquallySpaced(notch_centre,0,data);CHKERRQ(ierr);
      ierr = WeakZonesCentreOffset(notch_centre,1,data);CHKERRQ(ierr);
      break;

    case 1:
      ierr = WeakZoneCentreFromOptions(notch_centre, data);CHKERRQ(ierr);
      break;

    case 2:
      ierr = ComputeWeakZonesCentreEquallySpaced(notch_centre,0,data);CHKERRQ(ierr);
      ierr = WeakZonesCentreOffsetFromAngle(notch_centre,1,data);CHKERRQ(ierr);
      break;

    default:
      PetscPrintf(PETSC_COMM_WORLD,"No option set for the weak zone centre type, assuming equally spaced weak zones.\n");
      ierr = ComputeWeakZonesCentreEquallySpaced(notch_centre,0,data);CHKERRQ(ierr);
      ierr = WeakZonesCentreOffset(notch_centre,1,data);CHKERRQ(ierr);
      break;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetInitialWeakZone_Notches(MPntPStokesPl *mpprop_pls, double *position, PetscReal *notch_centre, ModelRiftNitscheCtx *data)
{
  PetscInt  i;
  PetscReal Oz_wz,Lz_wz;
  float     pls;
  short     yield;
  PetscFunctionBegin;

  /* Set an initial small random noise on plastic strain */
  pls = ptatin_RandomNumberGetDouble(0.0,0.03);
  /* Set yield to none */
  yield = 0;
  
  /* z position of the weak zones */
  //Oz_wz = 0.5*(data->Lz + data->Oz - data->wz_width); 
  //Lz_wz = 0.5*(data->Lz + data->Oz + data->wz_width);

  for (i=0; i<data->n_notches; i++) {
    PetscReal Ox_wz,Lx_wz;

    /* x position of the weak zones */
    Ox_wz = notch_centre[2*i    ] - 0.5*data->wz_width;
    Lx_wz = notch_centre[2*i    ] + 0.5*data->wz_width;
    /* z position of the weak zones */
    Oz_wz = notch_centre[2*i + 1] - 0.5*data->wz_width;
    Lz_wz = notch_centre[2*i + 1] + 0.5*data->wz_width;
  
    if (position[1] >= data->y_continent[2]) { // Only in lithospere
      if (position[0] >= Ox_wz && position[0] <= Lx_wz) {
        if (position[2] >= Oz_wz && position[2] <= Lz_wz) {
          /* Set higher initial plastic strain */
          pls = ptatin_RandomNumberGetDouble(0.1,0.8);
        }
      }
    }
  }
  MPntPStokesPlSetField_yield_indicator(mpprop_pls,yield);
  MPntPStokesPlSetField_plastic_strain(mpprop_pls,pls);

  PetscFunctionReturn(0);
}

static PetscReal Gaussian2D(PetscReal A, PetscReal a, PetscReal b, PetscReal c, PetscReal x, PetscReal x0, PetscReal z, PetscReal z0)
{ 
  PetscReal value=0.0;
  value = A * (PetscExpReal( -( a*(x-x0)*(x-x0) + 2*b*(x-x0)*(z-z0) + c*(z-z0)*(z-z0) ) ) );
  return value;
}

static PetscErrorCode ModelSetInitialWeakZone_Gaussians(MPntPStokesPl *mpprop_pls, double *position, PetscReal *notch_centre, ModelRiftNitscheCtx *data)
{
  PetscInt  i;
  PetscReal a,b,c;
  float     pls;
  short     yield;

  PetscFunctionBegin;

  yield = 0;
  /* Gaussian shape parameters */
  a = 0.5*data->wz_sigma[0]*data->wz_sigma[0];
  b = 0.0;
  c = 0.5*data->wz_sigma[1]*data->wz_sigma[1];
  /* Background plastic strain */
  pls = ptatin_RandomNumberGetDouble(0.0,0.03);
  if (position[1] >= data->y_continent[2]) { // lithospere only
    for (i=0; i<data->n_notches; i++) {
      pls += Gaussian2D(ptatin_RandomNumberGetDouble(0.0,1.0),a,b,c,position[0],notch_centre[2*i],position[2],notch_centre[2*i+1]);
    }
  }
  MPntPStokesPlSetField_yield_indicator(mpprop_pls,yield);
  MPntPStokesPlSetField_plastic_strain(mpprop_pls,pls);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetInitialWeakZoneDouble(MPntPStokesPl *mpprop_pls, double *position, ModelRiftNitscheCtx *data)
{
  PetscReal      coords_rt[] = {0.0,0.0,0.0};
  PetscReal      zcentre;
  float          pls;
  short          yield;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  /* Set an initial small random noise on plastic strain */
  pls = ptatin_RandomNumberGetDouble(0.0,0.03);
  /* Set yield to none */
  yield = 0;

  zcentre = 0.5*(data->Lz - data->Oz);

  if (position[1] >= data->y_continent[2]) {
    /* Rotate coordinate system */
    ierr = RotateReferential(position,coords_rt,PETSC_TRUE,data);CHKERRQ(ierr);
    /* x < 2*width of the wz in the rotated system */
    if (coords_rt[0] <= data->Ox + 2.0*data->wz_width) {
      if (coords_rt[2] >= zcentre - 0.5*(data->wz_offset + data->wz_width) && coords_rt[2] <= zcentre + 0.5*(data->wz_width - data->wz_offset)) {
        /* Set higher initial plastic strain */
        pls = ptatin_RandomNumberGetDouble(0.1,0.8);
      }
    }
  
    if (coords_rt[0] >= data->Lx - 2.0*data->wz_width) {
      if (coords_rt[2] >= zcentre + 0.5*(data->wz_offset - data->wz_width) && coords_rt[2] <= zcentre + 0.5*(data->wz_offset + data->wz_width)) {
        /* Set higher initial plastic strain */
        pls = ptatin_RandomNumberGetDouble(0.1,0.8);
      }
    }
  }
  MPntPStokesPlSetField_yield_indicator(mpprop_pls,yield);
  MPntPStokesPlSetField_plastic_strain(mpprop_pls,pls);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetInitialWeakZone_Oblique(MPntPStokesPl *mpprop_pls, double *position, ModelRiftNitscheCtx *data)
{
  PetscReal xcentre,zcentre,buffer,a=0.0,b=0.0;
  float     pls;
  short     yield;
  PetscFunctionBegin;

  /* Set an initial small random noise on plastic strain */
  pls = ptatin_RandomNumberGetDouble(0.0,0.03);
  /* Set yield to none */
  yield = 0;

  xcentre = 0.5*(data->Lx - data->Ox);
  zcentre = 0.5*(data->Lz - data->Oz);
  /* width in which the plastic strain attenuates on both sides of the WZ */
  buffer = 0.75*data->wz_width;

  if (position[1] >= data->y_continent[2]) {
    if ( (position[0] >= (zcentre - position[2])*PetscTanReal(data->wz_angle) + (xcentre - 0.5*data->wz_width/PetscCosReal(data->wz_angle))) &&
         (position[0] <= (zcentre - position[2])*PetscTanReal(data->wz_angle) + (xcentre + 0.5*data->wz_width/PetscCosReal(data->wz_angle))) ) {
      
      pls = ptatin_RandomNumberGetDouble(0.1,0.8);
    }

    if ( (position[0] >= (zcentre - position[2])*PetscTanReal(data->wz_angle) + (xcentre - 0.5*data->wz_width/PetscCosReal(data->wz_angle) - buffer/PetscCosReal(data->wz_angle))) &&
         (position[0] <= (zcentre - position[2])*PetscTanReal(data->wz_angle) + (xcentre - 0.5*data->wz_width/PetscCosReal(data->wz_angle))) ) {
      
      a = (zcentre - position[2])*PetscTanReal(data->wz_angle) + (xcentre - 0.5*data->wz_width/PetscCosReal(data->wz_angle));
      b = (zcentre - position[2])*PetscTanReal(data->wz_angle) + (xcentre - 0.5*data->wz_width/PetscCosReal(data->wz_angle) - buffer/PetscCosReal(data->wz_angle));
      pls = ptatin_RandomNumberGetDouble(0.1,0.8) * PetscAtanReal( (position[0] - b)/(a - b)*(0.5*M_PI) );
    }

    if ( (position[0] <= (zcentre - position[2])*PetscTanReal(data->wz_angle) + (xcentre + 0.5*data->wz_width/PetscCosReal(data->wz_angle) + buffer/PetscCosReal(data->wz_angle))) && 
         (position[0] >= (zcentre - position[2])*PetscTanReal(data->wz_angle) + (xcentre + 0.5*data->wz_width/PetscCosReal(data->wz_angle))) ) {
      
      a = (zcentre - position[2])*PetscTanReal(data->wz_angle) + (xcentre + 0.5*data->wz_width/PetscCosReal(data->wz_angle));
      b = (zcentre - position[2])*PetscTanReal(data->wz_angle) + (xcentre + 0.5*data->wz_width/PetscCosReal(data->wz_angle) + buffer/PetscCosReal(data->wz_angle));
      pls = ptatin_RandomNumberGetDouble(0.1,0.8) * PetscAtanReal( (position[0] - b)/(a - b)*(0.5*M_PI) );
    }
  }

  MPntPStokesPlSetField_yield_indicator(mpprop_pls,yield);
  MPntPStokesPlSetField_plastic_strain(mpprop_pls,pls);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetInitialWeakZone_ObliqueGaussian(MPntPStokesPl *mpprop_pls, double *position, ModelRiftNitscheCtx *data)
{
  PetscReal xcentre,zcentre,location;
  PetscReal a,b,c;
  float     pls;
  short     yield;
  PetscFunctionBegin;

  pls = ptatin_RandomNumberGetDouble(0.0,0.03);
  yield = 0;

  if (position[1] >= data->y_continent[2]) {
    xcentre = 0.5*(data->Lx - data->Ox);
    zcentre = 0.5*(data->Lz - data->Oz);
    location = (zcentre - position[2])*PetscTanReal(data->wz_angle) + xcentre;
  
    a = 0.5*data->wz_sigma[0]*data->wz_sigma[0];
    b = 0.0;
    c = 0.0;
    pls += Gaussian2D(ptatin_RandomNumberGetDouble(0.0,1.0),a,b,c,position[0],location,0.0,0.0);
  }
  MPntPStokesPlSetField_yield_indicator(mpprop_pls,yield);
  MPntPStokesPlSetField_plastic_strain(mpprop_pls,pls);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetInitialWeakZone_StraightGaussian(MPntPStokesPl *mpprop_pls, double *position, ModelRiftNitscheCtx *data)
{
  PetscReal zcentre;
  PetscReal a,b,c;
  float     pls;
  short     yield;
  PetscFunctionBegin;

  pls = ptatin_RandomNumberGetDouble(0.0,0.03);
  yield = 0;

  if (position[1] >= data->y_continent[2]) {
    zcentre = 0.5*(data->Lz - data->Oz);
  
    a = 0.0;
    b = 0.0;
    c = 0.5*data->wz_sigma[1]*data->wz_sigma[1];
    pls += Gaussian2D(ptatin_RandomNumberGetDouble(0.0,1.0),a,b,c,0.0,0.0,position[2],zcentre);
  }
  MPntPStokesPlSetField_yield_indicator(mpprop_pls,yield);
  MPntPStokesPlSetField_plastic_strain(mpprop_pls,pls);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetInitialWeakZoneGeometry_RiftNitsche(MPntPStokesPl *mpprop_pls, double *position, PetscReal *notch_centre, ModelRiftNitscheCtx *data)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  switch (data->wz_type) {
    case 0:
      ierr = ModelSetInitialWeakZone_Notches(mpprop_pls,position,notch_centre,data);CHKERRQ(ierr);
      break;

    case 1:
      ierr = ModelSetInitialWeakZone_Gaussians(mpprop_pls,position,notch_centre,data);CHKERRQ(ierr);
      break;

    case 2:
      ierr = ModelSetInitialWeakZone_Oblique(mpprop_pls,position,data);CHKERRQ(ierr);
      break;

    case 3:
      ierr = ModelSetInitialWeakZoneDouble(mpprop_pls,position,data);CHKERRQ(ierr);
      break;

    case 4:
      ierr = ModelSetInitialWeakZone_ObliqueGaussian(mpprop_pls,position,data);CHKERRQ(ierr);
      break;

    case 5:
      ierr = ModelSetInitialWeakZone_StraightGaussian(mpprop_pls,position,data);CHKERRQ(ierr);
      break;

    default:
      MPntPStokesPlSetField_yield_indicator(mpprop_pls,0);
      MPntPStokesPlSetField_plastic_strain(mpprop_pls, ptatin_RandomNumberGetDouble(0.0,0.03) );
      break;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyInitialMaterialGeometry_RiftNitsche(pTatinCtx c,ModelRiftNitscheCtx *data)
{
  DataBucket     db;
  DataField      PField_std,PField_pls;
  PetscReal      *notch_centre;
  int            n_mp_points,p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  ierr = pTatinGetMaterialPoints(c,&db,NULL);CHKERRQ(ierr);
  /* std variables */
  DataBucketGetDataFieldByName(db,MPntStd_classname,&PField_std);
  DataFieldGetAccess(PField_std);
  DataFieldVerifyAccess(PField_std,sizeof(MPntStd));
  /* Plastic strain variables */
  DataBucketGetDataFieldByName(db,MPntPStokesPl_classname,&PField_pls);
  DataFieldGetAccess(PField_pls);
  DataFieldVerifyAccess(PField_pls,sizeof(MPntPStokesPl));

  /* Weak zones centre coordinates: 0 ==> x
                                    1 ==> z */
  ierr = PetscMalloc1(2*data->n_notches,&notch_centre);CHKERRQ(ierr);
  ierr = SetWeakZonesCentreCoordinates(notch_centre,data);CHKERRQ(ierr);

  PetscPrintf(PETSC_COMM_WORLD,"************ Weak Zones Centre Coordinates ************\n",NULL);
  for (int II=0; II<data->n_notches; II++) {
    PetscPrintf(PETSC_COMM_WORLD,"notch_centre[2*%d + 0] = %f, notch_centre[2*%d + 1] = %f\n",II,notch_centre[2*II + 0],II,notch_centre[2*II + 1]);
  }

  DataBucketGetSizes(db,&n_mp_points,0,0);
  for (p=0; p<n_mp_points; p++) {
    MPntStd       *material_point;
    MPntPStokesPl *mpprop_pls;
    double        *position;

    DataFieldAccessPoint(PField_std,p,(void**)&material_point);
    DataFieldAccessPoint(PField_pls,p,(void**)&mpprop_pls);

    /* Access coordinates of the marker */
    MPntStdGetField_global_coord(material_point,&position);

    /* Layering geometry */
    ierr = ModelSetInitialMaterialLayering_RiftNitsche(material_point,position,data);CHKERRQ(ierr);
    /* Weak zone geometry */
    ierr = ModelSetInitialWeakZoneGeometry_RiftNitsche(mpprop_pls,position,notch_centre,data);CHKERRQ(ierr);
  }

  ierr = PetscFree(notch_centre);CHKERRQ(ierr);
  
  DataFieldRestoreAccess(PField_std);
  DataFieldRestoreAccess(PField_pls);

  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyInitialMaterialParameters_RiftNitsche(pTatinCtx c,void *ctx)
{
  ModelRiftNitscheCtx *data = (ModelRiftNitscheCtx*)ctx;
  PetscErrorCode      ierr;

  PetscFunctionBegin;

  ierr = ModelApplyInitialMaterialGeometry_RiftNitsche(c,data);CHKERRQ(ierr);
  /* Passive markers */
  ierr = PSwarmSetUp(data->pswarm);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyInitialStokesVariableMarkers_RiftNitsche(pTatinCtx c,Vec X,void *ctx)
{
  DM                         stokes_pack,dau,dap;
  PhysCompStokes             stokes;
  Vec                        Uloc,Ploc;
  PetscScalar                *LA_Uloc,*LA_Ploc;
  DataField                  PField;
  PetscErrorCode             ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);
  
  DataBucketGetDataFieldByName(c->material_constants,MaterialConst_MaterialType_classname,&PField);
  ierr = pTatinGetStokesContext(c,&stokes);CHKERRQ(ierr);
  stokes_pack = stokes->stokes_pack;

  ierr = DMCompositeGetEntries(stokes_pack,&dau,&dap);CHKERRQ(ierr);
  ierr = DMCompositeGetLocalVectors(stokes_pack,&Uloc,&Ploc);CHKERRQ(ierr);

  ierr = DMCompositeScatter(stokes_pack,X,Uloc,Ploc);CHKERRQ(ierr);
  ierr = VecGetArray(Uloc,&LA_Uloc);CHKERRQ(ierr);
  ierr = VecGetArray(Ploc,&LA_Ploc);CHKERRQ(ierr);
  ierr = pTatin_EvaluateRheologyNonlinearities(c,dau,LA_Uloc,dap,LA_Ploc);CHKERRQ(ierr);
  ierr = VecRestoreArray(Uloc,&LA_Uloc);CHKERRQ(ierr);
  ierr = VecRestoreArray(Ploc,&LA_Ploc);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelLoadTemperatureInitialSolution_FromFile(pTatinCtx c)
{
  PhysCompEnergyFV energy;
  PetscBool        flg = PETSC_FALSE,temperature_ic_from_file = PETSC_FALSE;
  char             fname[PETSC_MAX_PATH_LEN],temperature_file[PETSC_MAX_PATH_LEN];
  PetscViewer      viewer;
  PetscErrorCode   ierr;

  PetscFunctionBegin;

  ierr = pTatinGetContext_EnergyFV(c,&energy);CHKERRQ(ierr);
  /* If job is restarted skip that part (Temperature is loaded from checkpointed file) */
  if (!c->restart_from_file) {
    ierr = PetscOptionsGetBool(NULL,MODEL_NAME_R,"-temperature_ic_from_file",&temperature_ic_from_file,NULL);CHKERRQ(ierr);
    if (temperature_ic_from_file) {
      /* Check if a file is provided */
      ierr = PetscOptionsGetString(NULL,MODEL_NAME_R,"-temperature_file",temperature_file,PETSC_MAX_PATH_LEN-1,&flg);CHKERRQ(ierr);
      if (flg) {
        ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,temperature_file,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
      } else {
        PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s/temperature_steady.pbvec",c->outputpath);
        ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,fname,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
      }
      ierr = VecLoad(energy->T,viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Providing a temperature file for initial state is required\n");
    }
  }
  PetscFunctionReturn(0);
}

static PetscBool InitialAnalyticalVelocityFunction(PetscScalar position[],PetscScalar *val,void *ctx)
{
  ModelRiftNitscheCtx *data = (ModelRiftNitscheCtx*)ctx;
  PetscReal           u[3];
  PetscBool           impose=PETSC_TRUE;
  PetscErrorCode      ierr;

  PetscFunctionBegin;

  ierr = ModelAnalyticalVelocityFunction(position,u,data);CHKERRQ(ierr);

  *val = u[ data->component ];
  PetscFunctionReturn(impose);
}

static PetscBool BCListEvaluator_RotatedVelocityField(PetscScalar position[], PetscScalar *value, void *ctx)
{
  ModelRiftNitscheCtx *data = (ModelRiftNitscheCtx*)ctx;
  PetscReal           coords_rt[] = {0.0,0.0,0.0};
  PetscReal           u_xr[] = {0.0,0.0,0.0};
  PetscReal           r[] = {0.0,1.0,0.0};
  PetscReal           u_R[] = {0.0,0.0,0.0};
  PetscBool           impose=PETSC_TRUE;
  PetscErrorCode      ierr;

  PetscFunctionBegin;

  ierr = RotateReferential(position,coords_rt,PETSC_TRUE,data);CHKERRQ(ierr);
  ierr = ModelAnalyticalVelocityFunction(coords_rt,u_xr,data);CHKERRQ(ierr);
  Rotate_u(data->alpha_r,r,u_xr,u_R);
  *value = u_R[ data->component ];

  PetscFunctionReturn(impose);
}

static PetscErrorCode ModelApplyInitialVelocityField_ObliqueExtension(DM dau, Vec velocity,ModelRiftNitscheCtx *data)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  /* Initialize to zero the velocity vector */
  ierr = VecZeroEntries(velocity);CHKERRQ(ierr);

  /* x component */
  data->component = 0;
  ierr = DMDAVecTraverse3d(dau,velocity,data->component,InitialAnalyticalVelocityFunction,(void*)data);CHKERRQ(ierr);
  /* y component */
  data->component = 1;
  ierr = DMDAVecTraverse3d(dau,velocity,data->component,InitialAnalyticalVelocityFunction,(void*)data);CHKERRQ(ierr);
  /* z component */
  data->component = 2;
  ierr = DMDAVecTraverse3d(dau,velocity,data->component,InitialAnalyticalVelocityFunction,(void*)data);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyInitialVelocityField_RotatedVelocityField(DM dau, Vec velocity,ModelRiftNitscheCtx *data)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  /* Initialize to zero the velocity vector */
  ierr = VecZeroEntries(velocity);CHKERRQ(ierr);

  /* x component */
  data->component = 0;
  ierr = DMDAVecTraverse3d(dau,velocity,data->component,BCListEvaluator_RotatedVelocityField,(void*)data);CHKERRQ(ierr);
  /* y component */
  data->component = 1;
  ierr = DMDAVecTraverse3d(dau,velocity,data->component,BCListEvaluator_RotatedVelocityField,(void*)data);CHKERRQ(ierr);
  /* z component */
  data->component = 2;
  ierr = DMDAVecTraverse3d(dau,velocity,data->component,BCListEvaluator_RotatedVelocityField,(void*)data);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyInitialVelocityField_RiftNitsche(DM dau, Vec velocity,ModelRiftNitscheCtx *data)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  switch (data->bc_type)
  {
    case 0:
      ierr = ModelApplyInitialVelocityField_ObliqueExtension(dau,velocity,data);CHKERRQ(ierr);
      break;

    case 1:
      ierr = ModelApplyInitialVelocityField_ObliqueExtension(dau,velocity,data);CHKERRQ(ierr);
      break;

    case 2:
      ierr = ModelApplyInitialVelocityField_ObliqueExtension(dau,velocity,data);CHKERRQ(ierr);
      break;

    case 3:
      ierr = ModelApplyInitialVelocityField_RotatedVelocityField(dau,velocity,data);CHKERRQ(ierr);
      break;

    case 4:
      ierr = ModelApplyInitialVelocityField_RotatedVelocityField(dau,velocity,data);CHKERRQ(ierr);
      break;

    case 5:
      ierr = ModelApplyInitialVelocityField_RotatedVelocityField(dau,velocity,data);CHKERRQ(ierr);
      break;

    case 6:
      ierr = VecZeroEntries(velocity);CHKERRQ(ierr);
      break;

    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"No velocity boundary conditions type was given. Use one of the bc options\n");
      break;
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyInitialHydrostaticPressureField_RiftNitsche(pTatinCtx c, DM dau, DM dap, Vec pressure, ModelRiftNitscheCtx *data)
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

PetscErrorCode ModelApplyInitialSolution_RiftNitsche(pTatinCtx c,Vec X,void *ctx)
{
  ModelRiftNitscheCtx                          *data;
  DM                                           stokes_pack,dau,dap;
  Vec                                          velocity,pressure;
  PetscBool                                    active_energy;
  PetscErrorCode                               ierr;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  data = (ModelRiftNitscheCtx*)ctx;
  
  /* Access velocity and pressure vectors */
  stokes_pack = c->stokes_ctx->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dau,&dap);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);
  /* Velocity IC */
  ierr = ModelApplyInitialVelocityField_RiftNitsche(dau,velocity,data);CHKERRQ(ierr);
  /* Pressure IC */
  ierr = ModelApplyInitialHydrostaticPressureField_RiftNitsche(c,dau,dap,pressure,data);CHKERRQ(ierr);
  /* Attach solution vector (u, p) to passive markers */
  ierr = PSwarmAttachStateVecVelocityPressure(data->pswarm,X);CHKERRQ(ierr);
  /* Restore velocity and pressure vectors */
  ierr = DMCompositeRestoreAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);
  /* Temperature IC */
  ierr = pTatinContextValid_EnergyFV(c,&active_energy);CHKERRQ(ierr);
  if (active_energy) {
    ierr = ModelLoadTemperatureInitialSolution_FromFile(c);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode StrainRateBoundaryValue_Atan(PetscReal E[][3], PetscReal position[], ModelRiftNitscheCtx *data)
{
  PetscFunctionBegin;
  // Exx
  E[0][0] = 0.0;
  // Eyy
  E[1][1] = 0.0;
  // Ezz
  E[2][2] = (2.0*data->u_bc[2]*data->atan_sharpness) / ( M_PI * ( 1.0 + pow(data->atan_sharpness,2.0)*pow(position[2]-data->atan_offset,2.0) ) );
  // Exy
  E[0][1] = 0.0;
  // Exz
  E[0][2] = (data->u_bc[0]*data->atan_sharpness) / ( M_PI * ( 1.0 + pow(data->atan_sharpness,2.0)*pow(position[2]-data->atan_offset,2.0) ) );
  // Eyz
  E[1][2] = 0.0;

  E[1][0] = E[0][1];
  E[2][0] = E[0][2];
  E[2][1] = E[1][2];
  PetscFunctionReturn(0);
}

static PetscErrorCode RotateStrainRateBoundaryValue_Atan(PetscReal E_R[][3], PetscReal position[], ModelRiftNitscheCtx *data)
{
  PetscInt       i,j;
  PetscReal      r[] = {0.0,1.0,0.0};
  PetscReal      R[3][3],R_transpose[3][3],E[3][3],ERT[3][3];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = StrainRateBoundaryValue_Atan(E,position,data);CHKERRQ(ierr);
  RotationMatrix(data->alpha_r,r,R);

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
  
  PetscFunctionReturn(0);
}

static PetscErrorCode GeneralNavierSlipBC_Atan(Facet F,const PetscReal qp_coor[],
                                               PetscReal n_hat[],
                                               PetscReal t1_hat[],
                                               PetscReal epsS[],
                                               PetscReal H[],
                                               void *data)
{
  ModelRiftNitscheCtx *model_data = (ModelRiftNitscheCtx*)data;
  PetscInt            i,j;
  const PetscInt      indices_voigt[][3] = { {0, 3, 4}, {3, 1, 5}, {4, 5, 2} };
  PetscReal           E[3][3];
  PetscErrorCode      ierr;
  PetscFunctionBegin;

  switch (model_data->bc_type) {
    case 0:
      ierr = StrainRateBoundaryValue_Atan(E,qp_coor,model_data);CHKERRQ(ierr);
      break;

    case 1:
      ierr = StrainRateBoundaryValue_Atan(E,qp_coor,model_data);CHKERRQ(ierr);
      break;

    case 2:
      ierr = StrainRateBoundaryValue_Atan(E,qp_coor,model_data);CHKERRQ(ierr);
      break;

    case 3:
      ierr = RotateStrainRateBoundaryValue_Atan(E,qp_coor,model_data);CHKERRQ(ierr);
      break;

    case 4:
      break;

    case 5:
      ierr = RotateStrainRateBoundaryValue_Atan(E,qp_coor,model_data);CHKERRQ(ierr);
      break;

    default:
      ierr = StrainRateBoundaryValue_Atan(E,qp_coor,model_data);CHKERRQ(ierr);
      break;
  }

  for (i=0;i<3;i++) {
    for (j=0;j<3;j++) {
      epsS[ indices_voigt[i][j] ] = E[i][j];
    }
  }
  /* Fill the H tensor */
  for (i=0;i<6;i++) {
    H[i] = model_data->H[i];
  }
  /* Fill the arbitrary normal and one tangent vectors */
  for (j=0;j<3;j++) {
    t1_hat[j] = model_data->t1_hat[j];
    n_hat[j] = model_data->n_hat[j];
  }
  
  PetscFunctionReturn(0);
}


static PetscErrorCode GeneralNavierSlipBC(Facet F,const PetscReal qp_coor[],
                                          PetscReal n_hat[],
                                          PetscReal t1_hat[],
                                          PetscReal epsS[],
                                          PetscReal H[],
                                          void *data)
{
  ModelRiftNitscheCtx *model_data = (ModelRiftNitscheCtx*)data;
  PetscInt            i,j;
  PetscErrorCode      ierr;

  PetscFunctionBegin;

  ierr = ModelRotateStrainRateBoundaryValue(data);CHKERRQ(ierr);

  /* Fill the H tensor and the epsilon_s tensor */
  for (i=0;i<6;i++) {
    epsS[i] = model_data->epsilon_s[i];
    H[i] = model_data->H[i];
  }
  /* Fill the arbitrary normal and one tangent vectors */
  for (j=0;j<3;j++) {
    t1_hat[j] = model_data->t1_hat[j];
    n_hat[j] = model_data->n_hat[j];
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ConstantUdotN_NormalNavierSlip(Facet F,const PetscReal qp_coor[],PetscReal udotn[],void *data)
{
  PetscReal *input = (PetscReal*)data;
  udotn[0] = input[0];
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyGeneralNavierSlip_RiftNitsche(SurfBCList surflist,PetscBool insert_if_not_found,ModelRiftNitscheCtx *data)
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
  
  /* Apply on faces of normal x */
  {
    PetscInt       nsides;
    HexElementFace sides[] = {HEX_FACE_Nxi, HEX_FACE_Pxi}; //{ HEX_FACE_Nzeta, HEX_FACE_Pzeta };
    nsides = sizeof(sides) / sizeof(HexElementFace);
    ierr = MeshFacetMarkDomainFaces(facets,sc->fi,nsides,sides);CHKERRQ(ierr);
  }

  switch(data->u_func_type) {
    case 0:
      {
        SURFC_CHKSETVALS(SC_NITSCHE_GENERAL_SLIP,GeneralNavierSlipBC);
        if (!sc->dm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Missing sc->dm. Must call SurfaceConstraintSetDM() first");
        if (!sc->quadrature) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Missing sc->surfQ. Must call SurfaceConstraintSetQuadrature() first");
        if (!sc->facets->set_values_called) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Facets have not been selected");
        ierr = SurfaceConstraintSetValuesStrainRate_NITSCHE_GENERAL_SLIP(sc,(SurfCSetValuesNitscheGeneralSlip)GeneralNavierSlipBC,(void*)data);CHKERRQ(ierr);
      }
      break;

    case 1:
      {
        SURFC_CHKSETVALS(SC_NITSCHE_GENERAL_SLIP,GeneralNavierSlipBC_Atan);
        if (!sc->dm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Missing sc->dm. Must call SurfaceConstraintSetDM() first");
        if (!sc->quadrature) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Missing sc->surfQ. Must call SurfaceConstraintSetQuadrature() first");
        if (!sc->facets->set_values_called) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Facets have not been selected");
        ierr = SurfaceConstraintSetValuesStrainRate_NITSCHE_GENERAL_SLIP(sc,(SurfCSetValuesNitscheGeneralSlip)GeneralNavierSlipBC_Atan,(void*)data);CHKERRQ(ierr);
      }
      break;

    case 2:
      {
        SURFC_CHKSETVALS(SC_NITSCHE_GENERAL_SLIP,GeneralNavierSlipBC);
        if (!sc->dm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Missing sc->dm. Must call SurfaceConstraintSetDM() first");
        if (!sc->quadrature) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Missing sc->surfQ. Must call SurfaceConstraintSetQuadrature() first");
        if (!sc->facets->set_values_called) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Facets have not been selected");
        ierr = SurfaceConstraintSetValuesStrainRate_NITSCHE_GENERAL_SLIP(sc,(SurfCSetValuesNitscheGeneralSlip)GeneralNavierSlipBC,(void*)data);CHKERRQ(ierr);
      }
      break;

    default:
      {
        SURFC_CHKSETVALS(SC_NITSCHE_GENERAL_SLIP,GeneralNavierSlipBC);
        if (!sc->dm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Missing sc->dm. Must call SurfaceConstraintSetDM() first");
        if (!sc->quadrature) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Missing sc->surfQ. Must call SurfaceConstraintSetQuadrature() first");
        if (!sc->facets->set_values_called) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Facets have not been selected");
        ierr = SurfaceConstraintSetValuesStrainRate_NITSCHE_GENERAL_SLIP(sc,(SurfCSetValuesNitscheGeneralSlip)GeneralNavierSlipBC,(void*)data);CHKERRQ(ierr);
      }
      break;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyNormalNavierSlip_RiftNitsche(SurfBCList surflist,PetscBool insert_if_not_found,ModelRiftNitscheCtx *data)
{
  SurfaceConstraint sc;
  MeshEntity        facets;
  PetscErrorCode    ierr;
  
  ierr = SurfBCListGetConstraint(surflist,"boundary_x",&sc);CHKERRQ(ierr);
  if (!sc) {
    if (insert_if_not_found) {
      ierr = SurfBCListAddConstraint(surflist,"boundary_x",&sc);CHKERRQ(ierr);
      ierr = SurfaceConstraintSetType(sc,SC_NITSCHE_NAVIER_SLIP);CHKERRQ(ierr);
      ierr = SurfaceConstraintNitscheNavierSlip_SetPenalty(sc,1.0e3);CHKERRQ(ierr);
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Surface constraint not found");
  }
  ierr = SurfaceConstraintGetFacets(sc,&facets);CHKERRQ(ierr);
  
  {
    PetscInt       nsides;
    HexElementFace sides[] = { HEX_FACE_Nxi, HEX_FACE_Pxi };//, HEX_FACE_Neta, HEX_FACE_Nzeta, HEX_FACE_Pzeta };
    nsides = sizeof(sides) / sizeof(HexElementFace);
    ierr = MeshFacetMarkDomainFaces(facets,sc->fi,nsides,sides);CHKERRQ(ierr);
  }
  
  SURFC_CHKSETVALS(SC_NITSCHE_NAVIER_SLIP,ConstantUdotN_NormalNavierSlip);
  {
    PetscReal uD_c[] = {0.0};
    ierr = SurfaceConstraintSetValues(sc,(SurfCSetValuesGeneric)ConstantUdotN_NormalNavierSlip,(void*)uD_c);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelComputeBottomFlow_udotn(pTatinCtx c,Vec X, ModelRiftNitscheCtx *data)
{
  PhysCompStokes stokes;
  DM             dms;
  Vec            velocity,pressure;
  PetscReal      int_u_dot_n[HEX_EDGES];
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = pTatinGetStokesContext(c,&stokes);CHKERRQ(ierr);
  ierr = PhysCompStokesGetDMComposite(stokes,&dms);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(dms,X,&velocity,&pressure);CHKERRQ(ierr);  
  
  ierr = StokesComputeVdotN(stokes,velocity,int_u_dot_n);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"imin: %+1.4e\n",int_u_dot_n[ WEST_FACE  -1]);
  PetscPrintf(PETSC_COMM_WORLD,"imax: %+1.4e\n",int_u_dot_n[ EAST_FACE  -1]);
  PetscPrintf(PETSC_COMM_WORLD,"jmin: %+1.4e\n",int_u_dot_n[ SOUTH_FACE -1]);
  PetscPrintf(PETSC_COMM_WORLD,"jmax: [free surface] %+1.4e\n",int_u_dot_n[ NORTH_FACE -1]);
  PetscPrintf(PETSC_COMM_WORLD,"kmin: %+1.4e\n",int_u_dot_n[ BACK_FACE  -1]);
  PetscPrintf(PETSC_COMM_WORLD,"kmax: %+1.4e\n",int_u_dot_n[ FRONT_FACE -1]);

  ierr = DMCompositeRestoreAccess(dms,X,&velocity,&pressure);CHKERRQ(ierr);

  if (c->step == 0) {
    data->u_bc[1] = 2.0*data->u_bc[2]*(data->Lx - data->Ox)*(data->Ly - data->Oy)/((data->Lx - data->Ox)*(data->Lz - data->Oz));
  } else {
    /* Compute the vy velocity based on faces inflow/outflow except the top free surface */
    data->u_bc[1] = (int_u_dot_n[WEST_FACE-1]+int_u_dot_n[EAST_FACE-1]+int_u_dot_n[BACK_FACE-1]+int_u_dot_n[FRONT_FACE-1])/((data->Lx - data->Ox)*(data->Lz - data->Oz));
    PetscPrintf(PETSC_COMM_WORLD,"Vy = %+1.4e\n",data->u_bc[1]);    
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyObliqueExtensionPullApart_RiftNitsche(DM dav, BCList bclist,SurfBCList surflist,PetscBool insert_if_not_found,ModelRiftNitscheCtx *data)
{
  PetscReal      ux,uz,u_bot;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* Apply Oblique extension on IMAX and IMIN faces */
  ux = -data->u_bc[0];
  uz = -data->u_bc[2];
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,0,BCListEvaluator_constant,(void*)&ux);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,2,BCListEvaluator_constant,(void*)&uz);CHKERRQ(ierr);

  ux = data->u_bc[0];
  uz = data->u_bc[2];
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMAX_LOC,0,BCListEvaluator_constant,(void*)&ux);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMAX_LOC,2,BCListEvaluator_constant,(void*)&uz);CHKERRQ(ierr);

  /* Apply General Navier Slip BC on IMAX and IMIN faces */
  ierr = ModelApplyGeneralNavierSlip_RiftNitsche(surflist,insert_if_not_found,data);CHKERRQ(ierr);

  /* Apply base velocity from u.n */
  u_bot = data->u_bc[1];
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,1,BCListEvaluator_constant,(void*)&u_bot);CHKERRQ(ierr);
  //data->component = 1;
  //ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,data->component,InitialAnalyticalVelocityFunction,(void*)data);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyObliqueExtensionPullApartDirichlet_RiftNitsche(DM dav, BCList bclist,ModelRiftNitscheCtx *data)
{
  PetscReal      ux,uz,u_bot;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* Apply Oblique extension on IMAX and IMIN faces */
  ux = -data->u_bc[0];
  uz = -data->u_bc[2];
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,0,BCListEvaluator_constant,(void*)&ux);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,2,BCListEvaluator_constant,(void*)&uz);CHKERRQ(ierr);

  ux = data->u_bc[0];
  uz = data->u_bc[2];
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMAX_LOC,0,BCListEvaluator_constant,(void*)&ux);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMAX_LOC,2,BCListEvaluator_constant,(void*)&uz);CHKERRQ(ierr);

  /* Apply the linear analytical function on IMAX and IMIN faces */
  data->component = 0;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,data->component,InitialAnalyticalVelocityFunction,(void*)data);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,data->component,InitialAnalyticalVelocityFunction,(void*)data);CHKERRQ(ierr);
  data->component = 2;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,data->component,InitialAnalyticalVelocityFunction,(void*)data);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,data->component,InitialAnalyticalVelocityFunction,(void*)data);CHKERRQ(ierr);

  /* Apply base velocity from \int_{\partial \Omega} u.n */
  u_bot = data->u_bc[1];
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,1,BCListEvaluator_constant,(void*)&u_bot);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyOrthogonalExtensionFreeSlipNitsche_RiftNitsche(DM dav, BCList bclist,SurfBCList surflist,PetscBool insert_if_not_found,ModelRiftNitscheCtx *data)
{
  PetscReal      uz,u_bot;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* Extension on faces of normal z */
  uz = -data->norm_u;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,2,BCListEvaluator_constant,(void*)&uz);CHKERRQ(ierr);
  uz = data->norm_u;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMAX_LOC,2,BCListEvaluator_constant,(void*)&uz);CHKERRQ(ierr);

  /* Navier slip u.n = 0 on faces of normal x */
  ierr = ModelApplyNormalNavierSlip_RiftNitsche(surflist,PETSC_TRUE,data);CHKERRQ(ierr);

  /* Apply base velocity from \int_{\partial \Omega} u.n */
  u_bot = data->u_bc[1];
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,1,BCListEvaluator_constant,(void*)&u_bot);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyRotatedStrikeSlipNavierSlip_RiftNitsche(DM dav, BCList bclist,SurfBCList surflist,PetscBool insert_if_not_found,ModelRiftNitscheCtx *data)
{
  PetscReal      u_bot;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  data->component = 0;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,data->component,BCListEvaluator_RotatedVelocityField,(void*)data);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMAX_LOC,data->component,BCListEvaluator_RotatedVelocityField,(void*)data);CHKERRQ(ierr);
  data->component = 2;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,data->component,BCListEvaluator_RotatedVelocityField,(void*)data);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMAX_LOC,data->component,BCListEvaluator_RotatedVelocityField,(void*)data);CHKERRQ(ierr);

  /* Apply General Navier Slip BC on IMAX and IMIN faces */
  ierr = ModelApplyGeneralNavierSlip_RiftNitsche(surflist,insert_if_not_found,data);CHKERRQ(ierr);

  /* Apply base velocity from u.n */
  u_bot = data->u_bc[1];
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,1,BCListEvaluator_constant,(void*)&u_bot);CHKERRQ(ierr);
  //data->component = 1;
  //ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,data->component,BCListEvaluator_RotatedVelocityField,(void*)data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscBool BCListEvaluator_SplitFace(PetscScalar position[], PetscScalar *value, void *ctx)
{
  PetscReal *input = (PetscReal*)ctx;
  PetscReal x0,x1,v0,v1;
  PetscInt  dim;
  PetscBool impose = PETSC_TRUE;

  PetscFunctionBegin;
  x0 = input[0];
  x1 = input[1];
  v0 = input[2];
  v1 = -v0;
  dim = (int)input[3];

  if (position[dim] < x0) {
    *value = v0;
  } else if (position[dim] > x1) {
    *value = v1;
  } else {
    *value = v1+(position[dim]-x1)*(v0-v1)/(x0-x1);
  }
  PetscFunctionReturn(impose);
}

static PetscErrorCode ModelApplyAnalogueBoundaryConditions(DM dav, BCList bclist, ModelRiftNitscheCtx *data)
{
  PetscReal      bc_data[4],zero=0.0,u_bot;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  
  bc_data[0] = data->split_face_min[0];    // x0
  bc_data[1] = data->split_face_min[1];    // x1 
  bc_data[2] = data->u_bc[0];              // v0
  bc_data[3] = 2;                          // dim (typecasted to (int) in the function)

  /* x normal faces apply vx and -vx with a split in the middle of the face */
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,0,BCListEvaluator_SplitFace,(void*)bc_data);CHKERRQ(ierr);
  bc_data[0] = data->split_face_max[0];
  bc_data[1] = data->split_face_max[1];
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,0,BCListEvaluator_SplitFace,(void*)bc_data);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);

  /* z normal  */
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMAX_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);

  /* y normal apply vx and -vx with a split in the middle of the face */
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,0,BCListEvaluator_SplitFace,(void*)bc_data);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  /* Apply base y velocity from u.n */
  u_bot = data->u_bc[1];
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,1,BCListEvaluator_constant,(void*)&u_bot);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

static PetscErrorCode IncreaseVelocityWithTime(pTatinCtx ptatin, ModelRiftNitscheCtx *data)
{
  PetscReal      velocity_bc[2],time;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  /* Get time */
  ierr = pTatinGetTime(ptatin,&time);CHKERRQ(ierr);

  velocity_bc[1] = data->norm_u * cos(data->alpha_u);
  velocity_bc[0] = sqrt( pow(data->norm_u,2.0) - pow(velocity_bc[1],2.0) );

  if (time > data->time_full_velocity) {
    data->u_bc[0] = velocity_bc[0];
    data->u_bc[2] = velocity_bc[1];
  } else {
    data->u_bc[0] = velocity_bc[0] + (time - data->time_full_velocity) * velocity_bc[0] / data->time_full_velocity;
    data->u_bc[2] = velocity_bc[1] + (time - data->time_full_velocity) * velocity_bc[1] / data->time_full_velocity;
  }
  
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyBaseDrivenRotatedStrikeSlipNavierSlip_RiftNitsche(pTatinCtx ptatin, DM dav, BCList bclist,SurfBCList surflist,PetscBool insert_if_not_found,ModelRiftNitscheCtx *data)
{
  PetscReal      u_bot;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = IncreaseVelocityWithTime(ptatin,data);CHKERRQ(ierr);

  /* Apply rotated Couette flow on KMAX and KMIN faces */
  data->component = 0;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,data->component,BCListEvaluator_RotatedVelocityField,(void*)data);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMAX_LOC,data->component,BCListEvaluator_RotatedVelocityField,(void*)data);CHKERRQ(ierr);
  data->component = 2;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,data->component,BCListEvaluator_RotatedVelocityField,(void*)data);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMAX_LOC,data->component,BCListEvaluator_RotatedVelocityField,(void*)data);CHKERRQ(ierr);

  /* Apply General Navier Slip BC on IMAX and IMIN faces */
  ierr = ModelApplyGeneralNavierSlip_RiftNitsche(surflist,insert_if_not_found,data);CHKERRQ(ierr);

  /* Apply rotated Couette flow on JMIN face */
  data->component = 0;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,data->component,BCListEvaluator_RotatedVelocityField,(void*)data);CHKERRQ(ierr);
  data->component = 2;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,data->component,BCListEvaluator_RotatedVelocityField,(void*)data);CHKERRQ(ierr);
  /* Apply base y velocity from u.n */
  u_bot = data->u_bc[1];
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,1,BCListEvaluator_constant,(void*)&u_bot);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

static PetscBool MarkLithosphere(Facet facets, void *ctx)
{
  ModelRiftNitscheCtx *data = (ModelRiftNitscheCtx*)ctx;
  PetscBool impose = PETSC_FALSE;
  PetscFunctionBegin;
  /* Select the entire cell based on its centroid coordinate */
  if (facets->centroid[1] >= data->y_continent[2]) { impose = PETSC_TRUE; }
  PetscFunctionReturn(impose);
}

static PetscBool MarkAsthenosphere(Facet facets, void *ctx)
{
  ModelRiftNitscheCtx *data = (ModelRiftNitscheCtx*)ctx;
  PetscBool impose = PETSC_FALSE;
  PetscFunctionBegin;
  /* Select the entire cell based on its centroid coordinate */
  if (facets->centroid[1] < data->y_continent[2]) { impose = PETSC_TRUE; }
  PetscFunctionReturn(impose);
}

static PetscErrorCode MarkDirichletSubdomain(SurfBCList surflist,const char subdomain_name[],PetscInt n_domain_faces, HexElementFace *domain_face, PetscBool insert_if_not_found, PetscBool (*mark)(Facet,void*), void *ctx)
{
  MarkDomainFaceContext face_ctx;
  SurfaceConstraint     sc;
  MeshEntity            mesh_entity;
  PetscInt              f;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = SurfBCListGetConstraint(surflist,subdomain_name,&sc);CHKERRQ(ierr);
  if (!sc) {
    if (insert_if_not_found) {
      ierr = SurfBCListAddConstraint(surflist,subdomain_name,&sc);CHKERRQ(ierr);
      ierr = SurfaceConstraintSetType(sc,SC_DIRICHLET);CHKERRQ(ierr);
    } else { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Surface constraint not found"); }
  }
  ierr = SurfaceConstraintGetFacets(sc,&mesh_entity);CHKERRQ(ierr);

  ierr = MarkDomainFaceContextInit(&face_ctx);

  /* Faces to mark */
  face_ctx.n_domain_faces = n_domain_faces;
  for (f=0; f<n_domain_faces; f++) {
    face_ctx.domain_face[f] = domain_face[f];
  }
  
  face_ctx.mark = mark;
  face_ctx.user_data = ctx;
  /* Mark facets */
  ierr = MeshFacetMarkByBoundary(mesh_entity,sc->fi,NULL,(void*)&face_ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelMarkDirichlet(SurfBCList surflist, PetscBool insert_if_not_found, ModelRiftNitscheCtx *data)
{
  HexElementFace face;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  /* lithosphere */
  face = HEX_FACE_Nxi;
  ierr = MarkDirichletSubdomain(surflist,"Nxi_litho",1,&face,insert_if_not_found,MarkLithosphere,(void*)data);CHKERRQ(ierr);
  face = HEX_FACE_Pxi;
  ierr = MarkDirichletSubdomain(surflist,"Pxi_litho",1,&face,insert_if_not_found,MarkLithosphere,(void*)data);CHKERRQ(ierr);
  face = HEX_FACE_Nzeta;
  ierr = MarkDirichletSubdomain(surflist,"Nzeta_litho",1,&face,insert_if_not_found,MarkLithosphere,(void*)data);CHKERRQ(ierr);
  face = HEX_FACE_Pzeta;
  ierr = MarkDirichletSubdomain(surflist,"Pzeta_litho",1,&face,insert_if_not_found,MarkLithosphere,(void*)data);CHKERRQ(ierr);

  /* asthenosphere */
  face = HEX_FACE_Nxi;
  ierr = MarkDirichletSubdomain(surflist,"Nxi_asth",1,&face,insert_if_not_found,MarkAsthenosphere,(void*)data);CHKERRQ(ierr);
  face = HEX_FACE_Pxi;
  ierr = MarkDirichletSubdomain(surflist,"Pxi_asth",1,&face,insert_if_not_found,MarkAsthenosphere,(void*)data);CHKERRQ(ierr);
  face = HEX_FACE_Nzeta;
  ierr = MarkDirichletSubdomain(surflist,"Nzeta_asth",1,&face,insert_if_not_found,MarkAsthenosphere,(void*)data);CHKERRQ(ierr);
  face = HEX_FACE_Pzeta;
  ierr = MarkDirichletSubdomain(surflist,"Pzeta_asth",1,&face,insert_if_not_found,MarkAsthenosphere,(void*)data);CHKERRQ(ierr);

  /* bottom face */
  face = HEX_FACE_Neta;
  ierr = MarkDirichletSubdomain(surflist,"Neta",1,&face,insert_if_not_found,NULL,(void*)data);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyDirichlet_Neumann(pTatinCtx ptatin, DM dav, BCList bclist,SurfBCList surflist,PetscBool insert_if_not_found,ModelRiftNitscheCtx *data)
{
  SurfaceConstraint Nxi_litho,Pxi_litho,Neta,Nzeta_litho,Pzeta_litho;
  SurfaceConstraint Nxi_asth,Pxi_asth,Nzeta_asth,Pzeta_asth;
  PetscReal         ux;
  PetscErrorCode    ierr;
  PetscFunctionBegin;

  /* Mark facets */
  ierr = ModelMarkDirichlet(surflist,insert_if_not_found,data);CHKERRQ(ierr);

  /* Impose */
  /* XMIN LITHOSPHERE */
  ierr = SurfBCListGetConstraint(surflist,"Nxi_litho",&Nxi_litho);CHKERRQ(ierr);
  ux   = -data->u_bc[0];
  ierr = DMDABCListTraverseFacets3d(bclist,dav,Nxi_litho,0,BCListEvaluator_constant,(void*)&ux);CHKERRQ(ierr);
  /* XMAX LITHOSPHERE */
  ierr = SurfBCListGetConstraint(surflist,"Pxi_litho",&Pxi_litho);CHKERRQ(ierr);
  ux   = data->u_bc[0];
  ierr = DMDABCListTraverseFacets3d(bclist,dav,Pxi_litho,0,BCListEvaluator_constant,(void*)&ux);CHKERRQ(ierr);
  
  /* XMIN ASTHENOSPHERE */
  ierr = SurfBCListGetConstraint(surflist,"Nxi_asth",&Nxi_asth);CHKERRQ(ierr);
  ux   = data->u_bc[0];
  ierr = DMDABCListTraverseFacets3d(bclist,dav,Nxi_asth,0,BCListEvaluator_constant,(void*)&ux);CHKERRQ(ierr);
  /* XMAX ASTHENOSPHERE */
  ierr = SurfBCListGetConstraint(surflist,"Pxi_asth",&Pxi_asth);CHKERRQ(ierr);
  ux   = -data->u_bc[0];
  ierr = DMDABCListTraverseFacets3d(bclist,dav,Pxi_asth,0,BCListEvaluator_constant,(void*)&ux);CHKERRQ(ierr);

  /* ZMIN LITHOSPHERE */
  ierr = SurfBCListGetConstraint(surflist,"Nzeta_litho",&Nzeta_litho);CHKERRQ(ierr);
  ux   = -data->u_bc[2];
  ierr = DMDABCListTraverseFacets3d(bclist,dav,Nzeta_litho,2,BCListEvaluator_constant,(void*)&ux);CHKERRQ(ierr);
  /* ZMAX LITHOSPHERE */
  ierr = SurfBCListGetConstraint(surflist,"Pzeta_litho",&Pzeta_litho);CHKERRQ(ierr);
  ux   = data->u_bc[2];
  ierr = DMDABCListTraverseFacets3d(bclist,dav,Pzeta_litho,2,BCListEvaluator_constant,(void*)&ux);CHKERRQ(ierr);
  
  /* ZMIN ASTHENOSPHERE */
  ierr = SurfBCListGetConstraint(surflist,"Nzeta_asth",&Nzeta_asth);CHKERRQ(ierr);
  ux   = data->u_bc[2];
  ierr = DMDABCListTraverseFacets3d(bclist,dav,Nzeta_asth,2,BCListEvaluator_constant,(void*)&ux);CHKERRQ(ierr);
  /* ZMAX ASTHENOSPHERE */
  ierr = SurfBCListGetConstraint(surflist,"Pzeta_asth",&Pzeta_asth);CHKERRQ(ierr);
  ux   = -data->u_bc[2];
  ierr = DMDABCListTraverseFacets3d(bclist,dav,Pzeta_asth,2,BCListEvaluator_constant,(void*)&ux);CHKERRQ(ierr);

  /* YMIN */
  ierr = SurfBCListGetConstraint(surflist,"Neta",&Neta);CHKERRQ(ierr);
  ux   = data->u_bc[1];
  ierr = DMDABCListTraverseFacets3d(bclist,dav,Neta,1,BCListEvaluator_constant,(void*)&ux);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyBoundaryConditionsVelocity_RiftNitsche(pTatinCtx ptatin, DM dav, BCList bclist,SurfBCList surflist,PetscBool insert_if_not_found,ModelRiftNitscheCtx *data)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  /* See ModelSetBCType_RiftNitsche() for the bc case number */
  switch(data->bc_type) {
    case 0:
      /* Dirichlet on z normal faces and general navier slip on x normal faces */
      ierr = ModelApplyObliqueExtensionPullApart_RiftNitsche(dav,bclist,surflist,insert_if_not_found,data);CHKERRQ(ierr);
      break;

    case 1:
      /* Dirichlet on x and z normal faces */
      ierr = ModelApplyObliqueExtensionPullApartDirichlet_RiftNitsche(dav,bclist,data);CHKERRQ(ierr);
      break;

    case 2:
      /* Orthogonal extension with free- normal slip nitsche (u.n) = 0 on faces of normal x */
      ierr = ModelApplyOrthogonalExtensionFreeSlipNitsche_RiftNitsche(dav,bclist,surflist,insert_if_not_found,data);CHKERRQ(ierr);
      break;

    case 3:
      ierr = ModelApplyRotatedStrikeSlipNavierSlip_RiftNitsche(dav,bclist,surflist,insert_if_not_found,data);CHKERRQ(ierr);
      break;

    case 4:
      ierr = ModelApplyAnalogueBoundaryConditions(dav,bclist,data);CHKERRQ(ierr);
      break;

    case 5:
      ierr = ModelApplyBaseDrivenRotatedStrikeSlipNavierSlip_RiftNitsche(ptatin,dav,bclist,surflist,insert_if_not_found,data);CHKERRQ(ierr);
      break;

    case 6:
      ierr = ModelApplyDirichlet_Neumann(ptatin,dav,bclist,surflist,PETSC_TRUE,data);CHKERRQ(ierr);
      break;

    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"No velocity boundary conditions type was given. Use -bc_nitsche or -bc_dirichlet\n");
      break;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyTimeDependantEnergyBCs_RiftNitsche(pTatinCtx c,ModelRiftNitscheCtx *data)
{
  PhysCompEnergyFV energy;
  PetscReal        val_T;
  PetscErrorCode   ierr;
  
  PetscFunctionBegin;

  ierr = pTatinGetContext_EnergyFV(c,&energy);CHKERRQ(ierr);

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

static PetscErrorCode ModelApplyPoissonPressureBoundaryConditions_RiftNitsche(pTatinCtx ptatin)
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

PetscErrorCode ModelApplyBoundaryConditions_RiftNitsche(pTatinCtx c,void *ctx)
{
  ModelRiftNitscheCtx *data = (ModelRiftNitscheCtx*)ctx;
  PhysCompStokes      stokes;
  DM                  stokes_pack,dav,dap;
  Vec                 X = NULL;
  PetscBool           active_energy,active_poisson;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  /* Define velocity boundary conditions */
  ierr = pTatinGetStokesContext(c,&stokes);CHKERRQ(ierr);
  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);

  ierr = pTatinPhysCompGetData_Stokes(c,&X);CHKERRQ(ierr); 
  /* Compute uy as int_S v.n dS */
  ierr = ModelComputeBottomFlow_udotn(c,X,data);CHKERRQ(ierr);

  ierr = ModelApplyBoundaryConditionsVelocity_RiftNitsche(c,stokes->dav,stokes->u_bclist,stokes->surf_bclist,PETSC_TRUE,data);CHKERRQ(ierr);

  /* Define boundary conditions for any other physics */
  /* Temperature */
  ierr = pTatinContextValid_EnergyFV(c,&active_energy);CHKERRQ(ierr);
  if (active_energy) {
    ierr = ModelApplyTimeDependantEnergyBCs_RiftNitsche(c,data);CHKERRQ(ierr);
  }
  /* Poisson Pressure */
  ierr = pTatinContextValid_LithoP(c,&active_poisson);CHKERRQ(ierr);
  if (active_poisson) {
    ierr = ModelApplyPoissonPressureBoundaryConditions_RiftNitsche(c);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyBoundaryConditionMG_RiftNitsche(PetscInt nl,BCList bclist[],SurfBCList surf_bclist[],DM dav[],pTatinCtx c,void *ctx)
{
  ModelRiftNitscheCtx *data;
  PetscInt         n;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  data = (ModelRiftNitscheCtx*)ctx;
  /* Define velocity boundary conditions on each level within the MG hierarchy */
  for (n=0; n<nl; n++) {
    ierr = ModelApplyBoundaryConditionsVelocity_RiftNitsche(c,dav[n],bclist[n],surf_bclist[n],PETSC_FALSE,data);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplySurfaceRemeshing_RiftNitsche(DM dav, PetscReal dt, ModelRiftNitscheCtx *data)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  /* Dirichlet velocity imposed on z normal faces so we do the same here */
  ierr = UpdateMeshGeometry_ApplyDiffusionJMAX(dav,data->diffusivity_spm,dt,PETSC_FALSE,PETSC_FALSE,PETSC_TRUE,PETSC_TRUE,PETSC_FALSE);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyUpdateMeshGeometry_RiftNitsche(pTatinCtx c,Vec X,void *ctx)
{
  ModelRiftNitscheCtx *data;
  PhysCompStokes      stokes;
  DM                  stokes_pack,dav,dap;
  Vec                 velocity,pressure;
  PetscReal           dt;
  PetscErrorCode      ierr;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  data = (ModelRiftNitscheCtx*)ctx;
  
  /* fully lagrangian update */
  ierr = pTatinGetTimestep(c,&dt);CHKERRQ(ierr);
  ierr = pTatinGetStokesContext(c,&stokes);CHKERRQ(ierr);

  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);

  /* SURFACE REMESHING */
  ierr = ModelApplySurfaceRemeshing_RiftNitsche(dav,dt,data);CHKERRQ(ierr);

  ierr = UpdateMeshGeometry_FullLag_ResampleJMax_RemeshJMIN2JMAX(dav,velocity,NULL,dt);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);
 
  /* Update Mesh Refinement */
  ierr = ModelSetMeshRefinement_RiftNitsche(dav,data);CHKERRQ(ierr);
  ierr = DMDABilinearizeQ2Elements(dav);CHKERRQ(ierr);

  /* Passive markers update */
  ierr = PSwarmFieldUpdateAll(data->pswarm);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelOutputMarkerFields_RiftNitsche(pTatinCtx c,const char prefix[])
{
  DataBucket               materialpoint_db;
  int                      nf;
  const MaterialPointField mp_prop_list[] = { MPField_Std, MPField_Stokes, MPField_StokesPl};//, MPField_Energy };
  char                     mp_file_prefix[256];
  PetscErrorCode           ierr;

  PetscFunctionBegin;

  nf = sizeof(mp_prop_list)/sizeof(mp_prop_list[0]);

  ierr = pTatinGetMaterialPoints(c,&materialpoint_db,NULL);CHKERRQ(ierr);
  sprintf(mp_file_prefix,"%s_mpoints",prefix);
  ierr = SwarmViewGeneric_ParaView(materialpoint_db,nf,mp_prop_list,c->outputpath,mp_file_prefix);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelOutputEnergyFV_RiftNitsche(pTatinCtx c, const char prefix[], PetscBool been_here, ModelRiftNitscheCtx *data)
{
  PhysCompEnergyFV energy;
  char             root[PETSC_MAX_PATH_LEN],pvoutputdir[PETSC_MAX_PATH_LEN],fname[PETSC_MAX_PATH_LEN];
  char             pvdfilename[PETSC_MAX_PATH_LEN],vtkfilename[PETSC_MAX_PATH_LEN];
  char             stepprefix[PETSC_MAX_PATH_LEN];
  PetscErrorCode   ierr;

  PetscFunctionBegin;

  ierr = pTatinGetContext_EnergyFV(c,&energy);CHKERRQ(ierr);
  // PVD
  PetscSNPrintf(pvdfilename,PETSC_MAX_PATH_LEN-1,"%s/timeseries_T_fv.pvd",c->outputpath);
  if (prefix) { PetscSNPrintf(vtkfilename, PETSC_MAX_PATH_LEN-1, "%s_T_fv.pvts",prefix);
  } else {      PetscSNPrintf(vtkfilename, PETSC_MAX_PATH_LEN-1, "T_fv.pvts");           }
  
  PetscSNPrintf(stepprefix,PETSC_MAX_PATH_LEN-1,"step%D",c->step);
  if (!been_here) { /* new file */
    ierr = ParaviewPVDOpen(pvdfilename);CHKERRQ(ierr);
    ierr = ParaviewPVDAppend(pvdfilename,c->time,vtkfilename,stepprefix);CHKERRQ(ierr);
  } else {
    ierr = ParaviewPVDAppend(pvdfilename,c->time,vtkfilename,stepprefix);CHKERRQ(ierr);
  }
  
  ierr = PetscSNPrintf(root,PETSC_MAX_PATH_LEN-1,"%s",c->outputpath);CHKERRQ(ierr);
  ierr = PetscSNPrintf(pvoutputdir,PETSC_MAX_PATH_LEN-1,"%s/step%D",root,c->step);CHKERRQ(ierr);
  
  /* PetscVec */
  ierr = PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s_energy",prefix);CHKERRQ(ierr);
  ierr = FVDAView_JSON(energy->fv,pvoutputdir,fname);CHKERRQ(ierr); /* write meta data abour fv mesh, its DMDA and the coords */
  ierr = FVDAView_Heavy(energy->fv,pvoutputdir,fname);CHKERRQ(ierr);  /* write cell fields */
  ierr = PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s/%s_energy_T",pvoutputdir,prefix);CHKERRQ(ierr);
  ierr = PetscVecWriteJSON(energy->T,0,fname);CHKERRQ(ierr); /* write cell temperature */
  
  if (data->output_markers) {
    PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s/%s-Tfv",pvoutputdir,prefix);
    ierr = FVDAView_CellData(energy->fv,energy->T,PETSC_TRUE,fname);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelOutputPassiveMarkers(ModelRiftNitscheCtx *data)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = PSwarmView(data->pswarm,PSW_VT_SINGLETON);CHKERRQ(ierr);
  ierr = PSwarmViewInfo(data->pswarm);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode ModelOutput_RiftNitsche(pTatinCtx c,Vec X,const char prefix[],void *ctx)
{
  ModelRiftNitscheCtx         *data;
  PetscBool                   active_energy;
  const MaterialPointVariable mp_prop_list[] = { MPV_region, MPV_viscosity, MPV_density, MPV_plastic_strain }; //MPV_viscous_strain
  PetscErrorCode              ierr;
  static PetscBool            been_here = PETSC_FALSE;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  data = (ModelRiftNitscheCtx*)ctx;
  
  /* Output Velocity and pressure */
  ierr = pTatin3d_ModelOutputPetscVec_VelocityPressure_Stokes(c,X,prefix);CHKERRQ(ierr);
  
  /* Output markers cell fields (for production runs) */
  ierr = pTatin3dModelOutput_MarkerCellFieldsP0_PetscVec(c,PETSC_FALSE,sizeof(mp_prop_list)/sizeof(MaterialPointVariable),mp_prop_list,prefix);CHKERRQ(ierr);
  
  /* Output raw markers and vtu velocity and pressure (for testing and debugging) */
  if (data->output_markers) {
    ierr = pTatin3d_ModelOutput_VelocityPressure_Stokes(c,X,prefix);CHKERRQ(ierr);
    ierr = ModelOutputMarkerFields_RiftNitsche(c,prefix);CHKERRQ(ierr);
    /*{
      PhysCompStokes    stokes;
      char              root[PETSC_MAX_PATH_LEN];
      SurfaceConstraint sc;
      ierr = pTatinGetStokesContext(c,&stokes);CHKERRQ(ierr);
      ierr = SurfBCListGetConstraint(stokes->surf_bclist,"boundary",&sc);CHKERRQ(ierr);
      ierr = PetscSNPrintf(root,PETSC_MAX_PATH_LEN-1,"%s/step%D",c->outputpath,c->step);CHKERRQ(ierr);
      ierr = SurfaceConstraintViewParaview(sc, root, "boundary");CHKERRQ(ierr);
    }*/
  }
  /* Output passive markers */
  ierr = ModelOutputPassiveMarkers(data);CHKERRQ(ierr);

  /* Output temperature (FV) */
  ierr = pTatinContextValid_EnergyFV(c,&active_energy);CHKERRQ(ierr);
  if (active_energy) {
    ierr = ModelOutputEnergyFV_RiftNitsche(c,prefix,been_here,data);CHKERRQ(ierr);
  }
  been_here = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode SwarmMPntStd_CoordAssignment_FaceLatticeLayout3d_epsilon(DM da,PetscInt Nxp[],PetscReal perturb, PetscReal epsilon, PetscInt face_idx,DataBucket db)
{
  DataField      PField;
  PetscInt       e,ei,ej,ek,eij2d;
  Vec            gcoords;
  PetscScalar    *LA_coords;
  PetscScalar    el_coords[Q2_NODES_PER_EL_3D*NSD];
  int            ncells,ncells_face,np_per_cell,points_face,points_face_local=0;
  PetscInt       nel,nen,lmx,lmy,lmz,MX,MY,MZ;
  const PetscInt *elnidx;
  PetscInt       p,k,pi,pj;
  PetscReal      dxi,deta;
  int            np_current,np_new;
  PetscInt       si,sj,sk,M,N,P,lnx,lny,lnz;
  PetscBool      contains_east,contains_west,contains_north,contains_south,contains_front,contains_back;
  PetscErrorCode ierr;


  PetscFunctionBegin;

  ierr = DMDAGetElements_pTatinQ2P1(da,&nel,&nen,&elnidx);CHKERRQ(ierr);
  ncells = nel;
  ierr = DMDAGetLocalSizeElementQ2(da,&lmx,&lmy,&lmz);CHKERRQ(ierr);

  switch (face_idx) {
    case 0:// east-west
      ncells_face = lmy * lmz; // east
      break;
    case 1:
      ncells_face = lmy * lmz; // west
      break;

    case 2:// north-south
      ncells_face = lmx * lmz; // north
      break;
    case 3:
      ncells_face = lmx * lmz; // south
      break;

    case 4: // front-back
      ncells_face = lmx * lmy; // front
      break;
    case 5:
      ncells_face = lmx * lmy; // back
      break;

    default:
      SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_USER,"Unknown face index");
      break;
  }

  np_per_cell = Nxp[0] * Nxp[1];
  points_face = ncells_face * np_per_cell;

  if (perturb < 0.0) {
    SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_USER,"Cannot use a negative perturbation");
  }
  if (perturb > 1.0) {
    SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_USER,"Cannot use a perturbation greater than 1.0");
  }

  ierr = DMDAGetSizeElementQ2(da,&MX,&MY,&MZ);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&si,&sj,&sk,&lnx,&lny,&lnz);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,0, &M,&N,&P, 0,0,0, 0,0, 0,0,0, 0);CHKERRQ(ierr);

  contains_east  = PETSC_FALSE; if (si+lnx == M) { contains_east  = PETSC_TRUE; }
  contains_west  = PETSC_FALSE; if (si == 0)     { contains_west  = PETSC_TRUE; }
  contains_north = PETSC_FALSE; if (sj+lny == N) { contains_north = PETSC_TRUE; }
  contains_south = PETSC_FALSE; if (sj == 0)     { contains_south = PETSC_TRUE; }
  contains_front = PETSC_FALSE; if (sk+lnz == P) { contains_front = PETSC_TRUE; }
  contains_back  = PETSC_FALSE; if (sk == 0)     { contains_back  = PETSC_TRUE; }

  // re-size //
  switch (face_idx) {
    case 0:
      if (contains_east) points_face_local = points_face;
      break;
    case 1:
      if (contains_west) points_face_local = points_face;
      break;

    case 2:
      if (contains_north) points_face_local = points_face;
      break;
    case 3:
      if (contains_south) points_face_local = points_face;
      break;

    case 4:
      if (contains_front) points_face_local = points_face;
      break;
    case 5:
      if (contains_back) points_face_local = points_face;
      break;
  }
  
  DataBucketGetSizes(db,&np_current,NULL,NULL);
  np_new = np_current + points_face_local;
  
  DataBucketSetSizes(db,np_new,-1);

  /* setup for coords */
  ierr = DMGetCoordinatesLocal(da,&gcoords);CHKERRQ(ierr);
  ierr = VecGetArray(gcoords,&LA_coords);CHKERRQ(ierr);

  DataBucketGetDataFieldByName(db,MPntStd_classname,&PField);
  DataFieldGetAccess(PField);
  DataFieldVerifyAccess( PField,sizeof(MPntStd));

  dxi  = 2.0/(PetscReal)Nxp[0];
  deta = 2.0/(PetscReal)Nxp[1];

  p = np_current;
  for (e = 0; e < ncells; e++) {
    /* get coords for the element */
    ierr = DMDAGetElementCoordinatesQ2_3D(el_coords,(PetscInt*)&elnidx[nen*e],LA_coords);CHKERRQ(ierr);

    ek = e / (lmx*lmy);
    eij2d = e - ek * (lmx*lmy);
    ej = eij2d / lmx;
    ei = eij2d - ej * lmx;

    switch (face_idx) {
      case 0:// east-west
        if (!contains_east) { continue; }
        if (ei != lmx-1) { continue; }
        break;
      case 1:
        if (!contains_west) { continue; }
        if (ei != 0) { continue; }
        break;

      case 2:// north-south
        if (!contains_north) { continue; }
        if (ej != lmy-1) { continue; }
        break;
      case 3:
        if (!contains_south) { continue; }
        if (ej != 0) { continue; }
        break;

      case 4: // front-back
        if (!contains_front) { continue; }
        if (ek != lmz-1) { continue; }
        break;
      case 5:
        if (!contains_back) { continue; }
        if (ek != 0) { continue; }
        break;
    }
    
    for (pj=0; pj<Nxp[1]; pj++) {
      for (pi=0; pi<Nxp[0]; pi++) {
        MPntStd *marker;
        double xip2d[2],xip_shift2d[2],xip_rand2d[2];
        double xip[NSD],xp_rand[NSD],Ni[Q2_NODES_PER_EL_3D];

        /* define coordinates in 2d layout */
        xip2d[0] = -1.0 + dxi    * (pi + 0.5);
        xip2d[1] = -1.0 + deta   * (pj + 0.5);

        /* random between -0.5 <= shift <= 0.5 */
        xip_shift2d[0] = 1.0*(rand()/(RAND_MAX+1.0)) - 0.5;
        xip_shift2d[1] = 1.0*(rand()/(RAND_MAX+1.0)) - 0.5;

        xip_rand2d[0] = xip2d[0] + perturb * dxi    * xip_shift2d[0];
        xip_rand2d[1] = xip2d[1] + perturb * deta   * xip_shift2d[1];

        if (fabs(xip_rand2d[0]) > 1.0) {
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"fabs(x-point coord) greater than 1.0");
        }
        if (fabs(xip_rand2d[1]) > 1.0) {
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"fabs(y-point coord) greater than 1.0");
        }

        /* set to 3d dependnent on face */
        // case 0:// east-west
        // case 2:// north-south
        // case 4: // front-back
        switch (face_idx) {
          case 0:// east-west
            xip[0] = 1.0 - epsilon;
            xip[1] = xip_rand2d[0];
            xip[2] = xip_rand2d[1];
            break;
          case 1:
            xip[0] = -1.0 + epsilon;
            xip[1] = xip_rand2d[0];
            xip[2] = xip_rand2d[1];
            break;

          case 2:// north-south
            xip[0] = xip_rand2d[0];
            xip[1] = 1.0 - epsilon;
            xip[2] = xip_rand2d[1];
            break;
          case 3:
            xip[0] = xip_rand2d[0];
            xip[1] = -1.0 + epsilon;
            xip[2] = xip_rand2d[1];
            break;

          case 4: // front-back
            xip[0] = xip_rand2d[0];
            xip[1] = xip_rand2d[1];
            xip[2] = 1.0 - epsilon;
            break;
          case 5:
            xip[0] = xip_rand2d[0];
            xip[1] = xip_rand2d[1];
            xip[2] = -1.0 + epsilon;
            break;
        }

        pTatin_ConstructNi_Q2_3D(xip,Ni);

        xp_rand[0] = xp_rand[1] = xp_rand[2] = 0.0;
        for (k=0; k<Q2_NODES_PER_EL_3D; k++) {
          xp_rand[0] += Ni[k] * el_coords[NSD*k+0];
          xp_rand[1] += Ni[k] * el_coords[NSD*k+1];
          xp_rand[2] += Ni[k] * el_coords[NSD*k+2];
        }

        DataFieldAccessPoint(PField,p,(void**)&marker);

        marker->coor[0] = xp_rand[0];
        marker->coor[1] = xp_rand[1];
        marker->coor[2] = xp_rand[2];

        marker->xi[0] = xip[0];
        marker->xi[1] = xip[1];
        marker->xi[2] = xip[2];

        marker->wil    = e;
        marker->pid    = 0;
        p++;
      }
    }

  }
  DataFieldRestoreAccess(PField);
  ierr = VecRestoreArray(gcoords,&LA_coords);CHKERRQ(ierr);

  ierr = SwarmMPntStd_AssignUniquePointIdentifiers(PetscObjectComm((PetscObject)da),db,np_current,np_new);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyMaterialBoundaryCondition_RiftNitsche(pTatinCtx c,ModelRiftNitscheCtx *data)
{
  PhysCompStokes  stokes;
  DM              stokes_pack,dav,dap;
  PetscInt        Nxp[2];
  PetscInt        *face_list;
  PetscReal       perturb, epsilon;
  DataBucket      material_point_db,material_point_face_db;
  PetscInt        f, n_face_list;
  int             p,n_mp_points;
  MPAccess        mpX;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  
  ierr = pTatinGetStokesContext(c,&stokes);CHKERRQ(ierr);
  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);

  ierr = pTatinGetMaterialPoints(c,&material_point_db,NULL);CHKERRQ(ierr);

  /* create face storage for markers */
  DataBucketDuplicateFields(material_point_db,&material_point_face_db);
  
  n_face_list = 4;
  ierr = PetscMalloc1(n_face_list,&face_list);CHKERRQ(ierr);
  face_list[0] = 0;
  face_list[1] = 1;
  face_list[2] = 4;
  face_list[3] = 5;
  
  for (f=0; f<n_face_list; f++) {

    /* traverse */
    /* [0,1/east,west] ; [2,3/north,south] ; [4,5/front,back] */
    Nxp[0]  = 4;
    Nxp[1]  = 4;
    perturb = 0.1;

    /* reset size */
    DataBucketSetSizes(material_point_face_db,0,-1);

    /* assign coords */
    epsilon = 1.0e-6;
    ierr = SwarmMPntStd_CoordAssignment_FaceLatticeLayout3d_epsilon(dav,Nxp,perturb,epsilon,face_list[f],material_point_face_db);CHKERRQ(ierr);

    /* assign values */
    DataBucketGetSizes(material_point_face_db,&n_mp_points,0,0);
    ierr = MaterialPointGetAccess(material_point_face_db,&mpX);CHKERRQ(ierr);
    for (p=0; p<n_mp_points; p++) {
      ierr = MaterialPointSet_phase_index(mpX,p,MATERIAL_POINT_PHASE_UNASSIGNED);CHKERRQ(ierr);
    }
    ierr = MaterialPointRestoreAccess(material_point_face_db,&mpX);CHKERRQ(ierr);

    /* insert into volume bucket */
    DataBucketInsertValues(material_point_db,material_point_face_db);
  }

  /* Copy ALL values from nearest markers to newly inserted markers except (xi,xip,pid) */
  ierr = MaterialPointRegionAssignment_KDTree(material_point_db,PETSC_TRUE);CHKERRQ(ierr);

  /* delete */
  DataBucketDestroy(&material_point_face_db);
  
  ierr = PetscFree(face_list);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode MaterialPointResolutionMask_BoundaryFaces(DM dav, pTatinCtx ctx, PetscBool *popctrl_mask)
{
  PetscInt        nel,nen,el;
  const PetscInt  *elnidx;
  PetscInt        mx,my,mz;
  PetscInt        esi,esj,esk,lmx,lmy,lmz,e;
  PetscInt        iel,kel,jel;
  PetscErrorCode  ierr;
  
  PetscFunctionBegin;
  /* Get Q2 elements information */ 
  ierr = DMDAGetElements_pTatinQ2P1(dav,&nel,&nen,&elnidx);CHKERRQ(ierr);
  ierr = DMDAGetSizeElementQ2(dav,&mx,&my,&mz);CHKERRQ(ierr);
  ierr = DMDAGetCornersElementQ2(dav,&esi,&esj,&esk,&lmx,&lmy,&lmz);CHKERRQ(ierr);

  /* Set all to TRUE */
  for (el=0; el<nel; el++) {
    popctrl_mask[el] = PETSC_TRUE;
  }
  
  esi = esi/2;
  esj = esj/2;
  esk = esk/2;

  /* max(x) face */
  if (esi + lmx == mx) { 
    iel = lmx-1;
    for (kel=0; kel<lmz; kel++) {
      for (jel=0; jel<lmy; jel++) {
        e = iel + jel*lmx + kel*lmx*lmy;
        popctrl_mask[e] = PETSC_FALSE;
      }
    }
  }
  
  /* min(x) face */
  if (esi == 0) {
    iel = 0;
    for (kel=0; kel<lmz; kel++) {
      for (jel=0; jel<lmy; jel++) {
        e = iel + jel*lmx + kel*lmx*lmy;
        popctrl_mask[e] = PETSC_FALSE;
      }
    }
  }

  /* max(z) face */
  if (esk + lmz == mz) {
    kel = lmz-1;
    for (jel=0; jel<lmy; jel++) {
      for (iel=0; iel<lmx; iel++) {  
        e = iel + jel*lmx + kel*lmx*lmy;
        popctrl_mask[e] = PETSC_FALSE;
      }
    }
  }

  /* min(z) face */
  if (esk == 0) {
    kel = 0;
    for (jel=0; jel<lmy; jel++) {
      for (iel=0; iel<lmx; iel++) {  
        e = iel + jel*lmx + kel*lmx*lmy;
        popctrl_mask[e] = PETSC_FALSE;
      }
    }
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode MPPC_SimpleRemoval_Mask(PetscInt np_upper,DM da,DataBucket db,PetscBool reverse_order_removal, PetscBool *popctrl_mask)
{
  PetscInt        *cell_count,count;
  int             p32,npoints32;
  PetscInt        c,nel,nen;
  const PetscInt  *elnidx;
  DataField       PField;
  PetscLogDouble  t0,t1;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(PTATIN_MaterialPointPopulationControlRemove,0,0,0,0);CHKERRQ(ierr);

  ierr = DMDAGetElements_pTatinQ2P1(da,&nel,&nen,&elnidx);CHKERRQ(ierr);

  ierr = PetscMalloc( sizeof(PetscInt)*(nel),&cell_count );CHKERRQ(ierr);
  ierr = PetscMemzero( cell_count, sizeof(PetscInt)*(nel) );CHKERRQ(ierr);

  DataBucketGetSizes(db,&npoints32,NULL,NULL);

  /* compute number of points per cell */
  DataBucketGetDataFieldByName(db, MPntStd_classname ,&PField);
  DataFieldGetAccess(PField);
  for (p32=0; p32<npoints32; p32++) {
    MPntStd *marker_p;

    DataFieldAccessPoint(PField,p32,(void**)&marker_p);
    if (marker_p->wil < 0) { continue; }

    cell_count[ marker_p->wil ]++;
  }
  DataFieldRestoreAccess(PField);

  count = 0;
  for (c=0; c<nel; c++) {
    if (cell_count[c] > np_upper) {
      count++;
    }
  }

  if (count == 0) {
    ierr = PetscFree(cell_count);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(PTATIN_MaterialPointPopulationControlRemove,0,0,0,0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  PetscTime(&t0);

  if (!reverse_order_removal) {
    /* remove points from cells with excessive number */
    DataFieldGetAccess(PField);
    for (p32=0; p32<npoints32; p32++) {
      MPntStd *marker_p;
      int wil;

      DataFieldAccessPoint(PField,p32,(void**)&marker_p);
      wil = marker_p->wil;
      if (popctrl_mask[wil] == PETSC_TRUE) {
        if (cell_count[wil] > np_upper) {
          DataBucketRemovePointAtIndex(db,p32);

          DataBucketGetSizes(db,&npoints32,0,0); /* you need to update npoints as the list size decreases! */
          p32--; /* check replacement point */
          cell_count[wil]--;
        }
      }
    }
    DataFieldRestoreAccess(PField);
  }

  if (reverse_order_removal) {
    MPntStd *mp_std;
    int     wil;

    DataBucketGetDataFieldByName(db,MPntStd_classname,&PField);
    mp_std = PField->data;

    for (p32=npoints32-1; p32>=0; p32--) {

      wil = mp_std[p32].wil;
      if (wil < 0) { continue; }
  
      if (popctrl_mask[wil] == PETSC_TRUE) {
        if (cell_count[wil] > np_upper) {
          mp_std[p32].wil = -2;
          cell_count[wil]--;
        }
      }
    }

    for (p32=0; p32<npoints32; p32++) {
      wil = mp_std[p32].wil;
      if (wil == -2) {

        DataBucketRemovePointAtIndex(db,p32);
        DataBucketGetSizes(db,&npoints32,0,0); /* you need to update npoints as the list size decreases! */
        p32--; /* check replacement point */
        mp_std = PField->data;
      }
    }
  }

  PetscTime(&t1);

  ierr = PetscFree(cell_count);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PTATIN_MaterialPointPopulationControlRemove,0,0,0,0);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode AdaptMaterialPointResolution_Mask(pTatinCtx ctx)
{
  PetscErrorCode ierr;
  PetscInt       np_lower,np_upper,patch_extent,nxp,nyp,nzp;
  PetscReal      perturb;
  PetscBool      flg;
  PetscBool      *popctrl_mask; 
  DataBucket     db;
  PetscBool      reverse_order_removal;
  PetscInt       nel,nen;
  const PetscInt *elnidx;
  MPI_Comm       comm;

  PetscFunctionBegin;

  /* options for control number of points per cell */
  np_lower = 0;
  np_upper = 60;
  ierr = PetscOptionsGetInt(NULL,NULL,"-mp_popctrl_np_lower",&np_lower,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-mp_popctrl_np_upper",&np_upper,&flg);CHKERRQ(ierr);

  /* options for injection of markers */
  nxp = 2;
  nyp = 2;
  nzp = 2;
  ierr = PetscOptionsGetInt(NULL,NULL,"-mp_popctrl_nxp",&nxp,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-mp_popctrl_nyp",&nyp,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-mp_popctrl_nzp",&nzp,&flg);CHKERRQ(ierr);

  perturb = 0.1;
  ierr = PetscOptionsGetReal(NULL,NULL,"-mp_popctrl_perturb",&perturb,&flg);CHKERRQ(ierr);
  patch_extent = 1;
  ierr = PetscOptionsGetInt(NULL,NULL,"-mp_popctrl_patch_extent",&patch_extent,&flg);CHKERRQ(ierr);

  ierr = pTatinGetMaterialPoints(ctx,&db,NULL);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)ctx->stokes_ctx->dav,&comm);CHKERRQ(ierr);

  /* Get element number (nel)*/
  ierr = DMDAGetElements_pTatinQ2P1(ctx->stokes_ctx->dav,&nel,&nen,&elnidx);CHKERRQ(ierr);
  /* Allocate memory for the array */
  ierr = PetscMalloc1(nel,&popctrl_mask);CHKERRQ(ierr);
  
  ierr = MaterialPointResolutionMask_BoundaryFaces(ctx->stokes_ctx->dav,ctx,popctrl_mask);CHKERRQ(ierr);
  
  /* insertion */
  ierr = MPPC_NearestNeighbourPatch(np_lower,np_upper,patch_extent,nxp,nyp,nzp,perturb,ctx->stokes_ctx->dav,db);CHKERRQ(ierr);

  /* removal */
  if (np_upper != -1) {
    reverse_order_removal = PETSC_TRUE;
  ierr = MPPC_SimpleRemoval_Mask(np_upper,ctx->stokes_ctx->dav,db,reverse_order_removal,popctrl_mask);CHKERRQ(ierr);
  }

  ierr = PetscFree(popctrl_mask);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode ModelAdaptMaterialPointResolution_RiftNitsche(pTatinCtx c,void *ctx)
{
  ModelRiftNitscheCtx *data;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  data = (ModelRiftNitscheCtx*)ctx;

  /* Particles injection on faces */
  ierr = ModelApplyMaterialBoundaryCondition_RiftNitsche(c,data);CHKERRQ(ierr);

  /* Population control */
  //ierr = MaterialPointPopulationControl_v1(c);CHKERRQ(ierr);
  ierr = AdaptMaterialPointResolution_Mask(c);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode ModelDestroy_RiftNitsche(pTatinCtx c,void *ctx)
{
  ModelRiftNitscheCtx *data;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);
  data = (ModelRiftNitscheCtx*)ctx;

  /* Free contents of structure */
  /* destroy passive markers */
  ierr = PSwarmDestroy(&data->pswarm);CHKERRQ(ierr);
  /* Free structure */
  ierr = PetscFree(data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode pTatinModelRegister_RiftNitsche(void)
{
  ModelRiftNitscheCtx *data;
  pTatinModel         m;
  PetscErrorCode      ierr;

  PetscFunctionBegin;

  /* Allocate memory for the data structure for this model */
  ierr = PetscMalloc(sizeof(ModelRiftNitscheCtx),&data);CHKERRQ(ierr);
  ierr = PetscMemzero(data,sizeof(ModelRiftNitscheCtx));CHKERRQ(ierr);

  /* register user model */
  ierr = pTatinModelCreate(&m);CHKERRQ(ierr);

  /* Set name, model select via -ptatin_model NAME */
  ierr = pTatinModelSetName(m,"rift_nitsche");CHKERRQ(ierr);

  /* Set model data */
  ierr = pTatinModelSetUserData(m,data);CHKERRQ(ierr);

  /* Set function pointers */
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_INIT,                  (void (*)(void))ModelInitialize_RiftNitsche);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_MESH_GEOM,  (void (*)(void))ModelApplyInitialMeshGeometry_RiftNitsche);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_MAT_GEOM,   (void (*)(void))ModelApplyInitialMaterialParameters_RiftNitsche);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_STOKES_VARIABLE_MARKERS,(void (*)(void))ModelApplyInitialStokesVariableMarkers_RiftNitsche);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_SOLUTION,   (void (*)(void))ModelApplyInitialSolution_RiftNitsche);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_BC,              (void (*)(void))ModelApplyBoundaryConditions_RiftNitsche);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_BCMG,            (void (*)(void))ModelApplyBoundaryConditionMG_RiftNitsche);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_ADAPT_MP_RESOLUTION,   (void (*)(void))ModelAdaptMaterialPointResolution_RiftNitsche);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_UPDATE_MESH_GEOM,(void (*)(void))ModelApplyUpdateMeshGeometry_RiftNitsche);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_OUTPUT,                (void (*)(void))ModelOutput_RiftNitsche);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_DESTROY,               (void (*)(void))ModelDestroy_RiftNitsche);CHKERRQ(ierr);

  /* Insert model into list */
  ierr = pTatinModelRegister(m);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
