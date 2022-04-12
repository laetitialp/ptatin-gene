/*@ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**
**    Copyright (c) 2012
**        Dave A. May [dave.may@erdw.ethz.ch]
**        Institute of Geophysics
**        ETH Zürich
**        Sonneggstrasse 5
**        CH-8092 Zürich
**        Switzerland
**
**    project:    pTatin3d
**    filename:   model_ops_subduction_oblique.c
**
**
**    pTatin3d is free software: you can redistribute it and/or modify
**    it under the terms of the GNU General Public License as published
**    by the Free Software Foundation, either version 3 of the License,
**    or (at your option) any later version.
**
**    pTatin3d is distributed in the hope that it will be useful,
**    but WITHOUT ANY WARRANTY; without even the implied warranty of
**    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
**    See the GNU General Public License for more details.
**
**    You should have received a copy of the GNU General Public License
**    along with pTatin3d. If not, see <http://www.gnu.org/licenses/>.
**
** ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ @*/

/* Developed by Anthony Jourdon [jourdon.anthon@gmail.com] */

#include "petsc.h"
#include "ptatin3d.h"
#include "private/ptatin_impl.h"
#include "ptatin_utils.h"
#include "dmda_bcs.h"
#include "data_bucket.h"
#include "MPntStd_def.h"
#include "MPntPStokes_def.h"
#include "MPntPStokesPl_def.h"
#include "MPntPEnergy_def.h"
#include "output_material_points.h"
#include "output_material_points_p0.h"
#include "energy_output.h"
#include "output_paraview.h"
#include "material_point_std_utils.h"
#include "material_point_utils.h"
#include "material_point_popcontrol.h"
#include "stokes_form_function.h"
#include "ptatin_std_dirichlet_boundary_conditions.h"
#include "dmda_iterator.h"
#include "mesh_update.h"
#include "dmda_remesh.h"
#include "ptatin3d_stokes.h"
#include "ptatin3d_energy.h"
#include <ptatin3d_energyfv.h>
#include <ptatin3d_energyfv_impl.h>
#include "quadrature.h"
#include "QPntSurfCoefStokes_def.h"
#include "geometry_object.h"
#include <material_constants_energy.h>
#include "model_utils.h"
#include "rift2subd_ctx.h"

#include "element_utils_q1.h"
#include "element_type_Q2.h"
#include "dmda_element_q2p1.h"

static const char MODEL_NAME_RS[] = "model_RiftSubd_";

static PetscErrorCode ModelSetMaterialParameters(pTatinCtx c, ModelRiftSubdCtx *data);
PetscErrorCode SwarmMPntStd_CoordAssignment_FaceLatticeLayout3d_epsilon(DM da,PetscInt Nxp[],PetscReal perturb, PetscReal epsilon, PetscInt face_idx,DataBucket db);
static PetscReal Gaussian2D(PetscReal A, PetscReal a, PetscReal b, PetscReal c, PetscReal x, PetscReal x0, PetscReal z, PetscReal z0);


PetscErrorCode ModelInitialize_RiftSubd(pTatinCtx c,void *ctx)
{
  ModelRiftSubdCtx  *data;
  RheologyConstants *rheology;
  PetscInt          nn,i;
  PetscReal         cm_per_yer2m_per_sec = 1.0e-2 / ( 365.0 * 24.0 * 60.0 * 60.0 );
  PetscReal         Myr2sec = 1.0e6*3600.0*24.0*365.0;
  PetscBool         flg,found;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  data = (ModelRiftSubdCtx*)ctx;
  
  ierr = pTatinGetRheology(c,&rheology);CHKERRQ(ierr);
  rheology->rheology_type = RHEOLOGY_VP_STD;
  /* force energy equation to be introduced */
  ierr = PetscOptionsInsertString(NULL,"-activate_energyfv true");CHKERRQ(ierr);
  
  data->n_phases = 4;
  rheology->nphases_active = data->n_phases;
  rheology->apply_viscosity_cutoff_global = PETSC_TRUE;
  rheology->eta_upper_cutoff_global = 1.e+25;
  rheology->eta_lower_cutoff_global = 1.e+19;
  
  /* set the deffault values of the material constant for this particular model */
  /* scaling */
  data->length_bar    = 100.0 * 1.0e3;
  data->viscosity_bar = 1e22;
  data->velocity_bar  = 1.0e-10;
  /* cutoff */
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_RS,"-apply_viscosity_cutoff_global",&rheology->apply_viscosity_cutoff_global,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_RS,"-eta_lower_cutoff_global",&rheology->eta_lower_cutoff_global,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_RS,"-eta_upper_cutoff_global",&rheology->eta_upper_cutoff_global,NULL);CHKERRQ(ierr);
  
  /* box geometry, [m] */
  data->Lx = 1000.0e3; 
  data->Ly = 0.0e3;
  data->Lz = 600.0e3;
  data->Ox = 0.0e3;
  data->Oy = -680.0e3;
  data->Oz = 0.0e3;
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_RS,"-Lx",&data->Lx,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_RS,"-Ly",&data->Ly,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_RS,"-Lz",&data->Lz,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_RS,"-Ox",&data->Ox,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_RS,"-Oy",&data->Oy,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_RS,"-Oz",&data->Oz,&flg);CHKERRQ(ierr);
  /* reports before scaling */
  PetscPrintf(PETSC_COMM_WORLD,"********** Box Geometry **********\n",NULL);
  PetscPrintf(PETSC_COMM_WORLD,"[Ox,Lx] = [%+1.4e [m], %+1.4e [m]]\n", data->Ox ,data->Lx );
  PetscPrintf(PETSC_COMM_WORLD,"[Oy,Ly] = [%+1.4e [m], %+1.4e [m]]\n", data->Oy ,data->Ly );
  PetscPrintf(PETSC_COMM_WORLD,"[Oz,Lz] = [%+1.4e [m], %+1.4e [m]]\n", data->Oz ,data->Lz );
  
  data->y_continent[0] = -25.0e3; // Conrad
  data->y_continent[1] = -35.0e3; // Moho
  data->y_continent[2] = -120.0e3; // LAB
  
  data->wz = 20.0e3; // weak zone width
  
  /* Velocity */
  /* Orthogonal extension */
  data->v_extension = 1.0;
  /* Timing for BCs change */
  data->BC_time[0] = 10.0;
  data->BC_time[1] = 11.0;
  data->BC_time[2] = 21.0;
  data->BC_time[3] = 22.0;
  /* Oblique Compression */
  data->normV = 1.0;
  /* Angle of the velocity vector with the face on which it is applied */
  data->angle_v = 30.0;
  
  data->Ttop = 0.0; // Top temperature BC
  data->Tbottom = 1600.0; // Bottom temperature BC
  
  nn = 3;
  ierr = PetscOptionsGetRealArray(NULL,MODEL_NAME_RS,"-y_continent",data->y_continent,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != 3) {
      SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_USER,"Expected 3 values for -y_continent. Found %d",nn);
    }
  }
  
  /* Options for BCs */
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_RS,"-normV",        &data->normV,NULL);CHKERRQ(ierr);  
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_RS,"-angle_v",      &data->angle_v,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_RS,"-v_extension",  &data->v_extension,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_RS,"-wz_width",     &data->wz,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_RS,"-Ttop",         &data->Ttop,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_RS,"-Tbottom",      &data->Tbottom,NULL);CHKERRQ(ierr);
  nn = 10;
  ierr = PetscOptionsGetRealArray(NULL,MODEL_NAME_RS,"-bc_time",data->BC_time,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != 4) {
      SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_USER,"Expected 4 values for -bc_time. Found %d",nn);
    }
  }
  
  for (i=0;i<4;i++){
    PetscPrintf(PETSC_COMM_WORLD,"BC time [%d] = %1.3e Myr\n",i,data->BC_time[i]);
  }
  
  data->output_markers       = PETSC_FALSE;
  data->is_2D                = PETSC_FALSE;
  data->open_base            = PETSC_FALSE;
  data->freeslip_z           = PETSC_FALSE;
  data->litho_plitho_z       = PETSC_FALSE;
  data->full_face_plitho_z   = PETSC_FALSE;
  data->full_face_plithoKMAX = PETSC_FALSE;
  data->litho_plithoKMAX     = PETSC_FALSE;
  data->notches              = PETSC_FALSE;
  data->straight_wz          = PETSC_FALSE;
  data->one_notch            = PETSC_FALSE;
  data->use_v_dot_n          = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_RS,"-output_markers",       &data->output_markers,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_RS,"-is_2D",                &data->is_2D,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_RS,"-open_base",            &data->open_base,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_RS,"-freeslip_z",           &data->freeslip_z,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_RS,"-litho_plitho_z",       &data->litho_plitho_z,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_RS,"-full_face_plitho_z",   &data->full_face_plitho_z,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_RS,"-full_face_plithoKMAX", &data->full_face_plithoKMAX,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_RS,"-litho_plithoKMAX",     &data->litho_plithoKMAX,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_RS,"-notches",              &data->notches,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_RS,"-straight_wz",          &data->straight_wz,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_RS,"-one_notch",            &data->one_notch,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,MODEL_NAME_RS,"-use_v_dot_n",          &data->use_v_dot_n,NULL);CHKERRQ(ierr);
  
  /* Surface diffusion */
  data->Kero = 1.0e-6;
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_RS,"-Kero",&data->Kero,NULL);CHKERRQ(ierr);
  data->Kero = data->Kero / (data->length_bar*data->length_bar/data->time_bar);
  
  /* Compute additional scaling parameters */
  data->time_bar         = data->length_bar / data->velocity_bar;
  data->pressure_bar     = data->viscosity_bar/data->time_bar;
  data->density_bar      = data->pressure_bar * (data->time_bar*data->time_bar)/(data->length_bar*data->length_bar); // kg.m^-3
  data->acceleration_bar = data->length_bar / (data->time_bar*data->time_bar);
  
  /* Scale viscosity cutoff */
  rheology->eta_lower_cutoff_global = rheology->eta_lower_cutoff_global / data->viscosity_bar;
  rheology->eta_upper_cutoff_global = rheology->eta_upper_cutoff_global / data->viscosity_bar;
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
  data->wz             = data->wz             / data->length_bar;
  /* Scale velocity */
  data->v_extension = data->v_extension*cm_per_yer2m_per_sec/data->velocity_bar;
  data->normV       = data->normV*cm_per_yer2m_per_sec/data->velocity_bar;
  data->angle_v     = data->angle_v*M_PI/180.0;
  /* Scale time for BCs */
  for(i=0;i<4;i++){
    data->BC_time[i] = data->BC_time[i]*Myr2sec/data->time_bar;
  }
  
  ierr = ModelSetMaterialParameters(c,data);CHKERRQ(ierr);
  
  PetscPrintf(PETSC_COMM_WORLD,"[rift2subd]:  during the solve scaling is done using \n");
  PetscPrintf(PETSC_COMM_WORLD,"  L*    : %1.4e [m]\n",       data->length_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  U*    : %1.4e [m.s^-1]\n",  data->velocity_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  t*    : %1.4e [s]\n",       data->time_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  eta*  : %1.4e [Pa.s]\n",    data->viscosity_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  rho*  : %1.4e [kg.m^-3]\n", data->density_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  P*    : %1.4e [Pa]\n",      data->pressure_bar );
  PetscPrintf(PETSC_COMM_WORLD,"  a*    : %1.4e [m.s^-2]\n",  data->acceleration_bar );
  
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelSetMaterialParameters(pTatinCtx c, ModelRiftSubdCtx *data)
{
  DataField                 PField,PField_k,PField_Q;
  EnergyConductivityConst   *data_k;
  EnergySourceConst         *data_Q;
  DataBucket                materialconstants;
  EnergyMaterialConstants   *matconstants_e;
  PetscInt                  region_idx;
  int                       source_type[7] = {0, 0, 0, 0, 0, 0, 0};
  PetscReal                 *preexpA,*Ascale,*entalpy,*Vmol,*nexp,*Tref;
  PetscReal                 *phi,*phi_inf,*Co,*Co_inf,*Tens_cutoff,*Hst_cutoff,*eps_min,*eps_max;
  PetscReal                 *beta,*alpha,*rho,*heat_source,*conductivity;
  PetscReal                 phi_rad,phi_inf_rad,Cp;
  char                      *option_name;
  PetscErrorCode            ierr;
  
  PetscFunctionBegin;
  /* Material constants */
  ierr = pTatinGetMaterialConstants(c,&materialconstants);CHKERRQ(ierr);
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
  
  /* Allocate memory for arrays of material parameters */
  ierr = PetscMalloc1(data->n_phases,&preexpA     );CHKERRQ(ierr);
  ierr = PetscMalloc1(data->n_phases,&Ascale      );CHKERRQ(ierr);
  ierr = PetscMalloc1(data->n_phases,&entalpy     );CHKERRQ(ierr);
  ierr = PetscMalloc1(data->n_phases,&Vmol        );CHKERRQ(ierr);
  ierr = PetscMalloc1(data->n_phases,&nexp        );CHKERRQ(ierr);
  ierr = PetscMalloc1(data->n_phases,&Tref        );CHKERRQ(ierr);
  ierr = PetscMalloc1(data->n_phases,&phi         );CHKERRQ(ierr);
  ierr = PetscMalloc1(data->n_phases,&phi_inf     );CHKERRQ(ierr);
  ierr = PetscMalloc1(data->n_phases,&Co          );CHKERRQ(ierr);
  ierr = PetscMalloc1(data->n_phases,&Co_inf      );CHKERRQ(ierr);
  ierr = PetscMalloc1(data->n_phases,&Tens_cutoff );CHKERRQ(ierr);
  ierr = PetscMalloc1(data->n_phases,&Hst_cutoff  );CHKERRQ(ierr);
  ierr = PetscMalloc1(data->n_phases,&eps_min     );CHKERRQ(ierr);
  ierr = PetscMalloc1(data->n_phases,&eps_max     );CHKERRQ(ierr);
  ierr = PetscMalloc1(data->n_phases,&beta        );CHKERRQ(ierr);
  ierr = PetscMalloc1(data->n_phases,&alpha       );CHKERRQ(ierr);
  ierr = PetscMalloc1(data->n_phases,&rho         );CHKERRQ(ierr);
  ierr = PetscMalloc1(data->n_phases,&heat_source );CHKERRQ(ierr);
  ierr = PetscMalloc1(data->n_phases,&conductivity);CHKERRQ(ierr);
  /* Set default values for parameters */
  source_type[0] = ENERGYSOURCE_CONSTANT;
  Cp             = 800.0;
  for (region_idx=0;region_idx<data->n_phases;region_idx++) {
    preexpA[region_idx]     = 6.3e-6;
    Ascale[region_idx]      = 1.0e6;
    entalpy[region_idx]     = 156.0e3;
    Vmol[region_idx]        = 0.0;
    nexp[region_idx]        = 2.4;
    Tref[region_idx]        = 273.0;
    phi[region_idx]         = 30.0;
    phi_inf[region_idx]     = 5.0;
    Co[region_idx]          = 2.0e7;
    Co_inf[region_idx]      = 2.0e7;
    Tens_cutoff[region_idx] = 1.0e7;
    Hst_cutoff[region_idx]  = 2.0e8;
    eps_min[region_idx]     = 0.0;
    eps_max[region_idx]     = 0.5;
    beta[region_idx]        = 0.0;
    alpha[region_idx]       = 2.0e-5;
    rho[region_idx]         = 2700.0;
    heat_source[region_idx] = 0.0;
    conductivity[region_idx] = 1.0;
  }
  for (region_idx=0;region_idx<data->n_phases;region_idx++) {
    /* Set material constitutive laws */
    MaterialConstantsSetValues_MaterialType(materialconstants,region_idx,VISCOUS_ARRHENIUS_2,PLASTIC_DP,SOFTENING_LINEAR,DENSITY_BOUSSINESQ);

    /* VISCOUS PARAMETERS */
    if (asprintf (&option_name, "-preexpA_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_RS, option_name,&preexpA[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-Ascale_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_RS, option_name,&Ascale[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-entalpy_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_RS, option_name,&entalpy[region_idx],NULL);CHKERRQ(ierr);
    free (option_name); 
    if (asprintf (&option_name, "-Vmol_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_RS, option_name,&Vmol[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-nexp_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_RS, option_name,&nexp[region_idx],NULL);CHKERRQ(ierr);
    free (option_name); 
    if (asprintf (&option_name, "-Tref_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_RS, option_name,&Tref[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);
    /* Set viscous params for region_idx */
    MaterialConstantsSetValues_ViscosityArrh(materialconstants,region_idx,preexpA[region_idx],Ascale[region_idx],entalpy[region_idx],Vmol[region_idx],nexp[region_idx],Tref[region_idx]);  

    /* PLASTIC PARAMETERS */
    if (asprintf (&option_name, "-phi_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_RS, option_name,&phi[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-phi_inf_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_RS, option_name,&phi_inf[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-Co_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_RS, option_name,&Co[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-Co_inf_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_RS, option_name,&Co_inf[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-Tens_cutoff_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_RS, option_name,&Tens_cutoff[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-Hst_cutoff_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_RS, option_name,&Hst_cutoff[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-eps_min_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_RS, option_name,&eps_min[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-eps_max_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_RS, option_name,&eps_max[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);

    phi_rad     = M_PI * phi[region_idx]/180.0;
    phi_inf_rad = M_PI * phi_inf[region_idx]/180.0;
    /* Set plastic params for region_idx */
    MaterialConstantsSetValues_PlasticDP(materialconstants,region_idx,phi_rad,phi_inf_rad,Co[region_idx],Co_inf[region_idx],Tens_cutoff[region_idx],Hst_cutoff[region_idx]);
    MaterialConstantsSetValues_SoftLin(materialconstants,region_idx,eps_min[region_idx],eps_max[region_idx]);

    /* ENERGY PARAMETERS */
    if (asprintf (&option_name, "-alpha_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_RS, option_name,&alpha[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-beta_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_RS, option_name,&beta[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-rho_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_RS, option_name,&rho[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-heat_source_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_RS, option_name,&heat_source[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);
    if (asprintf (&option_name, "-conductivity_%d", (int)region_idx) < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"asprintf() failed");
    ierr = PetscOptionsGetReal(NULL,MODEL_NAME_RS, option_name,&conductivity[region_idx],NULL);CHKERRQ(ierr);
    free (option_name);
    
    /* Set energy params for region_idx */
    MaterialConstantsSetValues_EnergyMaterialConstants(region_idx,matconstants_e,alpha[region_idx],beta[region_idx],rho[region_idx],Cp,ENERGYDENSITY_CONSTANT,ENERGYCONDUCTIVITY_CONSTANT,source_type);
    MaterialConstantsSetValues_DensityBoussinesq(materialconstants,region_idx,rho[region_idx],alpha[region_idx],beta[region_idx]);
    EnergySourceConstSetField_HeatSource(&data_Q[region_idx],heat_source[region_idx]);
    EnergyConductivityConstSetField_k0(&data_k[region_idx],conductivity[region_idx]);
  }
  
  /* Report all material parameters values */
  for (region_idx=0; region_idx<data->n_phases;region_idx++) {
    MaterialConstantsPrintAll(materialconstants,region_idx);
    MaterialConstantsEnergyPrintAll(materialconstants,region_idx);
  }
  
  /* scale material properties */
  for (region_idx=0; region_idx<data->n_phases;region_idx++) {
    MaterialConstantsScaleAll(materialconstants,region_idx,data->length_bar,data->velocity_bar,data->time_bar,data->viscosity_bar,data->density_bar,data->pressure_bar);
    MaterialConstantsEnergyScaleAll(materialconstants,region_idx,data->length_bar,data->time_bar,data->pressure_bar);
  }
    
  ierr = PetscFree(preexpA     );CHKERRQ(ierr);
  ierr = PetscFree(Ascale      );CHKERRQ(ierr);
  ierr = PetscFree(entalpy     );CHKERRQ(ierr);
  ierr = PetscFree(Vmol        );CHKERRQ(ierr);
  ierr = PetscFree(nexp        );CHKERRQ(ierr);
  ierr = PetscFree(Tref        );CHKERRQ(ierr);
  ierr = PetscFree(phi         );CHKERRQ(ierr);
  ierr = PetscFree(phi_inf     );CHKERRQ(ierr);
  ierr = PetscFree(Co          );CHKERRQ(ierr);
  ierr = PetscFree(Co_inf      );CHKERRQ(ierr);
  ierr = PetscFree(Tens_cutoff );CHKERRQ(ierr);
  ierr = PetscFree(Hst_cutoff  );CHKERRQ(ierr);
  ierr = PetscFree(eps_min     );CHKERRQ(ierr);
  ierr = PetscFree(eps_max     );CHKERRQ(ierr);
  ierr = PetscFree(beta        );CHKERRQ(ierr);
  ierr = PetscFree(alpha       );CHKERRQ(ierr);
  ierr = PetscFree(rho         );CHKERRQ(ierr);
  ierr = PetscFree(heat_source );CHKERRQ(ierr);
  ierr = PetscFree(conductivity);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyInitialMeshGeometry_RiftSubd(pTatinCtx c,void *ctx)
{
  ModelRiftSubdCtx *data = (ModelRiftSubdCtx*)ctx;
  PhysCompStokes   stokes;
  DM               stokes_pack,dav,dap;
  PetscInt         dir,npoints;
  PetscReal        *xref,*xnat;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  ierr = pTatinGetStokesContext(c,&stokes);CHKERRQ(ierr);
  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(dav,data->Ox,data->Lx,data->Oy,data->Ly,data->Oz,data->Lz);CHKERRQ(ierr);
  if (data->is_2D) {
    ierr = pTatin3d_DefineVelocityMeshGeometryQuasi2D(c);CHKERRQ(ierr);
  }
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
  ierr = DMDABilinearizeQ2Elements(dav);CHKERRQ(ierr);
  
  PetscReal gvec[] = { 0.0, -9.8, 0.0 };
  ierr = PhysCompStokesSetGravityVector(c->stokes_ctx,gvec);CHKERRQ(ierr);
  ierr = PhysCompStokesScaleGravityVector(c->stokes_ctx,1.0/data->acceleration_bar);CHKERRQ(ierr);

  ierr = PetscFree(xref);CHKERRQ(ierr);
  ierr = PetscFree(xnat);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

static PetscReal Gaussian2D(PetscReal A, PetscReal a, PetscReal b, PetscReal c, PetscReal x, PetscReal x0, PetscReal z, PetscReal z0)
{ 
  PetscReal value=0.0;
  value = A * (PetscExpReal( -( a*(x-x0)*(x-x0) + 2*b*(x-x0)*(z-z0) + c*(z-z0)*(z-z0) ) ) );
  return value;
}

static PetscErrorCode ModelApplyInitialMaterialGeometry_RiftSubd_StraightWZ(pTatinCtx c,ModelRiftSubdCtx *data)
{
  DataBucket                db;
  DataField                 PField_std,PField_pls;
  PetscInt                  p;
  PetscReal                 x0,z0,sigma_x,sigma_z;
  int                       n_mp_points;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  
  /* define properties on material points */
  db = c->materialpoint_db;
  DataBucketGetDataFieldByName(db,MPntStd_classname,&PField_std);
  DataFieldGetAccess(PField_std);
  DataFieldVerifyAccess(PField_std,sizeof(MPntStd));

  DataBucketGetDataFieldByName(db,MPntPStokesPl_classname,&PField_pls);
  DataFieldGetAccess(PField_pls);
  DataFieldVerifyAccess(PField_pls,sizeof(MPntPStokesPl));
  
  /* Parameters for the gaussian weak zone */
  x0 = 0.5*data->Lx;
  z0 = 0.0/data->length_bar;
  sigma_x = 1.8;
  sigma_z = 1.0e-1;
  
  DataBucketGetSizes(db,&n_mp_points,0,0);
  for (p=0; p<n_mp_points; p++) {
    MPntStd       *material_point;
    MPntPStokesPl *mpprop_pls;
    PetscReal     a,b;
    int           region_idx;
    double        *position;
    float         pls;
    short         yield;

    DataFieldAccessPoint(PField_std,p,(void**)&material_point);
    DataFieldAccessPoint(PField_pls,p,(void**)&mpprop_pls);

    /* Access using the getter function */
    MPntStdGetField_global_coord(material_point,&position);
    /* Set an initial small random noise on plastic strain */
    pls = ptatin_RandomNumberGetDouble(0.0,0.03);
    /* Set yield to none */
    yield = 0;
    /* Set default region index to 0 */
    region_idx = 0;
    /* Attribute region index to layers */
    if (position[1] <= data->y_continent[0]) { region_idx = 1; }
    if (position[1] <= data->y_continent[1]) { region_idx = 2; }
    if (position[1] <= data->y_continent[2]) { region_idx = 3; }
    /* Set an initial plastic strain as a weak zone */
    if (position[1] >= data->y_continent[2]) {
      a = position[0]-x0;
      b = position[2]-z0;
      pls += ptatin_RandomNumberGetDouble(0.0,0.6) * PetscExpReal( -( PetscPowReal(a,2)/2.0*PetscPowReal(sigma_x,2) +
                                                                      PetscPowReal(b,2)/2.0*PetscPowReal(sigma_z,2) ) );
    }
    MPntStdSetField_phase_index(material_point,region_idx);
    MPntPStokesPlSetField_yield_indicator(mpprop_pls,yield);
    MPntPStokesPlSetField_plastic_strain(mpprop_pls,pls);
  }
  DataFieldRestoreAccess(PField_std);
  DataFieldRestoreAccess(PField_pls);
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyInitialMaterialGeometry_RiftSubd_2Notches(pTatinCtx c,ModelRiftSubdCtx *data)
{
  DataBucket                db;
  DataField                 PField_std,PField_pls;
  PetscInt                  p;
  PetscReal                 x0,z0,x1,z1,xc,sigma_x,sigma_z;
  int                       n_mp_points;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  
  /* define properties on material points */
  db = c->materialpoint_db;
  DataBucketGetDataFieldByName(db,MPntStd_classname,&PField_std);
  DataFieldGetAccess(PField_std);
  DataFieldVerifyAccess(PField_std,sizeof(MPntStd));

  DataBucketGetDataFieldByName(db,MPntPStokesPl_classname,&PField_pls);
  DataFieldGetAccess(PField_pls);
  DataFieldVerifyAccess(PField_pls,sizeof(MPntPStokesPl));
  
  /* Parameters for the gaussian weak zone */
  xc = data->Ox + (data->Lx - data->Ox)/2.0;
  x0 = xc - data->wz/2.0;
  x1 = xc + data->wz/2.0;
  
  z0 = 100.0e+3/data->length_bar;
  z1 = 500.0e+3/data->length_bar;
  
  sigma_x = 2.1;
  sigma_z = 1.0e-1;
  
  DataBucketGetSizes(db,&n_mp_points,0,0);
  for (p=0; p<n_mp_points; p++) {
    MPntStd       *material_point;
    MPntPStokesPl *mpprop_pls;
    PetscReal     a,b,n0,n1;
    int           region_idx;
    double        *position;
    float         pls;
    short         yield;

    DataFieldAccessPoint(PField_std,p,(void**)&material_point);
    DataFieldAccessPoint(PField_pls,p,(void**)&mpprop_pls);

    /* Access using the getter function */
    MPntStdGetField_global_coord(material_point,&position);
    /* Set an initial small random noise on plastic strain */
    pls = ptatin_RandomNumberGetDouble(0.0,0.03);
    /* Set yield to none */
    yield = 0;
    /* Set default region index to 0 */
    region_idx = 0;
    /* Attribute region index to layers */
    if (position[1] <= data->y_continent[0]) { region_idx = 1; }
    if (position[1] <= data->y_continent[1]) { region_idx = 2; }
    if (position[1] <= data->y_continent[2]) { region_idx = 3; }
    /* Set an initial plastic strain as a weak zone */
    if (position[1] >= data->y_continent[2]) {
      if (position[2] <= z0) {
        a  = position[0]-x0;
        b  = position[2]-z0;
        n0 = ptatin_RandomNumberGetDouble(0.0,0.6) * PetscExpReal( -( PetscPowReal(a,2)/2.0*PetscPowReal(sigma_x,2) +
                                                                      PetscPowReal(b,2)/2.0*PetscPowReal(sigma_z,2) ) );
      } else {
        n0 = 0.0;
      }
      if (position[2] >= z1) {
        a  = position[0]-x1;
        b  = position[2]-z1;
        n1 = ptatin_RandomNumberGetDouble(0.0,0.6) * PetscExpReal( -( PetscPowReal(a,2)/2.0*PetscPowReal(sigma_x,2) +
                                                                      PetscPowReal(b,2)/2.0*PetscPowReal(sigma_z,2) ) );
      } else {
        n1 = 0.0;
      }
      pls += n0 + n1;
    }
    MPntStdSetField_phase_index(material_point,region_idx);
    MPntPStokesPlSetField_yield_indicator(mpprop_pls,yield);
    MPntPStokesPlSetField_plastic_strain(mpprop_pls,pls);
  }
  DataFieldRestoreAccess(PField_std);
  DataFieldRestoreAccess(PField_pls);
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyInitialMaterialGeometry_RiftSubd_1Notch(pTatinCtx c,ModelRiftSubdCtx *data)
{
  DataBucket                db;
  DataField                 PField_std,PField_pls;
  PetscInt                  p,nn;
  PetscReal                 x0,z0,sigma_x,sigma_z,L_frac[2];
  PetscReal                 aa,bb,cc;
  PetscBool                 found=PETSC_FALSE;
  int                       n_mp_points;
  PetscErrorCode            ierr;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  
  /* define properties on material points */
  db = c->materialpoint_db;
  DataBucketGetDataFieldByName(db,MPntStd_classname,&PField_std);
  DataFieldGetAccess(PField_std);
  DataFieldVerifyAccess(PField_std,sizeof(MPntStd));

  DataBucketGetDataFieldByName(db,MPntPStokesPl_classname,&PField_pls);
  DataFieldGetAccess(PField_pls);
  DataFieldVerifyAccess(PField_pls,sizeof(MPntPStokesPl));
  
  /* Parameters for the gaussian weak zone */
  L_frac[0] = L_frac[1] = 0.5;
  nn = 10;
  ierr = PetscOptionsGetRealArray(NULL,MODEL_NAME_RS,"-notch_position",L_frac,&nn,&found);CHKERRQ(ierr);
  if (found) {
    if (nn != 2) {
      SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_USER,"Expected 2 values for -notch_position. Found %d",nn);
    }
  }
  x0 = data->Ox + (data->Lx - data->Ox)*L_frac[0];
  z0 = data->Oz + (data->Lz - data->Oz)*L_frac[1];
  
  sigma_x = 3.0e-1;
  sigma_z = 3.0e-1;
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_RS,"-sigma_x",&sigma_x,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,MODEL_NAME_RS,"-sigma_z",&sigma_z,NULL);CHKERRQ(ierr);
  aa = 0.5*sigma_x*sigma_x;
  bb = 0.0;
  cc = 0.5*sigma_z*sigma_z;

  DataBucketGetSizes(db,&n_mp_points,0,0);
  for (p=0; p<n_mp_points; p++) {
    MPntStd       *material_point;
    MPntPStokesPl *mpprop_pls;
    PetscReal     n0;
    int           region_idx;
    double        *position;
    float         pls;
    short         yield;

    DataFieldAccessPoint(PField_std,p,(void**)&material_point);
    DataFieldAccessPoint(PField_pls,p,(void**)&mpprop_pls);

    /* Access using the getter function */
    MPntStdGetField_global_coord(material_point,&position);
    /* Set an initial small random noise on plastic strain */
    pls = ptatin_RandomNumberGetDouble(0.0,0.03);
    /* Set yield to none */
    yield = 0;
    /* Set default region index to 0 */
    region_idx = 0;
    /* Attribute region index to layers */
    if (position[1] <= data->y_continent[0]) { region_idx = 1; }
    if (position[1] <= data->y_continent[1]) { region_idx = 2; }
    if (position[1] <= data->y_continent[2]) { region_idx = 3; }
    /* Set an initial plastic strain as a weak zone */
    if (position[1] >= data->y_continent[2]) {
      /* Compute a gaussian repartition for the random value set on the plastic strain */
      n0 = Gaussian2D(ptatin_RandomNumberGetDouble(0.0,0.8),aa,bb,cc,position[0],x0,position[2],z0);
      pls += n0;
    }
    MPntStdSetField_phase_index(material_point,region_idx);
    MPntPStokesPlSetField_yield_indicator(mpprop_pls,yield);
    MPntPStokesPlSetField_plastic_strain(mpprop_pls,pls);
  }
  DataFieldRestoreAccess(PField_std);
  DataFieldRestoreAccess(PField_pls);
  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyInitialMaterialGeometry_RiftSubd(pTatinCtx c,void *ctx)
{
  ModelRiftSubdCtx *data = (ModelRiftSubdCtx*)ctx;
  PetscInt         weak_zone_type;
  PetscErrorCode   ierr;
  PetscFunctionBegin;
  
  weak_zone_type = -1;
  if (data->straight_wz) {
    weak_zone_type = 0;
  } else if (data->notches) {
    weak_zone_type = 1;
  } else if (data->one_notch) {
    weak_zone_type = 2;
  }

  switch(weak_zone_type) 
  {
    case 0:
      ierr = ModelApplyInitialMaterialGeometry_RiftSubd_StraightWZ(c,data);CHKERRQ(ierr);
      break;

    case 1:
      ierr = ModelApplyInitialMaterialGeometry_RiftSubd_2Notches(c,data);CHKERRQ(ierr);
      break;

    case 2:
      ierr = ModelApplyInitialMaterialGeometry_RiftSubd_1Notch(c,data);CHKERRQ(ierr);
      break;

    default:
      PetscPrintf(PETSC_COMM_WORLD,"[[ WARNING ]] INITIAL WEAK ZONE NOT SET, RUNNING DEFAULT STRAIGHT WZ !!!\n");
      ierr = ModelApplyInitialMaterialGeometry_RiftSubd_StraightWZ(c,data);CHKERRQ(ierr);
      break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyInitialSolution_RiftSubd(pTatinCtx c,Vec X,void *ctx)
{
  ModelRiftSubdCtx                             *data;
  DM                                           stokes_pack,dau,dap;
  Vec                                          velocity,pressure;
  PetscReal                                    vxl,vxr,vzl,vzr;
  DMDAVecTraverse3d_HydrostaticPressureCalcCtx HPctx;
  DMDAVecTraverse3d_InterpCtx                  IntpCtx;
  PetscReal                                    MeshMin[3],MeshMax[3],domain_height;
  PetscBool                                    active_energy;
  PetscErrorCode                               ierr;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  data = (ModelRiftSubdCtx*)ctx;
  
  vxl = -data->v_extension;
  vxr =  data->v_extension;
  vzl = 0.0;
  vzr = 0.0;
  
  stokes_pack = c->stokes_ctx->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dau,&dap);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);
  
  ierr = VecZeroEntries(velocity);CHKERRQ(ierr);
  
  ierr = DMDAVecTraverse3d_InterpCtxSetUp_X(&IntpCtx,(vxr-vxl)/(data->Lx-data->Ox),vxl,0.0);CHKERRQ(ierr);
  ierr = DMDAVecTraverse3d(dau,velocity,0,DMDAVecTraverse3d_Interp,(void*)&IntpCtx);CHKERRQ(ierr);
  ierr = DMDAVecTraverse3d_InterpCtxSetUp_Y(&IntpCtx,0.0,0.0,0.0);CHKERRQ(ierr);
  ierr = DMDAVecTraverse3d(dau,velocity,1,DMDAVecTraverse3d_Interp,(void*)&IntpCtx);CHKERRQ(ierr);
  ierr = DMDAVecTraverse3d_InterpCtxSetUp_Z(&IntpCtx,(vzl-vzr)/(data->Lz-data->Oz),vzl,0.0);CHKERRQ(ierr);
  ierr = DMDAVecTraverse3d(dau,velocity,2,DMDAVecTraverse3d_Interp,(void*)&IntpCtx);CHKERRQ(ierr);
  
  /* Pressure IC */
  ierr = VecZeroEntries(pressure);CHKERRQ(ierr);
  ierr = DMGetBoundingBox(dau,MeshMin,MeshMax);CHKERRQ(ierr);
  domain_height = MeshMax[1] - MeshMin[1];

  HPctx.surface_pressure = 0.0;
  HPctx.ref_height = domain_height;
  HPctx.ref_N      = c->stokes_ctx->my-1;
  HPctx.grav       = 9.8 / data->acceleration_bar;
  HPctx.rho        = 3300.0 / data->density_bar;

  ierr = DMDAVecTraverseIJK(dap,pressure,0,DMDAVecTraverseIJK_HydroStaticPressure_v2,     (void*)&HPctx);CHKERRQ(ierr);
  ierr = DMDAVecTraverseIJK(dap,pressure,2,DMDAVecTraverseIJK_HydroStaticPressure_dpdy_v2,(void*)&HPctx);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);
  
  /* Temperature IC */
  ierr = pTatinContextValid_EnergyFV(c,&active_energy);CHKERRQ(ierr);
  if (active_energy) {
    PhysCompEnergyFV energy;
    PetscBool        flg = PETSC_FALSE,subduction_temperature_ic_from_file = PETSC_FALSE;
    char             fname[PETSC_MAX_PATH_LEN],temperature_file[PETSC_MAX_PATH_LEN];
    PetscViewer      viewer;
    
    ierr = pTatinGetContext_EnergyFV(c,&energy);CHKERRQ(ierr);
    /* If job is restarted skip that part (Temperature is loaded from checkpointed file) */
    if (!c->restart_from_file) {
      ierr = PetscOptionsGetBool(NULL,MODEL_NAME_RS,"-temperature_ic_from_file",&subduction_temperature_ic_from_file,NULL);CHKERRQ(ierr);
      if (subduction_temperature_ic_from_file) {
        /* Check if a file is provided */
        ierr = PetscOptionsGetString(NULL,MODEL_NAME_RS,"-temperature_file",temperature_file,PETSC_MAX_PATH_LEN-1,&flg);CHKERRQ(ierr);
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
  }
  
  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyInitialStokesVariableMarkers_RiftSubd(pTatinCtx c,Vec X,void *ctx)
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

PetscBool BCListEvaluator_Lithosphere(PetscScalar position[],PetscScalar *value,void *ctx)
{
  PetscBool      impose_dirichlet;
  BC_Lithosphere data_ctx = (BC_Lithosphere)ctx;
  
  PetscFunctionBegin;
  
  if (position[1] >= data_ctx->y_lab) {
    *value = data_ctx->v;
    impose_dirichlet = PETSC_TRUE;
  } else {
    impose_dirichlet = PETSC_FALSE;
  }
  return impose_dirichlet;
}

static PetscErrorCode ModelApplyBoundaryCondition_Transition(BCList bclist,DM dav,pTatinCtx c,ModelRiftSubdCtx *data)
{
  BC_Lithosphere bcdata;
  PetscReal      time,zero=0.0;
  PetscReal      vx,vz,vx_comp,vz_comp,vxl,vzl,vxr,vzr;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  ierr = PetscMalloc(sizeof(struct _p_BC_Lithosphere),&bcdata);CHKERRQ(ierr);
  
  ierr = pTatinGetTime(c,&time);
  /* Compression velocity */
  vz_comp = data->normV*PetscCosReal(data->angle_v);
  vx_comp = PetscSqrtReal(data->normV*data->normV - vz_comp*vz_comp);
  
  if (time >= data->BC_time[0] && time < data->BC_time[1]) {
    PetscPrintf(PETSC_COMM_WORLD,"[[ DECREASING VELOCITY ]]\n");
    vx = ((0.0-data->v_extension)/(data->BC_time[1]-data->BC_time[0])) * (time-data->BC_time[1]);
    vz = 0.0;
    vxl = -vx; vxr =  vx;
    vzl = -vz; vzr =  vz;
  } else if (time >= data->BC_time[1] && time < data->BC_time[2]) {
    PetscPrintf(PETSC_COMM_WORLD,"[[ RELAXING ]]\n");
    vxl = 0.0; vxr = 0.0;
    vzl = 0.0; vzr = 0.0;
  } else if (time >= data->BC_time[2] && time < data->BC_time[3]) {
    // Increase velocity to reach compression velocity
    PetscPrintf(PETSC_COMM_WORLD,"[[ INCREASING VELOCITY ]]\n");
    vx = ((0.0-vx_comp)/(data->BC_time[2]-data->BC_time[3])) * (time-data->BC_time[2]);
    vz = ((0.0-vz_comp)/(data->BC_time[2]-data->BC_time[3])) * (time-data->BC_time[2]);  
    vxl = vx; vxr = -vx;
    vzl = vz; vzr = -vz;
  }
  
  /* Apply the velocities */
  bcdata->y_lab = data->y_continent[2];
  bcdata->v = vxl;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,0,BCListEvaluator_Lithosphere,(void*)bcdata);CHKERRQ(ierr);
  bcdata->v = vzl;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,2,BCListEvaluator_Lithosphere,(void*)bcdata);CHKERRQ(ierr);
  bcdata->v = 0.0;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,1,BCListEvaluator_Lithosphere,(void*)bcdata);CHKERRQ(ierr);
  
  bcdata->y_lab = data->y_continent[2];
  bcdata->v = vxr;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,0,BCListEvaluator_Lithosphere,(void*)bcdata);CHKERRQ(ierr);
  bcdata->v = vzr;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,2,BCListEvaluator_Lithosphere,(void*)bcdata);CHKERRQ(ierr);
  bcdata->v = 0.0;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,1,BCListEvaluator_Lithosphere,(void*)bcdata);CHKERRQ(ierr);
  
  /* Free slip bottom */
  if (!data->open_base) {
    ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  }
  ierr = PetscFree(bcdata);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyBoundaryCondition_ObliqueCompression(BCList bclist,DM dav,pTatinCtx c,ModelRiftSubdCtx *data)
{
  BC_Lithosphere bcdata;
  PetscReal      vx,vz,vxl,vxr,vzl,vzr;
  PetscReal      zero = 0.0;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  
  ierr = PetscMalloc(sizeof(struct _p_BC_Lithosphere),&bcdata);CHKERRQ(ierr);

  /* Computing of the 2 velocity component required to get a vector of norm normV and angle angle_v */
  vz = data->normV*PetscCosReal(data->angle_v);
  vx = PetscSqrtReal(data->normV*data->normV - vz*vz);

  /* Left face */  
  vxl = vx;
  vzl = vz;
  /* Right face */
  vxr = -vx;
  vzr = -vz;

  bcdata->y_lab = data->y_continent[2];
  bcdata->v = vxl;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,0,BCListEvaluator_Lithosphere,(void*)bcdata);CHKERRQ(ierr);
  bcdata->v = vzl;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,2,BCListEvaluator_Lithosphere,(void*)bcdata);CHKERRQ(ierr);
  bcdata->v = 0.0;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,1,BCListEvaluator_Lithosphere,(void*)bcdata);CHKERRQ(ierr);
  
  bcdata->y_lab = data->y_continent[2];
  bcdata->v = vxr;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,0,BCListEvaluator_Lithosphere,(void*)bcdata);CHKERRQ(ierr);
  bcdata->v = vzr;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,2,BCListEvaluator_Lithosphere,(void*)bcdata);CHKERRQ(ierr);
  bcdata->v = 0.0;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,1,BCListEvaluator_Lithosphere,(void*)bcdata);CHKERRQ(ierr);
  
  /* Free slip bottom */
  if (!data->open_base) {
    ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,1,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  }
  ierr = PetscFree(bcdata);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyBoundaryCondition_OrthogonalExtension_Freeslip(BCList bclist,DM dav,pTatinCtx c,ModelRiftSubdCtx *data)
{
  PetscReal      vxl,vxr,vy,zero=0.0;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  
  vxl = -data->v_extension;
  vxr =  data->v_extension;
  /* Extension on faces of normal x */
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,0,BCListEvaluator_constant,(void*)&vxr);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,0,BCListEvaluator_constant,(void*)&vxl);CHKERRQ(ierr);
  /* Free slip on faces of normal z */
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMAX_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  /* Inflow on bottom face */
  vy = 2.0*data->v_extension*((data->Ly - data->Oy)*(data->Lz - data->Oz))/((data->Lx - data->Ox)*(data->Lz - data->Oz));
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,1,BCListEvaluator_constant,(void*)&vy);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyBoundaryCondition_OrthogonalExtension_Plitho(BCList bclist,DM dav,pTatinCtx c,ModelRiftSubdCtx *data, PetscBool full_face_extension)
{
  BC_Lithosphere bcdata;
  PetscReal      vxl,vxr;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  ierr = PetscMalloc(sizeof(struct _p_BC_Lithosphere),&bcdata);CHKERRQ(ierr);
  /* Depth at which Dirichlet BCs are no more applied in BCListEvaluator_Lithosphere */
  bcdata->y_lab = data->y_continent[2];

  vxl = -data->v_extension;
  vxr =  data->v_extension;
  /* Extension on faces of normal x */
  if (full_face_extension) {
    ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,0,BCListEvaluator_constant,(void*)&vxr);CHKERRQ(ierr);
    ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,0,BCListEvaluator_constant,(void*)&vxl);CHKERRQ(ierr);
  } else {
    bcdata->v = vxl;
    ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,0,BCListEvaluator_Lithosphere,(void*)bcdata);CHKERRQ(ierr);
    bcdata->v = vxr;
    ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,0,BCListEvaluator_Lithosphere,(void*)bcdata);CHKERRQ(ierr); 
  }

  bcdata->v = 0.0;
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,2,BCListEvaluator_Lithosphere,(void*)bcdata);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,2,BCListEvaluator_Lithosphere,(void*)bcdata);CHKERRQ(ierr);
  
  /* Neumann lithostatic pressure on faces of normal z */

  if (data->open_base) {
    /* If base is opened and we have a free surface we need to set vy somewhere */
    ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,1,BCListEvaluator_Lithosphere,(void*)bcdata);CHKERRQ(ierr);
    ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,1,BCListEvaluator_Lithosphere,(void*)bcdata);CHKERRQ(ierr);
  } else {
    /* Inflow on bottom face based on \int_Gamma v.n dS 
       If v.n not computed, vy = 0.0 */
    ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,1,BCListEvaluator_constant,(void*)&data->vy);CHKERRQ(ierr);
  }
  ierr = PetscFree(bcdata);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyBoundaryCondition_OrthogonalExtension_PlithoKMAXFreeslipKMIN(BCList bclist,DM dav,pTatinCtx c,ModelRiftSubdCtx *data, PetscBool full_face_extension)
{
  BC_Lithosphere bcdata;
  PetscReal      vxl,vxr,zero=0.0;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  ierr = PetscMalloc(sizeof(struct _p_BC_Lithosphere),&bcdata);CHKERRQ(ierr);
  /* Depth at which Dirichlet BCs are no more applied in BCListEvaluator_Lithosphere */
  bcdata->y_lab = data->y_continent[2];

  vxl = -data->v_extension;
  vxr =  data->v_extension;
  /* Extension on faces of normal x */
  if (full_face_extension) {
    PetscPrintf(PETSC_COMM_WORLD,"Applying extension on the whole face\n");
    ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,0,BCListEvaluator_constant,(void*)&vxr);CHKERRQ(ierr);
    ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,0,BCListEvaluator_constant,(void*)&vxl);CHKERRQ(ierr); 
  } else {
    PetscPrintf(PETSC_COMM_WORLD,"Applying extension in the lithosphere only\n");
    bcdata->v = vxr;
    ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,0,BCListEvaluator_Lithosphere,(void*)bcdata);CHKERRQ(ierr);
    bcdata->v = vxl;
    ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,0,BCListEvaluator_Lithosphere,(void*)bcdata);CHKERRQ(ierr); 
  }

  /* Free-slip KMIN */
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  /* Plitho KMAX */
  
  /* Base */
  if (data->open_base) {
    PetscPrintf(PETSC_COMM_WORLD,"Bottom boundary is open\n");
    /* If base is opened and we have a free surface we need to set vy somewhere */
    bcdata->v = 0.0;
    ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,1,BCListEvaluator_Lithosphere,(void*)bcdata);CHKERRQ(ierr);
    ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,1,BCListEvaluator_Lithosphere,(void*)bcdata);CHKERRQ(ierr);
  } else {
    /* Inflow on bottom face based on \int_Gamma v.n dS 
       If v.n not computed, vy = 0.0 */
    ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,1,BCListEvaluator_constant,(void*)&data->vy);CHKERRQ(ierr);
  }
  /* Free-surface */

  ierr = PetscFree(bcdata);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelComputeBottomFlow_Vdotn(pTatinCtx c,Vec X, ModelRiftSubdCtx *data)
{
  PhysCompStokes stokes;
  DM             dms;
  Vec            velocity,pressure;
  PetscReal      int_u_dot_n[HEX_EDGES];
  PetscErrorCode ierr;

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
    data->vy = 0.0;
  } else {
    /* Compute the vy velocity, if we want inflow vy > 0 so flip the sign */
    data->vy = (int_u_dot_n[WEST_FACE-1]+int_u_dot_n[EAST_FACE-1]+int_u_dot_n[BACK_FACE-1]+int_u_dot_n[FRONT_FACE-1])/((data->Lx - data->Ox)*(data->Lz - data->Oz));
    PetscPrintf(PETSC_COMM_WORLD,"Vy = %+1.4e\n",data->vy);    
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyBoundaryCondition_Transition_Freeslip(BCList bclist,DM dav,pTatinCtx c,ModelRiftSubdCtx *data)
{
  PetscReal      time,zero=0.0;
  PetscReal      vx,vxl,vxr,vy;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  
  ierr = pTatinGetTime(c,&time);
  
  if (time >= data->BC_time[0] && time < data->BC_time[1]) {
    PetscPrintf(PETSC_COMM_WORLD,"[[ DECREASING VELOCITY ]]\n");
    vx = ((0.0-data->v_extension)/(data->BC_time[1]-data->BC_time[0])) * (time-data->BC_time[1]);
    vxl = -vx;
    vxr =  vx;
    vy = 2.0*data->v_extension*((data->Ly - data->Oy)*(data->Lz - data->Oz))/((data->Lx - data->Ox)*(data->Lz - data->Oz));
  } else if (time >= data->BC_time[1] && time < data->BC_time[2]) {
    PetscPrintf(PETSC_COMM_WORLD,"[[ RELAXING ]]\n");
    vxl = 0.0;
    vxr = 0.0;
    vy  = 0.0;
  } else if (time >= data->BC_time[2] && time < data->BC_time[3]) {
    // Increase velocity to reach compression velocity
    PetscPrintf(PETSC_COMM_WORLD,"[[ INCREASING VELOCITY ]]\n");
    vx = ((0.0-data->normV)/(data->BC_time[2]-data->BC_time[3])) * (time-data->BC_time[2]);
    vxl = vx; 
    vxr = -vx;
    vy = -2.0*data->normV*((data->Ly - data->Oy)*(data->Lz - data->Oz))/((data->Lx - data->Ox)*(data->Lz - data->Oz));
  }
  
  /* Apply the velocities */
  /* Faces of normal x */
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,0,BCListEvaluator_constant,(void*)&vxl);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,0,BCListEvaluator_constant,(void*)&vxr);CHKERRQ(ierr);
  /* Freeslip on faces of normal z */
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMAX_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  /* Flow on bottom face */
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,1,BCListEvaluator_constant,(void*)&vy);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyBoundaryCondition_OrthogonalCompression_Freeslip(BCList bclist,DM dav,pTatinCtx c,ModelRiftSubdCtx *data)
{
  PetscReal      vx,vy,vxl,vxr;
  PetscReal      zero = 0.0;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  
  vx = data->normV;
  vy = -2.0*data->normV*((data->Ly - data->Oy)*(data->Lz - data->Oz))/((data->Lx - data->Ox)*(data->Lz - data->Oz));
  /* Left face */  
  vxl = vx;
  /* Right face */
  vxr = -vx;
  /* Compression on faces of normal x */
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMIN_LOC,0,BCListEvaluator_constant,(void*)&vxl);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_IMAX_LOC,0,BCListEvaluator_constant,(void*)&vxr);CHKERRQ(ierr);
  /* Freeslip on faces of normal z */
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMIN_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_KMAX_LOC,2,BCListEvaluator_constant,(void*)&zero);CHKERRQ(ierr);
  /* Outflow at the base */
  ierr = DMDABCListTraverse3d(bclist,dav,DMDABCList_JMIN_LOC,1,BCListEvaluator_constant,(void*)&vy);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyBoundaryConditionVelocity_RiftOnly(BCList bclist,DM dav,pTatinCtx c,ModelRiftSubdCtx *data)
{ 
  PetscInt       BC_case;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  BC_case = -1;
  if (data->litho_plitho_z) {
    BC_case = 0;
  } else if (data->full_face_plitho_z) {
    BC_case = 1;
  } else if (data->freeslip_z) {
    BC_case = 2;
  } else if (data->full_face_plithoKMAX) {
    BC_case = 3;
  } else if (data->litho_plithoKMAX) {
    BC_case = 4;
  }

  /* This function will apply BCs only for the rift case */
  switch(BC_case)
  {
    case 0:
      /* normal x faces: vx = v or -v, vy = 0, vz = 0 in the lithosphere plitho below
         normal z faces: plitho
         bottom:         vy inflow OR plitho (open base)
         surface:        free-surface */
      ierr = ModelApplyBoundaryCondition_OrthogonalExtension_Plitho(bclist,dav,c,data,PETSC_FALSE);CHKERRQ(ierr);
      break;

    case 1:
      /* normal x faces: vx = v or -v on the whole face
                         vz = 0 in the lithosphere
                         vy = 0 in the lithosphere if open base
         normal z faces: plitho
         bottom:         vy inflow OR plitho (open base)
         surface:        free-surface */
      ierr = ModelApplyBoundaryCondition_OrthogonalExtension_Plitho(bclist,dav,c,data,PETSC_TRUE);CHKERRQ(ierr);
      break;

    case 2:
      /* normal x faces: vx = v or -v on the whole face
         normal z faces: free-slip
         bottom:         vy inflow
         surface:        free-surface */
      ierr = ModelApplyBoundaryCondition_OrthogonalExtension_Freeslip(bclist,dav,c,data);CHKERRQ(ierr);
      break;

    case 3:
      /* normal x faces: vx = v or -v on the whole face
         normal z faces: zmin: free-slip
                         zmax: plitho
         bottom:         vy inflow OR plitho (open base)
         surface:        free-surface */
      ierr = ModelApplyBoundaryCondition_OrthogonalExtension_PlithoKMAXFreeslipKMIN(bclist,dav,c,data,PETSC_TRUE);CHKERRQ(ierr);
      break;

    case 4:
      /* normal x faces: vx = v or -v in the lithosphere, plitho below
         normal z faces: zmin: free-slip
                         zmax: plitho
         bottom:         vy inflow OR plitho (open base)
         surface:        free-surface */
      ierr = ModelApplyBoundaryCondition_OrthogonalExtension_PlithoKMAXFreeslipKMIN(bclist,dav,c,data,PETSC_FALSE);CHKERRQ(ierr);
      break;

    default:
      SETERRQ(PetscObjectComm((PetscObject)dav),PETSC_ERR_USER,"Unknown BC type");
      break;
  }

  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyBoundaryConditionVelocity_RiftSubd(BCList bclist,DM dav,pTatinCtx c,ModelRiftSubdCtx *data)
{
  /* BC_time:                                            t3________ velocity
     0 = end of extension                                /
     1 = start of quiescence phase           t1_________/t2
     2 = end of quiescence phase             / 
     3 = start of compression        _______/                     
                                           t0
                                     time ----->      
  */
  PetscReal time;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = pTatinGetTime(c,&time);CHKERRQ(ierr);
  /* Depending on time, apply the corresponding BC function */
  if (time < data->BC_time[0]) {
    ierr = ModelApplyBoundaryConditionVelocity_RiftOnly(bclist,dav,c,data);CHKERRQ(ierr);
  } else if (time >= data->BC_time[3]) {
    if (data->freeslip_z) {
      ierr = ModelApplyBoundaryCondition_OrthogonalCompression_Freeslip(bclist,dav,c,data);CHKERRQ(ierr);
    } else {
      ierr = ModelApplyBoundaryCondition_ObliqueCompression(bclist,dav,c,data);CHKERRQ(ierr);
    }
  } else {
    if (data->freeslip_z) {
      ierr = ModelApplyBoundaryCondition_Transition_Freeslip(bclist,dav,c,data);CHKERRQ(ierr);
    } else {
      ierr = ModelApplyBoundaryCondition_Transition(bclist,dav,c,data);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ModelApplyTimeDependantEnergyBCs(pTatinCtx c,ModelRiftSubdCtx *data)
{
  PhysCompEnergyFV energy;
  PetscReal        val_T;
  PetscInt         l;
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

  /* Iterate through all boundary faces, if there is inflow redefine the bc value and bc flux method */
  const DACellFace flist[] = { DACELL_FACE_W, DACELL_FACE_E, DACELL_FACE_B, DACELL_FACE_F };
  for (l=0; l<sizeof(flist)/sizeof(DACellFace); l++) {
    ierr = FVSetDirichletFromInflow(energy->fv,energy->T,flist[l]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyBoundaryCondition_RiftSubd(pTatinCtx c,void *ctx)
{
  ModelRiftSubdCtx *data;
  PhysCompStokes   stokes;
  DM               stokes_pack,dav,dap;
  Vec              X = NULL;
  PetscBool        active_energy;
  PetscErrorCode   ierr;

  PetscFunctionBegin;

  data = (ModelRiftSubdCtx*)ctx;

  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  /* Define velocity boundary conditions */
  ierr = pTatinGetStokesContext(c,&stokes);CHKERRQ(ierr);
  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);
  if (data->use_v_dot_n) {
    ierr = pTatinPhysCompGetData_Stokes(c,&X);CHKERRQ(ierr); 
    /* Compute vy as int_S v.n dS */
    ierr = ModelComputeBottomFlow_Vdotn(c,X,data);CHKERRQ(ierr);
  } else {
    /* Assume free-slip base */
    data->vy = 0.0;
  }

  ierr = ModelApplyBoundaryConditionVelocity_RiftSubd(stokes->u_bclist,dav,c,data);CHKERRQ(ierr);
  
  /* Define boundary conditions for any other physics */
  ierr = pTatinContextValid_EnergyFV(c,&active_energy);CHKERRQ(ierr);
  if (active_energy) {
    ierr = ModelApplyTimeDependantEnergyBCs(c,data);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyBoundaryConditionMG_RiftSubd(PetscInt nl,BCList bclist[],DM dav[],pTatinCtx c,void *ctx)
{
  ModelRiftSubdCtx *data;
  PetscInt         n;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);

  data = (ModelRiftSubdCtx*)ctx;
  /* Define velocity boundary conditions on each level within the MG hierarchy */
  for (n=0; n<nl; n++) {
    ierr = ModelApplyBoundaryConditionVelocity_RiftSubd(bclist[n],dav[n],c,data);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ModelApplyMaterialBoundaryCondition_RiftSubd(pTatinCtx c,void *ctx)
{
  ModelRiftSubdCtx *data;
  PhysCompStokes   stokes;
  DM               stokes_pack,dav,dap;
  PetscInt         Nxp[2];
  PetscInt         *face_list;
  PetscReal        perturb, epsilon;
  DataBucket       material_point_db,material_point_face_db;
  PetscInt         f, n_face_list;
  int              p,n_mp_points;
  MPAccess         mpX;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  data = (ModelRiftSubdCtx*)ctx;
  
  ierr = pTatinGetStokesContext(c,&stokes);CHKERRQ(ierr);
  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);

  ierr = pTatinGetMaterialPoints(c,&material_point_db,NULL);CHKERRQ(ierr);

  /* create face storage for markers */
  DataBucketDuplicateFields(material_point_db,&material_point_face_db);
  
  if (data->is_2D){
    n_face_list = 2;
    ierr = PetscMalloc1(n_face_list,&face_list);CHKERRQ(ierr);
    face_list[0] = 0;
    face_list[1] = 1;
  } else {
    n_face_list = 4;
    ierr = PetscMalloc1(n_face_list,&face_list);CHKERRQ(ierr);
    face_list[0] = 0;
    face_list[1] = 1;
    face_list[2] = 4;
    face_list[3] = 5;
  }
  
  for (f=0; f<n_face_list; f++) {
    /* traverse */
    /* [0,1/east,west] ; [2,3/north,south] ; [4,5/front,back] */
    Nxp[0]  = 8;
    Nxp[1]  = 8;
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

PetscErrorCode ModelApplyUpdateMeshGeometry_RiftSubd(pTatinCtx c,Vec X,void *ctx)
{
  ModelRiftSubdCtx *data;
  PhysCompStokes   stokes;
  DM               stokes_pack,dav,dap;
  Vec              velocity,pressure;
  PetscInt         npoints,dir;
  PetscReal        dt;
  PetscReal        *xref,*xnat;
  PetscErrorCode   ierr;
  
  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  data = (ModelRiftSubdCtx*)ctx;
  
  /* fully lagrangian update */
  ierr = pTatinGetTimestep(c,&dt);CHKERRQ(ierr);
  ierr = pTatinGetStokesContext(c,&stokes);CHKERRQ(ierr);

  stokes_pack = stokes->stokes_pack;
  ierr = DMCompositeGetEntries(stokes_pack,&dav,&dap);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);

  /* SURFACE REMESHING */
  ierr = UpdateMeshGeometry_ApplyDiffusionJMAX(dav,data->Kero,dt,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_TRUE,PETSC_FALSE);CHKERRQ(ierr);
  ierr = UpdateMeshGeometry_FullLag_ResampleJMax_RemeshJMIN2JMAX(dav,velocity,NULL,dt);
  ierr = DMCompositeRestoreAccess(stokes_pack,X,&velocity,&pressure);CHKERRQ(ierr);
 
  /* Update Mesh Refinement */
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
  ierr = DMDABilinearizeQ2Elements(dav);CHKERRQ(ierr);
  
  ierr = PetscFree(xref);CHKERRQ(ierr);
  ierr = PetscFree(xnat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ModelOutput_RiftSubd(pTatinCtx c,Vec X,const char prefix[],void *ctx)
{
  ModelRiftSubdCtx *data;
  PetscBool        active_energy;
  DataBucket       materialpoint_db;
  PetscErrorCode   ierr;
  static PetscBool been_here = PETSC_FALSE;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n",PETSC_FUNCTION_NAME);
  data = (ModelRiftSubdCtx*)ctx;
  
  /* Output Velocity and pressure */
  ierr = pTatin3d_ModelOutputPetscVec_VelocityPressure_Stokes(c,X,prefix);CHKERRQ(ierr);
  
  /* Output markers cell fields (for production runs) */
  {
    const MaterialPointVariable mp_prop_list[] = { MPV_region, MPV_viscosity, MPV_density, MPV_plastic_strain }; //MPV_viscous_strain,
    ierr = pTatin3dModelOutput_MarkerCellFieldsP0_PetscVec(c,PETSC_FALSE,sizeof(mp_prop_list)/sizeof(MaterialPointVariable),mp_prop_list,prefix);CHKERRQ(ierr);
  }
  
  /* Output raw markers (for testing and debugging) */
  if (data->output_markers)
  {
    ierr = pTatinGetMaterialPoints(c,&materialpoint_db,NULL);CHKERRQ(ierr);

    const int nf = 3;
    const MaterialPointField mp_prop_list[] = { MPField_Std, MPField_Stokes, MPField_StokesPl};//, MPField_Energy };
    char mp_file_prefix[256];

    sprintf(mp_file_prefix,"%s_mpoints",prefix);
    ierr = SwarmViewGeneric_ParaView(materialpoint_db,nf,mp_prop_list,c->outputpath,mp_file_prefix);CHKERRQ(ierr);
  }
  
  /* Output temperature (FV) */
  ierr = pTatinContextValid_EnergyFV(c,&active_energy);CHKERRQ(ierr);
  if (active_energy) {
    PhysCompEnergyFV energy;
    char             root[PETSC_MAX_PATH_LEN],pvoutputdir[PETSC_MAX_PATH_LEN],fname[PETSC_MAX_PATH_LEN];
    
    ierr = pTatinGetContext_EnergyFV(c,&energy);CHKERRQ(ierr);
    
    // PVD
    {
      char pvdfilename[PETSC_MAX_PATH_LEN],vtkfilename[PETSC_MAX_PATH_LEN];
      char stepprefix[PETSC_MAX_PATH_LEN];
      
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
    }
    ierr = PetscSNPrintf(root,PETSC_MAX_PATH_LEN-1,"%s",c->outputpath);CHKERRQ(ierr);
    ierr = PetscSNPrintf(pvoutputdir,PETSC_MAX_PATH_LEN-1,"%s/step%D",root,c->step);CHKERRQ(ierr);
    
    /* PetscVec */
    PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s_energy",prefix);
    ierr = FVDAView_JSON(energy->fv,pvoutputdir,fname);CHKERRQ(ierr); /* write meta data abour fv mesh, its DMDA and the coords */
    ierr = FVDAView_Heavy(energy->fv,pvoutputdir,fname);CHKERRQ(ierr);  /* write cell fields */
    PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s/%s_energy_T",pvoutputdir,prefix);
    ierr = PetscVecWriteJSON(energy->T,0,fname);CHKERRQ(ierr); /* write cell temperature */
    
    if (data->output_markers) {
      PetscSNPrintf(fname,PETSC_MAX_PATH_LEN-1,"%s/%s-Tfv",pvoutputdir,prefix);
      ierr = FVDAView_CellData(energy->fv,energy->T,PETSC_TRUE,fname);CHKERRQ(ierr);
    }
  }
  been_here = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode ModelDestroy_RiftSubd(pTatinCtx c,void *ctx)
{
  ModelRiftSubdCtx *data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"[[%s]]\n", PETSC_FUNCTION_NAME);
  data = (ModelRiftSubdCtx*)ctx;

  /* Free contents of structure */

  /* Free structure */
  ierr = PetscFree(data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode pTatinModelRegister_RiftSubd(void)
{
  ModelRiftSubdCtx *data;
  pTatinModel      m;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  /* Allocate memory for the data structure for this model */
  ierr = PetscMalloc(sizeof(ModelRiftSubdCtx),&data);CHKERRQ(ierr);
  ierr = PetscMemzero(data,sizeof(ModelRiftSubdCtx));CHKERRQ(ierr);

  /* register user model */
  ierr = pTatinModelCreate(&m);CHKERRQ(ierr);

  /* Set name, model select via -ptatin_model NAME */
  ierr = pTatinModelSetName(m,"rift2subd");CHKERRQ(ierr);

  /* Set model data */
  ierr = pTatinModelSetUserData(m,data);CHKERRQ(ierr);

  /* Set function pointers */
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_INIT,                  (void (*)(void))ModelInitialize_RiftSubd);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_MESH_GEOM,  (void (*)(void))ModelApplyInitialMeshGeometry_RiftSubd);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_MAT_GEOM,   (void (*)(void))ModelApplyInitialMaterialGeometry_RiftSubd);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_STOKES_VARIABLE_MARKERS,(void (*)(void))ModelApplyInitialStokesVariableMarkers_RiftSubd);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_INIT_SOLUTION,   (void (*)(void))ModelApplyInitialSolution_RiftSubd);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_BC,              (void (*)(void))ModelApplyBoundaryCondition_RiftSubd);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_BCMG,            (void (*)(void))ModelApplyBoundaryConditionMG_RiftSubd);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_MAT_BC,          (void (*)(void))ModelApplyMaterialBoundaryCondition_RiftSubd);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_APPLY_UPDATE_MESH_GEOM,(void (*)(void))ModelApplyUpdateMeshGeometry_RiftSubd);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_OUTPUT,                (void (*)(void))ModelOutput_RiftSubd);CHKERRQ(ierr);
  ierr = pTatinModelSetFunctionPointer(m,PTATIN_MODEL_DESTROY,               (void (*)(void))ModelDestroy_RiftSubd);CHKERRQ(ierr);

  /* Insert model into list */
  ierr = pTatinModelRegister(m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SwarmMPntStd_CoordAssignment_FaceLatticeLayout3d_epsilon(DM da,PetscInt Nxp[],PetscReal perturb, PetscReal epsilon, PetscInt face_idx,DataBucket db)
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