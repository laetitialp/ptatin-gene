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
 **    filename:   stokes_rheology_vp_std.c
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

#include "petsc.h"
#include "petscdm.h"

#include "ptatin3d_defs.h"
#include "ptatin3d.h"
#include "private/ptatin_impl.h"
#include "data_bucket.h"
#include "element_type_Q2.h"
#include "dmda_element_q2p1.h"
#include "element_utils_q2.h"
#include "dmdae.h"
#include "dmda_element_q1.h"
#include "element_utils_q1.h"
#include "phys_comp_energy.h"
#include "ptatin3d_energy.h"
#include "ptatin3d_energyfv.h"
#include "ptatin3d_energyfv_impl.h"
#include "fvda_impl.h"

#include "MPntStd_def.h"
#include "MPntPStokes_def.h"
#include "MPntPStokesPl_def.h"

#include "material_constants.h"

typedef enum { YTYPE_NONE=0, YTYPE_MISES=1, YTYPE_DP=2, YTYPE_TENSILE_FAILURE=3 } YieldTypeDefinition;

typedef struct {
  MaterialConst_MaterialType           *MatType_data;
  MaterialConst_DensityConst           *DensityConst_data;
  MaterialConst_DensityBoussinesq      *DensityBoussinesq_data;
  MaterialConst_DensityTable           *DensityTable_data;
  MaterialConst_ViscosityConst         *ViscConst_data;
  MaterialConst_ViscosityZ             *ViscZ_data;
  MaterialConst_ViscosityFK            *ViscFK_data;
  MaterialConst_ViscosityArrh          *ViscArrh_data;
  MaterialConst_ViscosityArrh_DislDiff *ViscArrhDislDiff_data;
  MaterialConst_PlasticMises           *PlasticMises_data;
  MaterialConst_PlasticDP              *PlasticDP_data;
  MaterialConst_SoftLin                *SoftLin_data;
  MaterialConst_SoftExpo               *SoftExpo_data;
} MaterialData;

typedef struct {
  MPntStd       *std;
  MPntPStokes   *stokes;
  MPntPStokesPl *pls;
} MaterialPointData;

typedef struct {
  MaterialData      *material_data;
  MaterialPointData *mp_data;
  PetscReal         *ux,*uy,*uz;
  PetscReal         *dNudx,*dNudy,*dNudz;
  PetscReal         *T,*pressure,*viscosity;
  long int          *npoints_yielded;
} RheologyData;


static inline void ComputeLinearSoft(float eplast,PetscReal emin,PetscReal emax, PetscReal X0, PetscReal Xinf, PetscReal *Xeff)
{
  *Xeff = X0;
  if (eplast > emin) {
    if (eplast > emax) {
      *Xeff = Xinf;
    } else {
      *Xeff  = X0 - (eplast-emin)/(emax-emin)*(X0-Xinf);
    }
  }
}

static inline void ComputeExponentialSoft(float eplast,PetscReal emin,PetscReal efold, PetscReal X0, PetscReal Xinf, PetscReal *Xeff)
{
  *Xeff = X0;
  if (eplast > emin) {
    *Xeff  = Xinf + (X0-Xinf) * exp(-(eplast-emin)/efold);
  }
}



static inline void ComputeStressIsotropic3d(PetscReal eta,double D[NSD][NSD],double T[NSD][NSD])
{
  const double two_eta = 2.0 * eta;

  T[0][0] = two_eta * D[0][0];  T[0][1] = two_eta * D[0][1];    T[0][2] = two_eta * D[0][2];
  T[1][0] =           T[0][1];  T[1][1] = two_eta * D[1][1];    T[1][2] = two_eta * D[1][2];
  T[2][0] =           T[0][2];  T[2][1] =           T[1][2];    T[2][2] = two_eta * D[2][2];
}

static inline void ComputeStrainRate3d(double ux[],double uy[],double uz[],double dNudx[],double dNudy[],double dNudz[],double D[NSD][NSD])
{
  int    k;
  double exx,eyy,ezz,exy,exz,eyz;

  exx=0.0;  eyy=0.0;  ezz=0.0;
  exy=0.0;  exz=0.0;  eyz=0.0;

  for (k=0; k<Q2_NODES_PER_EL_3D; k++) {
    exx += dNudx[k] * ux[k];
    eyy += dNudy[k] * uy[k];
    ezz += dNudz[k] * uz[k];

    exy += dNudy[k] * ux[k] + dNudx[k] * uy[k];
    exz += dNudz[k] * ux[k] + dNudx[k] * uz[k];
    eyz += dNudz[k] * uy[k] + dNudy[k] * uz[k];
  }
  exy = 0.5 * exy;
  exz = 0.5 * exz;
  eyz = 0.5 * eyz;

  D[0][0] = exx;    D[0][1] = exy;    D[0][2] = exz;
  D[1][0] = exy;    D[1][1] = eyy;    D[1][2] = eyz;
  D[2][0] = exz;    D[2][1] = eyz;    D[2][2] = ezz;
}

/*
   static inline void ComputeDeformationGradient3d(double ux[],double uy[],double uz[],double dNudx[],double dNudy[],double dNudz[],double L[NSD][NSD])
   {
   int i,j,k;

   for (i=0; i<3; i++) {
   for (j=0; j<3; j++) {
   L[i][j] = 0.0;
   }
   }
   for (k=0; k<Q2_NODES_PER_EL_3D; k++) {
// du/dx_i
L[0][0] += dNudx[k] * ux[k];
L[0][1] += dNudy[k] * ux[k];
L[0][2] += dNudz[k] * ux[k];
// dv/dx_i
L[1][0] += dNudx[k] * uy[k];
L[1][1] += dNudy[k] * uy[k];
L[1][2] += dNudz[k] * uy[k];
// dw/dx_i
L[2][0] += dNudx[k] * uz[k];
L[2][1] += dNudy[k] * uz[k];
L[2][2] += dNudz[k] * uz[k];
}
}
*/

static inline void ComputeSecondInvariant3d(double A[NSD][NSD],double *A2)
{
  int i,j;
  double sum = 0.0;
  for (i=0; i<3; i++) {
    for (j=0; j<3; j++) {
      sum = sum + A[i][j]*A[i][j];
    }
  }
  *A2 = sqrt( 0.5 * sum );
}

/*
   static inline void ComputeAverageTrace3d(double A[NSD][NSD],double *A2)
   {
   const double one_third = 0.333333333333333;

 *A2 = one_third * ( A[0][0] + A[1][1] + A[2][2] );

 }
 */
static inline PetscErrorCode EvaluateLinearSoftening(
  RheologyData *data,
  PetscReal    *Co_mp, // cohesion for DP, yield stress for Mises
  PetscReal    *phi_mp) // friction for DP, NULL for Mises
{
  int       region_idx   = data->mp_data->std->phase;
  int       plastic_type = data->material_data->MatType_data[ region_idx ].plastic_type;
  PetscReal emin         = data->material_data->SoftLin_data[ region_idx ].eps_min;
  PetscReal emax         = data->material_data->SoftLin_data[ region_idx ].eps_max;
  float     damage;
  PetscFunctionBegin;

  MPntPStokesPlGetField_damage(data->mp_data->pls,&damage);
  
  switch (plastic_type) {
    case PLASTIC_MISES:
    case PLASTIC_MISES_H:
    {
      MaterialConst_PlasticMises Mises_data = data->material_data->PlasticMises_data[ region_idx ];
      ComputeLinearSoft(damage,emin,emax,Mises_data.tau_yield,Mises_data.tau_yield_inf,Co_mp);
    }
      break;
    case PLASTIC_DP:
    case PLASTIC_DP_H:
    {
      MaterialConst_PlasticDP DP_data = data->material_data->PlasticDP_data[ region_idx ];
      ComputeLinearSoft(damage,emin,emax,DP_data.Co,DP_data.Co_inf,Co_mp);
      ComputeLinearSoft(damage,emin,emax,DP_data.phi,DP_data.phi_inf,phi_mp);
    }  
      break;
    
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"No default PlasticType set. Valid choices are PLASTIC_NONE, PLASTIC_MISES, PLASTIC_DP");
      break;
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode EvaluateExponentialSoftening(
  RheologyData *data,
  PetscReal *Co_mp, // cohesion for DP, yield stress for Mises
  PetscReal *phi_mp) // friction for DP, NULL for Mises
{
  int       region_idx   = data->mp_data->std->phase;
  int       plastic_type = data->material_data->MatType_data[ region_idx ].plastic_type;
  PetscReal emin         = data->material_data->SoftExpo_data[ region_idx ].eps_min;
  PetscReal efold        = data->material_data->SoftExpo_data[ region_idx ].eps_fold;
  float     damage;
  PetscFunctionBegin;

  MPntPStokesPlGetField_damage(data->mp_data->pls,&damage);

  switch (plastic_type) {
    case PLASTIC_MISES:
    case PLASTIC_MISES_H:
    {
      MaterialConst_PlasticMises Mises_data = data->material_data->PlasticMises_data[ region_idx ];
      ComputeExponentialSoft(damage,emin,efold,Mises_data.tau_yield,Mises_data.tau_yield_inf,Co_mp);
    }
      break;

    case PLASTIC_DP:
    case PLASTIC_DP_H:
    {
      MaterialConst_PlasticDP DP_data = data->material_data->PlasticDP_data[ region_idx ];
      ComputeExponentialSoft(damage,emin,efold,DP_data.Co,DP_data.Co_inf,Co_mp);
      ComputeExponentialSoft(damage,emin,efold,DP_data.phi,DP_data.phi_inf,phi_mp);
    }
      break;
    
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"No default PlasticType set. Valid choices are PLASTIC_NONE, PLASTIC_MISES, PLASTIC_DP");
      break;
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode EvaluateSofteningOnMarker(
  RheologyData *data,
  PetscReal *Co_mp, // cohesion for DP, yield stress for Mises
  PetscReal *phi_mp) // friction for DP, NULL for Mises
{
  int region_idx     = data->mp_data->std->phase;
  int softening_type = data->material_data->MatType_data[ region_idx ].softening_type;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  switch (softening_type) {
  case SOFTENING_NONE: 
    break;

  case SOFTENING_LINEAR: 
    ierr = EvaluateLinearSoftening(data,Co_mp,phi_mp);CHKERRQ(ierr);
    break;

  case SOFTENING_EXPONENTIAL: 
    ierr = EvaluateExponentialSoftening(data,Co_mp,phi_mp);CHKERRQ(ierr);
    break;

  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"No default SofteningType set. Valid choices are SOFTENING_NONE, SOFTENING_LINEAR, SOFTENING_EXPONENTIAL");
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode ViscosityArrheniusDislocationCreep(
  MaterialConst_ViscosityArrh *ViscArrh_data,
  RheologyData                *data, 
  int                         viscous_type)
{
  PetscReal R = 8.31440;
  PetscReal nexp      = ViscArrh_data->nexp;
  PetscReal entalpy   = ViscArrh_data->entalpy;
  PetscReal preexpA   = ViscArrh_data->preexpA;
  PetscReal Vmol      = ViscArrh_data->Vmol;
  PetscReal Tref      = ViscArrh_data->Tref;
  PetscReal Ascale    = ViscArrh_data->Ascale;
  PetscReal T_arrh    = *data->T + Tref ;
  PetscReal sr, pressure;
  double    D_mp[NSD][NSD], inv2_D_mp;


  PetscFunctionBegin;

  ComputeStrainRate3d(data->ux,data->uy,data->uz,data->dNudx,data->dNudy,data->dNudz,D_mp);
  ComputeSecondInvariant3d(D_mp,&inv2_D_mp);

  sr = inv2_D_mp/ViscArrh_data->Eta_scale*ViscArrh_data->P_scale;
  if (sr < 1.0e-17) { sr = 1.0e-17; }

  pressure = *data->pressure * ViscArrh_data->P_scale;
  if (pressure >= 0.0) { entalpy = entalpy + pressure*Vmol; }

  switch (viscous_type) {
    case VISCOUS_ARRHENIUS:
      *data->viscosity = Ascale*0.25*pow(sr,1.0/nexp - 1.0)*pow(0.75*preexpA,-1.0/nexp)*exp(entalpy/(nexp*R*T_arrh));
      break;
    
    case VISCOUS_ARRHENIUS_2:
      *data->viscosity = Ascale*pow(sr,1.0/nexp - 1.0)*pow(preexpA,-1.0/nexp)*exp(entalpy/(nexp*R*T_arrh));
      break;

    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"viscous_type can only be VISCOUS_ARRHENIUS, VISCOUS_ARRHENIUS_2");
      break;
  }
  *data->viscosity /= ViscArrh_data->Eta_scale;
  
  PetscFunctionReturn(0);
}

static inline PetscErrorCode ViscosityArrheniusDislocationDiffusionCreep(
  MaterialConst_ViscosityArrh_DislDiff *ViscArrhDislDiff_data,
  RheologyData *data)
{
  PetscScalar R            = 8.31440;
  PetscReal   preexpA_disl = ViscArrhDislDiff_data->preexpA_disl;
  PetscReal   Ascale_disl  = ViscArrhDislDiff_data->Ascale_disl;
  PetscReal   entalpy_disl = ViscArrhDislDiff_data->entalpy_disl;
  PetscReal   Vmol_disl    = ViscArrhDislDiff_data->Vmol_disl;
  PetscReal   nexp_disl    = ViscArrhDislDiff_data->nexp_disl;
  PetscReal   preexpA_diff = ViscArrhDislDiff_data->preexpA_diff;
  PetscReal   Ascale_diff  = ViscArrhDislDiff_data->Ascale_diff;
  PetscReal   entalpy_diff = ViscArrhDislDiff_data->entalpy_diff;
  PetscReal   Vmol_diff    = ViscArrhDislDiff_data->Vmol_diff;
  PetscReal   pexp_diff    = ViscArrhDislDiff_data->pexp_diff;
  PetscReal   gsize        = ViscArrhDislDiff_data->gsize;
  PetscReal   Tref         = ViscArrhDislDiff_data->Tref;
  PetscReal   T_arrh       = *data->T + Tref ;
  PetscReal   sr, eta_disl, eta_diff, pressure, eta;
  PetscScalar D_mp[NSD][NSD], inv2_D_mp;
    
    PetscFunctionBegin;

    ComputeStrainRate3d(data->ux,data->uy,data->uz,data->dNudx,data->dNudy,data->dNudz,D_mp);
    ComputeSecondInvariant3d(D_mp,&inv2_D_mp);
    
    sr = inv2_D_mp/ViscArrhDislDiff_data->Eta_scale*ViscArrhDislDiff_data->P_scale;
    if (sr < 1.0e-17) {
      sr = 1.0e-17;
    }
    
    pressure = *data->pressure * ViscArrhDislDiff_data->P_scale;
    
    if (pressure >= 0.0) {
      entalpy_disl += pressure*Vmol_disl;
      entalpy_diff += pressure*Vmol_diff;
    }
    /* dislocation creep viscosity */
    eta_disl = Ascale_disl*pow(sr,1.0/nexp_disl - 1.0)*pow(preexpA_disl,-1.0/nexp_disl)*exp(entalpy_disl/(nexp_disl*R*T_arrh));
    /* diffusion creep viscosity */
    eta_diff = Ascale_diff*1.0/(preexpA_diff)*pow(gsize,pexp_diff)*exp(entalpy_diff/(R*T_arrh));
    /* Combine the two viscosities */
    eta      = 1.0/(1.0/eta_disl + 1.0/eta_diff);
    /* Scale */
    *data->viscosity = eta/ViscArrhDislDiff_data->Eta_scale;
  PetscFunctionReturn(0);
}

static inline PetscErrorCode ViscosityPlasticMises(RheologyData *data)
{
  int            region_idx     = data->mp_data->std->phase;
  double         tau_yield_mp   = data->material_data->PlasticMises_data[ region_idx ].tau_yield;
  double         Tpred_mp[NSD][NSD], D_mp[NSD][NSD];
  double         inv2_D_mp, inv2_Tpred_mp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* softening */
  ierr = EvaluateSofteningOnMarker(data,&tau_yield_mp,NULL);CHKERRQ(ierr);
  /* mark all markers as not yielding */
  MPntPStokesPlSetField_yield_indicator(data->mp_data->pls,YTYPE_NONE);
  /* strain rate */
  ComputeStrainRate3d(data->ux,data->uy,data->uz,data->dNudx,data->dNudy,data->dNudz,D_mp);
  /* stress */
  ComputeStressIsotropic3d(*data->viscosity,D_mp,Tpred_mp);
  /* second inv stress */
  ComputeSecondInvariant3d(Tpred_mp,&inv2_Tpred_mp);

  if (inv2_Tpred_mp > tau_yield_mp) {
    ComputeSecondInvariant3d(D_mp,&inv2_D_mp);
    *data->viscosity = 0.5 * tau_yield_mp / inv2_D_mp;
    *data->npoints_yielded += 1;
    MPntPStokesPlSetField_yield_indicator(data->mp_data->pls,YTYPE_MISES);
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode ViscosityPlasticMisesH(RheologyData *data)
{
  int            region_idx     = data->mp_data->std->phase;
  double         tau_yield_mp   = data->material_data->PlasticMises_data[ region_idx ].tau_yield;
  double         D_mp[NSD][NSD];
  double         eta_flow_mp,eta_yield_mp, inv2_D_mp;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = EvaluateSofteningOnMarker(data,&tau_yield_mp,NULL);CHKERRQ(ierr);
  eta_flow_mp = *data->viscosity;

  /* strain rate */
  ComputeStrainRate3d(data->ux,data->uy,data->uz,data->dNudx,data->dNudy,data->dNudz,D_mp);
  ComputeSecondInvariant3d(D_mp,&inv2_D_mp);

  eta_yield_mp     = 0.5 * tau_yield_mp / inv2_D_mp;
  *data->viscosity = 1.0 / (  1.0/eta_flow_mp + 1.0/eta_yield_mp );
  *data->npoints_yielded += 1;
  PetscFunctionReturn(0);
}

static inline PetscErrorCode ViscosityPlasticDruckerPrager(RheologyData *data)
{
  int            region_idx = data->mp_data->std->phase;
  PetscReal      phi        = data->material_data->PlasticDP_data[ region_idx ].phi;
  PetscReal      Co         = data->material_data->PlasticDP_data[ region_idx ].Co;
  double         Tpred_mp[NSD][NSD], D_mp[NSD][NSD];
  double         tau_yield_mp, inv2_Tpred_mp, inv2_D_mp;
  short          yield_type;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = EvaluateSofteningOnMarker(data,&Co,&phi);CHKERRQ(ierr);

  /* mark all markers as not yielding */
  MPntPStokesPlSetField_yield_indicator(data->mp_data->pls,YTYPE_NONE);

  /* compute yield surface */
  tau_yield_mp = *data->pressure * sin(phi) + Co * cos(phi);

  /* identify yield type */
  yield_type = YTYPE_DP;
  if ( tau_yield_mp < data->material_data->PlasticDP_data[region_idx].tens_cutoff) {
    /* failure in tension cutoff */
    tau_yield_mp = data->material_data->PlasticDP_data[region_idx].tens_cutoff;
    yield_type   = YTYPE_TENSILE_FAILURE;
  } else if (tau_yield_mp > data->material_data->PlasticDP_data[region_idx].hst_cutoff) {
    /* failure at High stress cut off */
    tau_yield_mp = data->material_data->PlasticDP_data[region_idx].hst_cutoff;
    yield_type   = YTYPE_MISES;
  }
  /* strain rate */
  ComputeStrainRate3d(data->ux,data->uy,data->uz,data->dNudx,data->dNudy,data->dNudz,D_mp);
  /* stress */
  ComputeStressIsotropic3d(*data->viscosity,D_mp,Tpred_mp);
  /* second inv stress */
  ComputeSecondInvariant3d(Tpred_mp,&inv2_Tpred_mp);

  if (inv2_Tpred_mp > tau_yield_mp) {
    ComputeSecondInvariant3d(D_mp,&inv2_D_mp);
    *data->viscosity = 0.5 * tau_yield_mp / inv2_D_mp;
    *data->npoints_yielded += 1;
    MPntPStokesPlSetField_yield_indicator(data->mp_data->pls,yield_type);
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode EvaluateViscosityOnMarker_Viscous(RheologyData *data)
{
  int            region_idx   = data->mp_data->std->phase;
  int            viscous_type = data->material_data->MatType_data[ region_idx ].visc_type;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  switch (viscous_type) {
  case VISCOUS_CONSTANT: 
    *data->viscosity = data->material_data->ViscConst_data[ region_idx ].eta0;
  break;

  case VISCOUS_Z: 
  {
    MaterialConst_ViscosityZ ViscZ_data = data->material_data->ViscZ_data[ region_idx ];
    double         y = data->mp_data->std->coor[1];
    *data->viscosity = ViscZ_data.eta0*exp(-(ViscZ_data.zref-y)/ViscZ_data.zeta);
  }
  break;

  case VISCOUS_FRANKK: 
  {
    MaterialConst_ViscosityFK ViscFK_data = data->material_data->ViscFK_data[ region_idx ];
    PetscReal      T = *data->T;
    *data->viscosity = ViscFK_data.eta0*exp(-ViscFK_data.theta*T);
  }
  break;

  case VISCOUS_ARRHENIUS: 
  {
    MaterialConst_ViscosityArrh ViscArrh_data = data->material_data->ViscArrh_data[ region_idx ];
    ierr = ViscosityArrheniusDislocationCreep(&ViscArrh_data,data,VISCOUS_ARRHENIUS);CHKERRQ(ierr);
  }
    break;

  case VISCOUS_ARRHENIUS_2: 
  {
    MaterialConst_ViscosityArrh ViscArrh_data = data->material_data->ViscArrh_data[ region_idx ];
    ierr = ViscosityArrheniusDislocationCreep(&ViscArrh_data,data,VISCOUS_ARRHENIUS_2);CHKERRQ(ierr);
  }
    break;

  case VISCOUS_ARRHENIUS_DISLDIFF:
  {
    MaterialConst_ViscosityArrh_DislDiff ViscArrhDislDiff_data = data->material_data->ViscArrhDislDiff_data[ region_idx ];
    ierr = ViscosityArrheniusDislocationDiffusionCreep(&ViscArrhDislDiff_data,data);CHKERRQ(ierr);
  }
    break;

  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"No default ViscousType specified. Valid choices are VISCOUS_CONSTANT, VISCOUS_Z, VISCOUS_FRANKK, VISCOUS_ARRHENIUS, VISCOUS_ARRHENIUS_2");
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode EvaluateViscosityOnMarker_Plastic(RheologyData *data)
{
  int            plastic_type = data->material_data->MatType_data[ data->mp_data->std->phase ].plastic_type;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  switch (plastic_type) {
    case PLASTIC_NONE: 
      break;

    case PLASTIC_MISES: 
      ierr = ViscosityPlasticMises(data);CHKERRQ(ierr);
      break;

    case PLASTIC_MISES_H: 
      ierr = ViscosityPlasticMisesH(data);CHKERRQ(ierr);
      break;

    case PLASTIC_DP: 
      ierr = ViscosityPlasticDruckerPrager(data);CHKERRQ(ierr);
      break;

    case PLASTIC_DP_H: 
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"PlasticType PLASTIC_DP_H not implemented.");
      break;

    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"No default PlasticType set. Valid choices are PLASTIC_NONE, PLASTIC_MISES, PLASTIC_DP");
      break;
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode EvaluateViscosityOnMarker(RheologyData *data)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = EvaluateViscosityOnMarker_Viscous(data);CHKERRQ(ierr);
  ierr = EvaluateViscosityOnMarker_Plastic(data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static inline PetscErrorCode EvaluateDensityOnMarker(RheologyData *data)
{
  int region_idx   = data->mp_data->std->phase;
  int density_type = data->material_data->MatType_data[ region_idx ].density_type;
  PetscFunctionBegin;
  switch (density_type) {
    case DENSITY_CONSTANT: 
    {
      PetscReal rho_mp = data->material_data->DensityConst_data[ region_idx ].density;
      MPntPStokesSetField_density(data->mp_data->stokes,rho_mp);
    }
      break;

    case DENSITY_BOUSSINESQ: 
    {
      PetscReal rho0  = data->material_data->DensityBoussinesq_data[ region_idx ].density;
      PetscReal alpha = data->material_data->DensityBoussinesq_data[ region_idx ].alpha;
      PetscReal beta  = data->material_data->DensityBoussinesq_data[ region_idx ].beta;
      PetscReal rho_mp;
      PetscReal T_mp = *data->T;
      PetscReal pressure_mp = *data->pressure;

      rho_mp = rho0*(1-alpha*T_mp+beta*pressure_mp);
      MPntPStokesSetField_density(data->mp_data->stokes,rho_mp);
    }
      break;

    case DENSITY_TABLE: 
    {
      PhaseMap  phasemap = data->material_data->DensityTable_data[ region_idx ].map;
      PetscReal rho0     = data->material_data->DensityTable_data[ region_idx ].density;
      PetscReal rho_mp,xp[2];
      
      xp[0] = *data->T; 
      xp[1] = *data->pressure; 
      PhaseMapGetValue(phasemap,xp,&rho_mp);
      if (rho_mp == (double)PHASE_MAP_POINT_OUTSIDE) {
        rho_mp = rho0; 
      }
      /* this check that there is no 1.0 (change table) 
         if that happens the previous density stored on the marker is conserved */
      if (rho_mp > rho0/10.0) {
        MPntPStokesSetField_density(data->mp_data->stokes,rho_mp);
      }
    }
      break;

    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"No default DensityType set. Valid choices are DENSITY_CONSTANT, DENSITY_BOUSSINESQ, DENSITY_TABLE");
      break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode HealPlasticStrainMarker(
  MaterialConst_MaterialType *MatType_data, 
  MaterialConst_PlasticMises *PlasticMises_data,
  MaterialConst_PlasticDP    *PlasticDP_data,
  MPntStd *mpprop_std, 
  MPntPStokesPl *mpprop_pls, 
  PetscReal dt)
{
  int region_idx, plastic_type;
  double healing_rate;
  float damage;
  PetscFunctionBegin;
  
  MPntStdGetField_phase_index(mpprop_std,&region_idx);
  
  plastic_type = MatType_data[ region_idx ].plastic_type;
  switch (plastic_type) {
    case PLASTIC_DP:
      healing_rate = PlasticDP_data[ region_idx ].healing_rate;
      break;
    
    case PLASTIC_MISES:
      healing_rate = PlasticMises_data[ region_idx ].healing_rate;
      break;
    
    case PLASTIC_MISES_H:
      healing_rate = PlasticMises_data[ region_idx ].healing_rate;
      break;
    
    default:
      healing_rate = 0.0;
      break;
  }
  MPntPStokesPlGetField_damage(mpprop_pls,&damage);
  damage = damage - dt * healing_rate;
  /* Ensure plastic strain cannot be negative */
  if (damage < 0.0) { damage = 0.0; }
  MPntPStokesPlSetField_damage(mpprop_pls,damage);
  
  PetscFunctionReturn(0);
}

PetscErrorCode private_EvaluateRheologyNonlinearitiesMarkers_VPSTD(pTatinCtx user,DM dau,PetscScalar ufield[],DM dap,PetscScalar pfield[],DM daT,PetscScalar Tfield[])
{
  DM             cda;
  Vec            gcoords,gcoords_T;
  DataBucket     db,material_constants;
  DataField      PField_std,PField_stokes,PField_pls;
  DataField      PField_MatTypes;
  DataField      PField_DensityConst,PField_DensityBoussinesq,PField_DensityTable;
  DataField      PField_ViscConst,PField_ViscZ,PField_ViscFK,PField_ViscArrh,PField_ViscArrhDislDiff;
  DataField      PField_PlasticMises,PField_PlasticDP;
  DataField      PField_SoftLin,PField_SoftExpo;
  PetscScalar    min_eta,max_eta,min_eta_g,max_eta_g;
  PetscReal      *LA_gcoords,*LA_gcoords_T;
  PetscReal      elcoords[3*Q2_NODES_PER_EL_3D];
  PetscReal      elu[3*Q2_NODES_PER_EL_3D],elp[P_BASIS_FUNCTIONS];
  PetscReal      elT[Q1_NODES_PER_EL_3D];
  PetscReal      ux[Q2_NODES_PER_EL_3D],uy[Q2_NODES_PER_EL_3D],uz[Q2_NODES_PER_EL_3D];
  PetscReal      NI_T[Q1_NODES_PER_EL_3D];
  PetscReal      NI[Q2_NODES_PER_EL_3D],GNI[3][Q2_NODES_PER_EL_3D],NIp[P_BASIS_FUNCTIONS];
  PetscReal      dNudx[Q2_NODES_PER_EL_3D],dNudy[Q2_NODES_PER_EL_3D],dNudz[Q2_NODES_PER_EL_3D];
  PetscInt       nel,nen_u,nen_p,eidx,k;
  PetscInt       nel_T,nen_T;
  const PetscInt *elnidx_u;
  const PetscInt *elnidx_p;
  const PetscInt *elnidx_T;
  PetscInt       vel_el_lidx[3*U_BASIS_FUNCTIONS];
  double         eta_mp;
  int            pidx,n_mp_points;
  long int       npoints_yielded,npoints_yielded_g;
  /* structs for material constants */
  MaterialConst_MaterialType           *MatType_data;
  MaterialConst_DensityConst           *DensityConst_data;
  MaterialConst_DensityBoussinesq      *DensityBoussinesq_data;
  MaterialConst_DensityTable           *DensityTable_data;
  MaterialConst_ViscosityConst         *ViscConst_data;
  MaterialConst_ViscosityZ             *ViscZ_data;
  MaterialConst_ViscosityFK            *ViscFK_data;
  MaterialConst_ViscosityArrh          *ViscArrh_data;
  MaterialConst_ViscosityArrh_DislDiff *ViscArrhDislDiff_data;
  MaterialConst_PlasticMises           *PlasticMises_data;
  MaterialConst_PlasticDP              *PlasticDP_data;
  MaterialConst_SoftLin                *SoftLin_data;
  MaterialConst_SoftExpo               *SoftExpo_data;
  /* structs for viscosity evaluation inline functions */
  MaterialData      material_data;
  MaterialPointData mp_data;
  RheologyData      rheology_data;
  PetscLogDouble    t0,t1;
  PetscErrorCode    ierr;
  PetscFunctionBegin;

  PetscTime(&t0);

  /* access material point information */
  ierr = pTatinGetMaterialPoints(user,&db,NULL);CHKERRQ(ierr);
  DataBucketGetDataFieldByName(db,MPntStd_classname,&PField_std);
  DataFieldGetAccess(PField_std);

  /* PField_std global index marker, phase marker, ...*/
  /* PField_stokes contains: etaf, rhof */
  /* PField_pls contains: accumulated plastic strain, yield type */
  DataBucketGetDataFieldByName(db,MPntPStokes_classname,&PField_stokes);
  DataFieldGetAccess(PField_stokes);

  DataBucketGetDataFieldByName(db,MPntPStokesPl_classname,&PField_pls);
  DataFieldGetAccess(PField_pls);

  DataBucketGetSizes(db,&n_mp_points,0,0);

  /* setup for coords */
  ierr = DMGetCoordinateDM(dau,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dau,&gcoords);CHKERRQ(ierr);
  ierr = VecGetArray(gcoords,&LA_gcoords);CHKERRQ(ierr);

  /* get u,p element information */
  ierr = DMDAGetElements_pTatinQ2P1(dau,&nel,&nen_u,&elnidx_u);CHKERRQ(ierr);
  ierr = DMDAGetElements_pTatinQ2P1(dap,&nel,&nen_p,&elnidx_p);CHKERRQ(ierr);

  if (daT) {
    /* access the energy variables stored on the markers */
    //DataBucketGetDataFieldByName(db,MPntPEnergy_classname,&PField_energy);
    //DataFieldGetAccess(PField_energy);

    /* access the coordinates for the temperature mesh */ /* THIS IS NOT ACTUALLY NEEDED */
    ierr = DMGetCoordinatesLocal(daT,&gcoords_T);CHKERRQ(ierr);
    ierr = VecGetArray(gcoords_T,&LA_gcoords_T);CHKERRQ(ierr);

    ierr = DMDAGetElementsQ1(daT,&nel_T,&nen_T,&elnidx_T);CHKERRQ(ierr);

    if (nel_T != nel) {
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Require code update to utilize nested Q1 mesh for temperature");
    }
  }
  /* access material constants */
  ierr = pTatinGetMaterialConstants(user,&material_constants);CHKERRQ(ierr);

  DataBucketGetDataFieldByName(material_constants,MaterialConst_MaterialType_classname,  &PField_MatTypes);
  MatType_data           = (MaterialConst_MaterialType*)PField_MatTypes->data;

  DataBucketGetDataFieldByName(material_constants,MaterialConst_DensityConst_classname,  &PField_DensityConst);
  DensityConst_data      = (MaterialConst_DensityConst*)PField_DensityConst->data;

  DataBucketGetDataFieldByName(material_constants,MaterialConst_DensityBoussinesq_classname,  &PField_DensityBoussinesq);
  DensityBoussinesq_data = (MaterialConst_DensityBoussinesq*)PField_DensityBoussinesq->data;

  DataBucketGetDataFieldByName(material_constants,MaterialConst_DensityTable_classname,  &PField_DensityTable);
  DensityTable_data = (MaterialConst_DensityTable*)PField_DensityTable->data;

  DataBucketGetDataFieldByName(material_constants,MaterialConst_ViscosityConst_classname,&PField_ViscConst);
  ViscConst_data         = (MaterialConst_ViscosityConst*)PField_ViscConst->data;

  DataBucketGetDataFieldByName(material_constants,MaterialConst_ViscosityArrh_classname,&PField_ViscArrh);
  ViscArrh_data          = (MaterialConst_ViscosityArrh*)PField_ViscArrh->data;

  DataBucketGetDataFieldByName(material_constants,MaterialConst_ViscosityArrh_DislDiff_classname,&PField_ViscArrhDislDiff);
  ViscArrhDislDiff_data  = (MaterialConst_ViscosityArrh_DislDiff*)PField_ViscArrhDislDiff->data;

  DataBucketGetDataFieldByName(material_constants,MaterialConst_ViscosityFK_classname,&PField_ViscFK);
  ViscFK_data            = (MaterialConst_ViscosityFK*)PField_ViscFK->data;

  DataBucketGetDataFieldByName(material_constants,MaterialConst_ViscosityZ_classname,&PField_ViscZ);
  ViscZ_data             = (MaterialConst_ViscosityZ*)PField_ViscZ->data;

  DataBucketGetDataFieldByName(material_constants,MaterialConst_PlasticMises_classname,  &PField_PlasticMises);
  PlasticMises_data      = (MaterialConst_PlasticMises*)  PField_PlasticMises->data;

  DataBucketGetDataFieldByName(material_constants,MaterialConst_PlasticDP_classname,  &PField_PlasticDP);
  PlasticDP_data         = (MaterialConst_PlasticDP*)  PField_PlasticDP->data;

  DataBucketGetDataFieldByName(material_constants,MaterialConst_SoftLin_classname,  &PField_SoftLin);
  SoftLin_data           = (MaterialConst_SoftLin*)  PField_SoftLin->data;

  DataBucketGetDataFieldByName(material_constants,MaterialConst_SoftExpo_classname,  &PField_SoftExpo);
  SoftExpo_data          = (MaterialConst_SoftExpo*)  PField_SoftExpo->data;

  /* marker loop */
  min_eta = 1.0e100;
  max_eta = 1.0e-100;
  npoints_yielded = 0;

  /* Setup data structure */
  ierr = PetscMemzero(&rheology_data,sizeof(RheologyData));CHKERRQ(ierr);
  ierr = PetscMemzero(&material_data,sizeof(MaterialData));CHKERRQ(ierr);
  ierr = PetscMemzero(&mp_data,sizeof(MaterialPointData));CHKERRQ(ierr);

  material_data.MatType_data           = MatType_data;
  material_data.DensityConst_data      = DensityConst_data;
  material_data.DensityBoussinesq_data = DensityBoussinesq_data;
  material_data.DensityTable_data      = DensityTable_data;
  material_data.ViscConst_data         = ViscConst_data;
  material_data.ViscZ_data             = ViscZ_data;
  material_data.ViscFK_data            = ViscFK_data;
  material_data.ViscArrh_data          = ViscArrh_data;
  material_data.ViscArrhDislDiff_data  = ViscArrhDislDiff_data;
  material_data.PlasticMises_data      = PlasticMises_data;
  material_data.PlasticDP_data         = PlasticDP_data;
  material_data.SoftLin_data           = SoftLin_data;
  material_data.SoftExpo_data          = SoftExpo_data;

  rheology_data.material_data          = &material_data;
  rheology_data.npoints_yielded        = &npoints_yielded;

  for (pidx=0; pidx<n_mp_points; pidx++) {
    MPntStd       *mpprop_std;
    MPntPStokes   *mpprop_stokes;
    MPntPStokesPl *mpprop_pls;
    double        *xi_p;
    double        pressure_mp,T_mp;

    DataFieldAccessPoint(PField_std,   pidx,(void**)&mpprop_std);
    DataFieldAccessPoint(PField_stokes,pidx,(void**)&mpprop_stokes);
    DataFieldAccessPoint(PField_pls,   pidx,(void**)&mpprop_pls);

    /* Get index of element containing this marker */
    eidx = mpprop_std->wil;

    /* Get element indices */
    ierr = StokesVelocity_GetElementLocalIndices(vel_el_lidx,(PetscInt*)&elnidx_u[nen_u*eidx]);CHKERRQ(ierr);

    /* Get element coordinates */
    ierr = DMDAGetElementCoordinatesQ2_3D(elcoords,(PetscInt*)&elnidx_u[nen_u*eidx],LA_gcoords);CHKERRQ(ierr);

    /* Get element velocity */
    ierr = DMDAGetVectorElementFieldQ2_3D(elu,(PetscInt*)&elnidx_u[nen_u*eidx],ufield);CHKERRQ(ierr);

    /* Get element pressure */
    ierr = DMDAGetScalarElementField(elp,nen_p,(PetscInt*)&elnidx_p[nen_p*eidx],pfield);CHKERRQ(ierr);

    if (daT) {
      /* Get element temperature */
      ierr = DMDAEQ1_GetScalarElementField_3D(elT,(PetscInt*)&elnidx_T[nen_T*eidx],Tfield);CHKERRQ(ierr);
    }

    /* Get local coordinate of marker */
    xi_p = mpprop_std->xi;

    /* Prepare basis functions */
    /* grad.Ni */
    P3D_ConstructGNi_Q2_3D(xi_p,GNI);
    /* Mi */
    ConstructNi_pressure(xi_p,elcoords,NIp);

    /* Get shape function derivatives */
    P3D_evaluate_global_derivatives_Q2(elcoords,GNI,dNudx,dNudy,dNudz);

    /* Compute pressure at material point */
    pressure_mp = 0.0;
    for (k=0; k<P_BASIS_FUNCTIONS; k++) {
      pressure_mp += NIp[k] * elp[k];
    }
    /* get velocity components */
    for (k=0; k<Q2_NODES_PER_EL_3D; k++) {
      ux[k] = elu[3*k  ];
      uy[k] = elu[3*k+1];
      uz[k] = elu[3*k+2];
    }

    pTatin_ConstructNi_Q2_3D( xi_p, NI );

    T_mp = 0.0;
    if (daT) {
      /* Interpolate the temperature */
      /* NOTE: scaling is requred of xi_p if nested mesh is used */
      P3D_ConstructNi_Q1_3D(xi_p,NI_T);
      for (k=0; k<Q1_NODES_PER_EL_3D; k++) {
        T_mp += NI_T[k] * elT[k];
      }
    }

    /* fill data structure */
    mp_data.std             = mpprop_std;
    mp_data.stokes          = mpprop_stokes;
    mp_data.pls             = mpprop_pls;
    rheology_data.mp_data   = &mp_data;
    rheology_data.ux        = ux;
    rheology_data.uy        = uy;
    rheology_data.uz        = uz;
    rheology_data.dNudx     = dNudx;
    rheology_data.dNudy     = dNudy;
    rheology_data.dNudz     = dNudz;
    rheology_data.T         = &T_mp;
    rheology_data.pressure  = &pressure_mp;
    rheology_data.viscosity = &eta_mp;

    /* get viscosity on marker */
    ierr = EvaluateViscosityOnMarker(&rheology_data);CHKERRQ(ierr);
    /* update viscosity on marker */
    MPntPStokesSetField_eta_effective(mpprop_stokes,eta_mp);
    /* monitor bounds */
    if (eta_mp > max_eta) { max_eta = eta_mp; }
    if (eta_mp < min_eta) { min_eta = eta_mp; }

    /* get density on marker */
    ierr = EvaluateDensityOnMarker(&rheology_data);CHKERRQ(ierr);
    /* Global cutoffs for viscosity */
    /* Here should I store these? */
  }
  DataFieldRestoreAccess(PField_std);
  DataFieldRestoreAccess(PField_stokes);
  DataFieldRestoreAccess(PField_pls);

  ierr = VecRestoreArray(gcoords,&LA_gcoords);CHKERRQ(ierr);
  if (daT) {
    //DataFieldRestoreAccess(PField_energy);
    ierr = VecRestoreArray(gcoords_T,&LA_gcoords_T);CHKERRQ(ierr);
  }
  ierr = MPI_Allreduce(&min_eta,&min_eta_g,1, MPI_DOUBLE, MPI_MIN, PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&max_eta,&max_eta_g,1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&npoints_yielded,&npoints_yielded_g,1, MPI_LONG, MPI_SUM, PETSC_COMM_WORLD);CHKERRQ(ierr);

  PetscTime(&t1);

  PetscPrintf(PETSC_COMM_WORLD,"Update non-linearities (VPSTD) [mpoint]: (min,max)_eta %1.2e,%1.2e; log10(max/min) %1.2e; npoints_yielded %ld; cpu time %1.2e (sec)\n",
      min_eta_g, max_eta_g, log10(max_eta_g/min_eta_g), npoints_yielded_g, t1-t0 );

  PetscFunctionReturn(0);
}

PetscErrorCode private_EvaluateRheologyNonlinearitiesMarkers_VPSTD_FV(pTatinCtx user,DM dau,const PetscScalar ufield[],DM dap,const PetscScalar pfield[],
                                                                      PhysCompEnergyFV energy,const PetscScalar Tfield[])
{
  DM                   cda;
  Vec                  gcoords,fv_coor_local;
  FVReconstructionCell rcell;
  RheologyData         rheology_data;
  MaterialData         material_data;
  MaterialPointData    mp_data;
  DataBucket           db,material_constants;
  DataField            PField_std,PField_stokes,PField_pls;
  DataField            PField_MatTypes;
  DataField            PField_DensityConst,PField_DensityBoussinesq,PField_DensityTable;
  DataField            PField_ViscConst,PField_ViscZ,PField_ViscFK,PField_ViscArrh,PField_ViscArrhDislDiff;
  DataField            PField_PlasticMises,PField_PlasticDP;
  DataField            PField_SoftLin,PField_SoftExpo;
  PetscScalar          min_eta,max_eta,min_eta_g,max_eta_g;
  PetscReal            *LA_gcoords;
  PetscReal            elcoords[3*Q2_NODES_PER_EL_3D];
  PetscReal            elu[3*Q2_NODES_PER_EL_3D],elp[P_BASIS_FUNCTIONS];
  PetscReal            ux[Q2_NODES_PER_EL_3D],uy[Q2_NODES_PER_EL_3D],uz[Q2_NODES_PER_EL_3D];
  PetscReal            NI[Q2_NODES_PER_EL_3D],GNI[3][Q2_NODES_PER_EL_3D],NIp[P_BASIS_FUNCTIONS];
  PetscReal            dNudx[Q2_NODES_PER_EL_3D],dNudy[Q2_NODES_PER_EL_3D],dNudz[Q2_NODES_PER_EL_3D];
  const PetscReal      *_fv_coor;
  PetscInt             nel,nen_u,nen_p,eidx,k;
  const PetscInt       *elnidx_u;
  const PetscInt       *elnidx_p;
  PetscInt             vel_el_lidx[3*U_BASIS_FUNCTIONS];
  double               eta_mp;
  long int             npoints_yielded,npoints_yielded_g;
  int                  pidx,n_mp_points;
  PetscErrorCode       ierr;
  PetscLogDouble       t0,t1;
  /* structs or material constants */
  MaterialConst_MaterialType           *MatType_data;
  MaterialConst_DensityConst           *DensityConst_data;
  MaterialConst_DensityBoussinesq      *DensityBoussinesq_data;
  MaterialConst_DensityTable           *DensityTable_data;
  MaterialConst_ViscosityConst         *ViscConst_data;
  MaterialConst_ViscosityZ             *ViscZ_data;
  MaterialConst_ViscosityFK            *ViscFK_data;
  MaterialConst_ViscosityArrh          *ViscArrh_data;
  MaterialConst_ViscosityArrh_DislDiff *ViscArrhDislDiff_data;
  MaterialConst_PlasticMises           *PlasticMises_data;
  MaterialConst_PlasticDP              *PlasticDP_data;
  MaterialConst_SoftLin                *SoftLin_data;
  MaterialConst_SoftExpo               *SoftExpo_data;
  
  PetscFunctionBegin;
  
  PetscTime(&t0);
  
  /* access material point information */
  ierr = pTatinGetMaterialPoints(user,&db,NULL);CHKERRQ(ierr);
  DataBucketGetDataFieldByName(db,MPntStd_classname,&PField_std);
  DataFieldGetAccess(PField_std);
  
  /* PField_std global index marker, phase marker, ...*/
  /* PField_stokes contains: etaf, rhof */
  /* PField_pls contains: accumulated plastic strain, yield type */
  DataBucketGetDataFieldByName(db,MPntPStokes_classname,&PField_stokes);
  DataFieldGetAccess(PField_stokes);
  
  DataBucketGetDataFieldByName(db,MPntPStokesPl_classname,&PField_pls);
  DataFieldGetAccess(PField_pls);
  
  DataBucketGetSizes(db,&n_mp_points,0,0);
  
  /* setup for coords */
  ierr = DMGetCoordinateDM(dau,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dau,&gcoords);CHKERRQ(ierr);
  ierr = VecGetArray(gcoords,&LA_gcoords);CHKERRQ(ierr);
  
  /* get u,p element information */
  ierr = DMDAGetElements_pTatinQ2P1(dau,&nel,&nen_u,&elnidx_u);CHKERRQ(ierr);
  ierr = DMDAGetElements_pTatinQ2P1(dap,&nel,&nen_p,&elnidx_p);CHKERRQ(ierr);
  
  /* access material constants */
  ierr = pTatinGetMaterialConstants(user,&material_constants);CHKERRQ(ierr);
  
  DataBucketGetDataFieldByName(material_constants,MaterialConst_MaterialType_classname,  &PField_MatTypes);
  MatType_data           = (MaterialConst_MaterialType*)PField_MatTypes->data;
  
  DataBucketGetDataFieldByName(material_constants,MaterialConst_DensityConst_classname,  &PField_DensityConst);
  DensityConst_data      = (MaterialConst_DensityConst*)PField_DensityConst->data;
  
  DataBucketGetDataFieldByName(material_constants,MaterialConst_DensityBoussinesq_classname,  &PField_DensityBoussinesq);
  DensityBoussinesq_data = (MaterialConst_DensityBoussinesq*)PField_DensityBoussinesq->data;
  
  DataBucketGetDataFieldByName(material_constants,MaterialConst_DensityTable_classname,  &PField_DensityTable);
  DensityTable_data      = (MaterialConst_DensityTable*)PField_DensityTable->data;

  DataBucketGetDataFieldByName(material_constants,MaterialConst_ViscosityConst_classname,&PField_ViscConst);
  ViscConst_data         = (MaterialConst_ViscosityConst*)PField_ViscConst->data;
  
  DataBucketGetDataFieldByName(material_constants,MaterialConst_ViscosityArrh_classname,&PField_ViscArrh);
  ViscArrh_data          = (MaterialConst_ViscosityArrh*)PField_ViscArrh->data;
  
  DataBucketGetDataFieldByName(material_constants,MaterialConst_ViscosityArrh_DislDiff_classname,&PField_ViscArrhDislDiff);
  ViscArrhDislDiff_data  = (MaterialConst_ViscosityArrh_DislDiff*)PField_ViscArrhDislDiff->data;

  DataBucketGetDataFieldByName(material_constants,MaterialConst_ViscosityFK_classname,&PField_ViscFK);
  ViscFK_data            = (MaterialConst_ViscosityFK*)PField_ViscFK->data;
  
  DataBucketGetDataFieldByName(material_constants,MaterialConst_ViscosityZ_classname,&PField_ViscZ);
  ViscZ_data             = (MaterialConst_ViscosityZ*)PField_ViscZ->data;
  
  DataBucketGetDataFieldByName(material_constants,MaterialConst_PlasticMises_classname,  &PField_PlasticMises);
  PlasticMises_data      = (MaterialConst_PlasticMises*)  PField_PlasticMises->data;
  
  DataBucketGetDataFieldByName(material_constants,MaterialConst_PlasticDP_classname,  &PField_PlasticDP);
  PlasticDP_data         = (MaterialConst_PlasticDP*)  PField_PlasticDP->data;
  
  DataBucketGetDataFieldByName(material_constants,MaterialConst_SoftLin_classname,  &PField_SoftLin);
  SoftLin_data           = (MaterialConst_SoftLin*)  PField_SoftLin->data;
  
  DataBucketGetDataFieldByName(material_constants,MaterialConst_SoftExpo_classname,  &PField_SoftExpo);
  SoftExpo_data          = (MaterialConst_SoftExpo*)  PField_SoftExpo->data;
  
  PetscInt fv_start[3],fv_start_local[3],fv_ghost_offset[3],fv_ghost_range[3];
  
  ierr = DMDAGetCorners(energy->fv->dm_fv,&fv_start[0],&fv_start[1],&fv_start[2],NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(energy->fv->dm_fv,&fv_start_local[0],&fv_start_local[1],&fv_start_local[2],&fv_ghost_range[0],&fv_ghost_range[1],&fv_ghost_range[2]);CHKERRQ(ierr);
  fv_ghost_offset[0] = fv_start[0] - fv_start_local[0];
  fv_ghost_offset[1] = fv_start[1] - fv_start_local[1];
  fv_ghost_offset[2] = fv_start[2] - fv_start_local[2];
  
  ierr = DMGetCoordinatesLocal(energy->fv->dm_fv,&fv_coor_local);CHKERRQ(ierr);
  ierr = VecGetArrayRead(fv_coor_local,&_fv_coor);CHKERRQ(ierr);
  
  /* marker loop */
  min_eta = 1.0e100;
  max_eta = 1.0e-100;
  npoints_yielded = 0;

  /* fill data structure */
  ierr = PetscMemzero(&material_data,sizeof(MaterialData));CHKERRQ(ierr);
  ierr = PetscMemzero(&rheology_data,sizeof(RheologyData));CHKERRQ(ierr);
  ierr = PetscMemzero(&mp_data,sizeof(MaterialPointData));CHKERRQ(ierr);

  material_data.MatType_data           = MatType_data;
  material_data.DensityConst_data      = DensityConst_data;
  material_data.DensityBoussinesq_data = DensityBoussinesq_data;
  material_data.DensityTable_data      = DensityTable_data;
  material_data.ViscConst_data         = ViscConst_data;
  material_data.ViscZ_data             = ViscZ_data;
  material_data.ViscFK_data            = ViscFK_data;
  material_data.ViscArrh_data          = ViscArrh_data;
  material_data.ViscArrhDislDiff_data  = ViscArrhDislDiff_data;
  material_data.PlasticMises_data      = PlasticMises_data;
  material_data.PlasticDP_data         = PlasticDP_data;
  material_data.SoftLin_data           = SoftLin_data;
  material_data.SoftExpo_data          = SoftExpo_data;

  rheology_data.material_data          = &material_data;
  rheology_data.npoints_yielded        = &npoints_yielded;

  for (pidx=0; pidx<n_mp_points; pidx++) {
    MPntStd       *mpprop_std;
    MPntPStokes   *mpprop_stokes;
    MPntPStokesPl *mpprop_pls;
    double        *xi_p;
    double        pressure_mp,T_mp;
    
    DataFieldAccessPoint(PField_std,   pidx,(void**)&mpprop_std);
    DataFieldAccessPoint(PField_stokes,pidx,(void**)&mpprop_stokes);
    DataFieldAccessPoint(PField_pls,   pidx,(void**)&mpprop_pls);
    
    /* Get index of element containing this marker */
    eidx = mpprop_std->wil;
    
    /* Get element indices */
    ierr = StokesVelocity_GetElementLocalIndices(vel_el_lidx,(PetscInt*)&elnidx_u[nen_u*eidx]);CHKERRQ(ierr);
    
    /* Get element coordinates */
    ierr = DMDAGetElementCoordinatesQ2_3D(elcoords,(PetscInt*)&elnidx_u[nen_u*eidx],LA_gcoords);CHKERRQ(ierr);
    
    /* Get element velocity */
    ierr = DMDAGetVectorElementFieldQ2_3D(elu,(PetscInt*)&elnidx_u[nen_u*eidx],(PetscScalar*)ufield);CHKERRQ(ierr);
    
    /* Get element pressure */
    ierr = DMDAGetScalarElementField(elp,nen_p,(PetscInt*)&elnidx_p[nen_p*eidx],(PetscScalar*)pfield);CHKERRQ(ierr);
    
    /* Get local coordinate of marker */
    xi_p = mpprop_std->xi;
    
    /* Prepare basis functions */
    /* grad.Ni */
    P3D_ConstructGNi_Q2_3D(xi_p,GNI);
    /* Mi */
    ConstructNi_pressure(xi_p,elcoords,NIp);
    
    /* Get shape function derivatives */
    P3D_evaluate_global_derivatives_Q2(elcoords,GNI,dNudx,dNudy,dNudz);
    
    /* Compute pressure at material point */
    pressure_mp = 0.0;
    for (k=0; k<P_BASIS_FUNCTIONS; k++) {
      pressure_mp += NIp[k] * elp[k];
    }
    /* get velocity components */
    for (k=0; k<Q2_NODES_PER_EL_3D; k++) {
      ux[k] = elu[3*k  ];
      uy[k] = elu[3*k+1];
      uz[k] = elu[3*k+2];
    }
    pTatin_ConstructNi_Q2_3D( xi_p, NI );

    T_mp = 0.0;
    /* Interpolate the temperature */
    {
      PetscInt sub_fv_cell,macro_ijk[3],macro_fv_ijk[3],ijk[3];
      
      /* P0 */
      //printf("point %d: cell %d: xi %+1.4e,%+1.4e,%+1.4e\n",pidx,eidx,xi_p[0],xi_p[1],xi_p[2]);
      ptatin_macro_point_location_sub(eidx,energy->mi_parent,energy->nsubdivision,xi_p,&sub_fv_cell);
      //printf("  fv subcell (macro local) %d \n",sub_fv_cell);
      
      ierr = _cart_convert_index_to_ijk(eidx,energy->mi_parent,macro_ijk);CHKERRQ(ierr);
      ierr = _cart_convert_index_to_ijk(sub_fv_cell,energy->nsubdivision,macro_fv_ijk);CHKERRQ(ierr);
      
      ijk[0] = fv_ghost_offset[0] + macro_ijk[0] * energy->nsubdivision[0] + macro_fv_ijk[0];
      ijk[1] = fv_ghost_offset[1] + macro_ijk[1] * energy->nsubdivision[1] + macro_fv_ijk[1];
      ijk[2] = fv_ghost_offset[2] + macro_ijk[2] * energy->nsubdivision[2] + macro_fv_ijk[2];
      
      ierr = _cart_convert_ijk_to_index(ijk,fv_ghost_range,&sub_fv_cell);CHKERRQ(ierr);
      //printf("  fv subcell (local) %d \n",sub_fv_cell);
      
      T_mp = Tfield[sub_fv_cell];
      
      ierr = FVReconstructionP1Create(&rcell,energy->fv,sub_fv_cell,_fv_coor,Tfield);CHKERRQ(ierr);
      ierr = FVReconstructionP1Interpolate(&rcell,mpprop_std->coor,&T_mp);CHKERRQ(ierr);
    }
    /* fill data structure */
    mp_data.std             = mpprop_std;
    mp_data.stokes          = mpprop_stokes;
    mp_data.pls             = mpprop_pls;
    rheology_data.mp_data   = &mp_data;
    rheology_data.ux        = ux;
    rheology_data.uy        = uy;
    rheology_data.uz        = uz;
    rheology_data.dNudx     = dNudx;
    rheology_data.dNudy     = dNudy;
    rheology_data.dNudz     = dNudz;
    rheology_data.T         = &T_mp;
    rheology_data.pressure  = &pressure_mp;
    rheology_data.viscosity = &eta_mp;
    
    /* get viscosity on marker */
    ierr = EvaluateViscosityOnMarker(&rheology_data);CHKERRQ(ierr);
    /* update viscosity on marker */
    MPntPStokesSetField_eta_effective(mpprop_stokes,eta_mp);
    /* monitor bounds */
    if (eta_mp > max_eta) { max_eta = eta_mp; }
    if (eta_mp < min_eta) { min_eta = eta_mp; }
    
    /* get density on marker */
    ierr = EvaluateDensityOnMarker(&rheology_data);CHKERRQ(ierr);
    /* Global cutoffs for viscosity */
    /* Here should I store these? */
  }

  DataFieldRestoreAccess(PField_std);
  DataFieldRestoreAccess(PField_stokes);
  DataFieldRestoreAccess(PField_pls);
  
  ierr = VecRestoreArray(gcoords,&LA_gcoords);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(fv_coor_local,&_fv_coor);CHKERRQ(ierr);

  ierr = MPI_Allreduce(&min_eta,&min_eta_g,1, MPI_DOUBLE, MPI_MIN, PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&max_eta,&max_eta_g,1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&npoints_yielded,&npoints_yielded_g,1, MPI_LONG, MPI_SUM, PETSC_COMM_WORLD);CHKERRQ(ierr);
  
  PetscTime(&t1);
  
  PetscPrintf(PETSC_COMM_WORLD,"Update non-linearities (VPSTD-FV) [mpoint]: (min,max)_eta %1.2e,%1.2e; log10(max/min) %1.2e; npoints_yielded %ld; cpu time %1.2e (sec)\n",
              min_eta_g, max_eta_g, log10(max_eta_g/min_eta_g), npoints_yielded_g, t1-t0 );
  
  PetscFunctionReturn(0);
}

PetscErrorCode EvaluateRheologyNonlinearitiesMarkers_VPSTD(pTatinCtx user,DM dau,PetscScalar ufield[],DM dap,PetscScalar pfield[])
{
  PetscBool found,found_fv;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = pTatinContextValid_Energy(user,&found);CHKERRQ(ierr);
  ierr = pTatinContextValid_EnergyFV(user,&found_fv);CHKERRQ(ierr);
  if (found) {
    PhysCompEnergy energy;
    DM             daT;
    Vec            temperature,temperature_l;
    PetscScalar    *LA_temperature_l;
    
    ierr = pTatinGetContext_Energy(user,&energy);CHKERRQ(ierr);
    ierr = pTatinPhysCompGetData_Energy(user,&temperature,NULL);CHKERRQ(ierr);
    daT  = energy->daT;

    ierr = DMGetLocalVector(daT,&temperature_l);CHKERRQ(ierr);
    ierr = VecZeroEntries(temperature_l);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(daT,temperature,INSERT_VALUES,temperature_l);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd  (daT,temperature,INSERT_VALUES,temperature_l);CHKERRQ(ierr);
    ierr = VecGetArray(temperature_l,&LA_temperature_l);CHKERRQ(ierr);

    ierr = private_EvaluateRheologyNonlinearitiesMarkers_VPSTD(user,dau,ufield,dap,pfield,daT,LA_temperature_l);CHKERRQ(ierr);

    ierr = VecRestoreArray(temperature_l,&LA_temperature_l);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(daT,&temperature_l);CHKERRQ(ierr);

  } else if (found_fv) {

    PhysCompEnergyFV  energy;
    DM                daT;
    Vec               temperature,temperature_l;
    const PetscScalar *LA_temperature_l;
    
    ierr = pTatinGetContext_EnergyFV(user,&energy);CHKERRQ(ierr);
    daT  = energy->fv->dm_fv;
    temperature = energy->T;
    ierr = DMGetLocalVector(daT,&temperature_l);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(daT,temperature,INSERT_VALUES,temperature_l);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd  (daT,temperature,INSERT_VALUES,temperature_l);CHKERRQ(ierr);
    ierr = VecGetArrayRead(temperature_l,&LA_temperature_l);CHKERRQ(ierr);
    
    ierr = private_EvaluateRheologyNonlinearitiesMarkers_VPSTD_FV(user,dau,(const PetscScalar*)ufield,dap,(const PetscScalar*)pfield,energy,LA_temperature_l);CHKERRQ(ierr);
    
    ierr = VecRestoreArrayRead(temperature_l,&LA_temperature_l);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(daT,&temperature_l);CHKERRQ(ierr);

  } else {
    ierr = private_EvaluateRheologyNonlinearitiesMarkers_VPSTD(user,dau,ufield,dap,pfield,NULL,NULL);CHKERRQ(ierr);
  }


  PetscFunctionReturn(0);
}



PetscErrorCode ApplyViscosityCutOffMarkers_VPSTD(pTatinCtx user)
{
  PetscErrorCode    ierr;
  int               pidx,n_mp_points;
  DataBucket        db;
  DataField         PField_std,PField_stokes;
  double            min_eta,max_eta,min_eta_g,max_eta_g,min_cutoff,max_cutoff;
  PetscLogDouble    t0,t1;
  RheologyConstants *rheology;
  double            eta_mp;
  long int          npoints_cutoff[2],npoints_cutoff_g[2];

  PetscFunctionBegin;

  PetscTime(&t0);

  /* access material point information */
  ierr = pTatinGetMaterialPoints(user,&db,NULL);CHKERRQ(ierr);
  DataBucketGetDataFieldByName(db,MPntStd_classname,&PField_std);
  DataFieldGetAccess(PField_std);

  DataBucketGetDataFieldByName(db,MPntPStokes_classname,&PField_stokes);
  DataFieldGetAccess(PField_stokes);

  DataBucketGetSizes(db,&n_mp_points,0,0);
  rheology = &user->rheology_constants;
  /* marker loop */
  min_eta = 1.0e100;
  max_eta = 1.0e-100;
  npoints_cutoff[0] = 0;
  npoints_cutoff[1] = 0;

  min_cutoff = rheology->eta_lower_cutoff_global;
  max_cutoff = rheology->eta_upper_cutoff_global;

  for (pidx=0; pidx<n_mp_points; pidx++) {
    MPntStd     *mpprop_std;
    MPntPStokes *mpprop_stokes;
    PetscScalar min_cutoff_l,max_cutoff_l;
    int         region_idx;

    DataFieldAccessPoint(PField_std,   pidx,(void**)&mpprop_std);
    DataFieldAccessPoint(PField_stokes,pidx,(void**)&mpprop_stokes);

    /* Get marker types */
    region_idx = mpprop_std->phase;

    min_cutoff_l = rheology->eta_lower_cutoff[region_idx];
    if (min_cutoff_l < min_cutoff) { min_cutoff_l = min_cutoff;}

    max_cutoff_l = rheology->eta_upper_cutoff[region_idx];
    if (max_cutoff_l > max_cutoff) { max_cutoff_l = max_cutoff;}

    MPntPStokesGetField_eta_effective(mpprop_stokes,&eta_mp);

    if (eta_mp > max_cutoff_l) { eta_mp = max_cutoff_l; npoints_cutoff[1]++; }
    if (eta_mp < min_cutoff_l) { eta_mp = min_cutoff_l; npoints_cutoff[0]++; }

    /* update viscosity on marker */
    MPntPStokesSetField_eta_effective(mpprop_stokes,eta_mp);

    /* monitor bounds */
    if (eta_mp > max_eta) { max_eta = eta_mp; }
    if (eta_mp < min_eta) { min_eta = eta_mp; }
  }

  DataFieldRestoreAccess(PField_std);
  DataFieldRestoreAccess(PField_stokes);

  ierr = MPI_Allreduce(&min_eta,&min_eta_g,1, MPI_DOUBLE, MPI_MIN, PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&max_eta,&max_eta_g,1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Allreduce(npoints_cutoff,npoints_cutoff_g,2, MPI_LONG, MPI_SUM, PETSC_COMM_WORLD);CHKERRQ(ierr);

  PetscTime(&t1);

  if ((npoints_cutoff_g[0] > 0) || (npoints_cutoff_g[1] > 0)) {
    PetscPrintf(PETSC_COMM_WORLD,"Apply viscosity cutoff (VPSTD) [mpoint]: (min,max)_eta %1.2e,%1.2e; log10(max/min) %1.2e; npoints_cutoff (%ld,%ld); cpu time %1.2e (sec)\n", min_eta_g, max_eta_g, log10(max_eta_g/min_eta_g), npoints_cutoff_g[0],npoints_cutoff_g[1], t1-t0 );
  }

  PetscFunctionReturn(0);
}


PetscErrorCode StokesCoefficient_UpdateTimeDependentQuantities_VPSTD(pTatinCtx user,DM dau,PetscScalar ufield[],DM dap,PetscScalar pfield[])
{
  PetscErrorCode ierr;
  DM             cda;
  Vec            gcoords;
  PetscScalar    *LA_gcoords;
  int            pidx,n_mp_points;
  DataBucket     db,material_constants;
  DataField      PField_std,PField,PField_MatTypes,PField_PlasticMises,PField_PlasticDP;
  float          strain_mp,damage_mp;
  PetscInt       nel,nen_u,k;
  const PetscInt *elnidx_u;
  PetscReal      elcoords[3*Q2_NODES_PER_EL_3D];
  PetscReal      elu[3*Q2_NODES_PER_EL_3D];
  PetscReal      ux[Q2_NODES_PER_EL_3D],uy[Q2_NODES_PER_EL_3D],uz[Q2_NODES_PER_EL_3D];
  PetscReal      GNI[3][Q2_NODES_PER_EL_3D];
  PetscReal      dNudx[Q2_NODES_PER_EL_3D],dNudy[Q2_NODES_PER_EL_3D],dNudz[Q2_NODES_PER_EL_3D];
  int            eidx_mp;
  double         *xi_mp;
  double         D_mp[NSD][NSD];
  double         inv2_D_mp;
  short          yield_type;
  PetscReal      dt;

  MaterialConst_MaterialType *MatType_data;
  MaterialConst_PlasticMises *PlasticMises_data;
  MaterialConst_PlasticDP    *PlasticDP_data;
  PetscFunctionBegin;

  /* access current time step */
  ierr = pTatinGetTimestep(user,&dt);CHKERRQ(ierr);

  /* access material point information */
  ierr = pTatinGetMaterialPoints(user,&db,NULL);CHKERRQ(ierr);

  DataBucketGetDataFieldByName(db,MPntStd_classname,&PField_std);
  DataFieldGetAccess(PField_std);
  DataBucketGetDataFieldByName(db,MPntPStokesPl_classname,&PField);
  DataFieldGetAccess(PField);

  /* access material constants */
  ierr = pTatinGetMaterialConstants(user,&material_constants);CHKERRQ(ierr);

  DataBucketGetDataFieldByName(material_constants,MaterialConst_MaterialType_classname,  &PField_MatTypes);
  MatType_data           = (MaterialConst_MaterialType*)PField_MatTypes->data;

  DataBucketGetDataFieldByName(material_constants,MaterialConst_PlasticMises_classname,  &PField_PlasticMises);
  PlasticMises_data      = (MaterialConst_PlasticMises*)  PField_PlasticMises->data;

  DataBucketGetDataFieldByName(material_constants,MaterialConst_PlasticDP_classname,  &PField_PlasticDP);
  PlasticDP_data         = (MaterialConst_PlasticDP*)  PField_PlasticDP->data;

  DataBucketGetSizes(db,&n_mp_points,0,0);

  /* setup for coords */
  ierr = DMGetCoordinateDM(dau,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dau,&gcoords);CHKERRQ(ierr);
  ierr = VecGetArray(gcoords,&LA_gcoords);CHKERRQ(ierr);

  /* setup for elements */
  ierr = DMDAGetElements_pTatinQ2P1(dau,&nel,&nen_u,&elnidx_u);CHKERRQ(ierr);

  /* marker loop */
  for (pidx=0; pidx<n_mp_points; pidx++) {
    MPntStd       *mpprop_std;
    MPntPStokesPl *mpprop;

    DataFieldAccessPoint(PField_std, pidx,(void**)&mpprop_std);
    DataFieldAccessPoint(PField,     pidx,(void**)&mpprop);

    MPntPStokesPlGetField_yield_indicator(mpprop,&yield_type);
    if (yield_type > 0) {
      eidx_mp = mpprop_std->wil;

      /* Get element coordinates */
      ierr = DMDAGetElementCoordinatesQ2_3D(elcoords,(PetscInt*)&elnidx_u[nen_u*eidx_mp],LA_gcoords);CHKERRQ(ierr);
      /* Get element velocity */
      ierr = DMDAGetVectorElementFieldQ2_3D(elu,(PetscInt*)&elnidx_u[nen_u*eidx_mp],ufield);CHKERRQ(ierr);
      /* get velocity components */
      for (k=0; k<Q2_NODES_PER_EL_3D; k++) {
        ux[k] = elu[3*k  ];
        uy[k] = elu[3*k+1];
        uz[k] = elu[3*k+2];
      }

      /* Get local coordinate of marker */
      xi_mp = mpprop_std->xi;

      /* Prepare basis functions */
      /* grad.Ni */
      P3D_ConstructGNi_Q2_3D(xi_mp,GNI);
      /* Get shape function derivatives */
      P3D_evaluate_global_derivatives_Q2(elcoords,GNI,dNudx,dNudy,dNudz);

      /* strain rate */
      ComputeStrainRate3d(ux,uy,uz,dNudx,dNudy,dNudz,D_mp);
      /* second inv stress */
      ComputeSecondInvariant3d(D_mp,&inv2_D_mp);
      /* Plastic strain */
      MPntPStokesPlGetField_plastic_strain(mpprop,&strain_mp);
      strain_mp = strain_mp + dt * inv2_D_mp;
      MPntPStokesPlSetField_plastic_strain(mpprop,strain_mp);
      /* Damage */
      MPntPStokesPlGetField_damage(mpprop,&damage_mp);
      damage_mp = damage_mp + dt * inv2_D_mp;
      MPntPStokesPlSetField_damage(mpprop,damage_mp);
    }
    ierr = HealPlasticStrainMarker(MatType_data,PlasticMises_data,PlasticDP_data,mpprop_std,mpprop,dt);CHKERRQ(ierr);
  }

  DataFieldRestoreAccess(PField_std);
  DataFieldRestoreAccess(PField);
  ierr = VecRestoreArray(gcoords,&LA_gcoords);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

