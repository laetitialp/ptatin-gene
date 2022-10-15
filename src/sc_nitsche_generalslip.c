
#include <petsc.h>
#include <ptatin3d_defs.h>
#include <ptatin3d.h>
#include <private/ptatin_impl.h>
#include <quadrature.h>
#include <private/quadrature_impl.h>
#include <element_type_Q2.h>
#include <dmda_element_q2p1.h>
#include <element_utils_q2.h>
#include <ptatin3d_stokes.h>
#include <mesh_entity.h>
#include <surface_constraint.h>
#include <sc_generic.h>


/* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> */
// fe_form_compiler.py version: 8d4b0b5b8d2e57803682a919e42ac439d4c64103
// sympy version: 1.6.1
// using common substring elimination: True
// form file: nitsche-custom-h_IJ.py version: 53800e8dcfeb59279abb73274a0ef2bf16e58dc5

#include "nitsche_genslip_a_forms.h"
#include "nitsche_genslip_b_forms.h"
/* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> */

const PetscInt _voigt[][3] = { {0, 3, 4}, {3, 1, 5}, {4, 5, 2} };

typedef enum { V_X1=0, V_X2 } StokesSubVec;

typedef enum { M_A11=0, M_A12, M_A21, M_A22 } StokesSubMat;

typedef struct {
  PetscBool setup;
  /* user fields */
  PetscReal penalty;
} SCContextDemo;

typedef struct {
  QPntSurfCoefStokes *boundary_qp;
  /* user fields */
  double          *hat_n_qp;     // "hat_n"
  double          *hat_t1_qp;    // "hat_t1"
  double          *hat_t2_qp;    // "hat_t2"
  double          *tau_S_qp;     // "tau_S"
  double          *mathcal_H_qp; // "H"
} FormContextDemo;

static PetscErrorCode _compute_gamma(SCContextDemo *ctx, PetscReal eta, StokesForm *form, PetscReal *gamma)
{
  *gamma = ctx->penalty * eta * 4.0 / form->hF;
  PetscFunctionReturn(0);
}

static PetscErrorCode _destroy_demo(SurfaceConstraint sc)
{
  SCContextDemo *ctx;
  PetscErrorCode ierr;
  if (sc->data) {
    ctx = (SCContextDemo*)sc->data;
    ierr = PetscFree(ctx);CHKERRQ(ierr);
    sc->data = NULL;
  }
  PetscFunctionReturn(0);
}

/* surface constraint implementation specific */
static PetscErrorCode _form_access_demo(StokesForm *form)
{
  PetscErrorCode    ierr;
  SurfaceConstraint sc;
  SurfaceQuadrature boundary_q;
  SCContextDemo     *scdata = NULL;
  FormContextDemo   *formdata = NULL;
  
  //printf("Form[Demo]: access()\n");
  sc = form->sc;
  scdata = (SCContextDemo*)sc->data;
  
  formdata = (FormContextDemo*)form->data;
  
  //printf("data->setup %d\n",scdata->setup);
  boundary_q = sc->quadrature;
  ierr = SurfaceQuadratureGetAllCellData_Stokes(boundary_q,&formdata->boundary_qp);CHKERRQ(ierr);
  
  DataBucketGetEntriesdByName(sc->properties_db,"hat_n" ,(void**)&formdata->hat_n_qp);
  DataBucketGetEntriesdByName(sc->properties_db,"hat_t1",(void**)&formdata->hat_t1_qp);
  DataBucketGetEntriesdByName(sc->properties_db,"hat_t2",(void**)&formdata->hat_t2_qp);
  DataBucketGetEntriesdByName(sc->properties_db,"tau_S" ,(void**)&formdata->tau_S_qp);
  DataBucketGetEntriesdByName(sc->properties_db,"H" ,(void**)&formdata->mathcal_H_qp);

  PetscFunctionReturn(0);
}

/* surface constraint implementation specific */
static PetscErrorCode _form_restore_demo(StokesForm *form)
{
  SurfaceConstraint sc;
  FormContextDemo   *formdata = NULL;
  
  //printf("Form[Demo]: restore()\n");
  formdata = (FormContextDemo*)form->data;
  
  sc = form->sc;
  
  DataBucketRestoreEntriesdByName(sc->properties_db,"hat_n" ,(void**)&formdata->hat_n_qp);
  DataBucketRestoreEntriesdByName(sc->properties_db,"hat_t1",(void**)&formdata->hat_t1_qp);
  DataBucketRestoreEntriesdByName(sc->properties_db,"hat_t2",(void**)&formdata->hat_t2_qp);
  DataBucketRestoreEntriesdByName(sc->properties_db,"tau_S" ,(void**)&formdata->tau_S_qp);
  DataBucketRestoreEntriesdByName(sc->properties_db,"H" ,(void**)&formdata->mathcal_H_qp);
  formdata->boundary_qp = NULL;
  
  PetscFunctionReturn(0);
}

/* surface constraint implementation specific */
static PetscErrorCode StokesFormSetupContext_Demo(StokesForm *F,FormContextDemo *formdata)
{
  PetscErrorCode ierr;
  
  /* data */
  ierr = PetscMemzero(formdata,sizeof(FormContextDemo));CHKERRQ(ierr);
  F->data = (void*)formdata;
  
  /* methods */
  F->access  = _form_access_demo;
  F->restore = _form_restore_demo;
  F->apply   = NULL;
  PetscFunctionReturn(0);
}

/* surface constraint implementation specific */
static PetscErrorCode StokesFormSetup_Demo(StokesForm *form,SurfaceConstraint sc,FormContextDemo *formdata)
{
  PetscErrorCode ierr;
  ierr = StokesFormInit(form,FORM_UNINIT,sc);CHKERRQ(ierr);
  ierr = StokeFormSetFunctionSpace_Q2P1(form);CHKERRQ(ierr);
  ierr = StokesFormSetupContext_Demo(form,formdata);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* action (residual) */
/* point-wise kernels */
static PetscErrorCode _form_residual_F1(StokesForm *form,PetscReal ds[],PetscReal F[])
{
  SCContextDemo      *scdata = NULL;
  FormContextDemo    *formdata = NULL;
  PetscInt           sq_index,sc_index,qp_offset,II,JJ;
  double             eta,*normal,*nhat,*that1,*that2,*tauS,*H;
  double             gN,gamma,*Lambda[] = {NULL,NULL,NULL};
  PetscErrorCode     ierr;
  
  scdata   = (SCContextDemo*)form->sc->data;
  formdata = (FormContextDemo*)form->data;
  sc_index = form->facet_sc_i * form->nqp  + form->point_i;
  sq_index = form->facet_i * form->nqp  + form->point_i;
  
  eta    = (PetscReal) formdata->boundary_qp[ sq_index ].eta;
  normal = (PetscReal*)formdata->boundary_qp[ sq_index ].normal;
  
  qp_offset = 3*sc_index;
  nhat  = &formdata->hat_n_qp[qp_offset];
  that1 = &formdata->hat_t1_qp[qp_offset];
  that2 = &formdata->hat_t2_qp[qp_offset];
  qp_offset = 6*sc_index;
  tauS  = &formdata->tau_S_qp[qp_offset];
  H     = &formdata->mathcal_H_qp[qp_offset];

  gN = 0.0;
  ierr = _compute_gamma(scdata,eta,form,&gamma);CHKERRQ(ierr);
  
  nitsche_custom_h_a_q2_3d_residual_w(form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
                                      form->X[0]->W, form->X[0]->Wx, form->X[0]->Wy, form->X[0]->Wz,
                                      form->X[1]->W, form->X[1]->Wx, form->X[1]->Wy, form->X[1]->Wz,
                                      form->u_elfield_0,form->u_elfield_1,form->u_elfield_2,
                                      form->p_elfield_0,
                                      gN,  // parameter
                                      gamma,  // parameter
                                      normal,  // parameter
                                      nhat,  // parameter
                                      ds[0], F);
  
  Lambda[0] = nhat;
  Lambda[1] = that1;
  Lambda[2] = that2;
  
  for (II=0; II<3; II++) {
    for (JJ=0; JJ<3; JJ++) {
      double *L_I = Lambda[II];
      double *L_J = Lambda[JJ];
      
      nitsche_custom_h_b_q2_3d_residual_w(form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
                                          form->X[0]->W, form->X[0]->Wx, form->X[0]->Wy, form->X[0]->Wz,
                                          form->X[1]->W, form->X[1]->Wx, form->X[1]->Wy, form->X[1]->Wz,
                                          form->u_elfield_0,form->u_elfield_1,form->u_elfield_2,
                                          form->p_elfield_0,
                                          eta,  // parameter
                                          normal,  // parameter
                                          L_I,  // parameter
                                          L_J,  // parameter
                                          H[ _voigt[II][JJ] ],  // parameter
                                          tauS,   // parameter
                                          ds[0], F);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode _form_residual_F2(StokesForm *form,PetscReal ds[],PetscReal F[])
{
  SCContextDemo      *scdata = NULL;
  FormContextDemo    *formdata = NULL;
  PetscInt           sq_index,sc_index,qp_offset,II,JJ;
  double             eta,*normal,*nhat,*that1,*that2,*tauS,*H;
  double             gN,gamma;
  PetscErrorCode     ierr;
  
  scdata   = (SCContextDemo*)form->sc->data;
  formdata = (FormContextDemo*)form->data;
  sc_index = form->facet_sc_i * form->nqp  + form->point_i;
  sq_index = form->facet_i * form->nqp  + form->point_i;
  
  eta    = (PetscReal) formdata->boundary_qp[ sq_index ].eta;
  normal = (PetscReal*)formdata->boundary_qp[ sq_index ].normal;
  
  qp_offset = 3*sc_index;
  nhat  = &formdata->hat_n_qp[qp_offset];
  that1 = &formdata->hat_t1_qp[qp_offset];
  that2 = &formdata->hat_t2_qp[qp_offset];
  qp_offset = 6*sc_index;
  tauS  = &formdata->tau_S_qp[qp_offset];
  H     = &formdata->mathcal_H_qp[qp_offset];
  
  gN = 0.0;
  ierr = _compute_gamma(scdata,eta,form,&gamma);CHKERRQ(ierr);
  
  nitsche_custom_h_a_q2_3d_residual_q(form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
                                      form->X[0]->W, form->X[0]->Wx, form->X[0]->Wy, form->X[0]->Wz,
                                      form->X[1]->W, form->X[1]->Wx, form->X[1]->Wy, form->X[1]->Wz,
                                      form->u_elfield_0,form->u_elfield_1,form->u_elfield_2,
                                      form->p_elfield_0,
                                      gN,  // parameter
                                      normal,  //parameter
                                      nhat,  // parameter
                                      ds[0], F);
  
  for (II=0; II<3; II++) {
    for (JJ=0; JJ<3; JJ++) {
      
      nitsche_custom_h_b_q2_3d_residual_q(form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
                                          form->X[0]->W, form->X[0]->Wx, form->X[0]->Wy, form->X[0]->Wz,
                                          form->X[1]->W, form->X[1]->Wx, form->X[1]->Wy, form->X[1]->Wz,
                                          form->u_elfield_0,form->u_elfield_1,form->u_elfield_2,
                                          form->p_elfield_0,
                                          ds[0], F);
    }
  }
  PetscFunctionReturn(0);
}

/* point-wise kernel configuration */
static PetscErrorCode StoksFormConfigureAction_Residual(StokesForm *form,StokesSubVec op)
{
  PetscErrorCode ierr;
  ierr = StokesFormSetType(form,FORM_RESIDUAL);CHKERRQ(ierr);
  switch (op) {
    case V_X1:
      form->apply = _form_residual_F1;
      break;
    case V_X2:
      form->apply = _form_residual_F2;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Must be one of X1, X2");
      break;
  }
  PetscFunctionReturn(0);
}

/* surface constraint methods */
static PetscErrorCode sc_residual_F1(
  SurfaceConstraint sc, DM dmu,const PetscScalar ufield[], DM dmp,const PetscScalar pfield[], PetscScalar R[])
{
  PetscErrorCode  ierr;
  StokesForm      F;
  FormContextDemo formdata;
  
  //printf("_Residual_F1\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_Residual(&F,V_X1);CHKERRQ(ierr);
  ierr = generic_facet_action(&F, &F.u, dmu, dmu,ufield, dmp,pfield, R);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode sc_residual_F2(
  SurfaceConstraint sc, DM dmu,const PetscScalar ufield[], DM dmp,const PetscScalar pfield[], PetscScalar R[])
{
  PetscErrorCode  ierr;
  StokesForm      F;
  FormContextDemo formdata;
  
  //printf("_Residual_F2\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_Residual(&F,V_X2);CHKERRQ(ierr);
  ierr = generic_facet_action(&F, &F.p, dmu, dmu,ufield, dmp,pfield, R);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* action (spmv) */
/* point-wise kernels */
static PetscErrorCode _form_spmv_A11(StokesForm *form,PetscReal ds[],PetscReal F[])
{
  SCContextDemo      *scdata = NULL;
  FormContextDemo    *formdata = NULL;
  PetscInt           sq_index,sc_index,qp_offset,II,JJ;
  double             eta,*normal,*nhat,*that1,*that2,*tauS,*H;
  double             gN,gamma,*Lambda[] = {NULL,NULL,NULL};
  PetscErrorCode     ierr;
  
  scdata   = (SCContextDemo*)form->sc->data;
  formdata = (FormContextDemo*)form->data;
  sc_index = form->facet_sc_i * form->nqp  + form->point_i;
  sq_index = form->facet_i * form->nqp  + form->point_i;
  
  eta    = (PetscReal) formdata->boundary_qp[ sq_index ].eta;
  normal = (PetscReal*)formdata->boundary_qp[ sq_index ].normal;
  
  qp_offset = 3*sc_index;
  nhat  = &formdata->hat_n_qp[qp_offset];
  that1 = &formdata->hat_t1_qp[qp_offset];
  that2 = &formdata->hat_t2_qp[qp_offset];
  qp_offset = 6*sc_index;
  tauS  = &formdata->tau_S_qp[qp_offset];
  H     = &formdata->mathcal_H_qp[qp_offset];
  
  gN = 0.0;
  ierr = _compute_gamma(scdata,eta,form,&gamma);CHKERRQ(ierr);

  nitsche_custom_h_a_q2_3d_spmv_wu(form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
                                  form->trial->W, form->trial->Wx, form->trial->Wy, form->trial->Wz,
                                  form->u_elfield_0,form->u_elfield_1,form->u_elfield_2,
                                  gamma,  // parameter
                                  nhat,  // parameter
                                  ds[0], F);

  Lambda[0] = nhat;
  Lambda[1] = that1;
  Lambda[2] = that2;
  
  for (II=0; II<3; II++) {
    for (JJ=0; JJ<3; JJ++) {
      double *L_I = Lambda[II];
      double *L_J = Lambda[JJ];
      
      nitsche_custom_h_b_q2_3d_spmv_wu(form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
                                       form->trial->W, form->trial->Wx, form->trial->Wy, form->trial->Wz,
                                       form->u_elfield_0,form->u_elfield_1,form->u_elfield_2,
                                       eta,  // parameter
                                       normal,  // parameter
                                       L_I,  // parameter
                                       L_J,  // parameter
                                       H[ _voigt[II][JJ] ],   // parameter
                                       ds[0], F);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode _form_spmv_A12(StokesForm *form,PetscReal ds[],PetscReal F[])
{
  SCContextDemo      *scdata = NULL;
  FormContextDemo    *formdata = NULL;
  PetscInt           sq_index,sc_index,qp_offset,II,JJ;
  double             eta,*normal,*nhat,*that1,*that2,*tauS,*H;
  double             gN,gamma,*Lambda[] = {NULL,NULL,NULL};
  PetscErrorCode     ierr;
  
  scdata   = (SCContextDemo*)form->sc->data;
  formdata = (FormContextDemo*)form->data;
  sc_index = form->facet_sc_i * form->nqp  + form->point_i;
  sq_index = form->facet_i * form->nqp  + form->point_i;
  
  eta    = (PetscReal) formdata->boundary_qp[ sq_index ].eta;
  normal = (PetscReal*)formdata->boundary_qp[ sq_index ].normal;
  
  qp_offset = 3*sc_index;
  nhat  = &formdata->hat_n_qp[qp_offset];
  that1 = &formdata->hat_t1_qp[qp_offset];
  that2 = &formdata->hat_t2_qp[qp_offset];
  qp_offset = 6*sc_index;
  tauS  = &formdata->tau_S_qp[qp_offset];
  H     = &formdata->mathcal_H_qp[qp_offset];
  
  gN = 0.0;
  ierr = _compute_gamma(scdata,eta,form,&gamma);CHKERRQ(ierr);

  nitsche_custom_h_a_q2_3d_spmv_wp(form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
                                  form->trial->W, form->trial->Wx, form->trial->Wy, form->trial->Wz,
                                  form->p_elfield_0,
                                  normal,  // parameter
                                  ds[0], F);

  Lambda[0] = nhat;
  Lambda[1] = that1;
  Lambda[2] = that2;
  
  for (II=0; II<3; II++) {
    for (JJ=0; JJ<3; JJ++) {
      double *L_I = Lambda[II];
      double *L_J = Lambda[JJ];
      
      nitsche_custom_h_b_q2_3d_spmv_wp(form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
                                       form->trial->W, form->trial->Wx, form->trial->Wy, form->trial->Wz,
                                       form->p_elfield_0,
                                       ds[0], F);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode _form_spmv_A21(StokesForm *form,PetscReal ds[],PetscReal F[])
{
  SCContextDemo      *scdata = NULL;
  FormContextDemo    *formdata = NULL;
  PetscInt           sq_index,sc_index,qp_offset,II,JJ;
  double             eta,*normal,*nhat,*that1,*that2,*tauS,*H;
  double             gN,gamma,*Lambda[] = {NULL,NULL,NULL};
  PetscErrorCode     ierr;
  
  scdata   = (SCContextDemo*)form->sc->data;
  formdata = (FormContextDemo*)form->data;
  sc_index = form->facet_sc_i * form->nqp  + form->point_i;
  sq_index = form->facet_i * form->nqp  + form->point_i;
  
  eta    = (PetscReal) formdata->boundary_qp[ sq_index ].eta;
  normal = (PetscReal*)formdata->boundary_qp[ sq_index ].normal;
  
  qp_offset = 3*sc_index;
  nhat  = &formdata->hat_n_qp[qp_offset];
  that1 = &formdata->hat_t1_qp[qp_offset];
  that2 = &formdata->hat_t2_qp[qp_offset];
  qp_offset = 6*sc_index;
  tauS  = &formdata->tau_S_qp[qp_offset];
  H     = &formdata->mathcal_H_qp[qp_offset];
  
  gN = 0.0;
  ierr = _compute_gamma(scdata,eta,form,&gamma);CHKERRQ(ierr);

  nitsche_custom_h_a_q2_3d_spmv_qu(form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
                                  form->trial->W, form->trial->Wx, form->trial->Wy, form->trial->Wz,
                                  form->u_elfield_0,form->u_elfield_1,form->u_elfield_2,
                                  normal,  // parameter
                                  nhat,  // parameter
                                  ds[0], F);

  for (II=0; II<3; II++) {
    for (JJ=0; JJ<3; JJ++) {
      double *L_I = Lambda[II];
      double *L_J = Lambda[JJ];
      
      nitsche_custom_h_b_q2_3d_spmv_qu(form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
                                       form->trial->W, form->trial->Wx, form->trial->Wy, form->trial->Wz,
                                       form->u_elfield_0,form->u_elfield_1,form->u_elfield_2,
                                       ds[0], F);
    }
  }
  PetscFunctionReturn(0);
}

/* point-wise kernel configuration */
static PetscErrorCode StoksFormConfigureAction_SpMV(StokesForm *form,StokesSubMat op)
{
  PetscErrorCode ierr;
  ierr = StokesFormSetType(form,FORM_SPMV);CHKERRQ(ierr);
  switch (op) {
    case M_A11:
      form->apply = _form_spmv_A11;
      break;
    case M_A12:
      form->apply = _form_spmv_A12;
      break;
    case M_A21:
      form->apply = _form_spmv_A21;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Must be one of A11, A12, A21");
      break;
  }
  PetscFunctionReturn(0);
}

/* surface constraint methods */
static PetscErrorCode sc_spmv_A11(
  SurfaceConstraint sc, DM dmu,const PetscScalar ufield[], PetscScalar Y[])
{
  PetscErrorCode  ierr;
  StokesForm      F;
  FormContextDemo formdata;
  
  //printf("_SpMV_A11\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_SpMV(&F,M_A11);CHKERRQ(ierr);
  ierr = generic_facet_action(&F, &F.u, dmu, dmu,ufield, NULL,NULL, Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode sc_spmv_A12(
  SurfaceConstraint sc, DM dmu, DM dmp,const PetscScalar pfield[], PetscScalar Y[])
{
  PetscErrorCode  ierr;
  StokesForm      F;
  FormContextDemo formdata;
  
  //printf("_SpMV_A12\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_SpMV(&F,M_A12);CHKERRQ(ierr);
  ierr = generic_facet_action(&F, &F.u, dmu, dmu,NULL, dmp,pfield, Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode sc_spmv_A21(
  SurfaceConstraint sc, DM dmu,const PetscScalar ufield[], DM dmp, PetscScalar Y[])
{
  PetscErrorCode  ierr;
  StokesForm      F;
  FormContextDemo formdata;
  
  //printf("_SpMV_A21\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_SpMV(&F,M_A21);CHKERRQ(ierr);
  ierr = generic_facet_action(&F, &F.p, dmu, dmu,ufield, dmp,NULL, Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* assemble */
/* point-wise kernels */
static PetscErrorCode _form_asmb_A11(StokesForm *form,PetscReal ds[],PetscReal A[])
{
  SCContextDemo      *scdata = NULL;
  FormContextDemo    *formdata = NULL;
  PetscInt           sq_index,sc_index,qp_offset,II,JJ;
  double             eta,*normal,*nhat,*that1,*that2,*tauS,*H;
  double             gamma,*Lambda[] = {NULL,NULL,NULL};
  PetscErrorCode     ierr;
  
  scdata   = (SCContextDemo*)form->sc->data;
  formdata = (FormContextDemo*)form->data;
  sc_index = form->facet_sc_i * form->nqp  + form->point_i;
  sq_index = form->facet_i * form->nqp  + form->point_i;
  
  eta    = (PetscReal) formdata->boundary_qp[ sq_index ].eta;
  normal = (PetscReal*)formdata->boundary_qp[ sq_index ].normal;
  
  qp_offset = 3*sc_index;
  nhat  = &formdata->hat_n_qp[qp_offset];
  that1 = &formdata->hat_t1_qp[qp_offset];
  that2 = &formdata->hat_t2_qp[qp_offset];
  qp_offset = 6*sc_index;
  tauS  = &formdata->tau_S_qp[qp_offset];
  H     = &formdata->mathcal_H_qp[qp_offset];
  
  ierr = _compute_gamma(scdata,eta,form,&gamma);CHKERRQ(ierr);

  nitsche_custom_h_a_q2_3d_asmb_wu(form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
                                  form->trial->W, form->trial->Wx, form->trial->Wy, form->trial->Wz,
                                  gamma,  // parameter
                                  nhat,  // parameter
                                  ds[0], A);

  Lambda[0] = nhat;
  Lambda[1] = that1;
  Lambda[2] = that2;

  for (II=0; II<3; II++) {
    for (JJ=0; JJ<3; JJ++) {
      double *L_I = Lambda[II];
      double *L_J = Lambda[JJ];
      
      nitsche_custom_h_b_q2_3d_asmb_wu(form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
                                       form->trial->W, form->trial->Wx, form->trial->Wy, form->trial->Wz,
                                       eta,  // parameter
                                       normal,  // parameter
                                       L_I,  // parameter
                                       L_J,  // parameter
                                       H[ _voigt[II][JJ] ],  // parameter
                                       ds[0], A);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode _form_asmb_A12(StokesForm *form,PetscReal ds[],PetscReal A[])
{
  SCContextDemo      *scdata = NULL;
  FormContextDemo    *formdata = NULL;
  PetscInt           sq_index,sc_index,qp_offset,II,JJ;
  double             eta,*normal,*nhat,*that1,*that2,*tauS,*H;
  double             gamma,*Lambda[] = {NULL,NULL,NULL};
  PetscErrorCode     ierr;
  
  scdata   = (SCContextDemo*)form->sc->data;
  formdata = (FormContextDemo*)form->data;
  sc_index = form->facet_sc_i * form->nqp  + form->point_i;
  sq_index = form->facet_i * form->nqp  + form->point_i;
  
  eta    = (PetscReal) formdata->boundary_qp[ sq_index ].eta;
  normal = (PetscReal*)formdata->boundary_qp[ sq_index ].normal;
  
  qp_offset = 3*sc_index;
  nhat  = &formdata->hat_n_qp[qp_offset];
  that1 = &formdata->hat_t1_qp[qp_offset];
  that2 = &formdata->hat_t2_qp[qp_offset];
  qp_offset = 6*sc_index;
  tauS  = &formdata->tau_S_qp[qp_offset];
  H     = &formdata->mathcal_H_qp[qp_offset];
  
  ierr = _compute_gamma(scdata,eta,form,&gamma);CHKERRQ(ierr);

  nitsche_custom_h_a_q2_3d_asmb_wp(form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
                                  form->trial->W, form->trial->Wx, form->trial->Wy, form->trial->Wz,
                                  normal,  // parameter
                                  ds[0], A);

  for (II=0; II<3; II++) {
    for (JJ=0; JJ<3; JJ++) {
      double *L_I = Lambda[II];
      double *L_J = Lambda[JJ];
      
      nitsche_custom_h_b_q2_3d_asmb_wp(form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
                                       form->trial->W, form->trial->Wx, form->trial->Wy, form->trial->Wz,
                                       ds[0], A);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode _form_asmb_A21(StokesForm *form,PetscReal ds[],PetscReal A[])
{
  SCContextDemo      *scdata = NULL;
  FormContextDemo    *formdata = NULL;
  PetscInt           sq_index,sc_index,qp_offset,II,JJ;
  double             eta,*normal,*nhat,*that1,*that2,*tauS,*H;
  double             gamma,*Lambda[] = {NULL,NULL,NULL};
  PetscErrorCode     ierr;
  
  scdata   = (SCContextDemo*)form->sc->data;
  formdata = (FormContextDemo*)form->data;
  sc_index = form->facet_sc_i * form->nqp  + form->point_i;
  sq_index = form->facet_i * form->nqp  + form->point_i;
  
  eta    = (PetscReal) formdata->boundary_qp[ sq_index ].eta;
  normal = (PetscReal*)formdata->boundary_qp[ sq_index ].normal;
  
  qp_offset = 3*sc_index;
  nhat  = &formdata->hat_n_qp[qp_offset];
  that1 = &formdata->hat_t1_qp[qp_offset];
  that2 = &formdata->hat_t2_qp[qp_offset];
  qp_offset = 6*sc_index;
  tauS  = &formdata->tau_S_qp[qp_offset];
  H     = &formdata->mathcal_H_qp[qp_offset];
  
  ierr = _compute_gamma(scdata,eta,form,&gamma);CHKERRQ(ierr);

  nitsche_custom_h_a_q2_3d_asmb_qu(form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
                                  form->trial->W, form->trial->Wx, form->trial->Wy, form->trial->Wz,
                                  normal,  // parameter
                                  nhat,  // parameter
                                  ds[0], A);
  
  for (II=0; II<3; II++) {
    for (JJ=0; JJ<3; JJ++) {
      double *L_I = Lambda[II];
      double *L_J = Lambda[JJ];
      
      nitsche_custom_h_b_q2_3d_asmb_qu(form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
                                       form->trial->W, form->trial->Wx, form->trial->Wy, form->trial->Wz,
                                       ds[0], A);
    }
  }
  PetscFunctionReturn(0);
}

/* point-wise kernel configuration */
static PetscErrorCode StoksFormConfigureAction_Assemble(StokesForm *form,StokesSubMat op)
{
  PetscErrorCode ierr;
  ierr = StokesFormSetType(form,FORM_ASSEMBLE);CHKERRQ(ierr);
  switch (op) {
    case M_A11:
      form->apply = _form_asmb_A11;
      break;
    case M_A12:
      form->apply = _form_asmb_A12;
      break;
    case M_A21:
      form->apply = _form_asmb_A21;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Must be one of A11, A12, A21");
      break;
  }
  PetscFunctionReturn(0);
}

/* surface constraint methods */
static PetscErrorCode sc_asmb_A11(SurfaceConstraint sc, DM dmu, Mat A)
{
  PetscErrorCode  ierr;
  StokesForm      F;
  FormContextDemo formdata;
  
  //printf("_Assemble_A11\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_Assemble(&F,M_A11);CHKERRQ(ierr);
  ierr = generic_facet_assemble(&F, &F.u,&F.u, dmu, dmu, NULL, A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode sc_asmb_A12(SurfaceConstraint sc, DM dmu, DM dmp, Mat A)
{
  PetscErrorCode  ierr;
  StokesForm      F;
  FormContextDemo formdata;
  
  //printf("_Assemble_A12\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_Assemble(&F,M_A12);CHKERRQ(ierr);
  ierr = generic_facet_assemble(&F, &F.u,&F.p, dmu, dmu, dmp, A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode sc_asmb_A21(SurfaceConstraint sc, DM dmu, DM dmp, Mat A)
{
  PetscErrorCode  ierr;
  StokesForm      F;
  FormContextDemo formdata;
  
  //printf("_Assemble_A21\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_Assemble(&F,M_A21);CHKERRQ(ierr);
  ierr = generic_facet_assemble(&F, &F.p,&F.u, dmu, dmu, dmp, A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* assemble diagonal */
/* point-wise kernels */
static PetscErrorCode _form_asmbdiag_A11(StokesForm *form,PetscReal ds[],PetscReal diagA[])
{
  SCContextDemo      *scdata = NULL;
  FormContextDemo    *formdata = NULL;
  PetscInt           sq_index,sc_index,qp_offset,II,JJ;
  double             eta,*normal,*nhat,*that1,*that2,*tauS,*H;
  double             gamma,*Lambda[] = {NULL,NULL,NULL};
  PetscErrorCode     ierr;
  
  scdata   = (SCContextDemo*)form->sc->data;
  formdata = (FormContextDemo*)form->data;
  sc_index = form->facet_sc_i * form->nqp  + form->point_i;
  sq_index = form->facet_i * form->nqp  + form->point_i;
  
  eta    = (PetscReal) formdata->boundary_qp[ sq_index ].eta;
  normal = (PetscReal*)formdata->boundary_qp[ sq_index ].normal;
  
  qp_offset = 3*sc_index;
  nhat  = &formdata->hat_n_qp[qp_offset];
  that1 = &formdata->hat_t1_qp[qp_offset];
  that2 = &formdata->hat_t2_qp[qp_offset];
  qp_offset = 6*sc_index;
  tauS  = &formdata->tau_S_qp[qp_offset];
  H     = &formdata->mathcal_H_qp[qp_offset];
  
  ierr = _compute_gamma(scdata,eta,form,&gamma);CHKERRQ(ierr);

  nitsche_custom_h_a_q2_3d_asmbdiag_wu(form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
                                      form->trial->W, form->trial->Wx, form->trial->Wy, form->trial->Wz,
                                      gamma, // parameter
                                      nhat,  // parameter
                                      ds[0], diagA);
  
  Lambda[0] = nhat;
  Lambda[1] = that1;
  Lambda[2] = that2;

  for (II=0; II<3; II++) {
    for (JJ=0; JJ<3; JJ++) {
      double *L_I = Lambda[II];
      double *L_J = Lambda[JJ];
      
      nitsche_custom_h_b_q2_3d_asmbdiag_wu(form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
                                           form->trial->W, form->trial->Wx, form->trial->Wy, form->trial->Wz,
                                           eta, // parameter
                                           normal,  // parameter
                                           L_I,  // parameter
                                           L_J,  // parameter
                                           H[ _voigt[II][JJ] ],  // parameter
                                           ds[0], diagA);
    }
  }
  PetscFunctionReturn(0);
}

/* point-wise kernel configuration */
static PetscErrorCode StoksFormConfigureAction_AssembleDiagonal(StokesForm *form,StokesSubMat op)
{
  PetscErrorCode ierr;
  ierr = StokesFormSetType(form,FORM_ASSEMBLE_DIAG);CHKERRQ(ierr);
  switch (op) {
    case M_A11:
      form->apply = _form_asmbdiag_A11;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Can only be A11");
      break;
  }
  PetscFunctionReturn(0);
}

/* surface constraint methods */
static PetscErrorCode sc_asmbdiag_A11(SurfaceConstraint sc, DM dmu, Vec diagA)
{
  PetscErrorCode  ierr;
  StokesForm      F;
  FormContextDemo formdata;
  
  //printf("_AssembleDiagonal_A11\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_AssembleDiagonal(&F,M_A11);CHKERRQ(ierr);
  ierr = generic_facet_assemble_diagonal(&F, &F.u,dmu,  dmu, diagA);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode _form_spmv_wA(StokesForm *form,PetscReal ds[],PetscReal F[])
{
  SCContextDemo      *scdata = NULL;
  FormContextDemo    *formdata = NULL;
  PetscInt           sq_index,sc_index,qp_offset,II,JJ;
  double             eta,*normal,*nhat,*that1,*that2,*tauS,*H;
  double             gamma,*Lambda[] = {NULL,NULL,NULL};
  PetscErrorCode     ierr;
  
  scdata   = (SCContextDemo*)form->sc->data;
  formdata = (FormContextDemo*)form->data;
  sc_index = form->facet_sc_i * form->nqp  + form->point_i;
  sq_index = form->facet_i * form->nqp  + form->point_i;
  
  eta    = (PetscReal) formdata->boundary_qp[ sq_index ].eta;
  normal = (PetscReal*)formdata->boundary_qp[ sq_index ].normal;
  
  qp_offset = 3*sc_index;
  nhat  = &formdata->hat_n_qp[qp_offset];
  that1 = &formdata->hat_t1_qp[qp_offset];
  that2 = &formdata->hat_t2_qp[qp_offset];
  qp_offset = 6*sc_index;
  tauS  = &formdata->tau_S_qp[qp_offset];
  H     = &formdata->mathcal_H_qp[qp_offset];
  
  ierr = _compute_gamma(scdata,eta,form,&gamma);CHKERRQ(ierr);

  nitsche_custom_h_a_q2_3d_spmv_w_up(form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
                                    form->X[0]->W, form->X[0]->Wx, form->X[0]->Wy, form->X[0]->Wz,
                                    form->X[1]->W, form->X[1]->Wx, form->X[1]->Wy, form->X[1]->Wz,
                                    form->u_elfield_0,form->u_elfield_1,form->u_elfield_2,
                                    form->p_elfield_0,
                                    gamma,  // parameter
                                    normal,  // parameter
                                    nhat,  // parameter
                                    ds[0], F);

  Lambda[0] = nhat;
  Lambda[1] = that1;
  Lambda[2] = that2;

  for (II=0; II<3; II++) {
    for (JJ=0; JJ<3; JJ++) {
      double *L_I = Lambda[II];
      double *L_J = Lambda[JJ];
      
      nitsche_custom_h_b_q2_3d_spmv_w_up(form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
                                         form->X[0]->W, form->X[0]->Wx, form->X[0]->Wy, form->X[0]->Wz,
                                         form->X[1]->W, form->X[1]->Wx, form->X[1]->Wy, form->X[1]->Wz,
                                         form->u_elfield_0,form->u_elfield_1,form->u_elfield_2,
                                         form->p_elfield_0,
                                         eta,  // parameter
                                         normal,  // parameter
                                         L_I,  // parameter
                                         L_J,  // parameter
                                         H[ _voigt[II][JJ] ],  // parameter
                                         ds[0], F);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode _form_spmv_qA(StokesForm *form,PetscReal ds[],PetscReal F[])
{
  SCContextDemo      *scdata = NULL;
  FormContextDemo    *formdata = NULL;
  PetscInt           sq_index,sc_index,qp_offset,II,JJ;
  double             eta,*normal,*nhat,*that1,*that2,*tauS,*H;
  double             gamma,*Lambda[] = {NULL,NULL,NULL};
  PetscErrorCode     ierr;
  
  scdata   = (SCContextDemo*)form->sc->data;
  formdata = (FormContextDemo*)form->data;
  sc_index = form->facet_sc_i * form->nqp  + form->point_i;
  sq_index = form->facet_i * form->nqp  + form->point_i;
  
  eta    = (PetscReal) formdata->boundary_qp[ sq_index ].eta;
  normal = (PetscReal*)formdata->boundary_qp[ sq_index ].normal;
  
  qp_offset = 3*sc_index;
  nhat  = &formdata->hat_n_qp[qp_offset];
  that1 = &formdata->hat_t1_qp[qp_offset];
  that2 = &formdata->hat_t2_qp[qp_offset];
  qp_offset = 6*sc_index;
  tauS  = &formdata->tau_S_qp[qp_offset];
  H     = &formdata->mathcal_H_qp[qp_offset];
  
  ierr = _compute_gamma(scdata,eta,form,&gamma);CHKERRQ(ierr);

  nitsche_custom_h_a_q2_3d_spmv_q_up(form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
                                    form->X[0]->W, form->X[0]->Wx, form->X[0]->Wy, form->X[0]->Wz,
                                    form->X[1]->W, form->X[1]->Wx, form->X[1]->Wy, form->X[1]->Wz,
                                    form->u_elfield_0,form->u_elfield_1,form->u_elfield_2,
                                    form->p_elfield_0,
                                    normal,  // parameter
                                    nhat,  // parameter
                                    ds[0], F);
  
  for (II=0; II<3; II++) {
    for (JJ=0; JJ<3; JJ++) {
      double *L_I = Lambda[II];
      double *L_J = Lambda[JJ];
      
      nitsche_custom_h_b_q2_3d_spmv_q_up(form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
                                         form->X[0]->W, form->X[0]->Wx, form->X[0]->Wy, form->X[0]->Wz,
                                         form->X[1]->W, form->X[1]->Wx, form->X[1]->Wy, form->X[1]->Wz,
                                         form->u_elfield_0,form->u_elfield_1,form->u_elfield_2,
                                         form->p_elfield_0,
                                         ds[0], F);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode StoksFormConfigureAction_AuResidual(StokesForm *form,StokesSubVec op)
{
  PetscErrorCode ierr;
  ierr = StokesFormSetType(form,FORM_RESIDUAL);CHKERRQ(ierr);
  switch (op) {
    case V_X1:
      form->apply = _form_spmv_wA;
      break;
    case V_X2:
      form->apply = _form_spmv_qA;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Must be one of X1, X2");
      break;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode sc_spmv_A(SurfaceConstraint sc,
                                DM dmu,const PetscScalar ufield[],
                                DM dmp,const PetscScalar pfield[],
                                PetscScalar Yu[], PetscScalar Yp[])
{
  PetscErrorCode  ierr;
  StokesForm      F;
  FormContextDemo formdata;
  
  //printf("_SpMV_A\n");
  
  //printf("_Residual_A11X1_A12X2\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_AuResidual(&F,V_X1);CHKERRQ(ierr);
  ierr = generic_facet_action(&F, &F.u, dmu, dmu,ufield, dmp,pfield, Yu);CHKERRQ(ierr);
  
  //printf("_Residual_A21X1\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_AuResidual(&F,V_X2);CHKERRQ(ierr);
  ierr = generic_facet_action(&F, &F.p, dmu, dmu,ufield, dmp,pfield, Yp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode _SetType_NITSCHE_GENERAL_SLIP(SurfaceConstraint sc)
{
  SCContextDemo  *ctx;
  PetscErrorCode ierr;
  
  /* set methods */
  sc->ops.setup   = NULL; /* always null */
  sc->ops.destroy = _destroy_demo;
  
  sc->ops.residual_F  = NULL; /* always null */
  sc->ops.residual_Fu = sc_residual_F1;
  sc->ops.residual_Fp = sc_residual_F2;
  
  sc->ops.action_A   = sc_spmv_A;
  sc->ops.action_Auu = sc_spmv_A11;
  sc->ops.action_Aup = sc_spmv_A12;
  sc->ops.action_Apu = sc_spmv_A21;
  
  sc->ops.asmb_Auu = sc_asmb_A11;
  sc->ops.asmb_Aup = sc_asmb_A12;
  sc->ops.asmb_Apu = sc_asmb_A21;
  
  sc->ops.diag_Auu = sc_asmbdiag_A11;
  
  /* allocate implementation data */
  ierr = PetscMalloc1(1,&ctx);CHKERRQ(ierr);
  ctx->setup = PETSC_TRUE;
  sc->data = (void*)ctx;
  ctx->penalty = 20.0;
  
  /* insert properties into quadrature bucket */
  DataBucketRegister_double(sc->properties_db,"hat_n" ,3);
  DataBucketRegister_double(sc->properties_db,"hat_t1",3);
  DataBucketRegister_double(sc->properties_db,"hat_t2",3);
  DataBucketRegister_double(sc->properties_db,"tau_S" ,6);
  DataBucketRegister_double(sc->properties_db,"H"     ,6);
  
  DataBucketFinalize(sc->properties_db);
  PetscFunctionReturn(0);
}

PetscErrorCode user_nitsche_general_slip_set_init_values(Facet F,
                                                     const PetscReal qp_coor[],
                                                     PetscReal hat_n[],
                                                     PetscReal hat_t1[],
                                                     PetscReal tauS[],
                                                     PetscReal H[],
                                                     void *data)
{
  PetscErrorCode ierr;
  ierr = PetscMemzero(hat_n, sizeof(double)*3);CHKERRQ(ierr);
  ierr = PetscMemzero(hat_t1,sizeof(double)*3);CHKERRQ(ierr);
  ierr = PetscMemzero(tauS,  sizeof(double)*6);CHKERRQ(ierr);
  ierr = PetscMemzero(H,     sizeof(double)*6);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// c = a x b
static PetscErrorCode _vec_cross(PetscReal a[],PetscReal b[],PetscReal c[])
{
  c[0] =   a[1]*b[2] - a[2]*b[1];
  c[1] = -(a[0]*b[2] - a[2]*b[0]);
  c[2] =   a[0]*b[1] - a[1]*b[0];
  PetscFunctionReturn(0);
}

PetscErrorCode _resize_facet_quadrature_data(SurfaceConstraint sc);

PetscErrorCode SurfaceConstraintSetValues_NITSCHE_GENERAL_SLIP(SurfaceConstraint sc,
                                                              SurfCSetValuesNitscheGeneralSlip set,
                                                              void *data)
{
  PetscInt       e,facet_index,cell_side,cell_index,q,qp_offset;
  Facet          cell_facet;
  PetscReal      qp_coor[3],nhat[3],that1[3],that2[3],tauS[6],H[6];
  PetscReal      *nhat_qp,*that1_qp,*that2_qp,*tauS_qp,*H_qp;
  double         Ni[27];
  const PetscInt *elnidx;
  PetscInt       nel,nen;
  double         elcoords[3*Q2_NODES_PER_EL_3D];
  PetscErrorCode ierr;
  
  if (sc->type != SC_NITSCHE_GENERAL_SLIP) {
    PetscPrintf(PetscObjectComm((PetscObject)sc->dm),"[ignoring] SurfaceConstraintSetValues_NITSCHE_GENERAL_SLIP() called with different type on object with name \"%s\"\n",sc->name);
    PetscFunctionReturn(0);
  }
  
  if (!sc->dm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Missing sc->dm. Must call SurfaceConstraintSetDM() first");
  if (!sc->quadrature) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Missing sc->surfQ. Must call SurfaceConstraintSetQuadrature() first");
  if (!sc->facets->set_values_called) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Facets have not been selected");
  
  /* resize qp data */
  ierr = _resize_facet_quadrature_data(sc);CHKERRQ(ierr);
  
  DataBucketGetEntriesdByName(sc->properties_db,"hat_n", (void**)&nhat_qp);
  DataBucketGetEntriesdByName(sc->properties_db,"hat_t1",(void**)&that1_qp);
  DataBucketGetEntriesdByName(sc->properties_db,"hat_t2",(void**)&that2_qp);
  DataBucketGetEntriesdByName(sc->properties_db,"tau_S", (void**)&tauS_qp);
  DataBucketGetEntriesdByName(sc->properties_db,"H",     (void**)&H_qp);
  
  ierr = MeshFacetInfoGetCoords(sc->fi);CHKERRQ(ierr);
  ierr = FacetCreate(&cell_facet);CHKERRQ(ierr);
  ierr = DMDAGetElements_pTatinQ2P1(sc->fi->dm,&nel,&nen,&elnidx);CHKERRQ(ierr);
  
  for (e=0; e<sc->facets->n_entities; e++) {
    facet_index = sc->facets->local_index[e]; /* facet local index */
    cell_side  = sc->fi->facet_label[facet_index]; /* side label */
    cell_index = sc->fi->facet_cell_index[facet_index];
    
    ierr = FacetPack(cell_facet, facet_index, sc->fi);CHKERRQ(ierr);
    
    ierr = DMDAGetElementCoordinatesQ2_3D(elcoords,(PetscInt*)&elnidx[nen*cell_index],(PetscReal*)sc->fi->_mesh_coor);CHKERRQ(ierr);
    
    //qp_offset = sc->nqp_facet * facet_index; /* offset into entire domain qp list */
    qp_offset = sc->nqp_facet * e; /* offset into facet qp list */
    
    for (q=0; q<sc->nqp_facet; q++) {
      
      {
        PetscInt d,k;
        
        for (d=0; d<3; d++) { qp_coor[d] = 0.0; }
        sc->fi->element->basis_NI_3D(&sc->quadrature->gp3[cell_side][q],Ni);
        for (k=0; k<sc->fi->element->n_nodes_3D; k++) {
          for (d=0; d<3; d++) {
            qp_coor[d] += Ni[k] * elcoords[3*k+d];
          }
        }
      }
      
      ierr = PetscMemzero(nhat, sizeof(double)*3);CHKERRQ(ierr);
      ierr = PetscMemzero(that1,sizeof(double)*3);CHKERRQ(ierr);
      ierr = PetscMemzero(that2,sizeof(double)*3);CHKERRQ(ierr);
      ierr = PetscMemzero(tauS, sizeof(double)*6);CHKERRQ(ierr);
      ierr = PetscMemzero(H,    sizeof(double)*6);CHKERRQ(ierr);
      
      ierr = set(cell_facet, qp_coor, nhat, that1, tauS, H, data);CHKERRQ(ierr);
      {
        int d;
        double nrm_n=0.0,nrm_t=0.0;
        for (d=0; d<3; d++) { nrm_n += nhat[d]*nhat[d]; nrm_t += that1[d]*that1[d]; }
        for (d=0; d<3; d++) { nhat[d] = nhat[d]/sqrt(nrm_n);  that1[d] = that1[d]/sqrt(nrm_t); }
      }
      
      ierr = PetscMemcpy(&nhat_qp[3*(qp_offset+q)], nhat, sizeof(PetscReal)*3);CHKERRQ(ierr);

      ierr = PetscMemcpy(&that1_qp[3*(qp_offset+q)],that1,sizeof(PetscReal)*3);CHKERRQ(ierr);
      
      ierr = _vec_cross(nhat, that1, that2);
      ierr = PetscMemcpy(&that2_qp[3*(qp_offset+q)],that2,sizeof(PetscReal)*3);CHKERRQ(ierr);

      ierr = PetscMemcpy(&tauS_qp[6*(qp_offset+q)], tauS, sizeof(PetscReal)*6);CHKERRQ(ierr);

      ierr = PetscMemcpy(&H_qp[6*(qp_offset+q)],    H,    sizeof(PetscReal)*6);CHKERRQ(ierr);
    }
  }
  
  ierr = FacetDestroy(&cell_facet);CHKERRQ(ierr);
  ierr = MeshFacetInfoRestoreCoords(sc->fi);CHKERRQ(ierr);
  
  DataBucketRestoreEntriesdByName(sc->properties_db,"hat_n", (void**)&nhat_qp);
  DataBucketRestoreEntriesdByName(sc->properties_db,"hat_t1",(void**)&that1_qp);
  DataBucketRestoreEntriesdByName(sc->properties_db,"hat_t2",(void**)&that2_qp);
  DataBucketRestoreEntriesdByName(sc->properties_db,"tau_S", (void**)&tauS_qp);
  DataBucketRestoreEntriesdByName(sc->properties_db,"H",     (void**)&H_qp);
  PetscFunctionReturn(0);
}

PetscErrorCode SurfaceConstraintNitscheGeneralSlip_SetPenalty(SurfaceConstraint sc,PetscReal penalty)
{
  SCContextDemo   *scdata = NULL;
  if (sc->type != SC_NITSCHE_GENERAL_SLIP) PetscFunctionReturn(0);
  scdata = (SCContextDemo*)sc->data;
  scdata->penalty = penalty;
  PetscFunctionReturn(0);
}
