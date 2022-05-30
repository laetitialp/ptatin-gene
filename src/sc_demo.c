
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


typedef enum { V_X1=0, V_X2 } StokesSubVec;


typedef enum { M_A11=0, M_A12, M_A21, M_A22 } StokesSubMat;


typedef struct {
  PetscBool setup;
} SCContextDemo;


typedef struct {
  QPntSurfCoefStokes *boundary_qp;
} FormContextDemo;


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
  
  printf("Form[Demo]: access()\n");
  sc = form->sc;
  scdata = (SCContextDemo*)sc->data;
  
  formdata = (FormContextDemo*)form->data;
  
  printf("data->setup %d\n",scdata->setup);
  
  boundary_q = sc->quadrature;
  ierr = SurfaceQuadratureGetAllCellData_Stokes(boundary_q,&formdata->boundary_qp);CHKERRQ(ierr);
  
  //DataBucketGetEntriesdByName(sc->properties_db,"traction",(void**)&traction_qp);
  
  PetscFunctionReturn(0);
}

/* surface constraint implementation specific */
static PetscErrorCode _form_restore_demo(StokesForm *form)
{
  PetscErrorCode    ierr;
  SurfaceConstraint sc;
  FormContextDemo   *formdata = NULL;
  
  printf("Form[Demo]: restore()\n");
  formdata = (FormContextDemo*)form->data;
  
  sc = form->sc;
  
  //DataBucketRestoreEntriesdByName(sc->properties_db,"traction",(void**)&traction_qp);
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
  PetscErrorCode     ierr;
  SCContextDemo      *scdata = NULL;
  FormContextDemo    *formdata = NULL;
  PetscInt           sq_index,sc_index;
  PetscInt           bs_1=1,bs_2=1; /* block size of data 1 and data 2 */
  QPntSurfCoefStokes *qx = NULL;
  double             *normal,eta;
  FunctionSpace      *test = form->test;
  
  //printf("  Form[Demo]: apply_F1()\n");
  
  scdata   = (SCContextDemo*)form->sc->data;
  formdata = (FormContextDemo*)form->data;
  sc_index = form->facet_sc_i * bs_1 * form->nqp  + bs_1 * form->point_i;
  sq_index = form->facet_i * bs_2 * form->nqp  + bs_2 * form->point_i;
  
  qx = &formdata->boundary_qp[sq_index];
  normal = qx->normal;
  eta    = qx->eta;
  //qy = &formdata->constraint_qp[sc_index];
  
  /*
   auto-generated-func(
     test->W, test->Wx,test->Wy,test->Wz,
     form->X[0]->W, form->X[0]->Wx,form->X[0]->Wy,form->X[0]->Wz,
     form->X[1]->W, form->X[1]->Wx,form->X[1]->Wy,form->X[1]->Wz,
     form->u_elfield_0, form->u_elfield_1, form->u_elfield_2,
     form->p_elfield_0,
     qx->normal, qx->eta, formdata->xxx, scdata->yyy,
     *ds,F);
   
   for (q=0; q<form->nqp; q++) {
     PetscInt basis_offset_t = test->nbasis * q;
     PetscReal *wt = &test->W[basis_offset];
     PetscInt basis_offset_0 = form->X[0]->nbasis * q;
     PetscReal *w0 = &formX[0]->W[basis_offset_0];
     PetscInt basis_offset_1 = form->X[1]->nbasis * q;
     PetscReal *w1 = &formX[1]->W[basis_offset_1];
   
     auto-generated-func(
       wt, wtx,wty,wtz,
       w0, w0x,w0y,w0z,
       w1, w1x,w1y,w1z, ...
   
   }
   */

  
  PetscFunctionReturn(0);
}

static PetscErrorCode _form_residual_F2(StokesForm *form,PetscReal ds[],PetscReal F[])
{
  PetscFunctionReturn(0);
  }

/* point-wise kernel configuration */
PetscErrorCode StoksFormConfigureAction_Residual(StokesForm *form,StokesSubVec op)
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
  
  printf("_Residual_F1\n");
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
  
  printf("_Residual_F2\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_Residual(&F,V_X2);CHKERRQ(ierr);
  ierr = generic_facet_action(&F, &F.u, dmu, dmu,ufield, dmp,pfield, R);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* action (spmv) */
/* point-wise kernels */
static PetscErrorCode _form_spmv_A11(StokesForm *form,PetscReal ds[],PetscReal F[])
{
  PetscFunctionReturn(0);
}

static PetscErrorCode _form_spmv_A12(StokesForm *form,PetscReal ds[],PetscReal F[])
{
  PetscFunctionReturn(0);
}

static PetscErrorCode _form_spmv_A21(StokesForm *form,PetscReal ds[],PetscReal F[])
{
  PetscFunctionReturn(0);
}

/* point-wise kernel configuration */
PetscErrorCode StoksFormConfigureAction_SpMV(StokesForm *form,StokesSubMat op)
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
  
  printf("_SpMV_A11\n");
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
  
  printf("_SpMV_A12\n");
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
  
  printf("_SpMV_A21\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_SpMV(&F,M_A21);CHKERRQ(ierr);
  ierr = generic_facet_action(&F, &F.u, dmu, dmu,ufield, dmp,NULL, Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



/* assemble */
/* point-wise kernels */
static PetscErrorCode _form_asmb_A11(StokesForm *form,PetscReal ds[],PetscReal A[])
{
  PetscFunctionReturn(0);
}

static PetscErrorCode _form_asmb_A12(StokesForm *form,PetscReal ds[],PetscReal A[])
{
  PetscFunctionReturn(0);
}

static PetscErrorCode _form_asmb_A21(StokesForm *form,PetscReal ds[],PetscReal A[])
{
  PetscFunctionReturn(0);
}

/* point-wise kernel configuration */
PetscErrorCode StoksFormConfigureAction_Assemble(StokesForm *form,StokesSubMat op)
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
  
  printf("_Assemble_A11\n");
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
  
  printf("_Assemble_A12\n");
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
  
  printf("_Assemble_A21\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_Assemble(&F,M_A21);CHKERRQ(ierr);
  ierr = generic_facet_assemble(&F, &F.p,&F.u, dmu, dmu, dmp, A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



/* assemble diagonal */
/* point-wise kernels */
static PetscErrorCode _form_asmbdiag_A11(StokesForm *form,PetscReal ds[],PetscReal A[])
{
  PetscFunctionReturn(0);
}

/* point-wise kernel configuration */
PetscErrorCode StoksFormConfigureAction_AssembleDiagonal(StokesForm *form,StokesSubMat op)
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
static PetscErrorCode sc_asmbdiag_A11(SurfaceConstraint sc, DM dmu, PetscScalar A[])
{
  PetscErrorCode  ierr;
  StokesForm      F;
  FormContextDemo formdata;
  
  printf("_AssembleDiagonal_A11\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_AssembleDiagonal(&F,M_A11);CHKERRQ(ierr);
  //ierr = generic_facet_assemble_diag(&F, &F.u, dmu, A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}




PetscErrorCode _SetType_DEMO(SurfaceConstraint sc)
{
  SCContextDemo  *ctx;
  PetscErrorCode ierr;
  
  /* set methods */
  sc->ops.setup   = NULL; /* always null */
  sc->ops.destroy = _destroy_demo;
  
  sc->ops.residual_F   = NULL; /* always null */
  sc->ops.residual_Fu = sc_residual_F1;
  sc->ops.residual_Fp = sc_residual_F2;
  
  sc->ops.action_A    = NULL; /* todo */
  sc->ops.action_Auu  = sc_spmv_A11;
  sc->ops.action_Aup  = sc_spmv_A12;
  sc->ops.action_Apu  = sc_spmv_A21;
  
  sc->ops.asmb_A   = NULL; /* always null */
  sc->ops.asmb_Auu = sc_asmb_A11;
  sc->ops.asmb_Aup = sc_asmb_A12;
  sc->ops.asmb_Apu = sc_asmb_A21;
  
  sc->ops.diag_A   = NULL; /* always null */
  sc->ops.diag_Auu = sc_asmbdiag_A11;
  
  /* allocate implementation data */
  ierr = PetscMalloc1(1,&ctx);CHKERRQ(ierr);
  ctx->setup = PETSC_TRUE;
  sc->data = (void*)ctx;
  /* insert properties into quadrature bucket */
  DataBucketFinalize(sc->properties_db);
  //sc->type = SC_DEMO;
  PetscFunctionReturn(0);
}
