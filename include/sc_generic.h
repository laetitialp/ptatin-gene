
#ifndef __ptatin3d_sc_generic_h__
#define __ptatin3d_sc_generic_h__

#include <petsc.h>
#include <surface_constraint.h>


typedef enum { FORM_UNINIT=0, FORM_ASSEMBLE, FORM_ASSEMBLE_DIAG, FORM_SPMV, FORM_RESIDUAL } FormType;


typedef enum {
  BASIS_Q1_1=0, BASIS_Q1_3,
  BASIS_Q2_1, BASIS_Q2_3,
  BASIS_P0_1,
  BASIS_P1_1 } FEBasisType;


typedef struct {
  FEBasisType basis;
  PetscInt    nbasis;
  PetscInt    dim;
  PetscReal   *W, *Wx, *Wy, *Wz; /* basis and derivative */
} FunctionSpace;


typedef struct _p_StokesForm StokesForm;

struct _p_StokesForm {
  FormType          type;
  SurfaceConstraint sc;
  FunctionSpace     u,p,*X[2];
  /* data filled by form setup */
  void           *data;
  PetscErrorCode (*access)(StokesForm*);
  PetscErrorCode (*restore)(StokesForm*);
  PetscErrorCode (*apply)(StokesForm*,PetscReal*,PetscReal*);
  /* data filled by generic function */
  PetscInt  nqp;
  PetscInt  cell_i,facet_sc_i,facet_i,point_i;
  PetscReal *x_elfield;
  PetscReal *u_elfield_0,*u_elfield_1,*u_elfield_2;
  PetscReal *p_elfield_0;
  FunctionSpace *test,*trial;
};

PetscErrorCode FunctionSpaceSet_VelocityQ1(FunctionSpace *s);
PetscErrorCode FunctionSpaceSet_VelocityQ2(FunctionSpace *s);
PetscErrorCode FunctionSpaceSet_PressureP0(FunctionSpace *s);
PetscErrorCode FunctionSpaceSet_PressureP1(FunctionSpace *s);
PetscErrorCode FunctionSpaceSet_Q2P1(FunctionSpace *u,FunctionSpace *p);

PetscErrorCode StokeFormSetFunctionSpace_Q2P1(StokesForm *f);
PetscErrorCode StokesFormSetType(StokesForm *f,FormType type);
PetscErrorCode StokesFormInit(StokesForm *f,FormType type,SurfaceConstraint sc);
PetscErrorCode StokesFormFlush(StokesForm *f);

PetscErrorCode generic_facet_action(StokesForm *form,
                                           FunctionSpace *test_function,
                                           DM dm, /* always be q2 mesh */
                                           DM dau,const PetscScalar ufield[],
                                           DM dap,const PetscScalar pfield[],
                                           PetscScalar Y[]);

PetscErrorCode generic_facet_assemble(StokesForm *form,
                                      FunctionSpace *test_function,
                                      FunctionSpace *trial_function,
                                      DM dm, /* always be q2 mesh */
                                      DM dau,
                                      DM dap,
                                      Mat A);

PetscErrorCode generic_facet_assemble_diagonal(StokesForm *form,
                                               FunctionSpace *test_function,
                                               DM dm, /* always be q2 mesh */
                                               DM dmX,
                                               Vec diagA);
#endif
