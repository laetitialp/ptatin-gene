#include "petsc.h"

#include "ptatin3d_defs.h"
#include "ptatin3d.h"
#include "private/ptatin_impl.h"
#include "dmda_element_q2p1.h"
#include "quadrature.h"
#include "dmda_checkpoint.h"
#include "element_type_Q2.h"
#include "data_bucket.h"
//#include "mesh_entity.h"
#include "QPntVolCoefStokes_def.h"
#include "gravity.h"

const char *GravityTypeNames[] = {
  "gravity_constant",
  "gravity_radial_constant",
  "gravity_radial_var",
  "gravity_arbitrary",
  "gravity_poisson", 
  NULL
};

PetscErrorCode (*create_types[])(Gravity) = {
  GravityConstantCreateCtx, 
  GravityRadialConstantCreateCtx,
  0
};

PetscErrorCode GravityCreate(Gravity *gravity)
{
  Gravity        grav;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(struct _p_Gravity),&grav);CHKERRQ(ierr);
  ierr = PetscMemzero(grav,sizeof(struct _p_Gravity));CHKERRQ(ierr);
  *gravity = grav;
  PetscFunctionReturn(0);
}

PetscErrorCode GravitySetType(Gravity gravity, GravityType gtype)
{
  int            k = 0;
  const char     *name;
  PetscBool      found;
  PetscErrorCode (*create)(Gravity)=NULL; // Initialize function pointer to NULL
  PetscErrorCode ierr;
  PetscFunctionBegin;
  
  /* Assign the type of gravity */
  gravity->gravity_type = gtype;

  /* Initialize the name to the first entry of GravityTypeNames */
  name = GravityTypeNames[k];
  while (name != NULL) {
    /* Check if name matches the type given */
    ierr = PetscStrcmp(name,GravityTypeNames[ (int)gtype ],&found);CHKERRQ(ierr);
    if (found) {
      /* Assign function to the corresponding type and leave */
      create = create_types[k];
      break;
    }
    k++;
    /* Assign name to the next entry of the array and restart */
    name = GravityTypeNames[k];
  }

  /* Check that a function was assigned */
  if (!create) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"GravityType did not match any known type"); }
  /* Call the Create() function of the corresponding type */
  ierr = create(gravity);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode GravityScale(Gravity gravity, PetscReal scaling_factor)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = gravity->scale(gravity,scaling_factor);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode QuadratureSetGravity(PhysCompStokes stokes, Gravity gravity)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = gravity->quadrature_set(stokes,gravity);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode QuadratureUpdateGravity(PhysCompStokes stokes, Gravity gravity)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = gravity->update(stokes,gravity);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode pTatinQuadratureSetGravity(pTatinCtx ptatin)
{
  PhysCompStokes stokes;
  Gravity        gravity;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = pTatinGetGravityCtx(ptatin,&gravity);CHKERRQ(ierr);
  ierr = pTatinGetStokesContext(ptatin,&stokes);CHKERRQ(ierr);
  ierr = QuadratureSetGravity(stokes,gravity);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode pTatinQuadratureUpdateGravity(pTatinCtx ptatin)
{
  PhysCompStokes stokes;
  Gravity        gravity;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = pTatinGetGravityCtx(ptatin,&gravity);CHKERRQ(ierr);
  ierr = pTatinGetStokesContext(ptatin,&stokes);CHKERRQ(ierr);
  ierr = QuadratureUpdateGravity(stokes,gravity);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode GravityGetPointWiseVector(Gravity gravity, PetscInt eidx, PetscReal global_coords[], PetscReal local_coords[], PetscReal gvec[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = gravity->get_gvec(gravity,eidx,global_coords,local_coords,gvec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode pTatinGetGravityPointWiseVector(pTatinCtx ptatin, PetscInt eidx, PetscReal global_coords[], PetscReal local_coords[], PetscReal gvec[])
{
  Gravity        gravity;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = pTatinGetGravityCtx(ptatin,&gravity);CHKERRQ(ierr);
  ierr = GravityGetPointWiseVector(gravity,eidx,global_coords,local_coords,gvec);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode GravityDestroyCtx(Gravity *gravity)
{
  PetscFunctionBegin;
  if (!gravity)  { PetscFunctionReturn(0); }
  if (!*gravity) { PetscFunctionReturn(0); }
  (*gravity)->destroy(*gravity);
  *gravity = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode pTatinDestroyGravityCtx(pTatinCtx ptatin)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!ptatin->gravity_ctx) { PetscFunctionReturn(0); }
  ierr = GravityDestroyCtx(&ptatin->gravity_ctx);CHKERRQ(ierr);
  ptatin->gravity_ctx = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode pTatinContextValid_Gravity(pTatinCtx ptatin, PetscBool *exists)
{
  PetscFunctionBegin;
  *exists = PETSC_FALSE;
  if (ptatin->gravity_ctx) {
    *exists = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode pTatinGetGravityCtx(pTatinCtx ptatin, Gravity *ctx)
{
  Gravity gravity;

  PetscFunctionBegin;
  gravity = ptatin->gravity_ctx;
  *ctx = gravity;
  PetscFunctionReturn(0);
}

PetscErrorCode pTatinCreateGravity(pTatinCtx ptatin, GravityType gtype)
{
  Gravity        gravity;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  /* Create Gravity data structure */
  ierr = GravityCreate(&gravity);CHKERRQ(ierr);
  ierr = GravitySetType(gravity,gtype);CHKERRQ(ierr);
  
  ptatin->gravity_ctx = gravity;

  PetscFunctionReturn(0);
}

PetscErrorCode QuadratureSetBodyForcesOnPoint(QPntVolCoefStokes *cell_gausspoints, PetscInt qp_idx, PetscReal gravity_vector[])
{
  PetscInt d;
  double   density,Fu[3];
  PetscFunctionBegin;
  /* Get density on quadrature point */
  QPntVolCoefStokesGetField_rho_effective(&cell_gausspoints[qp_idx],&density);
  /* Compute rho*g on the quadrature point and attach it to the body force vector */
  for (d=0; d<NSD; d++) {
    Fu[d] = density * gravity_vector[d];
  }
  QPntVolCoefStokesSetField_momentum_rhs(&cell_gausspoints[qp_idx],Fu);
  PetscFunctionReturn(0);
}