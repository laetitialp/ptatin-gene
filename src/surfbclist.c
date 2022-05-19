
#include <petsc.h>
#include <petscdm.h>
#include <ptatin3d.h>
#include <quadrature.h>
#include <mesh_entity.h>
#include <surface_constraint.h>
#include <surfbclist.h>

static PetscBool _find_name_and_index(SurfBCList sl, const char name[], PetscInt *_index)
{
  PetscInt k;
  PetscBool found = PETSC_FALSE;
  PetscErrorCode ierr;
  
  *_index = -1;
  for (k=0; k<sl->sc_nreg; k++) {
    ierr = PetscStrcmp(sl->sc_list[k]->name,name,&found);
    if (found) {
      *_index = k;
      break;
    }
  }
  return found;
}

PetscErrorCode SurfBCListDestroy(SurfBCList *_sl)
{
  PetscErrorCode ierr;
  SurfBCList sl;
  PetscInt k;
  
  if (!_sl) PetscFunctionReturn(0);
  sl = *_sl;
  if (!sl) PetscFunctionReturn(0);
  //sl->dm = dm;
  //sl->surfQ = surfQ;
  ierr = MeshFacetInfoDestroy(&sl->mfi);CHKERRQ(ierr);
  for (k=0; k<sl->sc_nreg; k++) {
    ierr = SurfaceConstraintDestroy(&sl->sc_list[k]);CHKERRQ(ierr);
  }
  ierr = PetscFree(sl->sc_list);CHKERRQ(ierr);
  ierr = PetscFree(sl);CHKERRQ(ierr);
  *_sl = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode SurfBCListCreate(DM dm, SurfaceQuadrature surfQ, MeshFacetInfo mfi, SurfBCList *_sl)
{
  SurfBCList sl;
  PetscErrorCode ierr;
  
  ierr = PetscMalloc(sizeof(struct _p_SurfBCList),&sl);CHKERRQ(ierr);
  ierr = PetscMemzero(sl,sizeof(struct _p_SurfBCList));CHKERRQ(ierr);
  sl->dm = dm;
  sl->surfQ = surfQ;
  sl->mfi = mfi;
  ierr = MeshFacetInfoIncrementRef(mfi);CHKERRQ(ierr);
  
  ierr = PetscMalloc1(1,&sl->sc_list);CHKERRQ(ierr);
  sl->sc_list[0] = NULL;
  sl->sc_nreg = 0;

  *_sl = sl;
  
  PetscFunctionReturn(0);
}

PetscErrorCode SurfBCListAddConstraint(SurfBCList sl, const char name[], SurfaceConstraint *_sc)
{
  PetscBool found;
  PetscInt index;
  SurfaceConstraint sc;
  PetscErrorCode ierr;
  
  if (!sl->mfi) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"SurfBCList requires a non-NULL MeshFacetInfo object");
  if (!sl->surfQ) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"SurfBCList requires a non-NULL SurfaceQuadrature object");
  
  found = _find_name_and_index(sl,name,&index);
  if (found) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"SurfaceConstraint with name %s already registered",name);
  
  ierr = SurfaceConstraintCreateWithFacetInfo(sl->mfi,&sc);CHKERRQ(ierr);
  ierr = SurfaceConstraintSetQuadrature(sc,sl->surfQ);CHKERRQ(ierr);
  ierr = SurfaceConstraintSetName(sc,name);CHKERRQ(ierr);
  
  /* append to list */
  ierr = PetscRealloc(sizeof(SurfaceConstraint)*(sl->sc_nreg + 1),&sl->sc_list);CHKERRQ(ierr);
  sl->sc_list[sl->sc_nreg] = sc;
  sl->sc_nreg++;

  if (_sc) {
    *_sc = sc;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SurfBCListGetConstraint(SurfBCList sl, const char name[], SurfaceConstraint *_sc)
{
  PetscBool found;
  PetscInt index;
  
  *_sc = NULL;
  found = _find_name_and_index(sl,name,&index);
  //if (!found) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"SurfaceConstraint with name %s has not been registered",name);
  if (found) { *_sc = sl->sc_list[index]; }
  PetscFunctionReturn(0);
}

PetscErrorCode SurfBCListEvaluate(SurfBCList sl)
{
  
  PetscFunctionReturn(0);
}


PetscErrorCode SurfBCListViewer(SurfBCList sl,PetscViewer v)
{
  PetscErrorCode ierr;
  PetscInt k;
  PetscViewerASCIIPrintf(v,"SurfBCList:\n");
  PetscViewerASCIIPushTab(v);
  PetscViewerASCIIPrintf(v,"n_constraints: %D\n",sl->sc_nreg);
  for (k=0; k<sl->sc_nreg; k++) {
    PetscViewerASCIIPrintf(v,"constraint[%D]\n",k);
    PetscViewerASCIIPushTab(v);
    ierr = SurfaceConstraintViewer(sl->sc_list[k],v);CHKERRQ(ierr);
    PetscViewerASCIIPopTab(v);
  }
  PetscViewerASCIIPopTab(v);
  PetscFunctionReturn(0);
}

