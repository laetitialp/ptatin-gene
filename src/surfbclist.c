
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

PetscErrorCode SurfBCListInsertConstraint(SurfBCList sl, SurfaceConstraint sc, PetscBool *inserted)
{
  PetscBool found;
  PetscInt index;
  PetscErrorCode ierr;
  
  if (inserted) { *inserted = PETSC_FALSE; }
  if (!sc) { PetscFunctionReturn(0); }
  
  if (!sl->mfi) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"SurfBCList requires a non-NULL MeshFacetInfo object");
  if (!sl->surfQ) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"SurfBCList requires a non-NULL SurfaceQuadrature object");
  
  found = _find_name_and_index(sl,sc->name,&index);
  if (found) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"SurfaceConstraint with name %s already registered",sc->name);
  
  /* append to list */
  ierr = PetscRealloc(sizeof(SurfaceConstraint)*(sl->sc_nreg + 1),&sl->sc_list);CHKERRQ(ierr);
  sl->sc_list[sl->sc_nreg] = sc;
  sl->sc_nreg++;
  if (inserted) { *inserted = PETSC_TRUE; }  
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


PetscErrorCode SurfBCList_EvaluateFuFp(SurfBCList surfbc,
                                       DM dau,const PetscScalar ufield[],
                                       DM dap,const PetscScalar pfield[],
                                       PetscScalar Ru[],PetscScalar Rp[])
{
  PetscErrorCode ierr;
  PetscInt k;
  
  if (!surfbc) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"SurfBCList is NULL. This indiates a setup error has occurred");
  for (k=0; k<surfbc->sc_nreg; k++) {
    ierr = SurfaceConstraintOps_EvaluateFu(surfbc->sc_list[k],dau,ufield,dap,pfield,Ru, PETSC_FALSE);CHKERRQ(ierr);
    ierr = SurfaceConstraintOps_EvaluateFp(surfbc->sc_list[k],dau,ufield,dap,pfield,Rp, PETSC_FALSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SurfBCList_ActionA(SurfBCList surfbc,
                                       DM dau,const PetscScalar ufield[],
                                       DM dap,const PetscScalar pfield[],
                                       PetscScalar Ru[],PetscScalar Rp[])
{
  PetscErrorCode ierr;
  PetscInt k;
  
  if (!surfbc) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"SurfBCList is NULL. This indiates a setup error has occurred");
  for (k=0; k<surfbc->sc_nreg; k++) {
    ierr = SurfaceConstraintOps_ActionA(surfbc->sc_list[k],dau,ufield,dap,pfield,Ru,Rp, PETSC_FALSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SurfBCList_ActionA11(SurfBCList surfbc,
                                       DM dau,const PetscScalar ufield[],
                                       PetscScalar Yu[])
{
  PetscErrorCode ierr;
  PetscInt k;
  
  if (!surfbc) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"SurfBCList is NULL. This indiates a setup error has occurred");
  for (k=0; k<surfbc->sc_nreg; k++) {
    ierr = SurfaceConstraintOps_ActionA11(surfbc->sc_list[k],dau,ufield,Yu, PETSC_FALSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SurfBCList_ActionA12(SurfBCList surfbc,
                                       DM dau,
                                       DM dap,const PetscScalar pfield[],PetscScalar Yu[])
{
  PetscErrorCode ierr;
  PetscInt k;
  
  if (!surfbc) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"SurfBCList is NULL. This indiates a setup error has occurred");
  for (k=0; k<surfbc->sc_nreg; k++) {
    ierr = SurfaceConstraintOps_ActionA12(surfbc->sc_list[k],dau,dap,pfield,Yu, PETSC_FALSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SurfBCList_ActionA21(SurfBCList surfbc,
                                       DM dau,const PetscScalar ufield[],
                                       DM dap,
                                       PetscScalar Rp[])
{
  PetscErrorCode ierr;
  PetscInt k;
  
  if (!surfbc) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"SurfBCList is NULL. This indiates a setup error has occurred");
  for (k=0; k<surfbc->sc_nreg; k++) {
    ierr = SurfaceConstraintOps_ActionA21(surfbc->sc_list[k],dau,ufield,dap,Rp, PETSC_FALSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}




PetscErrorCode SurfBCList_AssembleAij(SurfBCList surfbc,
                                    PetscInt ij[],
                                    DM dau,
                                    DM dap,
                                    Mat A)
{
  PetscErrorCode ierr;
  PetscInt k;
  
  if (!surfbc) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"SurfBCList is NULL. This indiates a setup error has occurred");
  for (k=0; k<surfbc->sc_nreg; k++) {
    if (ij[0] == ij[1] == 0) {
      ierr = SurfaceConstraintOps_AssembleA11(surfbc->sc_list[k],dau,A, PETSC_FALSE);CHKERRQ(ierr);
    } else if (ij[0] == 0 && ij[1] == 1) {
      ierr = SurfaceConstraintOps_AssembleA12(surfbc->sc_list[k],dau,dap,A, PETSC_FALSE);CHKERRQ(ierr);
    } else if (ij[0] == 1 && ij[1] == 0) {
      ierr = SurfaceConstraintOps_AssembleA21(surfbc->sc_list[k],dau,dap,A, PETSC_FALSE);CHKERRQ(ierr);
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only assemble (0,0), (0,1) or (1,0)");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SurfBCList_AssembleDiagA11(SurfBCList surfbc,
                                      DM dau,
                                      PetscScalar Ae[])
{
  PetscErrorCode ierr;
  PetscInt k;
  
  if (!surfbc) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"SurfBCList is NULL. This indiates a setup error has occurred");
  for (k=0; k<surfbc->sc_nreg; k++) {
    ierr = SurfaceConstraintOps_AssembleDiagA11(surfbc->sc_list[k],dau,Ae, PETSC_FALSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}





