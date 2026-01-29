
#ifndef __private_ptatin3d_energyfv_impl_h__
#define __private_ptatin3d_energyfv_impl_h__

#include <petsc.h>
#include <petscvec.h>
#include <petscsnes.h>
#include <petscdm.h>
#include <fvda.h>

struct _p_PhysCompEnergyFV {
  PetscReal               time,dt;
  PetscInt                mi_parent[3],nsubdivision[3];
  PetscInt                npoints_macro;
  PetscReal               *xi_macro,**basis_macro;
  DM                      dmv;
  Vec                     velocity; /* interpolate velocity from the Q2 mesh */
  /* Temporal history */
  Vec                     Told; /* previous temperature solution vector */
  Vec                     Xold; /* <ALE> previous coordinate vector */
  FVDA                    fv; /* finite volume context */
  SNES                    snes; /* solver context for the discrete energy equation */
  Vec                     T,F,G; /* T solution and residual */
  Mat                     J;
};

#endif

