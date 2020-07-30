
#ifndef __fvda_private_h__
#define __fvda_private_h__

void EvaluateBasis_Q1_1D(const PetscReal xi[], PetscReal N[]);
void EvaluateBasisDerivative_Q1_1D(const PetscReal xi[], PetscReal dN[][DACELL1D_Q1_SIZE]);
void EvaluateBasis_Q1_2D(const PetscReal xi[], PetscReal N[]);
void EvaluateBasisDerivative_Q1_2D(const PetscReal xi[], PetscReal dN[][DACELL2D_Q1_SIZE]);
void EvaluateBasis_Q1_3D(const PetscReal xi[], PetscReal N[]);

PetscErrorCode DACellGeometry2d_GetFaceIndices(DM dm,DACellFace face,PetscInt fidx[]);
PetscErrorCode DACellGeometry3d_GetFaceIndices(DM dm,DACellFace face,PetscInt fidx[]);

#endif
