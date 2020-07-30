
#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <fvda_impl.h>
#include <fvda.h>
#include <fvda_private.h>
#include <fvda_utils.h>

typedef struct { PetscInt i; PetscInt j; PetscInt k; } CellTuple;



PetscErrorCode setup_lhs(PetscInt target,PetscInt nneigh,const PetscInt neigh[],const PetscReal cell_x[],PetscReal A[3][3],PetscReal iA[3][3])
{
  const PetscReal *cell_target_x;
  PetscReal t4, t6, t8, t10, t12, t14, t17;
  PetscInt  k;
  
  cell_target_x = &cell_x[3*target];
  
  A[0][0] = A[0][1] = A[0][2] = 0.0;
  A[1][0] = A[1][1] = A[1][2] = 0.0;
  A[2][0] = A[2][1] = A[2][2] = 0.0;

  for (k=0; k<nneigh; k++) {
    PetscReal      dx,dy,dz;
    const PetscInt idx = neigh[k];
    
    dx = cell_x[3*idx+0] - cell_target_x[0];
    dy = cell_x[3*idx+1] - cell_target_x[1];
    dz = cell_x[3*idx+2] - cell_target_x[2];
    
    A[0][0] += dx * dx;
    A[1][1] += dy * dy;
    A[2][2] += dz * dz;

    A[0][1] += dx * dy;
    A[0][2] += dx * dz;

    A[1][2] += dy * dz;
  }

  A[1][0] = A[0][1];
  A[2][0] = A[0][2];
  A[2][1] = A[1][2];
  
  /* get the inervse */
  t4  = A[2][0] * A[0][1];
  t6  = A[2][0] * A[0][2];
  t8  = A[1][0] * A[0][1];
  t10 = A[1][0] * A[0][2];
  t12 = A[0][0] * A[1][1];
  t14 = A[0][0] * A[1][2]; // 6
  t17 = 0.1e1 / (t4 * A[1][2] - t6 * A[1][1] - t8 * A[2][2] + t10 * A[2][1] + t12 * A[2][2] - t14 * A[2][1]);  // 12
  
  iA[0][0] = (A[1][1] * A[2][2] - A[1][2] * A[2][1]) * t17;  // 4
  iA[0][1] = -(A[0][1] * A[2][2] - A[0][2] * A[2][1]) * t17; // 5
  iA[0][2] = (A[0][1] * A[1][2] - A[0][2] * A[1][1]) * t17;  // 4
  iA[1][0] = -(-A[2][0] * A[1][2] + A[1][0] * A[2][2]) * t17;// 6
  iA[1][1] = (-t6 + A[0][0] * A[2][2]) * t17;                // 4
  iA[1][2] = -(-t10 + t14) * t17;                            // 4
  iA[2][0] = (-A[2][0] * A[1][1] + A[1][0] * A[2][1]) * t17; // 5
  iA[2][1] = -(-t4 + A[0][0] * A[2][1]) * t17;               // 5
  iA[2][2] = (-t8 + t12) * t17;                              // 3
  
  PetscFunctionReturn(0);
}

PetscErrorCode setup_coeff(FVDA fv,PetscInt target,PetscInt nneigh,const PetscInt neigh[],const PetscReal cell_x[],const PetscReal Q[],PetscReal coeff[])
{
  const PetscReal      *cell_target_x;
  PetscInt       k,i;
  PetscReal      A[3][3],iA[3][3],rhs[3];
  PetscErrorCode ierr;
  ////
  PetscInt       s[3],w[3],cij;
  CellTuple      cell,cellglobal;
  
  ierr = DMDAGetGhostCorners(fv->dm_fv,&s[0],&s[1],&s[2],&w[0],&w[1],&w[2]);CHKERRQ(ierr);
  
  cell.k = target / (w[0]*w[1]);
  cij = target - cell.k * (w[0]*w[1]);
  cell.j = cij / w[0];
  cell.i = cij - cell.j * w[0];
  
  cellglobal.i = cell.i + s[0];
  cellglobal.j = cell.j + s[1];
  cellglobal.k = cell.k + s[2];
  ////
  
  cell_target_x = &cell_x[3*target];
  
  rhs[0] = rhs[1] = rhs[2] = 0;
  
  for (k=0; k<nneigh; k++) {
    PetscReal      dx,dy,dz,dQ;
    const PetscInt idx = neigh[k];
    
    dx = cell_x[3*idx+0] - cell_target_x[0];
    dy = cell_x[3*idx+1] - cell_target_x[1];
    dz = cell_x[3*idx+2] - cell_target_x[2];

    dQ = Q[idx] - Q[target];

    //printf("cell global %d %d %d (%+1.4e,%+1.4e,%+1.4e) : neighbour %+1.4e,%+1.4e,%+1.4e \n",cellglobal.i,cellglobal.j,cellglobal.k,cell_target_x[0],cell_target_x[1],cell_target_x[2],cell_x[3*idx+0],cell_x[3*idx+1],cell_x[3*idx+2]);
    
    rhs[0] += dQ * dx;
    rhs[1] += dQ * dy;
    rhs[2] += dQ * dz;
  }
  
  ierr = setup_lhs(target,nneigh,neigh,cell_x,A,iA);CHKERRQ(ierr);

  coeff[0] = coeff[1] = coeff[2] = 0;
  for (i=0; i<3; i++) {
    for (k=0; k<3; k++) {
      coeff[i] += iA[i][k] * rhs[k];
    }
  }
  
  PetscFunctionReturn(0);
}

PetscErrorCode FVDAReconstructP1Evaluate(FVDA fv,
                  PetscReal x[],
                  PetscInt target,
                  const PetscReal cell_target_x[],const PetscReal Q[],
                  PetscReal coeff[],
                  PetscReal Q_hr[])
{
  Q_hr[0] = coeff[0] * (x[0] - cell_target_x[0])
          + coeff[1] * (x[1] - cell_target_x[1])
          + coeff[2] * (x[2] - cell_target_x[2])
          + Q[target];
  
  //
  {
    PetscInt       s[3],w[3],cij;
    CellTuple      cell,cellglobal;
    PetscErrorCode ierr;
    
    ierr = DMDAGetGhostCorners(fv->dm_fv,&s[0],&s[1],&s[2],&w[0],&w[1],&w[2]);CHKERRQ(ierr);

    cell.k = target / (w[0]*w[1]);
    cij = target - cell.k * (w[0]*w[1]);
    cell.j = cij / w[0];
    cell.i = cij - cell.j * w[0];
    
    cellglobal.i = cell.i + s[0];
    cellglobal.j = cell.j + s[1];
    cellglobal.k = cell.k + s[2];
    //printf("cell global %d %d %d (%+1.4e,%+1.4e,%+1.4e): recon %+1.12e\n",cellglobal.i,cellglobal.j,cellglobal.k,cell_target_x[0],cell_target_x[1],cell_target_x[2],Q_hr[0]);
  }
  //
  PetscFunctionReturn(0);
}

PetscErrorCode FVDAGetReconstructionStencil_AtCell(FVDA fv,PetscInt cijk,PetscInt *nn,PetscInt neigh[])
{
  PetscInt       s[3],w[3],cij,ii,jj,kk;
  CellTuple      cell,cellglobal;
  //PetscBool      interior = PETSC_TRUE;
  DM             dm = fv->dm_fv;
  PetscErrorCode ierr;
  
  ierr = DMDAGetGhostCorners(dm,&s[0],&s[1],&s[2],&w[0],&w[1],&w[2]);CHKERRQ(ierr);
  
  /* convert cijk into i,j,k in local dm_fv space */
  cell.k = cijk / (w[0]*w[1]);
  cij = cijk - cell.k * (w[0]*w[1]);
  cell.j = cij / w[0];
  cell.i = cij - cell.j * w[0];
  
  cellglobal.i = cell.i + s[0];
  cellglobal.j = cell.j + s[1];
  cellglobal.k = cell.k + s[2];
  
  /*
  if (cellglobal.i == s[0] || cellglobal.i == (s[0]+w[0]-1)) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot perform reconstruction on borders of fv local space. Range i: [%D,%D] -> Found: i %D",s[0],s[0]+w[0]-1,cellglobal.i);
  if (cellglobal.j == s[1] || cellglobal.j == (s[1]+w[1]-1)) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot perform reconstruction on borders of fv local space. Range j: [%D,%D] -> Found: j %D",s[1],s[1]+w[1]-1,cellglobal.j);
  if (cellglobal.k == s[2] || cellglobal.k == (s[2]+w[2]-1)) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot perform reconstruction on borders of fv local space. Range k: [%D,%D] -> Found: k %D",s[2],s[2]+w[2]-1,cellglobal.k);
  */
  if (cellglobal.i == s[0] && s[0] != 0) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot perform reconstruction on borders of fv local space. Range i: [%D,%D] -> Found: i %D",s[0],s[0]+w[0]-1,cellglobal.i);
  if (cellglobal.i == (s[0]+w[0]-1) && fv->Mi[0] != (s[0]+w[0])) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot perform reconstruction on borders of fv local space. Range i: [%D,%D] -> Found: i %D",s[0],s[0]+w[0]-1,cellglobal.i);

  if (cellglobal.j == s[1] && s[1] != 0) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot perform reconstruction on borders of fv local space. Range j: [%D,%D] -> Found: j %D",s[1],s[1]+w[1]-1,cellglobal.j);
  if (cellglobal.j == (s[1]+w[1]-1) && fv->Mi[1] != (s[1]+w[1])) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot perform reconstruction on borders of fv local space. Range j: [%D,%D] -> Found: j %D",s[1],s[1]+w[1]-1,cellglobal.j);

  if (cellglobal.k == s[2] && s[2] != 0) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot perform reconstruction on borders of fv local space. Range k: [%D,%D] -> Found: k %D",s[2],s[2]+w[2]-1,cellglobal.k);
  if (cellglobal.k == (s[2]+w[2]-1) && fv->Mi[2] != (s[2]+w[2])) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot perform reconstruction on borders of fv local space. Range k: [%D,%D] -> Found: k %D",s[2],s[2]+w[2]-1,cellglobal.k);
  
  
  *nn = 0;
  for (ii=-1; ii<=1; ii++) {
    for (jj=-1; jj<=1; jj++) {
      for (kk=-1; kk<=1; kk++) {
        PetscInt t[3];
        
        t[0] = cell.i + ii + s[0];
        t[1] = cell.j + jj + s[1];
        t[2] = cell.k + kk + s[2];
        
        if (t[0] < 0) continue;
        if (t[1] < 0) continue;
        if (t[2] < 0) continue;
        
        if (t[0] >= fv->Mi[0]) continue;
        if (t[1] >= fv->Mi[1]) continue;
        if (t[2] >= fv->Mi[2]) continue;

        if (t[0] < s[0]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"stencil range");
        if (t[1] < s[1]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"stencil range");
        if (t[2] < s[2]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"stencil range");
        
        if (t[0] >= s[0]+w[0]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"stencil range");
        if (t[1] >= s[1]+w[1]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"stencil range");
        if (t[2] >= s[2]+w[2]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"stencil range");

        
        neigh[*nn] = (cell.i + ii) + (cell.j + jj) * w[0] + (cell.k + kk) * w[0] * w[1];
        if (neigh[*nn] >= w[0]*w[1]*w[2]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Stencil index out-of-bounds of local fv space");
        //printf("cell global %d %d %d : nstencil [%d] ->  %d %d %d\n",cellglobal.i,cellglobal.j,cellglobal.k,*nn,t[0],t[1],t[2]);
        (*nn)++;
      }
    }
  }
  if (*nn > 27) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Stencil size out-of-bounds");
  //printf("cell global %d %d %d : nstencil %d\n",cellglobal.i,cellglobal.j,cellglobal.k,*nn);

  if (*nn < 4) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Stencil for reconstruction is not sufficiently large");
  
  
#if 0
  *nn = 0;
  if (interior) {
    
    neigh[0] = (cell.i - 1) + (cell.j + 0) * w[0] + (cell.k + 0) * w[0] * w[1];
    neigh[1] = (cell.i + 1) + (cell.j + 0) * w[0] + (cell.k + 0) * w[0] * w[1];

    neigh[2] = (cell.i + 0) + (cell.j - 1) * w[0] + (cell.k + 0) * w[0] * w[1];
    neigh[3] = (cell.i + 0) + (cell.j + 1) * w[0] + (cell.k + 0) * w[0] * w[1];

    neigh[4] = (cell.i + 0) + (cell.j + 0) * w[0] + (cell.k - 1) * w[0] * w[1];
    neigh[5] = (cell.i + 0) + (cell.j + 0) * w[0] + (cell.k + 1) * w[0] * w[1];

    *nn = 6;
  } else {

    for (ii=-1; ii<=1; ii++) {
      for (jj=-1; jj<=1; jj++) {
        for (kk=-1; kk<=1; kk++) {
          
          if (cell.i + ii + s[0] < 0) continue;
          if (cell.j + jj + s[1] < 0) continue;
          if (cell.k + kk + s[2] < 0) continue;

          if (cell.i + ii + s[0] >= fv->Mi[0]) continue;
          if (cell.j + jj + s[1] >= fv->Mi[1]) continue;
          if (cell.k + kk + s[2] >= fv->Mi[2]) continue;

          neigh[*nn] = (cell.i + ii) + (cell.j + jj) * w[0] + (cell.k + kk) * w[0] * w[1];
          (*nn)++;
        }
      }
    }
    
  }
#endif
  
  PetscFunctionReturn(0);
}

