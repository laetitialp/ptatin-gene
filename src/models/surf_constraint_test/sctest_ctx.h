#ifndef __ptatin3d_sctest_ctx_h__
#define __ptatin3d_sctest_ctx_h__

/* define user model */
typedef struct {
  PetscReal length_bar, viscosity_bar, velocity_bar, time_bar, pressure_bar, density_bar, acceleration_bar;
  PetscReal Lx, Ly, Lz, Ox, Oy, Oz;
  PetscReal layer1,layer2;
  PetscBool PolarMesh;
  PetscReal norm_u,uz0,ux0,alpha,theta;
  PetscReal n_hat[3],t1_hat[3],epsilon_s[6],H[6];
  PetscInt  direction_BC;
} ModelSCTestCtx;

#endif
