#ifndef __ptatin3d_sctest_ctx_h__
#define __ptatin3d_sctest_ctx_h__

/* define user model */
typedef struct {
  PetscReal length_bar, viscosity_bar, velocity_bar, time_bar, pressure_bar, density_bar, acceleration_bar;
  PetscReal Lx, Ly, Lz, Ox, Oy, Oz;
  PetscReal layer1,layer2;
} ModelSCTestCtx;

#endif
