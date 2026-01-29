#ifndef __ptatin3d_poisson_pressure_ctx_h__
#define __ptatin3d_poisson_pressure_ctx_h__

/* define user model */
typedef struct {
  PetscInt n_phases;
  /* Initial geometry */
  PetscInt  geometry_type;
  PetscReal Lx,Ly,Lz,Ox,Oy,Oz;
  PetscReal y_continent[3];
  /* Boundary conditions */
  PetscReal pressure_jmin,pressure_jmax;
  PetscBool dirichlet_jmin,dirichlet_jmax;
  /* scaling parameters */
  PetscReal length_bar,viscosity_bar,velocity_bar;
  PetscReal time_bar,pressure_bar,density_bar,acceleration_bar;
  /* Output */
  PetscBool output_markers;
} ModelPoissonPressureCtx;

#endif
