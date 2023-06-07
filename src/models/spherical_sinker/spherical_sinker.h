#ifndef __ptatinmodel_spherical_sinker_ctx_h__
#define __ptatinmodel_spherical_sinker_ctx_h__

typedef struct {
  PetscInt  n_phases,gravity_type;
  PetscReal O[3],L[3];
  PetscReal gravity_vector[3],gravity_magnitude;
  PetscReal inclusion_origin[3],inclusion_radius[3];
  PetscReal length_bar,viscosity_bar,velocity_bar,time_bar,pressure_bar,density_bar,acceleration_bar;
} ModelSphericalCtx;


#endif