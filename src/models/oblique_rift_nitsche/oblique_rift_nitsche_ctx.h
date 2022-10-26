#ifndef __ptatinmodel_oblique_rift_nitsche_ctx_h__
#define __ptatinmodel_oblique_rift_nitsche_ctx_h__

typedef struct {
  /* Bounding box */
  PetscReal Lx,Ly,Lz,Ox,Oy,Oz;
  /* Layering */
  PetscReal y_continent[3];
  /* Velocity BCs */
  PetscReal norm_u,alpha_u,u_bc[3];
  /* General Navier Slip BCs */
  PetscReal epsilon_s[6],H[6],t1_hat[3],n_hat[3];
  /* Number of materials */
  PetscInt  n_phases;
  /* Scaling values */
  PetscReal length_bar,viscosity_bar,velocity_bar;
  PetscReal time_bar,pressure_bar,density_bar,acceleration_bar;
  /* Viscosity cutoff */
  PetscBool eta_cutoff;
  PetscReal eta_max,eta_min;
  /* SPM parameters */
  PetscReal diffusivity_spm;
  /* Temperature BCs */
  PetscReal Ttop,Tbottom;
  /* Weak zone type */
  PetscBool wz_notch,wz_gauss,wz_oblique;
  PetscInt  n_notches;
  PetscReal wz_angle,wz_width,wz_sigma[2];
  /* Output */
  PetscBool output_markers;
} ModelRiftNitscheCtx;

#endif