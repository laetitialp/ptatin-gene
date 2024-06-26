#ifndef __ptatinmodel_oblique_rift_nitsche_ctx_h__
#define __ptatinmodel_oblique_rift_nitsche_ctx_h__

typedef struct {
  /* Bounding box */
  PetscReal L[3],O[3];
  /* Layering */
  PetscReal y_continent[3];
  /* Velocity BCs */
  PetscReal norm_u,alpha_u,u_bc[3],alpha_r;
  PetscInt  component,bc_type;
  PetscReal split_face_min[2],split_face_max[2];
  PetscReal time_full_velocity;
  /* General Navier Slip BCs */
  PetscReal epsilon_s[6],H[6],t1_hat[3],n_hat[3];
  /* Neumann BCs */
  PetscInt mark_type;
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
  PetscInt  n_notches,wz_type,wz_centre_type;
  PetscReal wz_angle,wz_width,wz_sigma[2];
  PetscReal wz_origin,wz_offset;
  /* Output */
  PetscBool output_markers,output_pressure;
  /* Passive markers */
  PSwarm pswarm;
  /* Poisson pressure */
  Mat      poisson_Jacobian;
  PetscInt prev_step;
} ModelRiftNitscheCtx;

#endif