#ifndef __ptatinmodel_stokes_law_ctx_h__
#define __ptatinmodel_stokes_law_ctx_h__

typedef struct {
  PetscInt  nregions,bc_type,component;
  PetscReal O[3],L[3];
  PetscReal r_s,sphere_centre[3];
  PetscReal gravity[3],u_T[3],time;
  PetscReal eta_f,eta_s,rho_f,rho_s;
  PetscReal length_bar,viscosity_bar,velocity_bar,time_bar,pressure_bar,density_bar,acceleration_bar;
  PetscReal Ttop,Tbottom,T_f,T_s;
  PetscBool output_markers,output_petscvec,refine_mesh;
  } ModelStokesLawCtx;

#endif