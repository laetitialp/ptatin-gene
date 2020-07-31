
#ifndef __fvda_impl_h__
#define __fvda_impl_h__

#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <fvda.h>



struct _p_FVDA {
  MPI_Comm comm;
  PetscInt dim,mi[3],Mi[3];
  DM       dm_geometry; /* overlap 1 : no coordinates */
  Vec      vertex_coor_geometry;
  DM       dm_fv; /* overlap 1 : has coords */
  /*DM       dm_fv_2;*/ /* overlap 2 - need for high resolution flux : has coords */
  /*Vec      cell_coor_geometry_local;*/ /* has different meanings depending on method */
  PetscInt *cell_ownership_i;
  PetscInt *cell_ownership_j;
  PetscInt *cell_ownership_k;
  
  PetscInt  ncoeff_cell,ncoeff_face;
  char      **cell_coeff_name,**face_coeff_name;
  PetscReal **cell_coefficient; /* default is likely A,B,C */
  PetscReal **face_coefficient; /* default is likely v.n */
  PetscInt  *cell_coeff_size,*face_coeff_size;
  
  PetscInt  ncells,nfaces;
  PetscInt  *flux_type; /* [nfaces] */
  
  PetscInt  *face_element_map; /* [2 * nfaces] */
  PetscInt  *face_fv_map; /* [2 * nfaces] */
  PetscReal *face_normal; /* [dim * nfaces] */
  PetscReal *face_centroid; /* [dim * nfaces] */
  DACellFaceLocation *face_location; /* [nfaces] */
  DACellFace         *face_type; /* [nfaces] */
  PetscInt nfaces_interior,nfaces_boundary;
  PetscInt boundary_ranges[7];
  PetscInt *face_idx_interior,*face_idx_boundary;
  
  FVFluxType *boundary_flux;
  PetscReal  *boundary_value;
  
  PetscBool setup;
  
  /* residual */
  PetscBool   q_dot;
  FVDAPDEType equation_type;
  PetscInt    numerical_flux;
  PetscInt    reconstruction;
  void           *ctx;
  PetscErrorCode (*ctx_destroy)(FVDA);
};


struct _p_FVALE {
  Vec       Q_k;
  Vec       vertex_coor_geometry_target;
  PetscReal dt;
};


struct _p_FVTD {
  Vec       Q_k;
  PetscReal dt;
};

#endif
