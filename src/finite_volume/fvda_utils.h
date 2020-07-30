
#ifndef __fvda_utils_h__
#define __fvda_utils_h__

/* fvda_property.c */
PetscErrorCode FVDAGetCellPropertyInfo(FVDA fv,PetscInt *len,const char ***name);
PetscErrorCode FVDARegisterCellProperty(FVDA fv,const char name[],PetscInt blocksize);
PetscErrorCode FVDAGetCellPropertyArray(FVDA fv,PetscInt index,PetscReal *data[]);
PetscErrorCode FVDAGetCellPropertyArrayRead(FVDA fv,PetscInt index,const PetscReal *data[]);
PetscErrorCode FVDAGetCellPropertyByNameArrayRead(FVDA fv,const char name[],const PetscReal *data[]);
PetscErrorCode FVDAGetCellPropertyByNameArray(FVDA fv,const char name[],PetscReal *data[]);
PetscErrorCode FVDACellPropertyGetInfo(FVDA fv,const char name[],PetscInt *index,PetscInt *len,PetscInt *bs);

PetscErrorCode FVDAGetFacePropertyInfo(FVDA fv,PetscInt *len,const char ***name);
PetscErrorCode FVDARegisterFaceProperty(FVDA fv,const char name[],PetscInt blocksize);
PetscErrorCode FVDAGetFacePropertyArray(FVDA fv,PetscInt index,PetscReal *data[]);
PetscErrorCode FVDAGetFacePropertyArrayRead(FVDA fv,PetscInt index,const PetscReal *data[]);
PetscErrorCode FVDAGetFacePropertyByNameArrayRead(FVDA fv,const char name[],const PetscReal *data[]);
PetscErrorCode FVDAGetFacePropertyByNameArray(FVDA fv,const char name[],PetscReal *data[]);
PetscErrorCode FVDAFacePropertyGetInfo(FVDA fv,const char name[],PetscInt *index,PetscInt *len,PetscInt *bs);


/* fvda_compatible_velocity.c */
PetscErrorCode FVDAPostProcessCompatibleVelocity(FVDA fv,const char name_v[],const char name_v_dot_n[],Vec source,KSP _ksp);
PetscErrorCode FVDAPPCompatibleVelocityCreate(FVDA fv,KSP *ksp);
PetscErrorCode FVDAIntegrateFlux(FVDA fv,const char field_name[],PetscBool f_dot_n,Vec R);
PetscErrorCode FVDAIntegrateFlux_Local(FVDA fv,const char field_name[],PetscBool f_dot_n,PetscReal factor,
                                     const PetscScalar _geom_coor[],PetscReal _r[]);
PetscErrorCode FVDAPPCompatibleVelocityCreate(FVDA fv,KSP *ksp);


/* fvda_view.c */
PetscErrorCode FVDAViewStatistics(FVDA fv,PetscBool collective);
PetscErrorCode FVDAView_CellGeom_local(FVDA fv);
PetscErrorCode FVDAView_BFaceGeom_local(FVDA fv);
PetscErrorCode FVDAView_FaceGeom_local(FVDA fv);
PetscErrorCode FVDAView_CellData_local(FVDA fv,Vec field,PetscBool view_cell_prop,const char prefix[]);
PetscErrorCode FVDAView_CellData(FVDA fv,Vec field,PetscBool view_cell_prop,const char prefix[]);
PetscErrorCode FVDAView_JSON(FVDA fv,const char path[],const char prefix[]);
PetscErrorCode PetscVecWriteJSON(Vec x,PetscInt format,const char suffix[]);
PetscErrorCode FVDAView_Heavy(FVDA fv,const char path[],const char suffix[]);


/* fv_ops_time_dep.c */
PetscErrorCode FVDASetup_TimeDep(FVDA fv);
PetscErrorCode FVDAAccessData_TimeDep(FVDA fv,PetscReal **dt,Vec *Qk);
PetscErrorCode FVDADestroy_TimeDep(FVDA fv);
PetscErrorCode fvda_eval_F_timedep(SNES snes,Vec X,Vec F,void *data);
PetscErrorCode fvda_eval_J_timedep(SNES snes,Vec X,Mat Ja,Mat Jb,void *data);
PetscErrorCode FVDAStep_FEuler(FVDA fv,PetscReal time,PetscReal dt,Vec X,Vec X_new);
PetscErrorCode FVDAStep_RK2SSP(FVDA fv,const PetscReal range[],PetscReal time,PetscReal dt,Vec X,Vec X_new);
PetscErrorCode FVEnforceBounds(FVDA fv,const PetscReal range[],Vec Q);
PetscErrorCode fvda_highres_eval_F_timedep(SNES snes,Vec X,Vec F,void *data);

/* fv_ops_ale.c */
PetscErrorCode FVDASetup_ALE(FVDA fv);
PetscErrorCode FVDAAccessData_ALE(FVDA fv,PetscReal **dt,Vec *Qk,Vec *coor_target);
PetscErrorCode FVDADestroy_ALE(FVDA fv);
PetscErrorCode fvda_eval_F_forward_ale(SNES snes,Vec X,Vec F,void *data);
PetscErrorCode fvda_eval_J_forward_ale(SNES snes,Vec X,Mat Ja,Mat Jb,void *data);
PetscErrorCode fvda_eval_F_backward_ale(SNES snes,Vec X,Vec F,void *data);


/* fvda_ale_utils.c */
PetscErrorCode FVDAALEComputeMeshVelocity(DM dmg,Vec x0,Vec x1,PetscReal dt,Vec v);
PetscErrorCode FVDAALEComputeFaceAverageVelocity_Interpolate(DM dmg,Vec x0,Vec v,FVDA fv,const char face_vec_name[]);


/* fvda_project.c */
PetscErrorCode FVDACellPropertyProjectToFace_HarmonicMean(FVDA fv,const char cell_field[],const char face_field[]);
PetscErrorCode FVDACellPropertyProjectToFace_ArithmeticMean(FVDA fv,const char cell_field[],const char face_field[]);
PetscErrorCode FVDACellPropertyProjectToFace_GeneralizedMean(FVDA fv,const char cell_field[],const char face_field[],PetscInt avg_type,PetscBool volume_weighted);
PetscErrorCode FVDAFieldSetUpProjectToVertex_Q1(FVDA fv,DM *dmf,Vec *field);
PetscErrorCode FVDAFieldProjectToVertex_Q1(FVDA fv,Vec fv_field,DM dmf,Vec field);
PetscErrorCode FVDAGradientProject(FVDA fv,Vec Q,Vec gradQ);
PetscErrorCode FVDAFieldProjectReconstructionToVertex_Q1(FVDA fv,Vec fv_field,PetscReal min,PetscReal max,DM dmf,Vec field);


/* fvda_reconstruction.c */
PetscErrorCode setup_coeff(FVDA fv,PetscInt target,PetscInt nneigh,const PetscInt neigh[],const PetscReal cell_x[],const PetscReal Q[],PetscReal coeff[]);
PetscErrorCode FVDAReconstructP1Evaluate(FVDA fv,
                                         PetscReal x[],
                                         PetscInt target,
                                         const PetscReal cell_target_x[],const PetscReal Q[],
                                         PetscReal coeff[],
                                         PetscReal Q_hr[]);
PetscErrorCode FVDAGetReconstructionStencil_AtCell(FVDA fv,PetscInt cijk,PetscInt *nn,PetscInt neigh[]);


/* fvda_bc_utils.c */
PetscErrorCode FVDABCMethod_SetNatural(FVDA fv,
                                       DACellFace face,
                                       PetscInt nfaces,
                                       const PetscReal coor[],
                                       const PetscReal normal[],
                                       const PetscInt cell[],
                                       PetscReal time,
                                       FVFluxType flux[],
                                       PetscReal bcvalue[],
                                       void *ctx);

PetscErrorCode FVDABCMethod_SetDirichlet(FVDA fv,
                                         DACellFace face,
                                         PetscInt nfaces,
                                         const PetscReal coor[],
                                         const PetscReal normal[],
                                         const PetscInt cell[],
                                         PetscReal time,
                                         FVFluxType flux[],
                                         PetscReal bcvalue[],
                                         void *ctx);

PetscErrorCode FVDABCMethod_SetNeumann(FVDA fv,
                                       DACellFace face,
                                       PetscInt nfaces,
                                       const PetscReal coor[],
                                       const PetscReal normal[],
                                       const PetscInt cell[],
                                       PetscReal time,
                                       FVFluxType flux[],
                                       PetscReal bcvalue[],
                                       void *ctx);

PetscErrorCode FVDABCMethod_SetNeumannWithVector(FVDA fv,
                                                 DACellFace face,
                                                 PetscInt nfaces,
                                                 const PetscReal coor[],
                                                 const PetscReal normal[],
                                                 const PetscInt cell[],
                                                 PetscReal time,
                                                 FVFluxType flux[],
                                                 PetscReal bcvalue[],
                                                 void *ctx);

/* dmda_warp.c */
PetscErrorCode DMDAWarpCoordinates_SinJMax(DM da,PetscReal amp,PetscReal omega[]);
PetscErrorCode DMDAWarpCoordinates_ExpJMax(DM da,PetscReal amp,PetscReal lambda[]);

#endif
