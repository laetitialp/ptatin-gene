
#ifndef __fvda_utils_h__
#define __fvda_utils_h__

/* fvda_property.c */
typedef enum {
  FVARRAY_DATA_SELF=0,
  FVARRAY_DATA_USER,
  FVARRAY_DATA_VEC
} FVArrayDataType;

typedef struct _p_FVArray *FVArray;
struct _p_FVArray {
  PetscInt        len,bs;
  PetscReal       *v;
  FVArrayDataType dtype;
  DM              dm;
  FVDA            fv;
  FVPrimitiveType type;
  void            *auxdata;
};

PetscErrorCode FVDARegisterCellProperty(FVDA fv,const char name[],PetscInt blocksize);
PetscErrorCode FVDAGetCellPropertyArray(FVDA fv,PetscInt index,PetscReal *data[]);
PetscErrorCode FVDAGetCellPropertyArrayRead(FVDA fv,PetscInt index,const PetscReal *data[]);
PetscErrorCode FVDAGetCellPropertyByNameArrayRead(FVDA fv,const char name[],const PetscReal *data[]);
PetscErrorCode FVDAGetCellPropertyByNameArray(FVDA fv,const char name[],PetscReal *data[]);
PetscErrorCode FVDACellPropertyGetInfo(FVDA fv,const char name[],PetscInt *index,PetscInt *len,PetscInt *bs);
PetscErrorCode FVDACellPropertyQuery(FVDA fv,const char name[],PetscBool *found);

PetscErrorCode FVDAGetFacePropertyInfo(FVDA fv,PetscInt *len,const char ***name);
PetscErrorCode FVDARegisterFaceProperty(FVDA fv,const char name[],PetscInt blocksize);
PetscErrorCode FVDAGetFacePropertyArray(FVDA fv,PetscInt index,PetscReal *data[]);
PetscErrorCode FVDAGetFacePropertyArrayRead(FVDA fv,PetscInt index,const PetscReal *data[]);
PetscErrorCode FVDAGetFacePropertyByNameArrayRead(FVDA fv,const char name[],const PetscReal *data[]);
PetscErrorCode FVDAGetFacePropertyByNameArray(FVDA fv,const char name[],PetscReal *data[]);
PetscErrorCode FVDAFacePropertyGetInfo(FVDA fv,const char name[],PetscInt *index,PetscInt *len,PetscInt *bs);
PetscErrorCode FVDAFacePropertyQuery(FVDA fv,const char name[],PetscBool *found);

PetscErrorCode FVArrayCreate(FVArray *a);
PetscErrorCode FVArrayDestroy(FVArray *a);
PetscErrorCode FVArrayCreateFromFVDAFaceProperty(FVDA fv,const char name[],FVArray *a);
PetscErrorCode FVArrayCreateFVDAFaceSpace(FVDA fv,PetscInt bs,FVArray *a);
PetscErrorCode FVArrayCreateFromFVDACellProperty(FVDA fv,const char name[],FVArray *a);
PetscErrorCode FVArrayCreateFVDACellSpace(FVDA fv,PetscInt bs,FVArray *a);
PetscErrorCode FVArraySetDM(FVArray a,DM dm);
PetscErrorCode FVArraySetFVDA(FVArray a,FVDA fv);
PetscErrorCode FVArrayCreateFromData(FVPrimitiveType t,PetscInt n,PetscInt b,const PetscReal x[],FVArray *a);
PetscErrorCode FVArrayZeroEntries(FVArray a);
PetscErrorCode FVArrayCreateFromVec(FVPrimitiveType t,Vec x,FVArray *a);

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
PetscErrorCode FVDAView_FaceData_local(FVDA fv,const char prefix[]);
PetscErrorCode FVDAView_CellData_local(FVDA fv,Vec field,PetscBool view_cell_prop,const char prefix[]);
PetscErrorCode FVDAView_CellData(FVDA fv,Vec field,PetscBool view_cell_prop,const char prefix[]);
PetscErrorCode FVDAView_JSON(FVDA fv,const char path[],const char prefix[]);
PetscErrorCode PetscVecWriteJSON(Vec x,PetscInt format,const char suffix[]);
PetscErrorCode FVDAView_Heavy(FVDA fv,const char path[],const char suffix[]);
PetscErrorCode FVDAOutputParaView(FVDA fv,Vec field,PetscBool view_cell_prop,const char path[],const char prefix[]);

/* fv_ops_time_dep.c */
PetscErrorCode FVDASetup_TimeDep(FVDA fv);
PetscErrorCode FVDAAccessData_TimeDep(FVDA fv,PetscReal **dt,Vec *Qk);
PetscErrorCode fvda_eval_F_timedep(SNES snes,Vec X,Vec F,void *data);
PetscErrorCode fvda_eval_J_timedep(SNES snes,Vec X,Mat Ja,Mat Jb,void *data);
PetscErrorCode FVDAStep_FEuler(FVDA fv,PetscReal time,PetscReal dt,Vec X,Vec X_new);
PetscErrorCode FVDAStep_RK2SSP(FVDA fv,const PetscReal range[],PetscReal time,PetscReal dt,Vec X,Vec X_new);
PetscErrorCode FVEnforceBounds(FVDA fv,const PetscReal range[],Vec Q);
PetscErrorCode fvda_highres_eval_F_timedep(SNES snes,Vec X,Vec F,void *data);


/* fv_ops_ale.c */
PetscErrorCode FVDASetup_ALE(FVDA fv);
PetscErrorCode FVDAAccessData_ALE(FVDA fv,PetscReal **dt,Vec *Qk,Vec *coor_target);
PetscErrorCode fvda_eval_F_forward_ale(SNES snes,Vec X,Vec F,void *data);
PetscErrorCode fvda_eval_J_forward_ale(SNES snes,Vec X,Mat Ja,Mat Jb,void *data);
PetscErrorCode fvda_eval_F_backward_ale(SNES snes,Vec X,Vec F,void *data);
PetscErrorCode fvda_highres_eval_F_forward_ale(SNES snes,Vec X,Vec F,void *data);


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
PetscErrorCode FVDAGradientProjectViaReconstruction(FVDA fv,FVArray Q,FVArray gradQ);
PetscErrorCode FVDAFieldProjectReconstructionToVertex_Q1(FVDA fv,Vec fv_field,PetscReal min,PetscReal max,DM dmf,Vec field);

typedef struct _p_FVProject *FVProject;
struct _p_FVProject {
  FVDA      fv;
  Vec       q;
  PetscReal range[2];
  PetscInt  type;
  DM              dmf;
  Vec             gf,lf;
  const PetscReal *_lf;
  PetscInt        bs;
  PetscErrorCode  (*setup)(FVProject);
  //PetscErrorCode (*eval)(FVProject,PetscInt cell,const PetscReal*,PetscReal*);
  PetscBool       issetup;
  PetscInt        nel,nen;
  const PetscInt  *e;
};

PetscErrorCode FVProjectCreate(FVDA fv,PetscInt type,Vec Q,FVProject *proj);
PetscErrorCode FVProjectSetBounds(FVProject proj,const PetscReal r[]);
PetscErrorCode FVProjectDestroyGlobalSpace(FVProject proj);
PetscErrorCode FVProjectSetup(FVProject proj);
PetscErrorCode FVProjectDestroy(FVProject *proj);
PetscErrorCode FVProjectEvaluate(FVProject proj,PetscInt cell,const PetscReal xi[],PetscReal val[]);

/* fvda_reconstruction.c */
typedef struct _p_FVReconstructionCell FVReconstructionCell;
struct _p_FVReconstructionCell {
  PetscInt  target_cell;
  PetscReal target_x[3];
  PetscReal target_Q;
  PetscReal coeff[3];
  PetscInt  n_neigh,neigh[125]; /* sufficient for a 5x5x5 patch, at most we need 3x3x3 */
};

PetscErrorCode setup_coeff(FVDA fv,PetscInt target,PetscInt nneigh,const PetscInt neigh[],const PetscReal cell_x[],const PetscReal Q[],PetscReal coeff[]);
PetscErrorCode FVDAReconstructP1Evaluate(FVDA fv,
                                         PetscReal x[],
                                         PetscInt target,
                                         const PetscReal cell_target_x[],const PetscReal Q[],
                                         PetscReal coeff[],
                                         PetscReal Q_hr[]);
PetscErrorCode FVDAGetReconstructionStencil_AtCell(FVDA fv,PetscInt cijk,PetscInt *nn,PetscInt neigh[]);

PetscErrorCode FVReconstructionP1Create(FVReconstructionCell *cell,
                                        FVDA fv,PetscInt local_fv_cell_index,
                                        const PetscReal _fv_coor[],const PetscReal _fv_field[]);

PetscErrorCode FVReconstructionP1Interpolate(FVReconstructionCell *cell,const PetscReal x[],PetscReal ival[]);

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

PetscErrorCode FVSetDirichletFromInflow(FVDA fv,Vec T,DACellFace face);

/* dmda_warp.c */
PetscErrorCode DMDAWarpCoordinates_SinJMax(DM da,PetscReal amp,PetscReal omega[]);
PetscErrorCode DMDAWarpCoordinates_ExpJMax(DM da,PetscReal amp,PetscReal lambda[]);

/* fvda_dimap.c */
/* Dense Integer Map object */
/* Note - the type DMap is part of the C11 standard */
typedef struct _p_DIMap *DIMap;
struct _p_DIMap {
  PetscInt  input_range[2];
  PetscBool negative_output_allowed; /* error if negative val found */
  PetscBool negative_input_ignored; /* return same value */
  PetscInt  *idx;
  PetscInt  len;
  PetscInt  range[2];
};

/* Dense integer map options */
typedef enum {
  DIMAP_IGNORE_NEGATIVE_INPUT  = 0,
  DIMAP_IGNORE_NEGATIVE_OUTPUT = 1,
  DIMAP_OPTION_MAX             = 2,
} DIMapOption;

PetscErrorCode DIMapCreate(DIMap *map);
PetscErrorCode DIMapDestroy(DIMap *map);
PetscErrorCode DIMapSetOptions(DIMap map,DIMapOption op,PetscBool val);
PetscErrorCode DIMapGetEntries(DIMap map,const PetscInt *idx[]);
PetscErrorCode DIMapCreate_FVDACell_RankLocalToLocal(FVDA fv,DIMap *map);
PetscErrorCode DIMapCreate_FVDACell_LocalToRankLocal(FVDA fv,DIMap *map);
PetscErrorCode DIMapApply(DIMap map,PetscInt i,PetscInt *j);
PetscErrorCode DIMapApplyN(DIMap map,PetscInt N,PetscInt i[],PetscInt j[]);
#define DIMAP_APPLY(map,i,j) (j) = (map)->idx[(i)];

#endif
