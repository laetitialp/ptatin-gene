
#include <petsc.h>
#include <fvda.h>
#include <fvda_impl.h>
#include <fvda_utils.h>


PetscErrorCode DIMapCreate(DIMap *map)
{
  PetscErrorCode ierr;
  DIMap          m;
  
  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(struct _p_DIMap),&m);CHKERRQ(ierr);
  ierr = PetscMemzero(m,sizeof(struct _p_DIMap));CHKERRQ(ierr);
  m->range[0] = -1;
  m->range[1] = -2;
  m->negative_output_allowed = PETSC_FALSE;
  m->negative_input_ignored = PETSC_FALSE;
  *map = m;
  PetscFunctionReturn(0);
}

PetscErrorCode DIMapDestroy(DIMap *map)
{
  PetscErrorCode ierr;
  DIMap          m;
  
  PetscFunctionBegin;
  if (!map) PetscFunctionReturn(0);
  m = *map;
  if (!m) PetscFunctionReturn(0);
  ierr = PetscFree(m->idx);CHKERRQ(ierr);
  ierr = PetscFree(m);CHKERRQ(ierr);
  *map = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode DIMapSetOptions(DIMap map,DIMapOption op,PetscBool val)
{
  PetscFunctionBegin;
  switch (op) {
    case DIMAP_IGNORE_NEGATIVE_INPUT:
      map->negative_input_ignored = val;
      break;

    case DIMAP_IGNORE_NEGATIVE_OUTPUT:
      map->negative_output_allowed = val;
      break;
      
    default:
      break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DIMapGetEntries(DIMap map,const PetscInt *idx[])
{
  PetscFunctionBegin;
  *idx = (const PetscInt*)map->idx;
  PetscFunctionReturn(0);
}

/*
 Rank local indices must always map to a positive local index.
 Hence inputs and outputs must always be positive
*/
PetscErrorCode DIMapCreate_FVDACell_RankLocalToLocal(FVDA fv,DIMap *map)
{
  PetscErrorCode ierr;
  DIMap          m;
  PetscInt       rl_start[3],g_start[3],rl_w[3],g_w[3],ghost_offset[3];
  PetscInt       i,j,k;
  
  
  PetscFunctionBegin;
  /* create and set options */
  ierr = DIMapCreate(&m);CHKERRQ(ierr);
  ierr = DIMapSetOptions(m,DIMAP_IGNORE_NEGATIVE_INPUT,PETSC_FALSE);CHKERRQ(ierr);
  ierr = DIMapSetOptions(m,DIMAP_IGNORE_NEGATIVE_OUTPUT,PETSC_FALSE);CHKERRQ(ierr);
  
  ierr = DMDAGetCorners(fv->dm_fv     ,&rl_start[0],&rl_start[1],&rl_start[2],&rl_w[0],&rl_w[1],&rl_w[2]);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(fv->dm_fv,&g_start[0],&g_start[1],&g_start[2],&g_w[0],&g_w[1],&g_w[2]);CHKERRQ(ierr);
  ghost_offset[0] = rl_start[0] - g_start[0];
  ghost_offset[1] = rl_start[1] - g_start[1];
  ghost_offset[2] = rl_start[2] - g_start[2];

  /* set input range */
  m->input_range[0] = 0;
  m->input_range[1] = fv->ncells;

  /* set size */
  m->len = rl_w[0] * rl_w[1] * rl_w[2];
  
  /* allocate and init */
  ierr = PetscCalloc1(m->len,&m->idx);CHKERRQ(ierr);
  for (k=0; k<m->len; k++) { m->idx[k] = -1; }
  
  /* set output range */
  m->range[0] = 0;
  m->range[1] = g_w[0] * g_w[1] * g_w[2];

  for (k=0; k<rl_w[2]; k++) {
    for (j=0; j<rl_w[1]; j++) {
      for (i=0; i<rl_w[0]; i++) {
        PetscInt rl_ijk[3],g_rijk[3],rl_idx,g_idx;
        rl_ijk[0] = i;
        rl_ijk[1] = j;
        rl_ijk[2] = k;
        ierr = _cart_convert_ijk_to_index(rl_ijk,rl_w,&rl_idx);CHKERRQ(ierr);

        g_rijk[0] = rl_ijk[0] + ghost_offset[0];
        g_rijk[1] = rl_ijk[1] + ghost_offset[1];
        g_rijk[2] = rl_ijk[2] + ghost_offset[2];
        ierr = _cart_convert_ijk_to_index(g_rijk,g_w,&g_idx);CHKERRQ(ierr);

        m->idx[ rl_idx ] = g_idx;
      }
    }
  }
  
  *map = m;
  
  PetscFunctionReturn(0);
}

/*
 Local indices will map to -1 or a positive rank local index in the range [0,ncells-1]
 Hence negative inputs allowed
 Hence negative outputs allowed
*/
PetscErrorCode DIMapCreate_FVDACell_LocalToRankLocal(FVDA fv,DIMap *map)
{
  PetscErrorCode ierr;
  DIMap          m;
  PetscInt       rl_start[3],g_start[3],rl_w[3],g_w[3],ghost_offset[3];
  PetscInt       i,j,k;

  
  PetscFunctionBegin;
  /* create and set options */
  ierr = DIMapCreate(&m);CHKERRQ(ierr);
  ierr = DIMapSetOptions(m,DIMAP_IGNORE_NEGATIVE_INPUT,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DIMapSetOptions(m,DIMAP_IGNORE_NEGATIVE_OUTPUT,PETSC_TRUE);CHKERRQ(ierr);
  
  ierr = DMDAGetCorners(fv->dm_fv     ,&rl_start[0],&rl_start[1],&rl_start[2],&rl_w[0],&rl_w[1],&rl_w[2]);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(fv->dm_fv,&g_start[0],&g_start[1],&g_start[2],&g_w[0],&g_w[1],&g_w[2]);CHKERRQ(ierr);
  ghost_offset[0] = rl_start[0] - g_start[0];
  ghost_offset[1] = rl_start[1] - g_start[1];
  ghost_offset[2] = rl_start[2] - g_start[2];

  /* set input range */
  m->input_range[0] = 0;
  m->input_range[1] = g_w[0] * g_w[1] * g_w[2];

  /* set size */
  m->len = g_w[0] * g_w[1] * g_w[2];
  
  /* allocate and init */
  ierr = PetscCalloc1(m->len,&m->idx);CHKERRQ(ierr);
  for (k=0; k<m->len; k++) { m->idx[k] = -1; }
  
  /* set output range */
  m->range[0] = 0;
  m->range[1] = fv->ncells;
  
  for (k=0; k<rl_w[2]; k++) {
    for (j=0; j<rl_w[1]; j++) {
      for (i=0; i<rl_w[0]; i++) {
        PetscInt rl_ijk[3],g_rijk[3],rl_idx,g_idx;
        rl_ijk[0] = i;
        rl_ijk[1] = j;
        rl_ijk[2] = k;
        ierr = _cart_convert_ijk_to_index(rl_ijk,rl_w,&rl_idx);CHKERRQ(ierr);
        
        g_rijk[0] = rl_ijk[0] + ghost_offset[0];
        g_rijk[1] = rl_ijk[1] + ghost_offset[1];
        g_rijk[2] = rl_ijk[2] + ghost_offset[2];
        ierr = _cart_convert_ijk_to_index(g_rijk,g_w,&g_idx);CHKERRQ(ierr);
        
        m->idx[ g_idx ] = rl_idx;
      }
    }
  }
  
  *map = m;

  PetscFunctionReturn(0);
}

/*
 j = map(i)
 if negative_input_ignored = true, return j = -i if i < 0
 if negative_output_allowed = false, return error if j < 0
 return error if i out of bounds of input_range[]
*/
PetscErrorCode DIMapApply(DIMap map,PetscInt i,PetscInt *j)
{
  PetscFunctionBegin;
  if ((i < 0) && (map->negative_input_ignored)) { *j = i; PetscFunctionReturn(0); }

#if defined(PETSC_USE_DEBUG)
  if (i < map->input_range[0]) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_SUP,"i %D outside range [%D,%D)",i,map->input_range[0],map->input_range[1]);
  if (i >= map->input_range[1]) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_SUP,"i %D outside range [%D,%D)",i,map->input_range[0],map->input_range[1]);
#endif

  *j = map->idx[i];
  
  if ((*j < 0) && (!map->negative_output_allowed)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"i %D maps to negative value %D : DIMap config does not allow negative output",i,*j);
  
  PetscFunctionReturn(0);
}

PetscErrorCode DIMapApplyN(DIMap map,PetscInt N,PetscInt i[],PetscInt j[])
{
  PetscInt       k;
  
  PetscFunctionBegin;
  if (map->negative_input_ignored) {
    for (k=0; k<N; k++) {
      if (i[k] < 0) j[k] = i[k];
      else {
#if defined(PETSC_USE_DEBUG)
        if (i[k] < map->input_range[0]) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_SUP,"i %D outside range [%D,%D)",i[k],map->input_range[0],map->input_range[1]);
        
        if (i[k] >= map->input_range[1]) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_SUP,"i %D outside range [%D,%D)",i[k],map->input_range[0],map->input_range[1]);
#endif
      }
    }
  } else {
#if defined(PETSC_USE_DEBUG)
    for (k=0; k<N; k++) {
      if (i[k] < map->input_range[0]) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_SUP,"i[%D] %D outside range [%D,%D)",k,i[k],map->input_range[0],map->input_range[1]);
      
      if (i[k] >= map->input_range[1]) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_SUP,"i[%D] %D outside range [%D,%D)",k,i[k],map->input_range[0],map->input_range[1]);
    }
#endif
  }
  
  for (k=0; k<N; k++) {
    j[k] = map->idx[i[k]];
  }

  if (!map->negative_output_allowed) {
    if (j[k] < 0) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_SUP,"i[%D] %D maps to negative value %D : DIMap config does not allow negative output",k,i,j[k]);
  }
  
  PetscFunctionReturn(0);
}
