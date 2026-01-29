/*@ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 **
 **    Copyright (c) 2012
 **        Dave A. May [dave.may@erdw.ethz.ch]
 **        Institute of Geophysics
 **        ETH Zürich
 **        Sonneggstrasse 5
 **        CH-8092 Zürich
 **        Switzerland
 **
 **    project:    pTatin3d
 **    filename:   phase_map.c
 **
 **
 **    pTatin3d is free software: you can redistribute it and/or modify
 **    it under the terms of the GNU General Public License as published
 **    by the Free Software Foundation, either version 3 of the License,
 **    or (at your option) any later version.
 **
 **    pTatin3d is distributed in the hope that it will be useful,
 **    but WITHOUT ANY WARRANTY; without even the implied warranty of
 **    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 **    See the GNU General Public License for more details.
 **
 **    You should have received a copy of the GNU General Public License
 **    along with pTatin3d. If not, see <http://www.gnu.org/licenses/>.
 **
 ** ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ @*/

#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"

#include "ptatin3d.h"
#include "ptatin3d_defs.h"
#include "phase_map.h"
#include "private/ptatin_impl.h"

void PhaseMapCreate(PhaseMap *map)
{
  PhaseMap pm;
  pm = malloc(sizeof(struct _p_PhaseMap));
  memset(pm,0,sizeof(struct _p_PhaseMap));
  *map = pm;
}

void PhaseMapDestroy(PhaseMap *map)
{
  PhaseMap pm;

  if (map==NULL) { return; }
  pm = *map;

  if (pm->data!=NULL) {
    free(pm->data);
    pm->data = NULL;
  }
  *map = NULL;
}

void PhaseMapGetIndex(PhaseMap pm,const int i,const int j, int *index)
{
  if (i<0) { printf("ERROR(%s): i = %d  <0 \n", __func__, i ); exit(EXIT_FAILURE); }
  if (j<0) { printf("ERROR(%s): j = %d < 0 \n", __func__, j ); exit(EXIT_FAILURE); }
  if (i>=pm->mx) { printf("ERROR(%s): i = %d > %d\n", __func__, i, pm->mx ); exit(EXIT_FAILURE); }
  if (j>=pm->my) { printf("ERROR(%s): j = %d > %d\n", __func__, j, pm->my ); exit(EXIT_FAILURE); }

  *index = i + j * pm->mx;
}

void PhaseMapLoadFromFile_ASCII(const char filename[],PhaseMap *map)
{
  FILE *fp = NULL;
  PhaseMap phasemap;
  char dummy[1000];
  int i,j;
  int index;

  /* open file to parse */
  fp = fopen(filename,"r");
  if (fp==NULL) {
    printf("Error(%s): Could not open file: %s \n",__func__, filename );
    exit(EXIT_FAILURE);
  }

  /* create data structure */
  PhaseMapCreate(&phasemap);

  /* read header information, mx,my,x0,y0,x1,y1 */
  //  fscanf(fp,"%s\n",dummy);
  if (!fgets(dummy,sizeof(dummy),fp)) {printf("fgets() failed. Exiting ungracefully.\n");exit(1);}
  if (fscanf(fp,"%d\n",&phasemap->mx) < 1) {printf("fscanf() failed. Exiting ungracefully1.\n");exit(1);}
  if (fscanf(fp,"%d\n",&phasemap->my) < 1) {printf("fscanf() failed. Exiting ungracefully2.\n");exit(1);}
  if (fscanf(fp,"%lf %lf %lf %lf\n",&phasemap->x0,&phasemap->y0,&phasemap->x1,&phasemap->y1) < 4) {printf("fscanf() failed. Exiting ungracefully3.\n");exit(1);}
  //
  phasemap->dx = (phasemap->x1 - phasemap->x0)/(double)(phasemap->mx);
  phasemap->dy = (phasemap->y1 - phasemap->y0)/(double)(phasemap->my);

  /* allocate data */
  phasemap->data = malloc( sizeof(double)* phasemap->mx * phasemap->my );

  /* parse phase map from file */
  index = 0;
  for (j=0; j<phasemap->my; j++) {
    for (i=0; i<phasemap->mx; i++) {
      fscanf(fp,"%lf ",&phasemap->data[index])  ;
      //printf("%f \n", phasemap->data[index]);
      index++;
    }
  }
  /* set pointer */
  *map = phasemap;
  fclose(fp);
}

int PhaseMapLoadFromFile_ASCII_ZIPPED(const char filename[],PhaseMap *map)
{
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Ascii zipped loading is not supported");
}

void PhaseMapLoadFromFile(const char filename[],PhaseMap *map)
{
  size_t len;
  int is_zipped;
  int matched_extension;

  is_zipped = 0;

  /* check extensions for common zipped file extensions */
  len = strlen(filename);
  matched_extension = strcmp(&filename[len-8],".tar.gz");
  if (matched_extension == 0) {
    printf("  Detected .tar.gz\n");
    is_zipped = 1;
  }
  matched_extension = strcmp(&filename[len-5],".tgz");
  if (matched_extension == 0) {
    printf("  Detected .tgz\n");
    is_zipped = 1;
  }
  matched_extension = strcmp(&filename[len-3],".Z");
  if (matched_extension == 0) {
    printf("  Detected .Z\n");
    is_zipped = 1;
  }

  if (is_zipped == 1) {
    PhaseMapLoadFromFile_ASCII_ZIPPED(filename,map);
  } else {
    PhaseMapLoadFromFile_ASCII(filename,map);
  }
}

void PhaseMapGetValue(PhaseMap phasemap,double xp[],double *val)
{
  int i,j,index;

  (*val) = (double)PHASE_MAP_POINT_OUTSIDE;

  if (xp[0] < phasemap->x0) { return; }
  if (xp[0] > phasemap->x1) { return; }
  if (xp[1] < phasemap->y0) { return; }
  if (xp[1] > phasemap->y1) { return; }

  i = (xp[0] - phasemap->x0)/phasemap->dx;
  j = (xp[1] - phasemap->y0)/phasemap->dy;
  if (i==phasemap->mx) { i--; }
  if (j==phasemap->my) { j--; }

  PhaseMapGetIndex(phasemap,i,j,&index);

  *val = phasemap->data[index];
}



/*

 gnuplot> set pm3d map
 gnuplot> splot "filename"

 */
void PhaseMapViewGnuplot(const char filename[],PhaseMap phasemap)
{
  FILE *fp = NULL;
  int i,j;

  /* open file to parse */
  fp = fopen(filename,"w");
  if (fp==NULL) {
    printf("Error(%s): Could not open file: %s \n",__func__, filename );
    exit(EXIT_FAILURE);
  }
  fprintf(fp,"# Phase map information \n");
  fprintf(fp,"# Phase map : (x0,y0) = (%lf,%lf) \n",phasemap->x0,phasemap->y0);
  fprintf(fp,"# Phase map : (x1,y1) = (%lf,%lf) \n",phasemap->x1,phasemap->y1);
  fprintf(fp,"# Phase map : (dx,dy) = (%lf,%lf) \n",phasemap->dx,phasemap->dy);
  fprintf(fp,"# Phase map : (mx,my) = (%d,%d) \n",phasemap->mx,phasemap->my);
 

  for (j=0; j<phasemap->my; j++) {
    for (i=0; i<phasemap->mx; i++) {
      double x,y;
      int index;

      x = phasemap->x0 + phasemap->dx * 0.5 + i * phasemap->dx;
      y = phasemap->y0 + phasemap->dy * 0.5 + j * phasemap->dy;
      PhaseMapGetIndex(phasemap,i,j,&index);

      fprintf(fp,"%lf %lf %lf \n", x,y,phasemap->data[index]);
    }fprintf(fp,"\n");
  }
  fclose(fp);
}





PetscErrorCode pTatinScalePhaseMap(PhaseMap phasemap,PetscScalar value_bar,PetscScalar y_bar,PetscScalar x_bar)
{
  PetscInt i,j,index; 
  PetscErrorCode ierr;

  phasemap->dy = phasemap->dy/y_bar;
  phasemap->y0 = phasemap->y0/y_bar;
  phasemap->y1 = phasemap->y1/y_bar;
  phasemap->dx = phasemap->dx/x_bar; 
  phasemap->x0 = phasemap->x0/x_bar; 
  phasemap->x1 = phasemap->x1/x_bar; 
  index = 0;
  for (j=0; j<phasemap->my; j++) {
    for (i=0; i<phasemap->mx; i++) {
      phasemap->data[index] = phasemap->data[index] /value_bar  ;
      index++;
    }
  }
  PetscFunctionReturn(0);
}





