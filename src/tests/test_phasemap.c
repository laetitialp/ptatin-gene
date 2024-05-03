
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
#include "ptatin_init.h"

// int main(void)
// {
//   PhaseMap phasemap;
//   double xp[2];
//   double dens;

//   //  PhaseMapLoadFromFile("test.bmp",&phasemap);
//   PhaseMapLoadFromFile("model_geometry",&phasemap);

//   xp[0] = 0.0;  xp[1] = 1.0;
//   PhaseMapGetDensity(phasemap,xp,&dens);
//   printf("x = ( %lf , %lf ) ==> phase = %f \n", xp[0],xp[1],dens);

//   xp[0] = 5.0;  xp[1] = 3.2;
//   PhaseMapGetDensity(phasemap,xp,&dens);
//   printf("x = ( %lf , %lf ) ==> density = %f \n", xp[0],xp[1],dens);

//   xp[0] = -1.0; xp[1] = 1.0;
//   PhaseMapGetDensity(phasemap,xp,&dens);
//   printf("x = ( %lf , %lf ) ==> phase = %f \n", xp[0],xp[1],dens);

//   PhaseMapViewGnuplot("test.gp",phasemap);

//   PhaseMapDestroy(&phasemap);

// }





static PetscErrorCode pTatin3d_StoreMaps(int argc,char **argv)
{
 pTatinCtx       ctx;
 PhaseMap phasemap ,phasemap2;
 PetscErrorCode  ierr;

  PetscFunctionBegin;
  
  ierr = pTatin3dCreateContext(&ctx);CHKERRQ(ierr);
  ierr = pTatin3dSetFromOptions(ctx);CHKERRQ(ierr);
  PhaseMapLoadFromFile("model_geometry",&phasemap);
  PhaseMapViewGnuplot("test1.gp",phasemap);
  
  ierr = pTatinScalePhaseMap(phasemap,1e-3,1e9,1.0);CHKERRQ(ierr);
  ierr = pTatinCtxAttachPhaseMap(ctx,phasemap, "ma_map"); CHKERRQ(ierr);

  ierr= pTatinCtxGetPhaseMap(ctx,&phasemap2, "ma_map"); CHKERRQ(ierr);
  PhaseMapViewGnuplot("test2.gp",phasemap2);

   PhaseMapDestroy(&phasemap);
   PhaseMapDestroy(&phasemap2);
 
  ierr = pTatin3dDestroyContext(&ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;

  ierr = pTatinInitialize(&argc,&argv,0,NULL);CHKERRQ(ierr);

  ierr = pTatin3d_StoreMaps(argc,argv);CHKERRQ(ierr);

  ierr = pTatinFinalize();CHKERRQ(ierr);
  return 0;
}
