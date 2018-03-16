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
 **    filename:   ptatin_utils.h
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

#ifndef __ptatin3d_ptatin_utils_h__
#define __ptatin3d_ptatin_utils_h__

PetscErrorCode pTatinCreateDirectory(const char dirname[]);
PetscErrorCode pTatinWriteOptionsFile(const char filename[]);
void pTatinGenerateFormattedTimestamp(char date_time[]);
void FileExists(const char *fname,int *exists);
void FileExistsRank(MPI_Comm comm,const char fname[],int *exists);
int StringEmpty(const char string[]);

void ptatin_RandomNumberSetSeed(unsigned seed);
void ptatin_RandomNumberSetSeedRank(MPI_Comm comm);
double ptatin_RandomNumberGetDouble(double min,double max);
int ptatin_RandomNumberGetInt(int min,int max);

PetscErrorCode pTatinGetRangeMaximumMemoryUsage(PetscReal range[]);
PetscErrorCode pTatinGetRangeCurrentMemoryUsage(PetscReal range[]);
PetscErrorCode pTatinTestDirectory(const char dirname[],char mode,PetscBool *_exists);
PetscErrorCode pTatinTestFile(const char filename[],char mode,PetscBool *_exists);

#endif
