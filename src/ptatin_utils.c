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
 **    filename:   ptatin_utils.c
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

#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include <pwd.h>


#include "petsc.h"
#include "ptatin_version_info.h"
#include "ptatin_utils.h"

PetscErrorCode pTatinCreateDirectory(const char dirname[])
{
	PetscMPIInt rank;
	int num,error_number;
	PetscErrorCode ierr;
	
	PetscFunctionBegin;
	ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
	
	/* Generate a new directory on proc 0 */
	if (rank == 0) {
		num = mkdir(dirname,S_IRWXU);
		error_number = errno;
	}
	ierr = MPI_Bcast(&num,1,MPI_INT,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
	ierr = MPI_Bcast(&error_number,1,MPI_INT,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
	
	if (error_number == EEXIST) {
		PetscPrintf(PETSC_COMM_WORLD,"[pTatin] Writing output to existing directory: %s \n",dirname);
	} else if (error_number == EACCES) {
		SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"[pTatin] Write permission is denied for the parent directory in which the new directory is to be added");
	} else if (error_number == EMLINK) {
		SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"[pTatin] The parent directory has too many links (entries)");
	} else if (error_number == ENOSPC) {
		SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"[pTatin] The file system doesn't have enough room to create the new directory");
	} else if (error_number == ENOSPC) {
		SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"[pTatin] The parent directory of the directory being created is on a read-only file system and cannot be modified");
	} else {
		PetscPrintf(PETSC_COMM_WORLD,"[pTatin] Created output directory: %s \n",dirname);
	}
	
	ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
	PetscFunctionReturn(0);
}

/* don't write out options this way, they cannot be loaded from file */
/*
 ierr = PetscOptionsGetAll(&copts);CHKERRQ(ierr);
 PetscPrintf(PETSC_COMM_WORLD,"All opts: %s \n", copts);
 ierr = PetscOptionsInsertFile(PETSC_COMM_WORLD,"test.opts",PETSC_FALSE);CHKERRQ(ierr);
 */
PetscErrorCode pTatinWriteOptionsFile(const char filename[])
{
	PetscViewer viewer;
	char username[PETSC_MAX_PATH_LEN];
	char date[PETSC_MAX_PATH_LEN];
	char machine[PETSC_MAX_PATH_LEN];
	char prgname[PETSC_MAX_PATH_LEN];
	PetscErrorCode ierr;
	
	if (!filename) {
		ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"ptatin.options",&viewer);CHKERRQ(ierr);
	} else {
		ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,filename,&viewer);CHKERRQ(ierr);
	}
	
	/* write header into options file */
	ierr = PetscGetUserName(username,PETSC_MAX_PATH_LEN-1);CHKERRQ(ierr);
	ierr = PetscGetDate(date,PETSC_MAX_PATH_LEN-1);CHKERRQ(ierr);
	ierr = PetscGetHostName(machine,PETSC_MAX_PATH_LEN-1);CHKERRQ(ierr);
	ierr = PetscGetProgramName(prgname,PETSC_MAX_PATH_LEN-1);CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(viewer,"## =================================================================================== \n");CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(viewer,"##\n");CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(viewer,"##   pTatin3d Options File\n");CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(viewer,"##     %s \n", PTATIN_VERSION_CNTR_REPO);
#ifdef PTATIN_DEVELOPMENT_VERSION
	ierr = PetscViewerASCIIPrintf(viewer,"##     %s \n", PTATIN_VERSION_CNTR_REVISION);
	ierr = PetscViewerASCIIPrintf(viewer,"##     %s \n", PTATIN_VERSION_CNTR_LOG);
  #ifdef PTATIN_GIT_REPO_STATUS
  ierr = PetscViewerASCIIPrintf(viewer,"##     %s \n", PTATIN_GIT_REPO_STATUS);
  #endif
#endif
#ifdef PTATIN_RELEASE
	ierr = PetscViewerASCIIPrintf(viewer,"##     %s \n", PTATIN_VERSION_CNTR_REVISION);
	ierr = PetscViewerASCIIPrintf(viewer,"##     Release v%d.%d-p%d  \n", PTATIN_VERSION_MAJOR,PTATIN_VERSION_MINOR,PTATIN_VERSION_PATCH);
#endif
	ierr = PetscViewerASCIIPrintf(viewer,"##\n");CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(viewer,"##   Generated by user: %s\n",username);CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(viewer,"##   Date             : %s\n",date);CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(viewer,"##   Machine          : %s\n",machine);CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(viewer,"##   Driver           : %s\n",prgname);CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(viewer,"##\n");CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(viewer,"## =================================================================================== \n");CHKERRQ(ierr);
	
	/* write options */
	ierr = PetscOptionsView(NULL,viewer);CHKERRQ(ierr);
	
	ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
	PetscPrintf(PETSC_COMM_WORLD,"[pTatin] Created options file: %s \n",filename);
	
	PetscFunctionReturn(0);
}

void pTatinGenerateFormattedTimestamp(char date_time[])
{
	time_t      currTime;
	struct tm*  timeInfo;
	int         adjustedYear;
	int         adjustedMonth;
	
	currTime = time( NULL );
	timeInfo = localtime( &currTime );
	/* See man localtime() for why to adjust these */
	adjustedYear = 1900 + timeInfo->tm_year;
	adjustedMonth = 1 + timeInfo->tm_mon;
	/* Format; MM(string) DD HH:MM:SS YYYY */	
	/*
	 printf( "%s %.2d %.2d:%.2d:%.2d %.4d \n",  
	 months[adjustedMonth], timeInfo->tm_mday,
	 timeInfo->tm_hour, timeInfo->tm_min, timeInfo->tm_sec, adjustedYear );
	 */
	sprintf( date_time, "%.4d.%.2d.%.2d_%.2d:%.2d:%.2d",  
					adjustedYear, adjustedMonth, timeInfo->tm_mday,
					timeInfo->tm_hour, timeInfo->tm_min, timeInfo->tm_sec );
}

void FileExists(const char fname[],int *exists)
{
	FILE *file = NULL;
	
	file = fopen(fname, "r");
	
	if (file) {
		fclose(file);
		*exists = 1;
	} else {
		*exists = 0;
	}
	MPI_Barrier(PETSC_COMM_WORLD);
}

void FileExistsRank(MPI_Comm comm,const char fname[],int *exists)
{
	int   rank,size;
	char  fname_rank[1024];
	FILE  *file = NULL;
	
	MPI_Comm_size(comm,&size);
	MPI_Comm_rank(comm,&rank);
	if (size == 1) {
		sprintf(fname_rank,"%s",fname);
	} else {
		sprintf(fname_rank,"%s_p%.5d",fname,rank);
	}
	
	file = fopen(fname_rank, "r");
	
	if (file) {
		fclose(file);
		*exists = 1;
	} else {
		*exists = 0;
	}
	MPI_Barrier(PETSC_COMM_WORLD);
}

int StringEmpty(const char string[])
{
	if (string) { /* AND (or &&) */
    if (string[0] == '\0') {
			return 1;
    }
	} else {
		return 1;
	}
	return 0;
}

void ptatin_RandomNumberSetSeed(unsigned seed)
{
	srand(seed);
}
void ptatin_RandomNumberSetSeedRank(MPI_Comm comm)
{
	int rank;
	
	MPI_Comm_rank(comm,&rank);
	srand((unsigned)rank);
}

double ptatin_RandomNumberGetDouble(double min,double max)
{
	double r,rr;
	
	r = rand()/((double)(RAND_MAX));
	rr =  min + (max - min) * r;
	return rr;
}

int ptatin_RandomNumberGetInt(int min,int max)
{
	double r,rr;
	int ri;
	
	r = rand()/((double)(RAND_MAX));
	rr =  ((double)min) + ((double)(max - min)) * r;
	ri = (int)rr;
	return ri;
}

PetscErrorCode pTatinGetRangeMaximumMemoryUsage(PetscReal range[])
{
	PetscErrorCode ierr;
	PetscLogDouble mem;
	double min,max,_mem;
	
	ierr = PetscMallocGetMaximumUsage(&mem);CHKERRQ(ierr);
	_mem = (double)mem;
	ierr = MPI_Allreduce(&_mem,&min,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD);CHKERRQ(ierr);
	ierr = MPI_Allreduce(&_mem,&max,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD);CHKERRQ(ierr);
	
	if (range) {
		range[0] = (PetscReal)min;
		range[1] = (PetscReal)max;
	} else {
		PetscPrintf(PETSC_COMM_WORLD,"pTatin3dMaxMemoryUsage = [%1.4e , %1.4e] (MB) \n",min*1.0e-6,max*1.0e-6);
	}
	
	PetscFunctionReturn(0);
}

PetscErrorCode pTatinGetRangeCurrentMemoryUsage(PetscReal range[])
{
	PetscErrorCode ierr;
	PetscLogDouble mem;
	double min,max,_mem;
	
	ierr = PetscMallocGetCurrentUsage(&mem);CHKERRQ(ierr);
	_mem = (double)mem;
	ierr = MPI_Allreduce(&_mem,&min,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD);CHKERRQ(ierr);
	ierr = MPI_Allreduce(&_mem,&max,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD);CHKERRQ(ierr);
	
	if (range) {
		range[0] = (PetscReal)min;
		range[1] = (PetscReal)max;
	} else {
		PetscPrintf(PETSC_COMM_WORLD,"pTatin3dCurrentMemoryUsage = [%1.4e , %1.4e] (MB) \n",min*1.0e-6,max*1.0e-6);
	}
	
	PetscFunctionReturn(0);
}

PetscErrorCode pTatinTestDirectory(const char dirname[],char mode,PetscBool *_exists)
{
  PetscMPIInt rank;
  int i_exists = 0;
  PetscBool exists = PETSC_FALSE;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  
  if (rank == 0) {
    ierr = PetscTestDirectory(dirname,mode,&exists);CHKERRQ(ierr);
    i_exists = (int)exists;
  }
  ierr = MPI_Bcast(&i_exists,1,MPI_INT,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  
  if (i_exists == 1) { *_exists = PETSC_TRUE; }
  else               { *_exists = PETSC_FALSE; }
  PetscFunctionReturn(0);
}

PetscErrorCode pTatinTestFile(const char filename[],char mode,PetscBool *_exists)
{
  PetscMPIInt rank;
  int i_exists = 0;
  PetscBool exists = PETSC_FALSE;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  
  if (rank == 0) {
    ierr = PetscTestFile(filename,mode,&exists);CHKERRQ(ierr);
    i_exists = (int)exists;
  }
  ierr = MPI_Bcast(&i_exists,1,MPI_INT,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  
  if (i_exists == 1) { *_exists = PETSC_TRUE; }
  else               { *_exists = PETSC_FALSE; }
  PetscFunctionReturn(0);
}
