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
 **    filename:   stokes_operators_tensor.c
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
// -*- indent-tabs-mode:t c-basic-offset:8 -*-

#include <petscfe.h>
#include <ptatin3d.h>
#include <ptatin3d_stokes.h>
#include <dmda_element_q2p1.h>
#include <stokes_operators.h>

extern PetscLogEvent MAT_MultMFA11_setup;
extern PetscLogEvent MAT_MultMFA11_copyto;
extern PetscLogEvent MAT_MultMFA11_kernel;
extern PetscLogEvent MAT_MultMFA11_copyfrom;
extern PetscLogEvent MAT_MultMFA11_merge;

typedef struct _p_MFA11CUDA *MFA11CUDA;

struct _p_MFA11CUDA {
  PetscObjectState state;

  PetscScalar *ufield;
  PetscReal   *LA_gcoords;
  PetscReal   *gaussdata;
  PetscInt    *elnidx_u;
  PetscScalar *Yu;
  PetscReal   *D;
  PetscReal   *B;
  PetscReal   *w;
};


template< typename T >
void check(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d \n",
                file, line, result);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}
#define CUDACHECK(val)       check( (val), #val, __FILE__, __LINE__ )


#define NQP 27			/* Number of quadrature points per element; must equal Q2_NODES_PER_EL_3D (27) */

#define WARPS_PER_BLOCK    4


typedef enum {
	GRAD,
	GRAD_TRANSPOSE
} GradMode;


__device__ double atomicAdd_double(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}


/*
 * Performs three tensor contractions: y[l,a,b,c] += T[a,k] S[b,j] R[c,i] x[l,k,j,i]
 */
static __device__ void TensorContract(PetscReal const *Rf,PetscReal const *Sf,PetscReal const *Tf,GradMode gmode,PetscReal const x[],PetscReal y[])
{
  PetscReal R[3][3],S[3][3],T[3][3];
  __shared__ PetscReal u[WARPS_PER_BLOCK][3][NQP],v[WARPS_PER_BLOCK][3][NQP];
  PetscInt i,j,k,l,kj,ji,a,b,c;
  PetscInt warp_in_block = threadIdx.x / 32;
  PetscInt id_in_warp = threadIdx.x % 32;

  for (j=0; j<3; j++) {
    for (i=0; i<3; i++) {
      R[i][j] = (gmode == GRAD ? Rf[3*i+j] : Rf[3*j + i]);
      S[i][j] = (gmode == GRAD ? Sf[3*i+j] : Sf[3*j + i]);
      T[i][j] = (gmode == GRAD ? Tf[3*i+j] : Tf[3*j + i]);
    }
  }

	// u[l,k,j,c] = R[c,i] x[l,k,j,i]
    for (i=0; i<3; ++i) {
      u[warp_in_block][i][id_in_warp] = 0;
      v[warp_in_block][i][id_in_warp] = x[i];
    }

    kj = id_in_warp / 3;
    c = id_in_warp % 3;
	for (l=0; l<3; l++) {
		//for (kj=0; kj<9; kj++) {
			for (i=0; i<3; i++) {
				//for (c=0; c<3; c++) {
					u[warp_in_block][l][id_in_warp] += R[c][i] * v[warp_in_block][l][kj*3+i];
				//}
			}
		//}
	}

	// v[l,k,b,c] = S[b,j] u[l,k,j,c]
    for (i=0; i<3; ++i) {
      v[warp_in_block][i][id_in_warp] = 0;
    }
    k = id_in_warp / 9;
    b = (id_in_warp % 9) / 3;
    c = id_in_warp % 3;
	for (l=0; l<3; l++) {
		//for (k=0; k<3; k++) {
			for (j=0; j<3; j++) {
				//for (c=0; c<3; c++) {
					//for (b=0; b<3; b++) {
						v[warp_in_block][l][id_in_warp] += S[b][j] * u[warp_in_block][l][(k*3+j)*3+c];
					//}
				//}
			}
		//}
	}

	// y[l,a,b,c] = T[a,k] v[l,k,b,c]
    a = id_in_warp / 9;
    ji = id_in_warp % 9;
	for (l=0; l<3; l++) {
		for (k=0; k<3; k++) {
			//for (a=0; a<3; a++) {
				//for (ji=0; ji<9; ji++) {
					y[l] += T[a][k] * v[warp_in_block][l][k*9+ji];
				//}
			//}
		}
	}
}

static __device__ void JacobianInvert(PetscScalar dx[3][3],PetscScalar &dxdet)
{
	PetscInt j,k;

		PetscScalar a[3][3];
		PetscScalar b0,b3,b6,det,idet;
		for (j=0; j<3; j++) {
			for (k=0; k<3; k++) {
				a[j][k] = dx[j][k];
			}
		}
		b0 =  (a[1][1]*a[2][2] - a[2][1]*a[1][2]);
		b3 = -(a[1][0]*a[2][2] - a[2][0]*a[1][2]);
		b6 =  (a[1][0]*a[2][1] - a[2][0]*a[1][1]);
		det = a[0][0]*b0 + a[0][1]*b3 + a[0][2]*b6;
		idet = 1.0 / det;
		dx[0][0] =  idet*b0;
		dx[0][1] = -idet*(a[0][1]*a[2][2] - a[2][1]*a[0][2]);
		dx[0][2] =  idet*(a[0][1]*a[1][2] - a[1][1]*a[0][2]);
		dx[1][0] =  idet*b3;
		dx[1][1] =  idet*(a[0][0]*a[2][2] - a[2][0]*a[0][2]);
		dx[1][2] = -idet*(a[0][0]*a[1][2] - a[1][0]*a[0][2]);
		dx[2][0] =  idet*b6;
		dx[2][1] = -idet*(a[0][0]*a[2][1] - a[2][0]*a[0][1]);
		dx[2][2] =  idet*(a[0][0]*a[1][1] - a[1][0]*a[0][1]);
		dxdet = det;
}

static __device__ void QuadratureAction(PetscScalar gaussdata_eta,
				       PetscScalar const dx[3][3],
				       PetscScalar dxdet,
				       PetscReal w,
				       PetscScalar const du[3][3],
				       PetscScalar dv[3][3])
{
	PetscInt l,k;

		PetscScalar Du[6],Dv[6]; /* Symmetric gradient with respect to physical coordinates, xx, yy, zz, xy+yx, xz+zx, yz+zy */

		PetscScalar dux[3][3];
		for (l=0; l<3; l++) { // fields
			for (k=0; k<3; k++) { // directions
				dux[k][l] = du[0][l] * dx[k][0] + du[1][l] * dx[k][1] + du[2][l] * dx[k][2];
			}
		}
		Du[0] = dux[0][0];
		Du[1] = dux[1][1];
		Du[2] = dux[2][2];
		Du[3] = 0.5*(dux[0][1] + dux[1][0]);
		Du[4] = 0.5*(dux[0][2] + dux[2][0]);
		Du[5] = 0.5*(dux[1][2] + dux[2][1]);

		for (k=0; k<6; k++) { /* Stress is coefficient of test function */
			Dv[k] = 2 * gaussdata_eta * Du[k];
		}

		PetscScalar dvx[3][3];
		dvx[0][0] = Dv[0];
		dvx[0][1] = Dv[3];
		dvx[0][2] = Dv[4];
		dvx[1][0] = Dv[3];
		dvx[1][1] = Dv[1];
		dvx[1][2] = Dv[5];
		dvx[2][0] = Dv[4];
		dvx[2][1] = Dv[5];
		dvx[2][2] = Dv[2];

		for (l=0; l<3; l++) { // fields
			for (k=0; k<3; k++) { // directions
				dv[k][l] = w * dxdet * (dvx[0][l] * dx[0][k] + dvx[1][l] * dx[1][k] + dvx[2][l] * dx[2][k]);
			}
		}
}

static __global__ void MFStokesWrapper_A11_CUDA_kernel(PetscInt nel,PetscInt nen_u,PetscInt const *elnidx_u,PetscReal const *LA_gcoords,PetscScalar const *ufield,PetscReal const *gaussdata,PetscScalar *Yu,
                                                          PetscReal const *D_global,PetscReal const *B_global,PetscReal const *w_global)
{
	PetscScalar el_uxv[3]; // unifies elu, elx, elv
	PetscScalar dx[3][3]={0},du[3][3]={0},dv[3][3]={0};
    PetscScalar dxdet = 0;
	PetscInt i,j,l;
    PetscInt elidx = (blockDim.x * blockIdx.x + threadIdx.x) / 32;  // one warp per element
    PetscInt id_in_warp = threadIdx.x % 32;
    PetscInt E = elnidx_u[nen_u*elidx+id_in_warp];
    __shared__ PetscReal D[3*3], B[3*3];

    if (threadIdx.x <  9) B[threadIdx.x] = B_global[threadIdx.x];
    if (threadIdx.x <  9) D[threadIdx.x] = D_global[threadIdx.x];

    __syncthreads();

    if (elidx >= nel)
      return;

	if (id_in_warp < Q2_NODES_PER_EL_3D) {

      for (l=0; l<3; l++) el_uxv[l] = LA_gcoords[3*E+l];
	  TensorContract(D,B,B,GRAD,el_uxv,dx[0]);
	  TensorContract(B,D,B,GRAD,el_uxv,dx[1]);
	  TensorContract(B,B,D,GRAD,el_uxv,dx[2]);

	  JacobianInvert(dx,dxdet);

      for (l=0; l<3; l++) el_uxv[l] = ufield[3*E+l];
	  TensorContract(D,B,B,GRAD,el_uxv,du[0]);
	  TensorContract(B,D,B,GRAD,el_uxv,du[1]);
	  TensorContract(B,B,D,GRAD,el_uxv,du[2]);

	  QuadratureAction(gaussdata[elidx*NQP + id_in_warp],dx,dxdet,w_global[id_in_warp],du,dv);

      for (l=0; l<3; l++) el_uxv[l] = 0;
	  TensorContract(D,B,B,GRAD_TRANSPOSE,dv[0],el_uxv);
	  TensorContract(B,D,B,GRAD_TRANSPOSE,dv[1],el_uxv);
	  TensorContract(B,B,D,GRAD_TRANSPOSE,dv[2],el_uxv);

      for (l=0; l<3; l++) {
        atomicAdd_double(Yu + 3*E+l, el_uxv[l]);
      }
    }
}

static __global__ void set_zero_CUDA_kernel(PetscScalar *Yu, PetscInt localsize)
{
   for (PetscInt i = blockDim.x * blockIdx.x + threadIdx.x; i<localsize; i += blockDim.x * gridDim.x)
     Yu[i] = 0;
}

extern "C" {

#undef __FUNCT__
#define __FUNCT__ "MFA11SetUp_CUDA"
PetscErrorCode MFA11SetUp_CUDA(MatA11MF mf)
{
  PetscErrorCode ierr;
  MFA11CUDA      ctx;
  PetscReal      x1[3],w1[3],B[3][3],D[3][3],w[NQP];
  PetscInt       i,j,k;

  PetscFunctionBegin;
  if (mf->ctx) PetscFunctionReturn(0);
  ierr = PetscMalloc1(1,&ctx);CHKERRQ(ierr);
  ctx->state = 0;

  ctx->ufield     = NULL;
  ctx->LA_gcoords = NULL;
  ctx->gaussdata  = NULL;
  ctx->elnidx_u   = NULL;
  ctx->Yu         = NULL;

  ierr = PetscDTGaussQuadrature(3,-1,1,x1,w1);CHKERRQ(ierr);
  for (i=0; i<3; i++) {
    B[i][0] = .5*(PetscSqr(x1[i]) - x1[i]);
    B[i][1] = 1 - PetscSqr(x1[i]);
    B[i][2] = .5*(PetscSqr(x1[i]) + x1[i]);
    D[i][0] = x1[i] - .5;
    D[i][1] = -2*x1[i];
    D[i][2] = x1[i] + .5;
  }
  for (i=0; i<3; i++)
    for (j=0; j<3; j++)
      for (k=0; k<3; k++)
        w[(i*3+j)*3+k] = w1[i] * w1[j] * w1[k];

  ierr = cudaMalloc(&ctx->D,  3 * 3 * sizeof(PetscReal));CUDACHECK(ierr);
  ierr = cudaMemcpy(ctx->D,D, 3 * 3 * sizeof(PetscReal),cudaMemcpyHostToDevice);CUDACHECK(ierr);

  ierr = cudaMalloc(&ctx->B,  3 * 3 * sizeof(PetscReal));CUDACHECK(ierr);
  ierr = cudaMemcpy(ctx->B,B, 3 * 3 * sizeof(PetscReal),cudaMemcpyHostToDevice);CUDACHECK(ierr);

  ierr = cudaMalloc(&ctx->w,  3 * 3 * 3 * sizeof(PetscReal));CUDACHECK(ierr);
  ierr = cudaMemcpy(ctx->w,w, 3 * 3 * 3 * sizeof(PetscReal),cudaMemcpyHostToDevice);CUDACHECK(ierr);

  mf->ctx = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFA11Destroy_CUDA"
PetscErrorCode MFA11Destroy_CUDA(MatA11MF mf)
{
  PetscErrorCode ierr;
  MFA11CUDA      ctx;

  PetscFunctionBegin;
  ctx = (MFA11CUDA)mf->ctx;
  if (!ctx) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"CUDA MF-SpMV implementation should have a valid context");
  /* Free internal members */
  ierr = cudaFree(ctx->ufield);CUDACHECK(ierr);
  ierr = cudaFree(ctx->LA_gcoords);CUDACHECK(ierr);
  ierr = cudaFree(ctx->gaussdata);CUDACHECK(ierr);
  ierr = cudaFree(ctx->elnidx_u);CUDACHECK(ierr);
  ierr = cudaFree(ctx->Yu);CUDACHECK(ierr);
  ierr = cudaFree(ctx->D);CUDACHECK(ierr);
  ierr = cudaFree(ctx->B);CUDACHECK(ierr);
  ierr = cudaFree(ctx->w);CUDACHECK(ierr);
  /* Free context */
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  mf->ctx = NULL;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFStokesWrapper_A11_CUDA"
PetscErrorCode MFStokesWrapper_A11_CUDA(MatA11MF mf,Quadrature volQ,DM dau,PetscScalar ufield[],PetscScalar Yu[])
{
	PetscErrorCode ierr;
	DM cda;
	Vec gcoords;
	const PetscReal *LA_gcoords;
	PetscInt nel,nen_u,e,i,localsize;
	const PetscInt *elnidx_u;
	QPntVolCoefStokes *all_gausspoints;
	const QPntVolCoefStokes *cell_gausspoints;
    MFA11CUDA      cudactx = (MFA11CUDA)mf->ctx;

	PetscFunctionBegin;
	ierr = PetscLogEventBegin(MAT_MultMFA11_setup,0,0,0,0);CHKERRQ(ierr);

	/* setup for coords */
	ierr = DMGetCoordinateDM( dau, &cda);CHKERRQ(ierr);
	ierr = DMGetCoordinatesLocal( dau,&gcoords );CHKERRQ(ierr);
	ierr = VecGetArrayRead(gcoords,&LA_gcoords);CHKERRQ(ierr);
    ierr = VecGetLocalSize(gcoords,&localsize);CHKERRQ(ierr);

	ierr = DMDAGetElements_pTatinQ2P1(dau,&nel,&nen_u,&elnidx_u);CHKERRQ(ierr);

	ierr = VolumeQuadratureGetAllCellData_Stokes(volQ,&all_gausspoints);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_MultMFA11_setup,0,0,0,0);CHKERRQ(ierr);

    /* Set up CUDA data */
	ierr = PetscLogEventBegin(MAT_MultMFA11_copyto,0,0,0,0);CHKERRQ(ierr);

    if (!cudactx->elnidx_u) {
      ierr = cudaMalloc(&cudactx->elnidx_u,        nel * nen_u * sizeof(PetscInt));CUDACHECK(ierr);
      ierr = cudaMemcpy(cudactx->elnidx_u,elnidx_u,nel * nen_u * sizeof(PetscInt),cudaMemcpyHostToDevice);CUDACHECK(ierr);
    }

    if (!cudactx->ufield) {
      ierr = cudaMalloc(&cudactx->ufield, localsize * sizeof(PetscScalar));CUDACHECK(ierr);
    }
    /* ufield always needs to be copied */
    ierr = cudaMemcpy(cudactx->ufield,ufield, localsize * sizeof(PetscScalar),cudaMemcpyHostToDevice);CUDACHECK(ierr);

    if (!cudactx->LA_gcoords) {
      ierr = cudaMalloc(&cudactx->LA_gcoords, localsize * sizeof(PetscReal));CUDACHECK(ierr);
    }
    
    if (!cudactx->gaussdata) {
      ierr = cudaMalloc(&cudactx->gaussdata,nel * NQP * sizeof(PetscReal));CUDACHECK(ierr);
    }

    if (mf->state != cudactx->state) {
      ierr = cudaMemcpy(cudactx->LA_gcoords,LA_gcoords, localsize * sizeof(PetscReal),cudaMemcpyHostToDevice);CUDACHECK(ierr);

      PetscReal *gaussdata_host;
      ierr = PetscMalloc(nel * NQP * sizeof(PetscReal), &gaussdata_host);CHKERRQ(ierr);
      for (e=0; e<nel; e++) {
        ierr = VolumeQuadratureGetCellData_Stokes(volQ,all_gausspoints,e,(QPntVolCoefStokes**)&cell_gausspoints);CHKERRQ(ierr);
        for (i=0; i<NQP; i++) gaussdata_host[e*NQP + i] = cell_gausspoints[i].eta;
      }
      ierr = cudaMemcpy(cudactx->gaussdata,gaussdata_host, nel * NQP * sizeof(PetscReal),cudaMemcpyHostToDevice);CUDACHECK(ierr);
      ierr = PetscFree(gaussdata_host);CHKERRQ(ierr);

      /* Save new state to avoid unnecessary subsequent copies */
      cudactx->state = mf->state;
    }

    if (!cudactx->Yu) {
      ierr = cudaMalloc(&cudactx->Yu, localsize * sizeof(PetscScalar));CUDACHECK(ierr);
    }

    ierr = cudaDeviceSynchronize();CUDACHECK(ierr);
    ierr = PetscLogEventEnd(MAT_MultMFA11_copyto,0,0,0,0);CHKERRQ(ierr);

    /* CUDA entry point
     *  - inputs: elnidx_u, LA_gcoords, ufield, gaussdata
     *  - output: Yu
     */
	ierr = PetscLogEventBegin(MAT_MultMFA11_kernel,0,0,0,0);CHKERRQ(ierr);
    set_zero_CUDA_kernel<<<256,256>>>(cudactx->Yu, localsize);
    MFStokesWrapper_A11_CUDA_kernel<<<(nel-1)/WARPS_PER_BLOCK + 1, WARPS_PER_BLOCK*32>>>(nel,nen_u,cudactx->elnidx_u,cudactx->LA_gcoords,cudactx->ufield,cudactx->gaussdata,cudactx->Yu,cudactx->D,cudactx->B,cudactx->w);
    ierr = cudaDeviceSynchronize();CUDACHECK(ierr);
    ierr = PetscLogEventEnd(MAT_MultMFA11_kernel,0,0,0,0);CHKERRQ(ierr);

    PetscLogFlops((nel * 9) * 3*NQP*(6+6+6));           /* 9 tensor contractions per element */
    PetscLogFlops(nel*NQP*(14 + 1/* division */ + 27)); /* 1 Jacobi inversion per element */
    PetscLogFlops((nel * 9) * 3*NQP*(6+6+6));           /* 1 quadrature action per element */

    /* Read back CUDA data */
	ierr = PetscLogEventBegin(MAT_MultMFA11_copyfrom,0,0,0,0);CHKERRQ(ierr);
    ierr = cudaMemcpy(Yu,cudactx->Yu,localsize * sizeof(PetscScalar),cudaMemcpyDeviceToHost);CUDACHECK(ierr);
    ierr = PetscLogEventEnd(MAT_MultMFA11_copyfrom,0,0,0,0);CHKERRQ(ierr);

	ierr = VecRestoreArrayRead(gcoords,&LA_gcoords);CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

} /* extern C */
