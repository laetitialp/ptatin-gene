/*@ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 **
 **    Copyright (c) 2023
 **        Dave A. May [dmay@ucsd.edu]
 **
 **    project:    pTatin3d
 **    filename:   stokes_operators_tensor_hip.hip
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

/*
perl hipify-perl stokes_operator_cuda.cu > stokes_operator_hip.hip
"CUDA error at -> "HIP error at
CUDACHECK -> HIPCHECK
CUDA_D -> HIP_D
CUDA_B -> HIP_B
*/

#include <petscfe.h>
#include <ptatin3d.h>
#include <ptatin3d_stokes.h>
#include <dmda_element_q2p1.h>
#include <stokes_operators.h>

#include <hip/hip_runtime.h>


extern PetscLogEvent MAT_MultMFA11_stp;
extern PetscLogEvent MAT_MultMFA11_cto;
extern PetscLogEvent MAT_MultMFA11_ker;
extern PetscLogEvent MAT_MultMFA11_cfr;

/* Constant memory for D and B matrices */
__constant__ PetscReal HIP_D[3*3], HIP_B[3*3];


static PetscBool log_ops = PETSC_FALSE;

template< typename T >
void check(T result, char const *const func, const char *const file, int const line)
{
  if (result)
  {
    fprintf(stderr, "HIP error at %s:%d code=%d (%s)\n",
        file, line, result, hipGetErrorString((hipError_t)result));
    hipDeviceReset();
    exit(EXIT_FAILURE);
  }
}
#define HIPCHECK(val)       check( (val), #val, __FILE__, __LINE__ )

#define WARPS_PER_BLOCK    16

#if defined(USE_ROCM)
  #define WARP_SIZE warpSize // = 64 or 32 (Defined in hip_runtime.h)
#else
  #define WARP_SIZE 32
#endif

/*
Taken from
  https://stackoverflow.com/questions/24095248/cudas-warp-shuffle-for-double-precision-data
*/
#if 0
/*for shuffle of double-precision point */
__device__ __inline__ double shfl_double(double x, int lane)
{
  // Split the double number into 2 32b registers.
  int lo, hi;
  asm volatile("mov.b64 {%0,%1}, %2;":"=r"(lo),"=r"(hi):"d"(x));
  // Shuffle the two 32b registers.
  lo = __shfl(lo,lane,WARP_SIZE);
  hi = __shfl(hi,lane,WARP_SIZE);
  // Recreate the 64b number.
  asm volatile("mov.b64 %0,{%1,%2};":"=d"(x):"r"(lo),"r"(hi));
  return x;
}
#endif

/* 
Note that instruction mov.b64 and the constraint "d" are not supported on the AMD GPU.
For this reason shfl_double() is modified.
Specifically I copied the provided double shuffle implementation from here
/opt/rocm/include/hip/hcc_detail/amd_warp_functions.h
*/

/*
Taken from
 /opt/rocm/include/hip/hcc_detail/amd_warp_functions.h
*/
__device__ inline double shfl_double(double var, int src_lane)
{
    static_assert(sizeof(double) == 2 * sizeof(int), "");
    static_assert(sizeof(double) == sizeof(uint64_t), "");

    int tmp[2]; __builtin_memcpy(tmp, &var, sizeof(tmp));
    tmp[0] = __shfl(tmp[0], src_lane, WARP_SIZE);
    tmp[1] = __shfl(tmp[1], src_lane, WARP_SIZE);

    uint64_t tmp0 = (static_cast<uint64_t>(tmp[1]) << 32ull) | static_cast<uint32_t>(tmp[0]);
    double tmp1;  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
    return tmp1;
}

/*
 * Performs three tensor contractions: y[l,a,b,c] += T[a,k] S[b,j] R[c,i] x[l,k,j,i]
 */
static __device__ void TensorContract(PetscReal const *R,PetscReal const *S,PetscReal const *T,PetscReal const x[],PetscReal y[])
{
  PetscInt id_in_warp = threadIdx.x % WARP_SIZE;

  PetscInt c = id_in_warp % 3;
  PetscInt kj = id_in_warp / 3;
  PetscInt k3 = (id_in_warp / 9) * 3;
  PetscInt ji = id_in_warp % 9;

  for (PetscInt l=0; l<3; l++) {

    // u[l,k,j,c] = R[c,i] x[l,k,j,i]
    PetscReal result = 0;
    for (PetscInt i=0; i<3; i++) result += R[i] * shfl_double(x[l], kj*3+i);

    // v[l,k,b,c] = S[b,j] u[l,k,j,c]
    PetscReal result2 = 0;
    for (PetscInt j=0; j<3; j++) result2 += S[j] * shfl_double(result, (k3+j)*3+c);

    // y[l,a,b,c] = T[a,k] v[l,k,b,c]
    for (PetscInt k=0; k<3; k++) y[l] += T[k] * shfl_double(result2, k*9+ji);

  } // for l
}

static __device__ void JacobianInvert(PetscScalar dx[3][3],PetscScalar &dxdet)
{
  PetscScalar a[3][3];
  PetscScalar b0,b3,b6,idet;
  for (PetscInt j=0; j<3; j++) {
    for (PetscInt k=0; k<3; k++) {
      a[j][k] = dx[j][k];
    }
  }
  b0 =  (a[1][1]*a[2][2] - a[2][1]*a[1][2]);
  b3 = -(a[1][0]*a[2][2] - a[2][0]*a[1][2]);
  b6 =  (a[1][0]*a[2][1] - a[2][0]*a[1][1]);
  dxdet = a[0][0]*b0 + a[0][1]*b3 + a[0][2]*b6;
  idet = 1.0 / dxdet;
  dx[0][0] =  idet*b0;
  dx[0][1] = -idet*(a[0][1]*a[2][2] - a[2][1]*a[0][2]);
  dx[0][2] =  idet*(a[0][1]*a[1][2] - a[1][1]*a[0][2]);
  dx[1][0] =  idet*b3;
  dx[1][1] =  idet*(a[0][0]*a[2][2] - a[2][0]*a[0][2]);
  dx[1][2] = -idet*(a[0][0]*a[1][2] - a[1][0]*a[0][2]);
  dx[2][0] =  idet*b6;
  dx[2][1] = -idet*(a[0][0]*a[2][1] - a[2][0]*a[0][1]);
  dx[2][2] =  idet*(a[0][0]*a[1][1] - a[1][0]*a[0][1]);
}

static __device__ void QuadratureAction(PetscScalar gaussdata_eta_w_dxdet,  // gaussdata_eta * w * dxdet
    PetscScalar const dx[3][3],
    PetscScalar const du[3][3],
    PetscScalar dv[3][3])
{
  /* Symmetric gradient with respect to physical coordinates, xx, yy, zz, xy+yx, xz+zx, yz+zy */

  PetscScalar dux[3][3];
  for (PetscInt l=0; l<3; l++) { // fields
    for (PetscInt k=0; k<3; k++) { // directions
      dux[k][l] = du[0][l] * dx[k][0] + du[1][l] * dx[k][1] + du[2][l] * dx[k][2];
    }
  }

  PetscScalar dvx[3][3];
  dvx[0][0] = 2 * gaussdata_eta_w_dxdet * dux[0][0];
  dvx[0][1] =     gaussdata_eta_w_dxdet * (dux[0][1] + dux[1][0]);
  dvx[0][2] =     gaussdata_eta_w_dxdet * (dux[0][2] + dux[2][0]);
  dvx[1][0] =     dvx[0][1];
  dvx[1][1] = 2 * gaussdata_eta_w_dxdet * dux[1][1];
  dvx[1][2] =     gaussdata_eta_w_dxdet * (dux[1][2] + dux[2][1]);
  dvx[2][0] =     dvx[0][2];
  dvx[2][1] =     dvx[1][2];
  dvx[2][2] = 2 * gaussdata_eta_w_dxdet * dux[2][2];

  for (PetscInt l=0; l<3; l++) { // fields
    for (PetscInt k=0; k<3; k++) { // directions
      dv[k][l] = (dvx[0][l] * dx[0][k] + dvx[1][l] * dx[1][k] + dvx[2][l] * dx[2][k]);
    }
  }
}

static __global__ void MFStokesWrapper_A11_HIP_kernel(PetscInt nel,PetscInt nen_u,PetscInt const *el_ids_colored,PetscInt const *elnidx_u,PetscReal const *LA_gcoords,PetscScalar const *ufield,PetscReal const *gaussdata_w,PetscScalar *Yu)
{
  PetscScalar el_x[3];
  PetscScalar el_uv[3]; // unifies elu, elv
  PetscScalar dx[3][3]={{0,0,0},{0,0,0},{0,0,0}},du[3][3]={{0,0,0},{0,0,0},{0,0,0}},dv[3][3]={{0,0,0},{0,0,0},{0,0,0}};
  PetscScalar dxdet = 0;
  PetscInt    elidx = (blockDim.x * blockIdx.x + threadIdx.x) / WARP_SIZE;  // one warp per colored element. elidx is here the index within the same color.
  PetscInt    id_in_warp = threadIdx.x % WARP_SIZE;
  PetscInt    E_times_3;
  PetscReal   R[3],S[3],T[3];
  PetscInt    c = id_in_warp % 3;
  PetscInt    b = (id_in_warp % 9) / 3;
  PetscInt    a = id_in_warp / 9;

  if (elidx >= nel)
    return;

  if (id_in_warp < Q2_NODES_PER_EL_3D) {

    elidx = el_ids_colored[elidx]; // get global element index
    E_times_3 = 3 * elnidx_u[nen_u*elidx+id_in_warp];

    for (PetscInt l=0; l<3; l++) {
      el_x[l] = LA_gcoords[E_times_3+l];
      el_uv[l] = ufield[E_times_3+l];
      R[l] = HIP_D[3*c+l];
      S[l] = HIP_B[3*b+l];
      T[l] = HIP_B[3*a+l];
    }
    TensorContract(R,S,T,el_x, dx[0]); //TensorContract(HIP_D,HIP_B,HIP_B,GRAD,el_uxv,dx[0]);
    TensorContract(R,S,T,el_uv,du[0]); //TensorContract(HIP_D,HIP_B,HIP_B,GRAD,el_uxv,du[0]);

    for (PetscInt l=0; l<3; l++) {
      R[l] = HIP_B[3*c+l];
      S[l] = HIP_D[3*b+l];
    }
    TensorContract(R,S,T,el_x, dx[1]); //TensorContract(HIP_B,HIP_D,HIP_B,GRAD,el_uxv,dx[1]);
    TensorContract(R,S,T,el_uv,du[1]); //TensorContract(HIP_B,HIP_D,HIP_B,GRAD,el_uxv,du[1]);

    for (PetscInt l=0; l<3; l++) {
      S[l] = HIP_B[3*b+l];
      T[l] = HIP_D[3*a+l];
    }
    TensorContract(R,S,T,el_x, dx[2]); //TensorContract(HIP_B,HIP_B,HIP_D,GRAD,el_uxv,dx[2]);
    TensorContract(R,S,T,el_uv,du[2]); //TensorContract(HIP_B,HIP_B,HIP_D,GRAD,el_uxv,du[2]);

    JacobianInvert(dx,dxdet);

    QuadratureAction(gaussdata_w[elidx*NQP + id_in_warp] * dxdet,dx,du,dv);

    for (PetscInt l=0; l<3; l++) {
      el_uv[l] = 0;
      R[l] = HIP_D[3*l + c];
      S[l] = HIP_B[3*l + b];
      T[l] = HIP_B[3*l + a];
    }
    TensorContract(R,S,T,dv[0],el_uv); //TensorContract(HIP_D,HIP_B,HIP_B,GRAD_TRANSPOSE,dv[0],el_uxv);
    for (PetscInt l=0; l<3; l++) {
      R[l] = HIP_B[3*l + c];
      S[l] = HIP_D[3*l + b];
    }
    TensorContract(R,S,T,dv[1],el_uv); //TensorContract(HIP_B,HIP_D,HIP_B,GRAD_TRANSPOSE,dv[1],el_uxv);
    for (PetscInt l=0; l<3; l++) {
      S[l] = HIP_B[3*l + b];
      T[l] = HIP_D[3*l + a];
    }
    TensorContract(R,S,T,dv[2],el_uv); //TensorContract(HIP_B,HIP_B,HIP_D,GRAD_TRANSPOSE,dv[2],el_uxv);

    for (PetscInt l=0; l<3; l++) {
      Yu[E_times_3+l] += el_uv[l];   // Note: Coloring ensures that there are no races here!
    }
  }
}

static __global__ void set_zero_HIP_kernel(PetscScalar *Yu, PetscInt localsize)
{
  for (PetscInt i = blockDim.x * blockIdx.x + threadIdx.x; i<localsize; i += blockDim.x * gridDim.x)
    Yu[i] = 0;
}

extern "C" {

PetscErrorCode MFA11SetUp_HIP(MatA11MF mf)
{
  PetscErrorCode ierr;
  MFA11CUDA      ctx;

  PetscFunctionBeginUser;
  if (mf->ctx) PetscFunctionReturn(0);
  ierr = PetscMalloc1(1,&ctx);CHKERRQ(ierr);
  ierr = MFA11HIP_SetUp(ctx);CHKERRQ(ierr);
  mf->ctx = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode MFA11HIP_SetUp(MFA11CUDA ctx)
{
    PetscErrorCode ierr;
    PetscReal      x1[3],w1[3],B[3][3],D[3][3];
    PetscInt       i;

    PetscFunctionBeginUser;
    ctx->state = 0;

    ctx->ufield             = NULL;
    ctx->LA_gcoords         = NULL;
    ctx->gaussdata_w        = NULL;
    ctx->Yu                 = NULL;
    ctx->element_colors     = 0;
    ctx->elements_per_color = NULL;
    ctx->el_ids_colored     = NULL;
    ctx->elnidx_u           = NULL;

    ierr = PetscDTGaussQuadrature(3,-1,1,x1,w1);CHKERRQ(ierr);
    for (i=0; i<3; i++) {
      B[i][0] = .5*(PetscSqr(x1[i]) - x1[i]);
      B[i][1] = 1 - PetscSqr(x1[i]);
      B[i][2] = .5*(PetscSqr(x1[i]) + x1[i]);
      D[i][0] = x1[i] - .5;
      D[i][1] = -2*x1[i];
      D[i][2] = x1[i] + .5;
    }

    ierr = hipMemcpyToSymbol(HIP_SYMBOL(HIP_D),D,3 * 3 * sizeof(PetscReal));HIPCHECK(ierr);
    ierr = hipMemcpyToSymbol(HIP_SYMBOL(HIP_B),B,3 * 3 * sizeof(PetscReal));HIPCHECK(ierr);

    PetscFunctionReturn(0);
}

PetscErrorCode MFA11Destroy_HIP(MatA11MF mf)
{
  PetscErrorCode ierr;
  MFA11CUDA      ctx;

  PetscFunctionBeginUser;
  ctx = (MFA11CUDA)mf->ctx;
  if (!ctx) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"HIP MF-SpMV implementation should have a valid context");
  ierr = MFA11HIP_CleanUp(ctx);CHKERRQ(ierr);
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  mf->ctx = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode MFA11HIP_CleanUp(MFA11CUDA ctx)
{
    PetscErrorCode ierr;
    PetscInt       i;

    PetscFunctionBeginUser;
    /* Free internal members */
    ierr = hipFree(ctx->ufield);HIPCHECK(ierr);
    ierr = hipFree(ctx->LA_gcoords);HIPCHECK(ierr);
    ierr = hipFree(ctx->gaussdata_w);HIPCHECK(ierr);
    for (i=0; i<ctx->element_colors; ++i) {
      ierr = hipFree(ctx->el_ids_colored[i]);HIPCHECK(ierr);
    }
    ierr = PetscFree(ctx->elements_per_color);HIPCHECK(ierr);
    ierr = PetscFree(ctx->el_ids_colored);HIPCHECK(ierr);
    ierr = hipFree(ctx->elnidx_u);HIPCHECK(ierr);
    ierr = hipFree(ctx->Yu);HIPCHECK(ierr);

    PetscFunctionReturn(0);
}

PetscErrorCode CopyTo_A11_HIP(MatA11MF mf,MFA11CUDA ctx,const PetscScalar *ufield,const PetscReal *LA_gcoords,const PetscReal *gaussdata_host,PetscInt nel,PetscInt nen_u,const PetscInt *elnidx_u,PetscInt nnodes_local)
{
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscInt       localsize = NSD*nnodes_local;

  PetscFunctionBeginUser;

  if (!ctx->elnidx_u) {
    ierr = hipMalloc(&ctx->elnidx_u,        nel * nen_u * sizeof(PetscInt));HIPCHECK(ierr);
    ierr = hipMemcpy(ctx->elnidx_u,elnidx_u,nel * nen_u * sizeof(PetscInt),hipMemcpyHostToDevice);HIPCHECK(ierr);
    if (log_ops) PetscPrintf(PETSC_COMM_SELF,"<copy to device> * elnidx_u bytes %1.6e\n",(double)(nel * nen_u * sizeof(PetscInt)));

    /* Assign colors to elements such that there is no overlap in writes to Yu if elements are processed concurrently */
    PetscInt elements_colored = 0;
    PetscInt *element_color;
    PetscInt *Yu_color; // scratchpad

    ierr = PetscMalloc(nel * sizeof(PetscInt), &element_color);CHKERRQ(ierr);
    for (i=0; i<nel; ++i) element_color[i] = -1;
    ierr = PetscMalloc(nnodes_local * sizeof(PetscInt), &Yu_color);CHKERRQ(ierr);
    for (i=0; i<nnodes_local; ++i) Yu_color[i] = -1;

    ctx->element_colors = 0;
    while (elements_colored < nel) {

      for (i=0; i<nel; ++i) {

        if (element_color[i] >= 0) continue;  /* element already has a color */

        /* Check if element can be colored: No element in Yu has current color */
        PetscInt can_be_colored = 1;
        for (j=0; j<nen_u; ++j) {
          if (Yu_color[elnidx_u[i*nen_u + j]] == ctx->element_colors) {
            can_be_colored = 0;
            break;
          }
        }

        /* Color element if possible, update Yu indices to current color */
        if (can_be_colored) {
          element_color[i] = ctx->element_colors;
          for (j=0; j<nen_u; ++j)
            Yu_color[elnidx_u[i*nen_u + j]] = ctx->element_colors;

          ++elements_colored;
        }
      }

      ++ctx->element_colors;
    }

    /* Generate HIP arrays with coloring information */
    ierr = PetscMalloc(ctx->element_colors * sizeof(PetscInt), &ctx->elements_per_color);CHKERRQ(ierr);
    ierr = PetscMalloc(ctx->element_colors * sizeof(PetscInt*), &ctx->el_ids_colored);CHKERRQ(ierr);

    for (i=0; i<ctx->element_colors; ++i) {
      /* count elements, collect element indices for this color and copy over to GPU: */
      ctx->elements_per_color[i] = 0;
      for (j=0; j<nel; ++j) {
        if (element_color[j] == i) {
          Yu_color[ctx->elements_per_color[i]] = j; /* Reusing Yu_color array here */
          ctx->elements_per_color[i] += 1;
        }
      }

      ierr = hipMalloc(&ctx->el_ids_colored[i],        ctx->elements_per_color[i] * sizeof(PetscInt));HIPCHECK(ierr);
      ierr = hipMemcpy(ctx->el_ids_colored[i],Yu_color,ctx->elements_per_color[i] * sizeof(PetscInt),hipMemcpyHostToDevice);HIPCHECK(ierr);
      if (log_ops) PetscPrintf(PETSC_COMM_SELF,"<copy to device> * Yu_color bytes %1.6e\n",(double)(ctx->elements_per_color[i] * sizeof(PetscInt)));
    }

    /* clean up */
    ierr = PetscFree(element_color);CHKERRQ(ierr);
    ierr = PetscFree(Yu_color);CHKERRQ(ierr);
  }

  if (!ctx->ufield) {
    ierr = hipMalloc(&ctx->ufield, localsize * sizeof(PetscScalar));HIPCHECK(ierr);
  }
  /* ufield always needs to be copied */
  ierr = hipMemcpy(ctx->ufield,ufield, localsize * sizeof(PetscScalar),hipMemcpyHostToDevice);HIPCHECK(ierr);
  if (log_ops) PetscPrintf(PETSC_COMM_SELF,"<copy to device> ufield bytes %1.6e\n",(double)(localsize * sizeof(PetscScalar)));


  if (!ctx->LA_gcoords) {
    ierr = hipMalloc(&ctx->LA_gcoords, localsize * sizeof(PetscReal));HIPCHECK(ierr);
  }

  if (!ctx->gaussdata_w) {
    ierr = hipMalloc(&ctx->gaussdata_w,nel * NQP * sizeof(PetscReal));HIPCHECK(ierr);
  }

  if (mf->state != ctx->state) {
    ierr = hipMemcpy(ctx->LA_gcoords,LA_gcoords, localsize * sizeof(PetscReal),hipMemcpyHostToDevice);HIPCHECK(ierr);
    if (log_ops) PetscPrintf(PETSC_COMM_SELF,"<copy to device> ** LA_gcoords bytes %1.6e\n",(double)(localsize * sizeof(PetscReal)));

    /* Note that we populate and free gaussdata_host outside this function,
       since this data may have come from another rank with SubRepart */
    ierr = hipMemcpy(ctx->gaussdata_w,gaussdata_host, nel * NQP * sizeof(PetscReal),hipMemcpyHostToDevice);HIPCHECK(ierr);
    if (log_ops) PetscPrintf(PETSC_COMM_SELF,"<copy to device> ** gaussdata_host bytes %1.6e\n",(double)(nel * NQP * sizeof(PetscReal)));

    /* Save new state to avoid unnecessary subsequent copies */
    ctx->state = mf->state;
  }

  if (!ctx->Yu) {
    ierr = hipMalloc(&ctx->Yu, localsize * sizeof(PetscScalar));HIPCHECK(ierr);
  }

  ierr = hipDeviceSynchronize();HIPCHECK(ierr);

  PetscFunctionReturn(0);
}


PetscErrorCode ProcessElements_A11_HIP(MFA11CUDA ctx,PetscInt nen_u,PetscInt localsize)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  set_zero_HIP_kernel<<<256,256>>>(ctx->Yu, localsize);
  for (i=0; i<ctx->element_colors; ++i) {
    MFStokesWrapper_A11_HIP_kernel<<<(ctx->elements_per_color[i]-1)/WARPS_PER_BLOCK + 1, WARPS_PER_BLOCK*WARP_SIZE>>>(ctx->elements_per_color[i],nen_u,ctx->el_ids_colored[i],ctx->elnidx_u,ctx->LA_gcoords,ctx->ufield,ctx->gaussdata_w,ctx->Yu);
  }
  ierr = hipDeviceSynchronize();HIPCHECK(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode CopyFrom_A11_HIP(MFA11CUDA ctx,PetscScalar *Yu,PetscInt localsize)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = hipMemcpy(Yu,ctx->Yu,localsize * sizeof(PetscScalar),hipMemcpyDeviceToHost);HIPCHECK(ierr);
  if (log_ops) PetscPrintf(PETSC_COMM_SELF,"<copy from device> Yu bytes %1.6e\n",(double)(localsize * sizeof(PetscScalar)));
  PetscFunctionReturn(0);
}

PetscErrorCode MFStokesWrapper_A11_HIP(MatA11MF mf,Quadrature volQ,DM dau,PetscScalar ufield[],PetscScalar Yu[])
{
  PetscErrorCode          ierr;
  DM                      cda;
  Vec                     gcoords;
  const PetscReal         *LA_gcoords;
  PetscInt                nel,nen_u,e,i,j,k,localsize;
  const PetscInt          *elnidx_u;
  QPntVolCoefStokes       *all_gausspoints;
  const QPntVolCoefStokes *cell_gausspoints;
  MFA11CUDA               ctx = (MFA11CUDA)mf->ctx;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(MAT_MultMFA11_stp,0,0,0,0);CHKERRQ(ierr);

  /* setup for coords */
  ierr = DMGetCoordinateDM( dau, &cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal( dau,&gcoords );CHKERRQ(ierr);
  ierr = VecGetArrayRead(gcoords,&LA_gcoords);CHKERRQ(ierr);
  ierr = VecGetLocalSize(gcoords,&localsize);CHKERRQ(ierr);

  ierr = DMDAGetElements_pTatinQ2P1(dau,&nel,&nen_u,&elnidx_u);CHKERRQ(ierr);

  ierr = VolumeQuadratureGetAllCellData_Stokes(volQ,&all_gausspoints);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_MultMFA11_stp,0,0,0,0);CHKERRQ(ierr);

  /* Set up HIP data */
  ierr = PetscLogEventBegin(MAT_MultMFA11_cto,0,0,0,0);CHKERRQ(ierr);
  {
    PetscReal *gaussdata_host=NULL;
    if(!ctx->state) {
      PetscReal x1[3],w1[3],w[NQP];

      ierr = PetscDTGaussQuadrature(3,-1,1,x1,w1);CHKERRQ(ierr);
      for (i=0; i<3; i++)
        for (j=0; j<3; j++)
          for (k=0; k<3; k++)
            w[(i*3+j)*3+k] = w1[i] * w1[j] * w1[k];

      ierr = PetscMalloc(nel * NQP * sizeof(PetscReal), &gaussdata_host);CHKERRQ(ierr);
      for (e=0; e<nel; e++) {
        ierr = VolumeQuadratureGetCellData_Stokes(volQ,all_gausspoints,e,(QPntVolCoefStokes**)&cell_gausspoints);CHKERRQ(ierr);
        for (i=0; i<NQP; i++) gaussdata_host[e*NQP + i] = cell_gausspoints[i].eta * w[i];
      }

    }
    ierr = CopyTo_A11_HIP(mf,ctx,ufield,LA_gcoords,gaussdata_host,nel,nen_u,elnidx_u,localsize/NSD);CHKERRQ(ierr);

    if(gaussdata_host) {
      ierr = PetscFree(gaussdata_host);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(MAT_MultMFA11_cto,0,0,0,0);CHKERRQ(ierr);

  /* HIP entry point
   *  - inputs: elnidx_u, LA_gcoords, ufield, gaussdata_w
   *  - output: Yu
   */

    ierr = PetscLogEventBegin(MAT_MultMFA11_ker,0,0,0,0);CHKERRQ(ierr);

    ierr = ProcessElements_A11_HIP(ctx,nen_u,localsize);CHKERRQ(ierr);
    {
      PetscLogDouble ff;

      ff = ((PetscLogDouble)nel) * 9 * 3*NQP*(6+6+6);  /* 9 tensor contractions per element */
      ff += ((PetscLogDouble)nel) * NQP*(14 + 1 + 27); /* 1 Jacobi inversion per element */
      ff += ((PetscLogDouble)nel) * NQP*(5*9+6+6+6*9); /* 1 quadrature action per element */
      PetscLogFlops(ff);
    }
    ierr = PetscLogEventEnd(MAT_MultMFA11_ker,0,0,0,0);CHKERRQ(ierr);

    /* Read back HIP data */
    ierr = PetscLogEventBegin(MAT_MultMFA11_cfr,0,0,0,0);CHKERRQ(ierr);
    ierr = CopyFrom_A11_HIP(ctx,Yu,localsize);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_MultMFA11_cfr,0,0,0,0);CHKERRQ(ierr);

    ierr = VecRestoreArrayRead(gcoords,&LA_gcoords);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

} /* extern C */
