
//
// fe_form_compiler.py version: 8d4b0b5b8d2e57803682a919e42ac439d4c64103
// sympy version: 1.6.1
// using common substring elimination: True
// form file: nitsche-custom-h_IJ.py version: 53800e8dcfeb59279abb73274a0ef2bf16e58dc5
//

//#include <stdio.h>
//#include <math.h>

//
// key: wu
//

// ---------------------------------------------------
// test function[0] coeff:   [w0[i], w1[i], w2[i]]
// test function[0]:         wNt[i]
// test function[0] derivs:  [wdNtx0[i], wdNtx1[i], wdNtx2[i]]
//
// trial function[0] coeff:  [u0[j], u1[j], u2[j]]
// trial function[0]:        uN[j]
// trial function[0] derivs: [udNx0[j], udNx1[j], udNx2[j]]
//
// test function[0] dim: 3 spatial dim: 3 numcoeff:  27
// trial function[0] dim: 3 spatial dim: 3 numcoeff:  27
// ---------------------------------------------------
void nitsche_custom_h_a_q2_3d_asmb_wu(
double wNt[], double wdNtx0[], double wdNtx1[], double wdNtx2[],
double uN[], double udNx0[], double udNx1[], double udNx2[],
double gamma,  // parameter
double nhat[],  // parameter
double scale, double A[])
{
  int i,j;
  for (i=0; i<27; i++) { // w_nbasis
    for (j=0; j<27; j++) { // u_nbasis
      A[(3*i + 0)*81 + (3*j + 0)] += scale * (gamma*pow(nhat[0], 2)*uN[j]*wNt[i]);
      A[(3*i + 0)*81 + (3*j + 1)] += scale * (gamma*nhat[0]*nhat[1]*uN[j]*wNt[i]);
      A[(3*i + 0)*81 + (3*j + 2)] += scale * (gamma*nhat[0]*nhat[2]*uN[j]*wNt[i]);
      A[(3*i + 1)*81 + (3*j + 0)] += scale * (gamma*nhat[0]*nhat[1]*uN[j]*wNt[i]);
      A[(3*i + 1)*81 + (3*j + 1)] += scale * (gamma*pow(nhat[1], 2)*uN[j]*wNt[i]);
      A[(3*i + 1)*81 + (3*j + 2)] += scale * (gamma*nhat[1]*nhat[2]*uN[j]*wNt[i]);
      A[(3*i + 2)*81 + (3*j + 0)] += scale * (gamma*nhat[0]*nhat[2]*uN[j]*wNt[i]);
      A[(3*i + 2)*81 + (3*j + 1)] += scale * (gamma*nhat[1]*nhat[2]*uN[j]*wNt[i]);
      A[(3*i + 2)*81 + (3*j + 2)] += scale * (gamma*pow(nhat[2], 2)*uN[j]*wNt[i]);
  }}
}

// ---------------------------------------------------
// test function[0] coeff:   [w0[i], w1[i], w2[i]]
// test function[0]:         wNt[i]
// test function[0] derivs:  [wdNtx0[i], wdNtx1[i], wdNtx2[i]]
//
// trial function[0] coeff:  [u0[j], u1[j], u2[j]]
// trial function[0]:        uN[j]
// trial function[0] derivs: [udNx0[j], udNx1[j], udNx2[j]]
//
// test function[0] dim: 3 spatial dim: 3 numcoeff:  27
// trial function[0] dim: 3 spatial dim: 3 numcoeff:  27
// ---------------------------------------------------
void nitsche_custom_h_a_q2_3d_asmbdiag_wu(
double wNt[], double wdNtx0[], double wdNtx1[], double wdNtx2[],
double uN[], double udNx0[], double udNx1[], double udNx2[],
double gamma,  // parameter
double nhat[],  // parameter
double scale, double F[])
{
  int i,j;
  for (i=0; i<27; i++) { // w_nbasis
    j = i;
    F[3*i + 0] += scale * (gamma*pow(nhat[0], 2)*uN[j]*wNt[i]);
    F[3*i + 1] += scale * (gamma*pow(nhat[1], 2)*uN[j]*wNt[i]);
    F[3*i + 2] += scale * (gamma*pow(nhat[2], 2)*uN[j]*wNt[i]);
  }
}

// ---------------------------------------------------
// test function[0] coeff:   [w0[i], w1[i], w2[i]]
// test function[0]:         wNt[i]
// test function[0] derivs:  [wdNtx0[i], wdNtx1[i], wdNtx2[i]]
//
// trial function[0] coeff:  [u0[j], u1[j], u2[j]]
// trial function[0]:        uN[j]
// trial function[0] derivs: [udNx0[j], udNx1[j], udNx2[j]]
//
// test function[0] dim: 3 spatial dim: 3 numcoeff:  27
// trial function[0] dim: 3 spatial dim: 3 numcoeff:  27
// ---------------------------------------------------
void nitsche_custom_h_a_q2_3d_spmv_wu(
double wNt[], double wdNtx0[], double wdNtx1[], double wdNtx2[],
double uN[], double udNx0[], double udNx1[], double udNx2[],
double u0[], double u1[], double u2[],
double gamma,  // parameter
double nhat[],  // parameter
double scale, double F[])
{
  int i,j;
  double u0j_uNj = 0.0;
  double u1j_uNj = 0.0;
  double u2j_uNj = 0.0;
  for (j=0; j<27; j++) { // u_nbasis_0
    u0j_uNj += u0[j]*uN[j];
    u1j_uNj += u1[j]*uN[j];
    u2j_uNj += u2[j]*uN[j];
  }
  for (i=0; i<27; i++) { // w_nbasis
    F[3*i + 0] += scale * (gamma*nhat[0]*wNt[i]*(nhat[0]*u0j_uNj + nhat[1]*u1j_uNj + nhat[2]*u2j_uNj));
    F[3*i + 1] += scale * (gamma*nhat[1]*wNt[i]*(nhat[0]*u0j_uNj + nhat[1]*u1j_uNj + nhat[2]*u2j_uNj));
    F[3*i + 2] += scale * (gamma*nhat[2]*wNt[i]*(nhat[0]*u0j_uNj + nhat[1]*u1j_uNj + nhat[2]*u2j_uNj));
  }
}

//
// key: wp
//

// ---------------------------------------------------
// test function[0] coeff:   [w0[i], w1[i], w2[i]]
// test function[0]:         wNt[i]
// test function[0] derivs:  [wdNtx0[i], wdNtx1[i], wdNtx2[i]]
//
// trial function[0] coeff:  [p0[j]]
// trial function[0]:        pN[j]
// trial function[0] derivs: [pdNx0[j], pdNx1[j], pdNx2[j]]
//
// test function[0] dim: 3 spatial dim: 3 numcoeff:  27
// trial function[0] dim: 1 spatial dim: 3 numcoeff:   4
// ---------------------------------------------------
void nitsche_custom_h_a_q2_3d_asmb_wp(
double wNt[], double wdNtx0[], double wdNtx1[], double wdNtx2[],
double pN[], double pdNx0[], double pdNx1[], double pdNx2[],
double n[],  // parameter
double scale, double A[])
{
  int i,j;
  for (i=0; i<27; i++) { // w_nbasis
    for (j=0; j<4; j++) { // p_nbasis
      A[(3*i + 0)*4 + (1*j + 0)] += scale * (n[0]*pN[j]*wNt[i]);
      A[(3*i + 1)*4 + (1*j + 0)] += scale * (n[1]*pN[j]*wNt[i]);
      A[(3*i + 2)*4 + (1*j + 0)] += scale * (n[2]*pN[j]*wNt[i]);
  }}
}

// ---------------------------------------------------
// test function[0] coeff:   [w0[i], w1[i], w2[i]]
// test function[0]:         wNt[i]
// test function[0] derivs:  [wdNtx0[i], wdNtx1[i], wdNtx2[i]]
//
// trial function[0] coeff:  [p0[j]]
// trial function[0]:        pN[j]
// trial function[0] derivs: [pdNx0[j], pdNx1[j], pdNx2[j]]
//
// test function[0] dim: 3 spatial dim: 3 numcoeff:  27
// trial function[0] dim: 1 spatial dim: 3 numcoeff:   4
// ---------------------------------------------------
void nitsche_custom_h_a_q2_3d_spmv_wp(
double wNt[], double wdNtx0[], double wdNtx1[], double wdNtx2[],
double pN[], double pdNx0[], double pdNx1[], double pdNx2[],
double p0[],
double n[],  // parameter
double scale, double F[])
{
  int i,j;
  double p0j_pNj = 0.0;
  for (j=0; j<4; j++) { // p_nbasis_0
    p0j_pNj += p0[j]*pN[j];
  }
  for (i=0; i<27; i++) { // w_nbasis
    F[3*i + 0] += scale * (n[0]*p0j_pNj*wNt[i]);
    F[3*i + 1] += scale * (n[1]*p0j_pNj*wNt[i]);
    F[3*i + 2] += scale * (n[2]*p0j_pNj*wNt[i]);
  }
}

//
// key: qu
//

// ---------------------------------------------------
// test function[0] coeff:   [q0[i]]
// test function[0]:         qNt[i]
// test function[0] derivs:  [qdNtx0[i], qdNtx1[i], qdNtx2[i]]
//
// trial function[0] coeff:  [u0[j], u1[j], u2[j]]
// trial function[0]:        uN[j]
// trial function[0] derivs: [udNx0[j], udNx1[j], udNx2[j]]
//
// test function[0] dim: 1 spatial dim: 3 numcoeff:   4
// trial function[0] dim: 3 spatial dim: 3 numcoeff:  27
// ---------------------------------------------------
void nitsche_custom_h_a_q2_3d_asmb_qu(
double qNt[], double qdNtx0[], double qdNtx1[], double qdNtx2[],
double uN[], double udNx0[], double udNx1[], double udNx2[],
double nhat[],  // parameter
double scale, double A[])
{
  int i,j;
  for (i=0; i<4; i++) { // q_nbasis
    for (j=0; j<27; j++) { // u_nbasis
      A[(1*i + 0)*81 + (3*j + 0)] += scale * (nhat[0]*qNt[i]*uN[j]);
      A[(1*i + 0)*81 + (3*j + 1)] += scale * (nhat[1]*qNt[i]*uN[j]);
      A[(1*i + 0)*81 + (3*j + 2)] += scale * (nhat[2]*qNt[i]*uN[j]);
  }}
}

// ---------------------------------------------------
// test function[0] coeff:   [q0[i]]
// test function[0]:         qNt[i]
// test function[0] derivs:  [qdNtx0[i], qdNtx1[i], qdNtx2[i]]
//
// trial function[0] coeff:  [u0[j], u1[j], u2[j]]
// trial function[0]:        uN[j]
// trial function[0] derivs: [udNx0[j], udNx1[j], udNx2[j]]
//
// test function[0] dim: 1 spatial dim: 3 numcoeff:   4
// trial function[0] dim: 3 spatial dim: 3 numcoeff:  27
// ---------------------------------------------------
void nitsche_custom_h_a_q2_3d_spmv_qu(
double qNt[], double qdNtx0[], double qdNtx1[], double qdNtx2[],
double uN[], double udNx0[], double udNx1[], double udNx2[],
double u0[], double u1[], double u2[],
double nhat[],  // parameter
double scale, double F[])
{
  int i,j;
  double u0j_uNj = 0.0;
  double u1j_uNj = 0.0;
  double u2j_uNj = 0.0;
  for (j=0; j<27; j++) { // u_nbasis_0
    u0j_uNj += u0[j]*uN[j];
    u1j_uNj += u1[j]*uN[j];
    u2j_uNj += u2[j]*uN[j];
  }
  for (i=0; i<4; i++) { // q_nbasis
    F[1*i + 0] += scale * (qNt[i]*(nhat[0]*u0j_uNj + nhat[1]*u1j_uNj + nhat[2]*u2j_uNj));
  }
}

//
// key: qp
//

// ---------------------------------------------------
// test function[0] coeff:   [q0[i]]
// test function[0]:         qNt[i]
// test function[0] derivs:  [qdNtx0[i], qdNtx1[i], qdNtx2[i]]
//
// trial function[0] coeff:  [p0[j]]
// trial function[0]:        pN[j]
// trial function[0] derivs: [pdNx0[j], pdNx1[j], pdNx2[j]]
//
// test function[0] dim: 1 spatial dim: 3 numcoeff:   4
// trial function[0] dim: 1 spatial dim: 3 numcoeff:   4
// ---------------------------------------------------
void nitsche_custom_h_a_q2_3d_asmb_qp(
double qNt[], double qdNtx0[], double qdNtx1[], double qdNtx2[],
double pN[], double pdNx0[], double pdNx1[], double pdNx2[],
double scale, double A[])
{
  int i,j;
  for (i=0; i<4; i++) { // q_nbasis
    for (j=0; j<4; j++) { // p_nbasis
      A[(1*i + 0)*4 + (1*j + 0)] += scale * (0);
  }}
}

// ---------------------------------------------------
// test function[0] coeff:   [q0[i]]
// test function[0]:         qNt[i]
// test function[0] derivs:  [qdNtx0[i], qdNtx1[i], qdNtx2[i]]
//
// trial function[0] coeff:  [p0[j]]
// trial function[0]:        pN[j]
// trial function[0] derivs: [pdNx0[j], pdNx1[j], pdNx2[j]]
//
// test function[0] dim: 1 spatial dim: 3 numcoeff:   4
// trial function[0] dim: 1 spatial dim: 3 numcoeff:   4
// ---------------------------------------------------
void nitsche_custom_h_a_q2_3d_asmbdiag_qp(
double qNt[], double qdNtx0[], double qdNtx1[], double qdNtx2[],
double pN[], double pdNx0[], double pdNx1[], double pdNx2[],
double scale, double F[])
{
  int i,j;
  for (i=0; i<4; i++) { // q_nbasis
    j = i;
    F[1*i + 0] += scale * (0);
  }
}

// ---------------------------------------------------
// test function[0] coeff:   [q0[i]]
// test function[0]:         qNt[i]
// test function[0] derivs:  [qdNtx0[i], qdNtx1[i], qdNtx2[i]]
//
// trial function[0] coeff:  [p0[j]]
// trial function[0]:        pN[j]
// trial function[0] derivs: [pdNx0[j], pdNx1[j], pdNx2[j]]
//
// test function[0] dim: 1 spatial dim: 3 numcoeff:   4
// trial function[0] dim: 1 spatial dim: 3 numcoeff:   4
// ---------------------------------------------------
void nitsche_custom_h_a_q2_3d_spmv_qp(
double qNt[], double qdNtx0[], double qdNtx1[], double qdNtx2[],
double pN[], double pdNx0[], double pdNx1[], double pdNx2[],
double p0[],
double scale, double F[])
{
  int i,j;
  for (i=0; i<4; i++) { // q_nbasis
    F[1*i + 0] += scale * (0);
  }
}

//
// key: w_up
//

// ---------------------------------------------------
// test function[0] coeff:   [w0[i], w1[i], w2[i]]
// test function[0]:         wNt[i]
// test function[0] derivs:  [wdNtx0[i], wdNtx1[i], wdNtx2[i]]
//
// trial function[0] coeff:  [u0[j], u1[j], u2[j]]
// trial function[0]:        uN[j]
// trial function[0] derivs: [udNx0[j], udNx1[j], udNx2[j]]
//
// trial function[1] coeff:  [p0[j]]
// trial function[1]:        pN[j]
// trial function[1] derivs: [pdNx0[j], pdNx1[j], pdNx2[j]]
//
// test function[0] dim: 3 spatial dim: 3 numcoeff:  27
// trial function[0] dim: 3 spatial dim: 3 numcoeff:  27
// trial function[1] dim: 1 spatial dim: 3 numcoeff:   4
// ---------------------------------------------------
void nitsche_custom_h_a_q2_3d_spmv_w_up(
double wNt[], double wdNtx0[], double wdNtx1[], double wdNtx2[],
double uN[], double udNx0[], double udNx1[], double udNx2[],
double pN[], double pdNx0[], double pdNx1[], double pdNx2[],
double u0[], double u1[], double u2[],
double p0[],
double gamma,  // parameter
double n[],  // parameter
double nhat[],  // parameter
double scale, double F[])
{
  int i,j;
  double p0j_pNj = 0.0;
  double u0j_uNj = 0.0;
  double u1j_uNj = 0.0;
  double u2j_uNj = 0.0;
  for (j=0; j<27; j++) { // u_nbasis_0
    u0j_uNj += u0[j]*uN[j];
    u1j_uNj += u1[j]*uN[j];
    u2j_uNj += u2[j]*uN[j];
  }
  for (j=0; j<4; j++) { // p_nbasis_1
    p0j_pNj += p0[j]*pN[j];
  }
  for (i=0; i<27; i++) { // w_nbasis
    {
    double tce0;
    tce0 = gamma*nhat[0];
    F[3*i + 0] += scale * (wNt[i]*(gamma*pow(nhat[0], 2)*u0j_uNj + n[0]*p0j_pNj + nhat[1]*tce0*u1j_uNj + nhat[2]*tce0*u2j_uNj));
    }
    {
    double tce0;
    tce0 = gamma*nhat[1];
    F[3*i + 1] += scale * (wNt[i]*(gamma*pow(nhat[1], 2)*u1j_uNj + n[1]*p0j_pNj + nhat[0]*tce0*u0j_uNj + nhat[2]*tce0*u2j_uNj));
    }
    {
    double tce0;
    tce0 = gamma*nhat[2];
    F[3*i + 2] += scale * (wNt[i]*(gamma*pow(nhat[2], 2)*u2j_uNj + n[2]*p0j_pNj + nhat[0]*tce0*u0j_uNj + nhat[1]*tce0*u1j_uNj));
    }
  }
}

//
// key: q_up
//

// ---------------------------------------------------
// test function[0] coeff:   [q0[i]]
// test function[0]:         qNt[i]
// test function[0] derivs:  [qdNtx0[i], qdNtx1[i], qdNtx2[i]]
//
// trial function[0] coeff:  [u0[j], u1[j], u2[j]]
// trial function[0]:        uN[j]
// trial function[0] derivs: [udNx0[j], udNx1[j], udNx2[j]]
//
// trial function[1] coeff:  [p0[j]]
// trial function[1]:        pN[j]
// trial function[1] derivs: [pdNx0[j], pdNx1[j], pdNx2[j]]
//
// test function[0] dim: 1 spatial dim: 3 numcoeff:   4
// trial function[0] dim: 3 spatial dim: 3 numcoeff:  27
// trial function[1] dim: 1 spatial dim: 3 numcoeff:   4
// ---------------------------------------------------
void nitsche_custom_h_a_q2_3d_spmv_q_up(
double qNt[], double qdNtx0[], double qdNtx1[], double qdNtx2[],
double uN[], double udNx0[], double udNx1[], double udNx2[],
double pN[], double pdNx0[], double pdNx1[], double pdNx2[],
double u0[], double u1[], double u2[],
double p0[],
double nhat[],  // parameter
double scale, double F[])
{
  int i,j;
  double u0j_uNj = 0.0;
  double u1j_uNj = 0.0;
  double u2j_uNj = 0.0;
  for (j=0; j<27; j++) { // u_nbasis_0
    u0j_uNj += u0[j]*uN[j];
    u1j_uNj += u1[j]*uN[j];
    u2j_uNj += u2[j]*uN[j];
  }
  for (i=0; i<4; i++) { // q_nbasis
    F[1*i + 0] += scale * (qNt[i]*(nhat[0]*u0j_uNj + nhat[1]*u1j_uNj + nhat[2]*u2j_uNj));
  }
}

//
// key: w
//

// ---------------------------------------------------
// test function[0] coeff:   [w0[i], w1[i], w2[i]]
// test function[0]:         wNt[i]
// test function[0] derivs:  [wdNtx0[i], wdNtx1[i], wdNtx2[i]]
//
// trial function[0] coeff:  [u0[j], u1[j], u2[j]]
// trial function[0]:        uN[j]
// trial function[0] derivs: [udNx0[j], udNx1[j], udNx2[j]]
//
// trial function[1] coeff:  [p0[j]]
// trial function[1]:        pN[j]
// trial function[1] derivs: [pdNx0[j], pdNx1[j], pdNx2[j]]
//
// test function[0] dim: 3 spatial dim: 3 numcoeff:  27
// trial function[0] dim: 3 spatial dim: 3 numcoeff:  27
// trial function[1] dim: 1 spatial dim: 3 numcoeff:   4
// ---------------------------------------------------
void nitsche_custom_h_a_q2_3d_residual_w(
double wNt[], double wdNtx0[], double wdNtx1[], double wdNtx2[],
double uN[], double udNx0[], double udNx1[], double udNx2[],
double pN[], double pdNx0[], double pdNx1[], double pdNx2[],
double u0[], double u1[], double u2[],
double p0[],
double gN,  // parameter
double gamma,  // parameter
double n[],  // parameter
double nhat[],  // parameter
double scale, double F[])
{
  int i,j;
  double p0j_pNj = 0.0;
  double u0j_uNj = 0.0;
  double u1j_uNj = 0.0;
  double u2j_uNj = 0.0;
  for (j=0; j<27; j++) { // u_nbasis_0
    u0j_uNj += u0[j]*uN[j];
    u1j_uNj += u1[j]*uN[j];
    u2j_uNj += u2[j]*uN[j];
  }
  for (j=0; j<4; j++) { // p_nbasis_1
    p0j_pNj += p0[j]*pN[j];
  }
  for (i=0; i<27; i++) { // w_nbasis
    {
    double tce0;
    tce0 = gamma*nhat[0];
    F[3*i + 0] += scale * (wNt[i]*(gN*tce0 + gamma*pow(nhat[0], 2)*u0j_uNj + n[0]*p0j_pNj + nhat[1]*tce0*u1j_uNj + nhat[2]*tce0*u2j_uNj));
    }
    {
    double tce0;
    tce0 = gamma*nhat[1];
    F[3*i + 1] += scale * (wNt[i]*(gN*tce0 + gamma*pow(nhat[1], 2)*u1j_uNj + n[1]*p0j_pNj + nhat[0]*tce0*u0j_uNj + nhat[2]*tce0*u2j_uNj));
    }
    {
    double tce0;
    tce0 = gamma*nhat[2];
    F[3*i + 2] += scale * (wNt[i]*(gN*tce0 + gamma*pow(nhat[2], 2)*u2j_uNj + n[2]*p0j_pNj + nhat[0]*tce0*u0j_uNj + nhat[1]*tce0*u1j_uNj));
    }
  }
}

//
// key: q
//

// ---------------------------------------------------
// test function[0] coeff:   [q0[i]]
// test function[0]:         qNt[i]
// test function[0] derivs:  [qdNtx0[i], qdNtx1[i], qdNtx2[i]]
//
// trial function[0] coeff:  [u0[j], u1[j], u2[j]]
// trial function[0]:        uN[j]
// trial function[0] derivs: [udNx0[j], udNx1[j], udNx2[j]]
//
// trial function[1] coeff:  [p0[j]]
// trial function[1]:        pN[j]
// trial function[1] derivs: [pdNx0[j], pdNx1[j], pdNx2[j]]
//
// test function[0] dim: 1 spatial dim: 3 numcoeff:   4
// trial function[0] dim: 3 spatial dim: 3 numcoeff:  27
// trial function[1] dim: 1 spatial dim: 3 numcoeff:   4
// ---------------------------------------------------
void nitsche_custom_h_a_q2_3d_residual_q(
double qNt[], double qdNtx0[], double qdNtx1[], double qdNtx2[],
double uN[], double udNx0[], double udNx1[], double udNx2[],
double pN[], double pdNx0[], double pdNx1[], double pdNx2[],
double u0[], double u1[], double u2[],
double p0[],
double gN,  // parameter
double nhat[],  // parameter
double scale, double F[])
{
  int i,j;
  double u0j_uNj = 0.0;
  double u1j_uNj = 0.0;
  double u2j_uNj = 0.0;
  for (j=0; j<27; j++) { // u_nbasis_0
    u0j_uNj += u0[j]*uN[j];
    u1j_uNj += u1[j]*uN[j];
    u2j_uNj += u2[j]*uN[j];
  }
  for (i=0; i<4; i++) { // q_nbasis
    F[1*i + 0] += scale * (qNt[i]*(gN + nhat[0]*u0j_uNj + nhat[1]*u1j_uNj + nhat[2]*u2j_uNj));
  }
}
