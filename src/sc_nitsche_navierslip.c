
#include <petsc.h>
#include <ptatin3d_defs.h>
#include <ptatin3d.h>
#include <data_bucket.h>
#include <private/ptatin_impl.h>
#include <quadrature.h>
#include <private/quadrature_impl.h>
#include <element_type_Q2.h>
#include <dmda_element_q2p1.h>
#include <element_utils_q2.h>
#include <ptatin3d_stokes.h>
#include <mesh_entity.h>
#include <surface_constraint.h>
#include <sc_generic.h>


//
// -gamma*uDn*(n[0]*w0[i]*wNt[i] + n[1]*w1[i]*wNt[i] + n[2]*w2[i]*wNt[i]) + gamma*(n[0]*u0[j]*uN[j] + n[1]*u1[j]*uN[j] + n[2]*u2[j]*uN[j])*(n[0]*w0[i]*wNt[i] + n[1]*w1[i]*wNt[i] + n[2]*w2[i]*wNt[i]) + uDn*(n[0]*(2.0*eta*n[1]*(0.5*w0[i]*wdNtx1[i] + 0.5*w1[i]*wdNtx0[i]) + 2.0*eta*n[2]*(0.5*w0[i]*wdNtx2[i] + 0.5*w2[i]*wdNtx0[i]) + n[0]*(2.0*eta*w0[i]*wdNtx0[i] - 1.0*q0[i]*qNt[i])) + n[1]*(2.0*eta*n[0]*(0.5*w0[i]*wdNtx1[i] + 0.5*w1[i]*wdNtx0[i]) + 2.0*eta*n[2]*(0.5*w1[i]*wdNtx2[i] + 0.5*w2[i]*wdNtx1[i]) + n[1]*(2.0*eta*w1[i]*wdNtx1[i] - 1.0*q0[i]*qNt[i])) + n[2]*(2.0*eta*n[0]*(0.5*w0[i]*wdNtx2[i] + 0.5*w2[i]*wdNtx0[i]) + 2.0*eta*n[1]*(0.5*w1[i]*wdNtx2[i] + 0.5*w2[i]*wdNtx1[i]) + n[2]*(2.0*eta*w2[i]*wdNtx2[i] - 1.0*q0[i]*qNt[i]))) + (n[0]*(2.0*eta*n[1]*(0.5*u0[j]*udNx1[j] + 0.5*u1[j]*udNx0[j]) + 2.0*eta*n[2]*(0.5*u0[j]*udNx2[j] + 0.5*u2[j]*udNx0[j]) + n[0]*(2.0*eta*u0[j]*udNx0[j] - 1.0*p0[j]*pN[j])) + n[1]*(2.0*eta*n[0]*(0.5*u0[j]*udNx1[j] + 0.5*u1[j]*udNx0[j]) + 2.0*eta*n[2]*(0.5*u1[j]*udNx2[j] + 0.5*u2[j]*udNx1[j]) + n[1]*(2.0*eta*u1[j]*udNx1[j] - 1.0*p0[j]*pN[j])) + n[2]*(2.0*eta*n[0]*(0.5*u0[j]*udNx2[j] + 0.5*u2[j]*udNx0[j]) + 2.0*eta*n[1]*(0.5*u1[j]*udNx2[j] + 0.5*u2[j]*udNx1[j]) + n[2]*(2.0*eta*u2[j]*udNx2[j] - 1.0*p0[j]*pN[j])))*(-n[0]*w0[i]*wNt[i] - n[1]*w1[i]*wNt[i] - n[2]*w2[i]*wNt[i]) + (n[0]*(2.0*eta*n[1]*(0.5*w0[i]*wdNtx1[i] + 0.5*w1[i]*wdNtx0[i]) + 2.0*eta*n[2]*(0.5*w0[i]*wdNtx2[i] + 0.5*w2[i]*wdNtx0[i]) + n[0]*(2.0*eta*w0[i]*wdNtx0[i] - 1.0*q0[i]*qNt[i])) + n[1]*(2.0*eta*n[0]*(0.5*w0[i]*wdNtx1[i] + 0.5*w1[i]*wdNtx0[i]) + 2.0*eta*n[2]*(0.5*w1[i]*wdNtx2[i] + 0.5*w2[i]*wdNtx1[i]) + n[1]*(2.0*eta*w1[i]*wdNtx1[i] - 1.0*q0[i]*qNt[i])) + n[2]*(2.0*eta*n[0]*(0.5*w0[i]*wdNtx2[i] + 0.5*w2[i]*wdNtx0[i]) + 2.0*eta*n[1]*(0.5*w1[i]*wdNtx2[i] + 0.5*w2[i]*wdNtx1[i]) + n[2]*(2.0*eta*w2[i]*wdNtx2[i] - 1.0*q0[i]*qNt[i])))*(-n[0]*u0[j]*uN[j] - n[1]*u1[j]*uN[j] - n[2]*u2[j]*uN[j])
//

// key: wu
//
// ---------------------------------------------------
//
// test function[0] coeff:   [w0[i], w1[i], w2[i]]
// test function[0]:         wNt[i]
// test function[0] derivs:  [wdNtx0[i], wdNtx1[i], wdNtx2[i]]
//
// trial function[0] coeff:  [u0[j], u1[j], u2[j]]
// trial function[0]:        uN[j]
// trial function[0] derivs: [udNx0[j], udNx1[j], udNx2[j]]
//
// test function[0] dim:          3
// test function[0] spatial dim:  3
// test function[0] numcoeff:     27
//
// trial function[0] dim:         3
// trial function[0] spatial dim: 3
// trial function[0] numcoeff:    27
//
// ---------------------------------------------------
//
void nitsche_navier_q2_3d_asmb_wu(
                                  double wNt[], double wdNtx0[], double wdNtx1[], double wdNtx2[],
                                  double uN[], double udNx0[], double udNx1[], double udNx2[],
                                  double eta,  // parameter
                                  double gamma,  // parameter
                                  double n[],  // parameter
                                  double scale, double A[])
{
  int i,j;
  for (i=0; i<27; i++) { // w_nbasis
    for (j=0; j<27; j++) { // u_nbasis
      {
        double tce0, tce1, tce2, tce3;
        tce0 = 2.0*eta;
        tce1 = n[0]*tce0;
        tce2 = n[1]*tce0;
        tce3 = n[2]*tce0;
        A[(3*i + 0)*81 + (3*j + 0)] += scale * (pow(n[0], 2)*(gamma*uN[j]*wNt[i] - tce1*uN[j]*wdNtx0[i] - tce1*udNx0[j]*wNt[i] - tce2*uN[j]*wdNtx1[i] - tce2*udNx1[j]*wNt[i] - tce3*uN[j]*wdNtx2[i] - tce3*udNx2[j]*wNt[i]));
      }
      {
        double tce0, tce1, tce2, tce3;
        tce0 = 2.0*eta;
        tce1 = n[0]*tce0;
        tce2 = n[1]*tce0;
        tce3 = n[2]*tce0;
        A[(3*i + 0)*81 + (3*j + 1)] += scale * (n[0]*n[1]*(gamma*uN[j]*wNt[i] - tce1*uN[j]*wdNtx0[i] - tce1*udNx0[j]*wNt[i] - tce2*uN[j]*wdNtx1[i] - tce2*udNx1[j]*wNt[i] - tce3*uN[j]*wdNtx2[i] - tce3*udNx2[j]*wNt[i]));
      }
      {
        double tce0, tce1, tce2, tce3;
        tce0 = 2.0*eta;
        tce1 = n[0]*tce0;
        tce2 = n[1]*tce0;
        tce3 = n[2]*tce0;
        A[(3*i + 0)*81 + (3*j + 2)] += scale * (n[0]*n[2]*(gamma*uN[j]*wNt[i] - tce1*uN[j]*wdNtx0[i] - tce1*udNx0[j]*wNt[i] - tce2*uN[j]*wdNtx1[i] - tce2*udNx1[j]*wNt[i] - tce3*uN[j]*wdNtx2[i] - tce3*udNx2[j]*wNt[i]));
      }
      {
        double tce0, tce1, tce2, tce3;
        tce0 = 2.0*eta;
        tce1 = n[0]*tce0;
        tce2 = n[1]*tce0;
        tce3 = n[2]*tce0;
        A[(3*i + 1)*81 + (3*j + 0)] += scale * (n[0]*n[1]*(gamma*uN[j]*wNt[i] - tce1*uN[j]*wdNtx0[i] - tce1*udNx0[j]*wNt[i] - tce2*uN[j]*wdNtx1[i] - tce2*udNx1[j]*wNt[i] - tce3*uN[j]*wdNtx2[i] - tce3*udNx2[j]*wNt[i]));
      }
      {
        double tce0, tce1, tce2, tce3;
        tce0 = 2.0*eta;
        tce1 = n[0]*tce0;
        tce2 = n[1]*tce0;
        tce3 = n[2]*tce0;
        A[(3*i + 1)*81 + (3*j + 1)] += scale * (pow(n[1], 2)*(gamma*uN[j]*wNt[i] - tce1*uN[j]*wdNtx0[i] - tce1*udNx0[j]*wNt[i] - tce2*uN[j]*wdNtx1[i] - tce2*udNx1[j]*wNt[i] - tce3*uN[j]*wdNtx2[i] - tce3*udNx2[j]*wNt[i]));
      }
      {
        double tce0, tce1, tce2, tce3;
        tce0 = 2.0*eta;
        tce1 = n[0]*tce0;
        tce2 = n[1]*tce0;
        tce3 = n[2]*tce0;
        A[(3*i + 1)*81 + (3*j + 2)] += scale * (n[1]*n[2]*(gamma*uN[j]*wNt[i] - tce1*uN[j]*wdNtx0[i] - tce1*udNx0[j]*wNt[i] - tce2*uN[j]*wdNtx1[i] - tce2*udNx1[j]*wNt[i] - tce3*uN[j]*wdNtx2[i] - tce3*udNx2[j]*wNt[i]));
      }
      {
        double tce0, tce1, tce2, tce3;
        tce0 = 2.0*eta;
        tce1 = n[0]*tce0;
        tce2 = n[1]*tce0;
        tce3 = n[2]*tce0;
        A[(3*i + 2)*81 + (3*j + 0)] += scale * (n[0]*n[2]*(gamma*uN[j]*wNt[i] - tce1*uN[j]*wdNtx0[i] - tce1*udNx0[j]*wNt[i] - tce2*uN[j]*wdNtx1[i] - tce2*udNx1[j]*wNt[i] - tce3*uN[j]*wdNtx2[i] - tce3*udNx2[j]*wNt[i]));
      }
      {
        double tce0, tce1, tce2, tce3;
        tce0 = 2.0*eta;
        tce1 = n[0]*tce0;
        tce2 = n[1]*tce0;
        tce3 = n[2]*tce0;
        A[(3*i + 2)*81 + (3*j + 1)] += scale * (n[1]*n[2]*(gamma*uN[j]*wNt[i] - tce1*uN[j]*wdNtx0[i] - tce1*udNx0[j]*wNt[i] - tce2*uN[j]*wdNtx1[i] - tce2*udNx1[j]*wNt[i] - tce3*uN[j]*wdNtx2[i] - tce3*udNx2[j]*wNt[i]));
      }
      {
        double tce0, tce1, tce2, tce3;
        tce0 = 2.0*eta;
        tce1 = n[0]*tce0;
        tce2 = n[1]*tce0;
        tce3 = n[2]*tce0;
        A[(3*i + 2)*81 + (3*j + 2)] += scale * (pow(n[2], 2)*(gamma*uN[j]*wNt[i] - tce1*uN[j]*wdNtx0[i] - tce1*udNx0[j]*wNt[i] - tce2*uN[j]*wdNtx1[i] - tce2*udNx1[j]*wNt[i] - tce3*uN[j]*wdNtx2[i] - tce3*udNx2[j]*wNt[i]));
      }
    }}
}
//
// ---------------------------------------------------
//
// test function[0] coeff:   [w0[i], w1[i], w2[i]]
// test function[0]:         wNt[i]
// test function[0] derivs:  [wdNtx0[i], wdNtx1[i], wdNtx2[i]]
//
// trial function[0] coeff:  [u0[j], u1[j], u2[j]]
// trial function[0]:        uN[j]
// trial function[0] derivs: [udNx0[j], udNx1[j], udNx2[j]]
//
// test function[0] dim:          3
// test function[0] spatial dim:  3
// test function[0] numcoeff:     27
//
// trial function[0] dim:         3
// trial function[0] spatial dim: 3
// trial function[0] numcoeff:    27
//
// ---------------------------------------------------
//
void nitsche_navier_q2_3d_asmbdiag_wu(
                                      double wNt[], double wdNtx0[], double wdNtx1[], double wdNtx2[],
                                      double uN[], double udNx0[], double udNx1[], double udNx2[],
                                      double eta,  // parameter
                                      double gamma,  // parameter
                                      double n[],  // parameter
                                      double scale, double F[])
{
  int i,j;
  for (i=0; i<27; i++) { // w_nbasis
    j = i;
    {
      double tce0, tce1, tce2, tce3;
      tce0 = 2.0*eta;
      tce1 = n[0]*tce0;
      tce2 = n[1]*tce0;
      tce3 = n[2]*tce0;
      F[3*i + 0] += scale * (pow(n[0], 2)*(gamma*uN[j]*wNt[i] - tce1*uN[j]*wdNtx0[i] - tce1*udNx0[j]*wNt[i] - tce2*uN[j]*wdNtx1[i] - tce2*udNx1[j]*wNt[i] - tce3*uN[j]*wdNtx2[i] - tce3*udNx2[j]*wNt[i]));
    }
    {
      double tce0, tce1, tce2, tce3;
      tce0 = 2.0*eta;
      tce1 = n[0]*tce0;
      tce2 = n[1]*tce0;
      tce3 = n[2]*tce0;
      F[3*i + 1] += scale * (pow(n[1], 2)*(gamma*uN[j]*wNt[i] - tce1*uN[j]*wdNtx0[i] - tce1*udNx0[j]*wNt[i] - tce2*uN[j]*wdNtx1[i] - tce2*udNx1[j]*wNt[i] - tce3*uN[j]*wdNtx2[i] - tce3*udNx2[j]*wNt[i]));
    }
    {
      double tce0, tce1, tce2, tce3;
      tce0 = 2.0*eta;
      tce1 = n[0]*tce0;
      tce2 = n[1]*tce0;
      tce3 = n[2]*tce0;
      F[3*i + 2] += scale * (pow(n[2], 2)*(gamma*uN[j]*wNt[i] - tce1*uN[j]*wdNtx0[i] - tce1*udNx0[j]*wNt[i] - tce2*uN[j]*wdNtx1[i] - tce2*udNx1[j]*wNt[i] - tce3*uN[j]*wdNtx2[i] - tce3*udNx2[j]*wNt[i]));
    }
  }
}
//
// ---------------------------------------------------
//
// test function[0] coeff:   [w0[i], w1[i], w2[i]]
// test function[0]:         wNt[i]
// test function[0] derivs:  [wdNtx0[i], wdNtx1[i], wdNtx2[i]]
//
// trial function[0] coeff:  [u0[j], u1[j], u2[j]]
// trial function[0]:        uN[j]
// trial function[0] derivs: [udNx0[j], udNx1[j], udNx2[j]]
//
// test function[0] dim:          3
// test function[0] spatial dim:  3
// test function[0] numcoeff:     27
//
// trial function[0] dim:         3
// trial function[0] spatial dim: 3
// trial function[0] numcoeff:    27
//
// ---------------------------------------------------
//
void nitsche_navier_q2_3d_spmv_wu(
                                  double wNt[], double wdNtx0[], double wdNtx1[], double wdNtx2[],
                                  double uN[], double udNx0[], double udNx1[], double udNx2[],
                                  double u0[], double u1[], double u2[],
                                  double eta,  // parameter
                                  double gamma,  // parameter
                                  double n[],  // parameter
                                  double scale, double F[])
{
  int i,j;
  double u2j_udNx0j = 0.0;
  double u2j_udNx1j = 0.0;
  double u1j_udNx2j = 0.0;
  double u0j_uNj = 0.0;
  double u1j_uNj = 0.0;
  double u0j_udNx1j = 0.0;
  double u0j_udNx2j = 0.0;
  double u0j_udNx0j = 0.0;
  double u1j_udNx1j = 0.0;
  double u1j_udNx0j = 0.0;
  double u2j_udNx2j = 0.0;
  double u2j_uNj = 0.0;
  for (j=0; j<27; j++) { // u_nbasis_0
    u0j_uNj += u0[j]*uN[j];
    u1j_uNj += u1[j]*uN[j];
    u2j_uNj += u2[j]*uN[j];
    u0j_udNx0j += u0[j]*udNx0[j];
    u0j_udNx1j += u0[j]*udNx1[j];
    u0j_udNx2j += u0[j]*udNx2[j];
    u1j_udNx0j += u1[j]*udNx0[j];
    u1j_udNx1j += u1[j]*udNx1[j];
    u1j_udNx2j += u1[j]*udNx2[j];
    u2j_udNx0j += u2[j]*udNx0[j];
    u2j_udNx1j += u2[j]*udNx1[j];
    u2j_udNx2j += u2[j]*udNx2[j];
  }
  for (i=0; i<27; i++) { // w_nbasis
    {
      double tce0, tce1, tce2, tce3, tce4, tce5, tce6, tce7, tce8, tce9, tce10, tce11, tce12, tce13, tce14, tce15, tce16;
      tce0 = gamma*wNt[i];
      tce1 = n[0]*u0j_uNj;
      tce2 = n[1]*u1j_uNj;
      tce3 = n[2]*u2j_uNj;
      tce4 = 2.0*eta;
      tce5 = n[1]*tce4;
      tce6 = tce5*wdNtx1[i];
      tce7 = n[0]*wNt[i];
      tce8 = tce5*tce7;
      tce9 = tce2*tce4;
      tce10 = n[0]*wdNtx0[i];
      tce11 = n[2]*tce4;
      tce12 = tce11*tce7;
      tce13 = n[2]*tce5*wNt[i];
      tce14 = pow(n[0], 2)*tce4;
      tce15 = pow(n[1], 2)*tce4;
      tce16 = pow(n[2], 2)*tce4;
      F[3*i + 0] += scale * (n[0]*(-n[2]*tce9*wdNtx2[i] + tce0*tce1 + tce0*tce2 + tce0*tce3 - tce1*tce11*wdNtx2[i] - tce1*tce6 - tce10*tce3*tce4 - tce10*tce9 - tce12*u0j_udNx2j - tce12*u2j_udNx0j - tce13*u1j_udNx2j - tce13*u2j_udNx1j - tce14*u0j_uNj*wdNtx0[i] - tce14*u0j_udNx0j*wNt[i] - tce15*u1j_uNj*wdNtx1[i] - tce15*u1j_udNx1j*wNt[i] - tce16*u2j_uNj*wdNtx2[i] - tce16*u2j_udNx2j*wNt[i] - tce3*tce6 - tce8*u0j_udNx1j - tce8*u1j_udNx0j));
    }
    {
      double tce0, tce1, tce2, tce3, tce4, tce5, tce6, tce7, tce8, tce9, tce10, tce11, tce12, tce13, tce14, tce15, tce16;
      tce0 = gamma*wNt[i];
      tce1 = n[0]*u0j_uNj;
      tce2 = n[1]*u1j_uNj;
      tce3 = n[2]*u2j_uNj;
      tce4 = 2.0*eta;
      tce5 = n[1]*tce4;
      tce6 = tce5*wdNtx1[i];
      tce7 = n[0]*wNt[i];
      tce8 = tce5*tce7;
      tce9 = tce2*tce4;
      tce10 = n[0]*wdNtx0[i];
      tce11 = n[2]*tce4;
      tce12 = tce11*tce7;
      tce13 = n[2]*tce5*wNt[i];
      tce14 = pow(n[0], 2)*tce4;
      tce15 = pow(n[1], 2)*tce4;
      tce16 = pow(n[2], 2)*tce4;
      F[3*i + 1] += scale * (n[1]*(-n[2]*tce9*wdNtx2[i] + tce0*tce1 + tce0*tce2 + tce0*tce3 - tce1*tce11*wdNtx2[i] - tce1*tce6 - tce10*tce3*tce4 - tce10*tce9 - tce12*u0j_udNx2j - tce12*u2j_udNx0j - tce13*u1j_udNx2j - tce13*u2j_udNx1j - tce14*u0j_uNj*wdNtx0[i] - tce14*u0j_udNx0j*wNt[i] - tce15*u1j_uNj*wdNtx1[i] - tce15*u1j_udNx1j*wNt[i] - tce16*u2j_uNj*wdNtx2[i] - tce16*u2j_udNx2j*wNt[i] - tce3*tce6 - tce8*u0j_udNx1j - tce8*u1j_udNx0j));
    }
    {
      double tce0, tce1, tce2, tce3, tce4, tce5, tce6, tce7, tce8, tce9, tce10, tce11, tce12, tce13, tce14, tce15, tce16;
      tce0 = gamma*wNt[i];
      tce1 = n[0]*u0j_uNj;
      tce2 = n[1]*u1j_uNj;
      tce3 = n[2]*u2j_uNj;
      tce4 = 2.0*eta;
      tce5 = n[1]*tce4;
      tce6 = tce5*wdNtx1[i];
      tce7 = n[0]*wNt[i];
      tce8 = tce5*tce7;
      tce9 = tce2*tce4;
      tce10 = n[0]*wdNtx0[i];
      tce11 = n[2]*tce4;
      tce12 = tce11*tce7;
      tce13 = n[2]*tce5*wNt[i];
      tce14 = pow(n[0], 2)*tce4;
      tce15 = pow(n[1], 2)*tce4;
      tce16 = pow(n[2], 2)*tce4;
      F[3*i + 2] += scale * (n[2]*(-n[2]*tce9*wdNtx2[i] + tce0*tce1 + tce0*tce2 + tce0*tce3 - tce1*tce11*wdNtx2[i] - tce1*tce6 - tce10*tce3*tce4 - tce10*tce9 - tce12*u0j_udNx2j - tce12*u2j_udNx0j - tce13*u1j_udNx2j - tce13*u2j_udNx1j - tce14*u0j_uNj*wdNtx0[i] - tce14*u0j_udNx0j*wNt[i] - tce15*u1j_uNj*wdNtx1[i] - tce15*u1j_udNx1j*wNt[i] - tce16*u2j_uNj*wdNtx2[i] - tce16*u2j_udNx2j*wNt[i] - tce3*tce6 - tce8*u0j_udNx1j - tce8*u1j_udNx0j));
    }
  }
}


// key: wp
//
// ---------------------------------------------------
//
// test function[0] coeff:   [w0[i], w1[i], w2[i]]
// test function[0]:         wNt[i]
// test function[0] derivs:  [wdNtx0[i], wdNtx1[i], wdNtx2[i]]
//
// trial function[0] coeff:  [p0[j]]
// trial function[0]:        pN[j]
// trial function[0] derivs: [pdNx0[j], pdNx1[j], pdNx2[j]]
//
// test function[0] dim:          3
// test function[0] spatial dim:  3
// test function[0] numcoeff:     27
//
// trial function[0] dim:         1
// trial function[0] spatial dim: 3
// trial function[0] numcoeff:    4
//
// ---------------------------------------------------
//
void nitsche_navier_q2_3d_asmb_wp(
                                  double wNt[], double wdNtx0[], double wdNtx1[], double wdNtx2[],
                                  double pN[], double pdNx0[], double pdNx1[], double pdNx2[],
                                  double n[],  // parameter
                                  double scale, double A[])
{
  int i,j;
  for (i=0; i<27; i++) { // w_nbasis
    for (j=0; j<4; j++) { // p_nbasis
      A[(3*i + 0)*4 + (1*j + 0)] += scale * (1.0*n[0]*pN[j]*wNt[i]*(pow(n[0], 2) + pow(n[1], 2) + pow(n[2], 2)));
      A[(3*i + 1)*4 + (1*j + 0)] += scale * (1.0*n[1]*pN[j]*wNt[i]*(pow(n[0], 2) + pow(n[1], 2) + pow(n[2], 2)));
      A[(3*i + 2)*4 + (1*j + 0)] += scale * (1.0*n[2]*pN[j]*wNt[i]*(pow(n[0], 2) + pow(n[1], 2) + pow(n[2], 2)));
    }}
}
//
// ---------------------------------------------------
//
// test function[0] coeff:   [w0[i], w1[i], w2[i]]
// test function[0]:         wNt[i]
// test function[0] derivs:  [wdNtx0[i], wdNtx1[i], wdNtx2[i]]
//
// trial function[0] coeff:  [p0[j]]
// trial function[0]:        pN[j]
// trial function[0] derivs: [pdNx0[j], pdNx1[j], pdNx2[j]]
//
// test function[0] dim:          3
// test function[0] spatial dim:  3
// test function[0] numcoeff:     27
//
// trial function[0] dim:         1
// trial function[0] spatial dim: 3
// trial function[0] numcoeff:    4
//
// ---------------------------------------------------
//
void nitsche_navier_q2_3d_spmv_wp(
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
    F[3*i + 0] += scale * (1.0*n[0]*p0j_pNj*wNt[i]*(pow(n[0], 2) + pow(n[1], 2) + pow(n[2], 2)));
    F[3*i + 1] += scale * (1.0*n[1]*p0j_pNj*wNt[i]*(pow(n[0], 2) + pow(n[1], 2) + pow(n[2], 2)));
    F[3*i + 2] += scale * (1.0*n[2]*p0j_pNj*wNt[i]*(pow(n[0], 2) + pow(n[1], 2) + pow(n[2], 2)));
  }
}


// key: qu
//
// ---------------------------------------------------
//
// test function[0] coeff:   [q0[i]]
// test function[0]:         qNt[i]
// test function[0] derivs:  [qdNtx0[i], qdNtx1[i], qdNtx2[i]]
//
// trial function[0] coeff:  [u0[j], u1[j], u2[j]]
// trial function[0]:        uN[j]
// trial function[0] derivs: [udNx0[j], udNx1[j], udNx2[j]]
//
// test function[0] dim:          1
// test function[0] spatial dim:  3
// test function[0] numcoeff:     4
//
// trial function[0] dim:         3
// trial function[0] spatial dim: 3
// trial function[0] numcoeff:    27
//
// ---------------------------------------------------
//
void nitsche_navier_q2_3d_asmb_qu(
                                  double qNt[], double qdNtx0[], double qdNtx1[], double qdNtx2[],
                                  double uN[], double udNx0[], double udNx1[], double udNx2[],
                                  double n[],  // parameter
                                  double scale, double A[])
{
  int i,j;
  for (i=0; i<4; i++) { // q_nbasis
    for (j=0; j<27; j++) { // u_nbasis
      A[(1*i + 0)*81 + (3*j + 0)] += scale * (1.0*n[0]*qNt[i]*uN[j]*(pow(n[0], 2) + pow(n[1], 2) + pow(n[2], 2)));
      A[(1*i + 0)*81 + (3*j + 1)] += scale * (1.0*n[1]*qNt[i]*uN[j]*(pow(n[0], 2) + pow(n[1], 2) + pow(n[2], 2)));
      A[(1*i + 0)*81 + (3*j + 2)] += scale * (1.0*n[2]*qNt[i]*uN[j]*(pow(n[0], 2) + pow(n[1], 2) + pow(n[2], 2)));
    }}
}
//
// ---------------------------------------------------
//
// test function[0] coeff:   [q0[i]]
// test function[0]:         qNt[i]
// test function[0] derivs:  [qdNtx0[i], qdNtx1[i], qdNtx2[i]]
//
// trial function[0] coeff:  [u0[j], u1[j], u2[j]]
// trial function[0]:        uN[j]
// trial function[0] derivs: [udNx0[j], udNx1[j], udNx2[j]]
//
// test function[0] dim:          1
// test function[0] spatial dim:  3
// test function[0] numcoeff:     4
//
// trial function[0] dim:         3
// trial function[0] spatial dim: 3
// trial function[0] numcoeff:    27
//
// ---------------------------------------------------
//
void nitsche_navier_q2_3d_spmv_qu(
                                  double qNt[], double qdNtx0[], double qdNtx1[], double qdNtx2[],
                                  double uN[], double udNx0[], double udNx1[], double udNx2[],
                                  double u0[], double u1[], double u2[],
                                  double n[],  // parameter
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
    {
      double tce0, tce1, tce2, tce3, tce4, tce5;
      tce0 = pow(n[1], 2);
      tce1 = n[0]*u0j_uNj;
      tce2 = pow(n[2], 2);
      tce3 = pow(n[0], 2);
      tce4 = n[1]*u1j_uNj;
      tce5 = n[2]*u2j_uNj;
      F[1*i + 0] += scale * (1.0*qNt[i]*(pow(n[0], 3)*u0j_uNj + pow(n[1], 3)*u1j_uNj + pow(n[2], 3)*u2j_uNj + tce0*tce1 + tce0*tce5 + tce1*tce2 + tce2*tce4 + tce3*tce4 + tce3*tce5));
    }
  }
}


// key: qp
//
// ---------------------------------------------------
//
// test function[0] coeff:   [q0[i]]
// test function[0]:         qNt[i]
// test function[0] derivs:  [qdNtx0[i], qdNtx1[i], qdNtx2[i]]
//
// trial function[0] coeff:  [p0[j]]
// trial function[0]:        pN[j]
// trial function[0] derivs: [pdNx0[j], pdNx1[j], pdNx2[j]]
//
// test function[0] dim:          1
// test function[0] spatial dim:  3
// test function[0] numcoeff:     4
//
// trial function[0] dim:         1
// trial function[0] spatial dim: 3
// trial function[0] numcoeff:    4
//
// ---------------------------------------------------
//
void nitsche_navier_q2_3d_asmb_qp(
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
//
// ---------------------------------------------------
//
// test function[0] coeff:   [q0[i]]
// test function[0]:         qNt[i]
// test function[0] derivs:  [qdNtx0[i], qdNtx1[i], qdNtx2[i]]
//
// trial function[0] coeff:  [p0[j]]
// trial function[0]:        pN[j]
// trial function[0] derivs: [pdNx0[j], pdNx1[j], pdNx2[j]]
//
// test function[0] dim:          1
// test function[0] spatial dim:  3
// test function[0] numcoeff:     4
//
// trial function[0] dim:         1
// trial function[0] spatial dim: 3
// trial function[0] numcoeff:    4
//
// ---------------------------------------------------
//
void nitsche_navier_q2_3d_asmbdiag_qp(
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
//
// ---------------------------------------------------
//
// test function[0] coeff:   [q0[i]]
// test function[0]:         qNt[i]
// test function[0] derivs:  [qdNtx0[i], qdNtx1[i], qdNtx2[i]]
//
// trial function[0] coeff:  [p0[j]]
// trial function[0]:        pN[j]
// trial function[0] derivs: [pdNx0[j], pdNx1[j], pdNx2[j]]
//
// test function[0] dim:          1
// test function[0] spatial dim:  3
// test function[0] numcoeff:     4
//
// trial function[0] dim:         1
// trial function[0] spatial dim: 3
// trial function[0] numcoeff:    4
//
// ---------------------------------------------------
//
void nitsche_navier_q2_3d_spmv_qp(
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


// key: w
//
// ---------------------------------------------------
//
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
// test function[0] dim:          3
// test function[0] spatial dim:  3
// test function[0] numcoeff:     27
//
// trial function[0] dim:         3
// trial function[0] spatial dim: 3
// trial function[0] numcoeff:    27
//
// trial function[1] dim:         1
// trial function[1] spatial dim: 3
// trial function[1] numcoeff:    4
//
// ---------------------------------------------------
//
void nitsche_navier_q2_3d_residual_w(
                                     double wNt[], double wdNtx0[], double wdNtx1[], double wdNtx2[],
                                     double uN[], double udNx0[], double udNx1[], double udNx2[],
                                     double pN[], double pdNx0[], double pdNx1[], double pdNx2[],
                                     double u0[], double u1[], double u2[],
                                     double p0[],
                                     double eta,  // parameter
                                     double gamma,  // parameter
                                     double n[],  // parameter
                                     double uDn,  // parameter
                                     double scale, double F[])
{
  int i,j;
  double u2j_udNx0j = 0.0;
  double p0j_pNj = 0.0;
  double u2j_udNx1j = 0.0;
  double u1j_udNx2j = 0.0;
  double u0j_uNj = 0.0;
  double u1j_uNj = 0.0;
  double u0j_udNx1j = 0.0;
  double u0j_udNx2j = 0.0;
  double u0j_udNx0j = 0.0;
  double u1j_udNx1j = 0.0;
  double u1j_udNx0j = 0.0;
  double u2j_udNx2j = 0.0;
  double u2j_uNj = 0.0;
  for (j=0; j<27; j++) { // u_nbasis_0
    u0j_uNj += u0[j]*uN[j];
    u1j_uNj += u1[j]*uN[j];
    u2j_uNj += u2[j]*uN[j];
    u0j_udNx0j += u0[j]*udNx0[j];
    u0j_udNx1j += u0[j]*udNx1[j];
    u0j_udNx2j += u0[j]*udNx2[j];
    u1j_udNx0j += u1[j]*udNx0[j];
    u1j_udNx1j += u1[j]*udNx1[j];
    u1j_udNx2j += u1[j]*udNx2[j];
    u2j_udNx0j += u2[j]*udNx0[j];
    u2j_udNx1j += u2[j]*udNx1[j];
    u2j_udNx2j += u2[j]*udNx2[j];
  }
  for (j=0; j<4; j++) { // p_nbasis_1
    p0j_pNj += p0[j]*pN[j];
  }
  for (i=0; i<27; i++) { // w_nbasis
    {
      double tce0, tce1, tce2, tce3, tce4, tce5, tce6, tce7, tce8, tce9, tce10, tce11, tce12, tce13, tce14, tce15, tce16, tce17, tce18, tce19, tce20, tce21, tce22;
      tce0 = gamma*wNt[i];
      tce1 = n[0]*u0j_uNj;
      tce2 = n[1]*u1j_uNj;
      tce3 = n[2]*u2j_uNj;
      tce4 = 2.0*eta;
      tce5 = tce4*uDn;
      tce6 = n[0]*wdNtx0[i];
      tce7 = n[1]*wdNtx1[i];
      tce8 = n[2]*wdNtx2[i];
      tce9 = pow(n[0], 2);
      tce10 = tce9*wNt[i];
      tce11 = 1.0*p0j_pNj;
      tce12 = pow(n[1], 2);
      tce13 = tce11*wNt[i];
      tce14 = pow(n[2], 2);
      tce15 = tce1*tce4;
      tce16 = n[0]*tce4*wNt[i];
      tce17 = n[1]*tce16;
      tce18 = tce4*tce6;
      tce19 = n[2]*tce16;
      tce20 = n[1]*n[2]*tce4*wNt[i];
      tce21 = tce12*tce4;
      tce22 = tce14*tce4;
      F[3*i + 0] += scale * (n[0]*(tce0*tce1 + tce0*tce2 + tce0*tce3 - tce0*uDn + tce10*tce11 - tce10*tce4*u0j_udNx0j + tce12*tce13 + tce13*tce14 - tce15*tce7 - tce15*tce8 - tce17*u0j_udNx1j - tce17*u1j_udNx0j - tce18*tce2 - tce18*tce3 - tce19*u0j_udNx2j - tce19*u2j_udNx0j - tce2*tce4*tce8 - tce20*u1j_udNx2j - tce20*u2j_udNx1j - tce21*u1j_uNj*wdNtx1[i] - tce21*u1j_udNx1j*wNt[i] - tce22*u2j_uNj*wdNtx2[i] - tce22*u2j_udNx2j*wNt[i] - tce3*tce4*tce7 - tce4*tce9*u0j_uNj*wdNtx0[i] + tce5*tce6 + tce5*tce7 + tce5*tce8));
    }
    {
      double tce0, tce1, tce2, tce3, tce4, tce5, tce6, tce7, tce8, tce9, tce10, tce11, tce12, tce13, tce14, tce15, tce16, tce17, tce18, tce19, tce20, tce21, tce22;
      tce0 = gamma*wNt[i];
      tce1 = n[0]*u0j_uNj;
      tce2 = n[1]*u1j_uNj;
      tce3 = n[2]*u2j_uNj;
      tce4 = 2.0*eta;
      tce5 = tce4*uDn;
      tce6 = n[0]*wdNtx0[i];
      tce7 = n[1]*wdNtx1[i];
      tce8 = n[2]*wdNtx2[i];
      tce9 = pow(n[0], 2);
      tce10 = tce9*wNt[i];
      tce11 = 1.0*p0j_pNj;
      tce12 = pow(n[1], 2);
      tce13 = tce11*wNt[i];
      tce14 = pow(n[2], 2);
      tce15 = tce1*tce4;
      tce16 = n[0]*tce4*wNt[i];
      tce17 = n[1]*tce16;
      tce18 = tce4*tce6;
      tce19 = n[2]*tce16;
      tce20 = n[1]*n[2]*tce4*wNt[i];
      tce21 = tce12*tce4;
      tce22 = tce14*tce4;
      F[3*i + 1] += scale * (n[1]*(tce0*tce1 + tce0*tce2 + tce0*tce3 - tce0*uDn + tce10*tce11 - tce10*tce4*u0j_udNx0j + tce12*tce13 + tce13*tce14 - tce15*tce7 - tce15*tce8 - tce17*u0j_udNx1j - tce17*u1j_udNx0j - tce18*tce2 - tce18*tce3 - tce19*u0j_udNx2j - tce19*u2j_udNx0j - tce2*tce4*tce8 - tce20*u1j_udNx2j - tce20*u2j_udNx1j - tce21*u1j_uNj*wdNtx1[i] - tce21*u1j_udNx1j*wNt[i] - tce22*u2j_uNj*wdNtx2[i] - tce22*u2j_udNx2j*wNt[i] - tce3*tce4*tce7 - tce4*tce9*u0j_uNj*wdNtx0[i] + tce5*tce6 + tce5*tce7 + tce5*tce8));
    }
    {
      double tce0, tce1, tce2, tce3, tce4, tce5, tce6, tce7, tce8, tce9, tce10, tce11, tce12, tce13, tce14, tce15, tce16, tce17, tce18, tce19, tce20, tce21, tce22;
      tce0 = gamma*wNt[i];
      tce1 = n[0]*u0j_uNj;
      tce2 = n[1]*u1j_uNj;
      tce3 = n[2]*u2j_uNj;
      tce4 = 2.0*eta;
      tce5 = tce4*uDn;
      tce6 = n[0]*wdNtx0[i];
      tce7 = n[1]*wdNtx1[i];
      tce8 = n[2]*wdNtx2[i];
      tce9 = pow(n[0], 2);
      tce10 = tce9*wNt[i];
      tce11 = 1.0*p0j_pNj;
      tce12 = pow(n[1], 2);
      tce13 = tce11*wNt[i];
      tce14 = pow(n[2], 2);
      tce15 = tce1*tce4;
      tce16 = n[0]*tce4*wNt[i];
      tce17 = n[1]*tce16;
      tce18 = tce4*tce6;
      tce19 = n[2]*tce16;
      tce20 = n[1]*n[2]*tce4*wNt[i];
      tce21 = tce12*tce4;
      tce22 = tce14*tce4;
      F[3*i + 2] += scale * (n[2]*(tce0*tce1 + tce0*tce2 + tce0*tce3 - tce0*uDn + tce10*tce11 - tce10*tce4*u0j_udNx0j + tce12*tce13 + tce13*tce14 - tce15*tce7 - tce15*tce8 - tce17*u0j_udNx1j - tce17*u1j_udNx0j - tce18*tce2 - tce18*tce3 - tce19*u0j_udNx2j - tce19*u2j_udNx0j - tce2*tce4*tce8 - tce20*u1j_udNx2j - tce20*u2j_udNx1j - tce21*u1j_uNj*wdNtx1[i] - tce21*u1j_udNx1j*wNt[i] - tce22*u2j_uNj*wdNtx2[i] - tce22*u2j_udNx2j*wNt[i] - tce3*tce4*tce7 - tce4*tce9*u0j_uNj*wdNtx0[i] + tce5*tce6 + tce5*tce7 + tce5*tce8));
    }
  }
}


// key: q
//
// ---------------------------------------------------
//
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
// test function[0] dim:          1
// test function[0] spatial dim:  3
// test function[0] numcoeff:     4
//
// trial function[0] dim:         3
// trial function[0] spatial dim: 3
// trial function[0] numcoeff:    27
//
// trial function[1] dim:         1
// trial function[1] spatial dim: 3
// trial function[1] numcoeff:    4
//
// ---------------------------------------------------
//
void nitsche_navier_q2_3d_residual_q(
                                     double qNt[], double qdNtx0[], double qdNtx1[], double qdNtx2[],
                                     double uN[], double udNx0[], double udNx1[], double udNx2[],
                                     double pN[], double pdNx0[], double pdNx1[], double pdNx2[],
                                     double u0[], double u1[], double u2[],
                                     double p0[],
                                     double n[],  // parameter
                                     double uDn,  // parameter
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
    {
      double tce0, tce1, tce2, tce3, tce4, tce5;
      tce0 = pow(n[0], 2);
      tce1 = pow(n[1], 2);
      tce2 = pow(n[2], 2);
      tce3 = n[0]*u0j_uNj;
      tce4 = n[1]*u1j_uNj;
      tce5 = n[2]*u2j_uNj;
      F[1*i + 0] += scale * (1.0*qNt[i]*(pow(n[0], 3)*u0j_uNj + pow(n[1], 3)*u1j_uNj + pow(n[2], 3)*u2j_uNj + tce0*tce4 + tce0*tce5 - tce0*uDn + tce1*tce3 + tce1*tce5 - tce1*uDn + tce2*tce3 + tce2*tce4 - tce2*uDn));
    }
  }
}

/* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> */

typedef enum { V_X1=0, V_X2 } StokesSubVec;


typedef enum { M_A11=0, M_A12, M_A21, M_A22 } StokesSubMat;


typedef struct {
  PetscBool setup;
  PetscReal penalty;
} SCContextDemo;


typedef struct {
  QPntSurfCoefStokes *boundary_qp;
  PetscReal          *sc_udotn_qp;
} FormContextDemo;


static PetscErrorCode _destroy_demo(SurfaceConstraint sc)
{
  SCContextDemo *ctx;
  PetscErrorCode ierr;
  if (sc->data) {
    ctx = (SCContextDemo*)sc->data;
    ierr = PetscFree(ctx);CHKERRQ(ierr);
    sc->data = NULL;
  }
  PetscFunctionReturn(0);
}

/* surface constraint implementation specific */
static PetscErrorCode _form_access_demo(StokesForm *form)
{
  PetscErrorCode    ierr;
  SurfaceConstraint sc;
  SurfaceQuadrature boundary_q;
  SCContextDemo     *scdata = NULL;
  FormContextDemo   *formdata = NULL;
  int               bs;
  
  printf("Form[-]: access()\n");
  sc = form->sc;
  scdata = (SCContextDemo*)sc->data;
  
  formdata = (FormContextDemo*)form->data;
  
  printf("data->setup %d\n",scdata->setup);
  
  boundary_q = sc->quadrature;
  ierr = SurfaceQuadratureGetAllCellData_Stokes(boundary_q,&formdata->boundary_qp);CHKERRQ(ierr);
  DataBucketGetArray_double(sc->properties_db,"u.n",&bs,(double**)&formdata->sc_udotn_qp);
  //DataBucketGetEntriesdByName(sc->properties_db,"traction",(void**)&traction_qp);
  
  PetscFunctionReturn(0);
}

/* surface constraint implementation specific */
static PetscErrorCode _form_restore_demo(StokesForm *form)
{
  SurfaceConstraint sc;
  FormContextDemo   *formdata = NULL;
  
  printf("Form[-]: restore()\n");
  formdata = (FormContextDemo*)form->data;
  
  sc = form->sc;
  
  DataBucketRestoreArray_double(sc->properties_db,"u.n",(double**)&formdata->sc_udotn_qp);
  //DataBucketRestoreEntriesdByName(sc->properties_db,"traction",(void**)&traction_qp);
  formdata->boundary_qp = NULL;
  
  PetscFunctionReturn(0);
}

/* surface constraint implementation specific */
static PetscErrorCode StokesFormSetupContext_Demo(StokesForm *F,FormContextDemo *formdata)
{
  PetscErrorCode ierr;
  
  /* data */
  ierr = PetscMemzero(formdata,sizeof(FormContextDemo));CHKERRQ(ierr);
  F->data = (void*)formdata;
  
  /* methods */
  F->access  = _form_access_demo;
  F->restore = _form_restore_demo;
  F->apply   = NULL;
  PetscFunctionReturn(0);
}

/* surface constraint implementation specific */
static PetscErrorCode StokesFormSetup_Demo(StokesForm *form,SurfaceConstraint sc,FormContextDemo *formdata)
{
  PetscErrorCode ierr;
  ierr = StokesFormInit(form,FORM_UNINIT,sc);CHKERRQ(ierr);
  ierr = StokeFormSetFunctionSpace_Q2P1(form);CHKERRQ(ierr);
  ierr = StokesFormSetupContext_Demo(form,formdata);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* action (residual) */
/* point-wise kernels */
static PetscErrorCode _form_residual_F1(StokesForm *form,PetscReal ds[],PetscReal F[])
{
  SCContextDemo   *scdata;
  FormContextDemo *formdata;
  PetscInt        qp_offset;
  PetscReal       gamma,eta,uDn,*normal;
  
  scdata   = (void*)form->sc->data;
  formdata = (void*)form->data;
  
  eta    = (PetscReal) formdata->boundary_qp[ form->nqp * form->facet_i + form->point_i ].eta;
  normal = (PetscReal*)formdata->boundary_qp[ form->nqp * form->facet_i + form->point_i ].normal;
  
  qp_offset = 1*(form->nqp * form->facet_sc_i + form->point_i);
  uDn  = formdata->sc_udotn_qp[qp_offset];
  
  gamma = scdata->penalty * eta * 4.0 / form->hF;
  
  nitsche_navier_q2_3d_residual_w(form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
                                     form->X[0]->W, form->X[0]->Wx, form->X[0]->Wy, form->X[0]->Wz,
                                     form->X[1]->W, form->X[1]->Wx, form->X[1]->Wy, form->X[1]->Wz,
                                     form->u_elfield_0,form->u_elfield_1,form->u_elfield_2,
                                     form->u_elfield_0,
                                     eta, gamma,
                                     normal, uDn,
                                     ds[0], F);
  PetscFunctionReturn(0);
}

static PetscErrorCode _form_residual_F2(StokesForm *form,PetscReal ds[],PetscReal F[])
{
  SCContextDemo   *scdata;
  FormContextDemo *formdata;
  PetscInt        qp_offset;
  PetscReal       gamma,eta,uDn,*normal;
  
  scdata   = (void*)form->sc->data;
  formdata = (void*)form->data;
  
  eta    = (PetscReal) formdata->boundary_qp[ form->nqp * form->facet_i + form->point_i ].eta;
  normal = (PetscReal*)formdata->boundary_qp[ form->nqp * form->facet_i + form->point_i ].normal;
  
  qp_offset = 1*(form->nqp * form->facet_sc_i + form->point_i);
  uDn  = formdata->sc_udotn_qp[qp_offset];
  
  gamma = scdata->penalty * eta * 4.0 / form->hF;
  
  nitsche_navier_q2_3d_residual_q(form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
                                     form->X[0]->W, form->X[0]->Wx, form->X[0]->Wy, form->X[0]->Wz,
                                     form->X[1]->W, form->X[1]->Wx, form->X[1]->Wy, form->X[1]->Wz,
                                     form->u_elfield_0,form->u_elfield_1,form->u_elfield_2,
                                     form->u_elfield_0,
                                     normal, uDn,
                                     ds[0], F);
  PetscFunctionReturn(0);
}

/* point-wise kernel configuration */
static PetscErrorCode StoksFormConfigureAction_Residual(StokesForm *form,StokesSubVec op)
{
  PetscErrorCode ierr;
  ierr = StokesFormSetType(form,FORM_RESIDUAL);CHKERRQ(ierr);
  switch (op) {
    case V_X1:
      form->apply = _form_residual_F1;
      break;
    case V_X2:
      form->apply = _form_residual_F2;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Must be one of X1, X2");
      break;
  }
  PetscFunctionReturn(0);
}

/* surface constraint methods */
static PetscErrorCode sc_residual_F1(
  SurfaceConstraint sc, DM dmu,const PetscScalar ufield[], DM dmp,const PetscScalar pfield[], PetscScalar R[])
{
  PetscErrorCode  ierr;
  StokesForm      F;
  FormContextDemo formdata;
  
  printf("_Residual_F1\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_Residual(&F,V_X1);CHKERRQ(ierr);
  ierr = generic_facet_action(&F, &F.u, dmu, dmu,ufield, dmp,pfield, R);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode sc_residual_F2(
  SurfaceConstraint sc, DM dmu,const PetscScalar ufield[], DM dmp,const PetscScalar pfield[], PetscScalar R[])
{
  PetscErrorCode  ierr;
  StokesForm      F;
  FormContextDemo formdata;
  
  printf("_Residual_F2\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_Residual(&F,V_X2);CHKERRQ(ierr);
  ierr = generic_facet_action(&F, &F.p, dmu, dmu,ufield, dmp,pfield, R);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* action (spmv) */
/* point-wise kernels */
static PetscErrorCode _form_spmv_A11(StokesForm *form,PetscReal ds[],PetscReal F[])
{
  SCContextDemo   *scdata;
  FormContextDemo *formdata;
  PetscInt        qp_offset;
  PetscReal       gamma,eta,uDn,*normal;
  
  scdata   = (void*)form->sc->data;
  formdata = (void*)form->data;
  
  eta    = (PetscReal) formdata->boundary_qp[ form->nqp * form->facet_i + form->point_i ].eta;
  normal = (PetscReal*)formdata->boundary_qp[ form->nqp * form->facet_i + form->point_i ].normal;
  
  qp_offset = 1*(form->nqp * form->facet_sc_i + form->point_i);
  uDn  = formdata->sc_udotn_qp[qp_offset];
  
  gamma = scdata->penalty * eta * 4.0 / form->hF;

  nitsche_navier_q2_3d_spmv_wu(form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
                                  form->trial->W, form->trial->Wx, form->trial->Wy, form->trial->Wz,
                                  form->u_elfield_0,form->u_elfield_1,form->u_elfield_2,
                                  eta, gamma, normal,  // parameter
                                  ds[0], F);
  PetscFunctionReturn(0);
}

static PetscErrorCode _form_spmv_A12(StokesForm *form,PetscReal ds[],PetscReal F[])
{
  SCContextDemo   *scdata;
  FormContextDemo *formdata;
  PetscInt        qp_offset;
  PetscReal       gamma,eta,uDn,*normal;
  
  scdata   = (void*)form->sc->data;
  formdata = (void*)form->data;
  
  eta    = (PetscReal) formdata->boundary_qp[ form->nqp * form->facet_i + form->point_i ].eta;
  normal = (PetscReal*)formdata->boundary_qp[ form->nqp * form->facet_i + form->point_i ].normal;
  
  qp_offset = 1*(form->nqp * form->facet_sc_i + form->point_i);
  uDn  = formdata->sc_udotn_qp[qp_offset];
  
  gamma = scdata->penalty * eta * 4.0 / form->hF;
  
  nitsche_navier_q2_3d_spmv_wp(form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
                                  form->trial->W, form->trial->Wx, form->trial->Wy, form->trial->Wz,
                                  form->p_elfield_0,
                                  normal,  // parameter
                                  ds[0], F);
  PetscFunctionReturn(0);
}

static PetscErrorCode _form_spmv_A21(StokesForm *form,PetscReal ds[],PetscReal F[])
{
  SCContextDemo   *scdata;
  FormContextDemo *formdata;
  PetscInt        qp_offset;
  PetscReal       gamma,eta,uDn,*normal;
  
  scdata   = (void*)form->sc->data;
  formdata = (void*)form->data;
  
  eta    = (PetscReal) formdata->boundary_qp[ form->nqp * form->facet_i + form->point_i ].eta;
  normal = (PetscReal*)formdata->boundary_qp[ form->nqp * form->facet_i + form->point_i ].normal;
  
  qp_offset = 1*(form->nqp * form->facet_sc_i + form->point_i);
  uDn  = formdata->sc_udotn_qp[qp_offset];
  
  gamma = scdata->penalty * eta * 4.0 / form->hF;
  
  nitsche_navier_q2_3d_spmv_qu(form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
                                  form->trial->W, form->trial->Wx, form->trial->Wy, form->trial->Wz,
                                  form->u_elfield_0,form->u_elfield_1,form->u_elfield_2,
                                  normal,  // parameter
                                  ds[0], F);
  PetscFunctionReturn(0);
}

/* point-wise kernel configuration */
static PetscErrorCode StoksFormConfigureAction_SpMV(StokesForm *form,StokesSubMat op)
{
  PetscErrorCode ierr;
  ierr = StokesFormSetType(form,FORM_SPMV);CHKERRQ(ierr);
  switch (op) {
    case M_A11:
      form->apply = _form_spmv_A11;
      break;
    case M_A12:
      form->apply = _form_spmv_A12;
      break;
    case M_A21:
      form->apply = _form_spmv_A21;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Must be one of A11, A12, A21");
      break;
  }
  PetscFunctionReturn(0);
}

/* surface constraint methods */
static PetscErrorCode sc_spmv_A11(
  SurfaceConstraint sc, DM dmu,const PetscScalar ufield[], PetscScalar Y[])
{
  PetscErrorCode  ierr;
  StokesForm      F;
  FormContextDemo formdata;
  
  printf("_SpMV_A11\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_SpMV(&F,M_A11);CHKERRQ(ierr);
  ierr = generic_facet_action(&F, &F.u, dmu, dmu,ufield, NULL,NULL, Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode sc_spmv_A12(
  SurfaceConstraint sc, DM dmu, DM dmp,const PetscScalar pfield[], PetscScalar Y[])
{
  PetscErrorCode  ierr;
  StokesForm      F;
  FormContextDemo formdata;
  
  printf("_SpMV_A12\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_SpMV(&F,M_A12);CHKERRQ(ierr);
  ierr = generic_facet_action(&F, &F.u, dmu, dmu,NULL, dmp,pfield, Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode sc_spmv_A21(
  SurfaceConstraint sc, DM dmu,const PetscScalar ufield[], DM dmp, PetscScalar Y[])
{
  PetscErrorCode  ierr;
  StokesForm      F;
  FormContextDemo formdata;
  
  printf("_SpMV_A21\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_SpMV(&F,M_A21);CHKERRQ(ierr);
  ierr = generic_facet_action(&F, &F.u, dmu, dmu,ufield, dmp,NULL, Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



/* assemble */
/* point-wise kernels */
static PetscErrorCode _form_asmb_A11(StokesForm *form,PetscReal ds[],PetscReal A[])
{
  SCContextDemo   *scdata;
  FormContextDemo *formdata;
  PetscInt        qp_offset;
  PetscReal       gamma,eta,uDn,*normal;

  scdata   = (void*)form->sc->data;
  formdata = (void*)form->data;
  
  eta    = (PetscReal) formdata->boundary_qp[ form->nqp * form->facet_i + form->point_i ].eta;
  normal = (PetscReal*)formdata->boundary_qp[ form->nqp * form->facet_i + form->point_i ].normal;
  
  qp_offset = 1*(form->nqp * form->facet_sc_i + form->point_i);
  uDn  = formdata->sc_udotn_qp[qp_offset];
  
  gamma = scdata->penalty * eta * 4.0 / form->hF;
  
  nitsche_navier_q2_3d_asmb_wu(form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
                                  form->trial->W, form->trial->Wx, form->trial->Wy, form->trial->Wz,
                                  eta,  // parameter
                                  gamma,  // parameter
                                  normal,  // parameter
                                  ds[0], A);
  PetscFunctionReturn(0);
}

static PetscErrorCode _form_asmb_A12(StokesForm *form,PetscReal ds[],PetscReal A[])
{
  SCContextDemo   *scdata;
  FormContextDemo *formdata;
  PetscInt        qp_offset;
  PetscReal       gamma,eta,uDn,*normal;
  
  scdata   = (void*)form->sc->data;
  formdata = (void*)form->data;
  
  eta    = (PetscReal) formdata->boundary_qp[ form->nqp * form->facet_i + form->point_i ].eta;
  normal = (PetscReal*)formdata->boundary_qp[ form->nqp * form->facet_i + form->point_i ].normal;
  
  qp_offset = 1*(form->nqp * form->facet_sc_i + form->point_i);
  uDn  = formdata->sc_udotn_qp[qp_offset];
  
  gamma = scdata->penalty * eta * 4.0 / form->hF;
  
  nitsche_navier_q2_3d_asmb_wp(form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
                                  form->trial->W, form->trial->Wx, form->trial->Wy, form->trial->Wz,
                                  normal,  // parameter
                                  ds[0], A);
  PetscFunctionReturn(0);
}

static PetscErrorCode _form_asmb_A21(StokesForm *form,PetscReal ds[],PetscReal A[])
{
  SCContextDemo   *scdata;
  FormContextDemo *formdata;
  PetscInt        qp_offset;
  PetscReal       gamma,eta,uDn,*normal;
  
  scdata   = (void*)form->sc->data;
  formdata = (void*)form->data;
  
  eta    = (PetscReal) formdata->boundary_qp[ form->nqp * form->facet_i + form->point_i ].eta;
  normal = (PetscReal*)formdata->boundary_qp[ form->nqp * form->facet_i + form->point_i ].normal;
  
  qp_offset = 1*(form->nqp * form->facet_sc_i + form->point_i);
  uDn  = formdata->sc_udotn_qp[qp_offset];
  
  gamma = scdata->penalty * eta * 4.0 / form->hF;
  
  nitsche_navier_q2_3d_asmb_qu(form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
                                  form->trial->W, form->trial->Wx, form->trial->Wy, form->trial->Wz,
                                  normal,  // parameter
                                  ds[0], A);
  PetscFunctionReturn(0);
}

/* point-wise kernel configuration */
static PetscErrorCode StoksFormConfigureAction_Assemble(StokesForm *form,StokesSubMat op)
{
  PetscErrorCode ierr;
  ierr = StokesFormSetType(form,FORM_ASSEMBLE);CHKERRQ(ierr);
  switch (op) {
    case M_A11:
      form->apply = _form_asmb_A11;
      break;
    case M_A12:
      form->apply = _form_asmb_A12;
      break;
    case M_A21:
      form->apply = _form_asmb_A21;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Must be one of A11, A12, A21");
      break;
  }
  PetscFunctionReturn(0);
}

/* surface constraint methods */
static PetscErrorCode sc_asmb_A11(SurfaceConstraint sc, DM dmu, Mat A)
{
  PetscErrorCode  ierr;
  StokesForm      F;
  FormContextDemo formdata;
  
  printf("_Assemble_A11\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_Assemble(&F,M_A11);CHKERRQ(ierr);
  ierr = generic_facet_assemble(&F, &F.u,&F.u, dmu, dmu, NULL, A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode sc_asmb_A12(SurfaceConstraint sc, DM dmu, DM dmp, Mat A)
{
  PetscErrorCode  ierr;
  StokesForm      F;
  FormContextDemo formdata;
  
  printf("_Assemble_A12\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_Assemble(&F,M_A12);CHKERRQ(ierr);
  ierr = generic_facet_assemble(&F, &F.u,&F.p, dmu, dmu, dmp, A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode sc_asmb_A21(SurfaceConstraint sc, DM dmu, DM dmp, Mat A)
{
  PetscErrorCode  ierr;
  StokesForm      F;
  FormContextDemo formdata;
  
  printf("_Assemble_A21\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_Assemble(&F,M_A21);CHKERRQ(ierr);
  ierr = generic_facet_assemble(&F, &F.p,&F.u, dmu, dmu, dmp, A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



/* assemble diagonal */
/* point-wise kernels */
static PetscErrorCode _form_asmbdiag_A11(StokesForm *form,PetscReal ds[],PetscReal A[])
{
  PetscFunctionReturn(0);
}

/* point-wise kernel configuration */
static PetscErrorCode StoksFormConfigureAction_AssembleDiagonal(StokesForm *form,StokesSubMat op)
{
  PetscErrorCode ierr;
  ierr = StokesFormSetType(form,FORM_ASSEMBLE_DIAG);CHKERRQ(ierr);
  switch (op) {
    case M_A11:
      form->apply = _form_asmbdiag_A11;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Can only be A11");
      break;
  }
  PetscFunctionReturn(0);
}

/* surface constraint methods */
static PetscErrorCode sc_asmbdiag_A11(SurfaceConstraint sc, DM dmu, PetscScalar A[])
{
  PetscErrorCode  ierr;
  StokesForm      F;
  FormContextDemo formdata;
  
  printf("_AssembleDiagonal_A11\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_AssembleDiagonal(&F,M_A11);CHKERRQ(ierr);
  //ierr = generic_facet_assemble_diagonal(&F, &F.u, dmu, A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


static PetscErrorCode _form_spmv_wA(StokesForm *form,PetscReal ds[],PetscReal F[])
{
  SCContextDemo   *scdata;
  FormContextDemo *formdata;
  PetscReal       gamma,eta,uDn, *normal;
  
  scdata   = (void*)form->sc->data;
  formdata = (void*)form->data;
  
  eta    = (PetscReal) formdata->boundary_qp[ form->nqp * form->facet_i + form->point_i ].eta;
  normal = (PetscReal*)formdata->boundary_qp[ form->nqp * form->facet_i + form->point_i ].normal;
  
  //udotn  = &formdata->sc_udotn_qp[qp_offset];
  uDn = 0.0;
  
  gamma = scdata->penalty * eta * 4.0 / form->hF;
  
  nitsche_navier_q2_3d_residual_w(form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
                                     form->X[0]->W, form->X[0]->Wx, form->X[0]->Wy, form->X[0]->Wz,
                                     form->X[1]->W, form->X[1]->Wx, form->X[1]->Wy, form->X[1]->Wz,
                                     form->u_elfield_0,form->u_elfield_1,form->u_elfield_2,
                                     form->u_elfield_0,
                                     eta, gamma,
                                     normal, uDn,
                                     ds[0], F);
  
  PetscFunctionReturn(0);
}


static PetscErrorCode _form_spmv_qA(StokesForm *form,PetscReal ds[],PetscReal F[])
{
  SCContextDemo   *scdata;
  FormContextDemo *formdata;
  PetscReal       gamma,eta,uDn, *normal;
  
  scdata   = (void*)form->sc->data;
  formdata = (void*)form->data;
  
  eta    = (PetscReal) formdata->boundary_qp[ form->nqp * form->facet_i + form->point_i ].eta;
  normal = (PetscReal*)formdata->boundary_qp[ form->nqp * form->facet_i + form->point_i ].normal;
  
  uDn = 0.0;
  
  gamma = scdata->penalty * eta * 4.0 / form->hF;
  
  nitsche_navier_q2_3d_residual_q(form->test->W, form->test->Wx, form->test->Wy, form->test->Wz,
                                     form->X[0]->W, form->X[0]->Wx, form->X[0]->Wy, form->X[0]->Wz,
                                     form->X[1]->W, form->X[1]->Wx, form->X[1]->Wy, form->X[1]->Wz,
                                     form->u_elfield_0,form->u_elfield_1,form->u_elfield_2,
                                     form->u_elfield_0,
                                     normal, uDn,
                                     ds[0], F);
  
  PetscFunctionReturn(0);
}

static PetscErrorCode StoksFormConfigureAction_AuResidual(StokesForm *form,StokesSubVec op)
{
  PetscErrorCode ierr;
  ierr = StokesFormSetType(form,FORM_RESIDUAL);CHKERRQ(ierr);
  switch (op) {
    case V_X1:
      form->apply = _form_spmv_wA;
      break;
    case V_X2:
      form->apply = _form_spmv_qA;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Must be one of X1, X2");
      break;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode sc_spmv_A(
                                  SurfaceConstraint sc,
                                  DM dmu,const PetscScalar ufield[],
                                  DM dmp,const PetscScalar pfield[],
                                  PetscScalar Yu[], PetscScalar Yp[])
{
  PetscErrorCode  ierr;
  StokesForm      F;
  FormContextDemo formdata;
  
  printf("_SpMV_A\n");

  printf("_Residual_A11X1_A12X2\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_AuResidual(&F,V_X1);CHKERRQ(ierr);
  ierr = generic_facet_action(&F, &F.u, dmu, dmu,ufield, dmp,pfield, Yu);CHKERRQ(ierr);

  printf("_Residual_A21X1\n");
  ierr = StokesFormSetup_Demo(&F,sc,&formdata);CHKERRQ(ierr);
  ierr = StoksFormConfigureAction_AuResidual(&F,V_X2);CHKERRQ(ierr);
  ierr = generic_facet_action(&F, &F.p, dmu, dmu,ufield, dmp,pfield, Yp);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


PetscErrorCode _SetType_NITSCHE_NAVIER_SLIP(SurfaceConstraint sc)
{
  SCContextDemo  *ctx;
  PetscErrorCode ierr;
  
  /* set methods */
  sc->ops.setup   = NULL; /* always null */
  sc->ops.destroy = _destroy_demo;
  
  sc->ops.residual_F   = NULL; /* always null */
  sc->ops.residual_Fu = sc_residual_F1;
  sc->ops.residual_Fp = sc_residual_F2;
  
  sc->ops.action_A    = sc_spmv_A;
  sc->ops.action_Auu  = sc_spmv_A11;
  sc->ops.action_Aup  = sc_spmv_A12;
  sc->ops.action_Apu  = sc_spmv_A21;
  
  sc->ops.asmb_A   = NULL; /* always null */
  sc->ops.asmb_Auu = sc_asmb_A11;
  sc->ops.asmb_Aup = sc_asmb_A12;
  sc->ops.asmb_Apu = sc_asmb_A21;
  
  sc->ops.diag_A   = NULL; /* always null */
  sc->ops.diag_Auu = sc_asmbdiag_A11;
  
  /* allocate implementation data */
  ierr = PetscMalloc1(1,&ctx);CHKERRQ(ierr);
  ctx->setup = PETSC_TRUE;
  ctx->penalty = 200.0 * 1.0e2 * 0.5;
  sc->data = (void*)ctx;
  
  /* insert properties into quadrature bucket */
  DataBucketRegister_double(sc->properties_db,"u.n",1);
  DataBucketFinalize(sc->properties_db);
  
  PetscFunctionReturn(0);
}


PetscErrorCode user_nitsche_navier_slip_set_constant(Facet F,
                                          const PetscReal qp_coor[],
                                          PetscReal uD[],
                                          void *data)
{
  PetscReal *input;
  input = (PetscReal*)data;
  *uD = input[0];
  PetscFunctionReturn(0);
}

PetscErrorCode _resize_facet_quadrature_data(SurfaceConstraint sc);

PetscErrorCode SurfaceConstraintSetValues_NITSCHE_NAVIER_SLIP(SurfaceConstraint sc,
                                                   SurfCSetValuesNitscheNavierSlip set,
                                                   void *data)
{
  PetscInt e,facet_index,cell_side,cell_index,q,qp_offset;
  Facet cell_facet;
  PetscReal qp_coor[3],uD[3];
  PetscErrorCode ierr;
  PetscReal *uD_qp;
  double Ni[27];
  const PetscInt *elnidx;
  PetscInt       nel,nen;
  double         elcoords[3*Q2_NODES_PER_EL_3D];
  
  
  if (sc->type != SC_NITSCHE_NAVIER_SLIP) {
    PetscPrintf(PetscObjectComm((PetscObject)sc->dm),"[ignoring] SurfaceConstraintSetValues_NITSCHE_NAVIER_SLIP() called with different type on object with name \"%s\"\n",sc->name);
    PetscFunctionReturn(0);
  }
  
  if (!sc->dm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Missing sc->dm. Must call SurfaceConstraintSetDM() first");
  if (!sc->quadrature) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Missing sc->surfQ. Must call SurfaceConstraintSetQuadrature() first");
  if (!sc->facets->set_values_called) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Facets have not been selected");
  
  /* resize qp data */
  ierr = _resize_facet_quadrature_data(sc);CHKERRQ(ierr);
  
  DataBucketGetEntriesdByName(sc->properties_db,"u.n",(void**)&uD_qp);
  
  ierr = MeshFacetInfoGetCoords(sc->fi);CHKERRQ(ierr);
  ierr = FacetCreate(&cell_facet);CHKERRQ(ierr);
  ierr = DMDAGetElements_pTatinQ2P1(sc->fi->dm,&nel,&nen,&elnidx);CHKERRQ(ierr);
  
  for (e=0; e<sc->facets->n_entities; e++) {
    facet_index = sc->facets->local_index[e]; /* facet local index */
    cell_side  = sc->fi->facet_label[facet_index]; /* side label */
    cell_index = sc->fi->facet_cell_index[facet_index];
    //printf("cell_side %d\n",cell_side);
    
    ierr = FacetPack(cell_facet, facet_index, sc->fi);CHKERRQ(ierr);
    
    ierr = DMDAGetElementCoordinatesQ2_3D(elcoords,(PetscInt*)&elnidx[nen*cell_index],(PetscReal*)sc->fi->_mesh_coor);CHKERRQ(ierr);
    
    //qp_offset = sc->nqp_facet * facet_index; /* offset into entire domain qp list */
    qp_offset = sc->nqp_facet * e; /* offset into facet qp list */
    for (q=0; q<sc->nqp_facet; q++) {
      
      {
        PetscInt d,k;
        
        for (d=0; d<3; d++) { qp_coor[d] = 0.0; }
        sc->fi->element->basis_NI_3D(&sc->quadrature->gp3[cell_side][q],Ni);
        for (k=0; k<sc->fi->element->n_nodes_3D; k++) {
          for (d=0; d<3; d++) {
            qp_coor[d] += Ni[k] * elcoords[3*k+d];
          }
        }
      }
      
      ierr = set(cell_facet, qp_coor, uD, data);CHKERRQ(ierr);
      
      ierr = PetscMemcpy(&uD_qp[1*(qp_offset+q)],uD,sizeof(PetscReal)*1);CHKERRQ(ierr);
    }
  }
  
  ierr = FacetDestroy(&cell_facet);CHKERRQ(ierr);
  ierr = MeshFacetInfoRestoreCoords(sc->fi);CHKERRQ(ierr);
  
  DataBucketRestoreEntriesdByName(sc->properties_db,"u.n",(void**)&uD_qp);
  
  PetscFunctionReturn(0);
}
