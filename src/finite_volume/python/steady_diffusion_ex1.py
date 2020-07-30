
import sympy as spy

def grad3(Q):
  _grad = [ Q.diff(x), Q.diff(y), Q.diff(z) ]
  return _grad

def div3(F):
  _div = F[0].diff(x) + F[1].diff(y) + F[2].diff(z)
  return _div

x,y,z = spy.symbols(['x[0]','x[1]','x[2]'])

Q = spy.sin(spy.pi * 2.1 * x) * spy.cos(spy.pi * 1.1 * y) * spy.sin(spy.pi * 0.4 * z)

k = 1

#gradQ = [ Q.diff(x), Q.diff(y), Q.diff(z) ]
gradQ = grad3(Q)

flux = [ -k * gradQ[0], -k * gradQ[1], -k * gradQ[2] ]

f = - div3(flux)

print('Q[0] = ' + spy.ccode(Q) + ';')

print('gradQ[0] = ' + spy.ccode(gradQ[0]) + ';')
print('gradQ[1] = ' + spy.ccode(gradQ[1]) + ';')
print('gradQ[2] = ' + spy.ccode(gradQ[2]) + ';')

print('F[0] = ' + spy.ccode(flux[0]) + ';')
print('F[1] = ' + spy.ccode(flux[1]) + ';')
print('F[2] = ' + spy.ccode(flux[2]) + ';')

print('f[0] = ' + spy.ccode(f) + ';')

