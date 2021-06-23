#####################################################################
# This script is meant to extract cross-sections from pTatin model  #
# For now its only purpose is to look rapidly the result at a       #
# given step. It does not make nice post-processing                 #
# Anthony Jourdon                                                   #
#                                                                   #
# Python     version: 3.9.5                                         #
# Numpy      version: 1.20.3                                        #
# Matplotlib version: 3.4.2                                         #
#####################################################################
import os
import sys
import struct
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time

def pTatinReadBinary(binary_filename,NX,NY,NZ,Cross_section=False,dim=3,plan=None,inode=None):
  """ Extract values from binary .vec files outputed from pTatin
      binary_filename = name of the .vec file (string)
      NX = number of nodes in X direction (integer)
      NY = number of nodes in Y direction (integer)
      NZ = number of nodes in Z direction (integer)
      Cross_section = Boolean, True if we want a cross-section/False if we want all the points
      plan = determine along which make the cross-section
           0 = xz (Horizontal section = inode along y)
           1 = zy (Vertical section = inode along x)
           2 = xy (Vertical section = inode along z)
      inode = location of the node at which make the cross-section
      dim = 1, 2, or 3 number of dimension the variable has
  """
  start_time = time.time()
  if dim == 1:
    # Big Endian, 1 values
    dimtype = '>d'
  elif dim == 2:
    # Big Endian, 2 values
    dimtype = '>dd'
  elif dim == 3:
    # Big Endian, 3 values
    dimtype = '>ddd'
  else:
    raise Exception('dim can only be 1, 2 or 3. The value of dim was: {}'.format(dim))
  
  if Cross_section == False:
    NX0 = 0; NX1 = NX
    NY0 = 0; NY1 = NY
    NZ0 = 0; NZ1 = NZ
  elif Cross_section == True:
    # Replace by the value of the node at which we make the cross-section depending on the cross-section direction
    if plan == 0:
      NX0 = 0;       NX1 = NX
      NY0 = inode-1; NY1 = inode
      NZ0 = 0;       NZ1 = NZ
    elif plan == 1:
      NX0 = inode-1; NX1 = inode
      NY0 = 0;       NY1 = NY
      NZ0 = 0;       NZ1 = NZ
    elif plan == 2:
      NX0 = 0;       NX1 = NX
      NY0 = 0;       NY1 = NY
      NZ0 = inode-1; NZ1 = inode
    else:
      raise Exception('plan can only be 0, 1 or 2. The value of plan was: {}'.format(plan))
  
  # Create an empty variable to collect data
  C = []
  # Open the binary file
  with open(binary_filename,"rb") as binary_file:
    # Outer loop goes along z values
    for k in np.arange(NZ0,NZ1,1,dtype='int32'):
      # Intermediate loop goes along y values
      for j in np.arange(NY0,NY1,1,dtype='int32'):
        # Inner loop goes along x values
        for i in np.arange(NX0,NX1,1,dtype='int32'):
          # Compute byte offset (I don't know which 8 is precision or seek?)
          offset = NX*NY*k + NX*j + i
          offset = offset * dim * 8 + 8
          binary_file.seek(offset,0)
          f8_bytes = binary_file.read(dim*8)
          v = struct.unpack(dimtype,f8_bytes)
          C.append(v)
          
  C = np.array(C,dtype='float64')
  print("\t Extracted %s in --- %s seconds ---" % (binary_filename,time.time() - start_time))
  return C

def Reshape_for_gradient(NX,NY,NZ,V,V_coords):
  # Reshape to get:
  #                       y = j                     y = j + 1
  #    x = i     [ Vx(z = k) Vx(z = k + 1) ] [ Vx(z = k) Vx(z = k + 1) ] 
  #    x = i + 1 [ Vx(z = k) Vx(z = k + 1) ] [ Vx(z = k) Vx(z = k + 1) ]
  Vx = np.reshape(V[:,0],(NX,NY,NZ),order='F')
  Vy = np.reshape(V[:,1],(NX,NY,NZ),order='F')
  Vz = np.reshape(V[:,2],(NX,NY,NZ),order='F')
  V_xcoords = np.reshape(V_coords[:,0],(NX,NY,NZ),order='F')
  V_ycoords = np.reshape(V_coords[:,1],(NX,NY,NZ),order='F')
  V_zcoords = np.reshape(V_coords[:,2],(NX,NY,NZ),order='F')
  
  # Keep coords as 1D array for each direction 
  #(np.gradient needs that to know the spacing between values in each direction)
  V_xcoords = V_xcoords[:,0,0]
  V_ycoords = V_ycoords[0,:,0]
  V_zcoords = V_zcoords[0,0,:]
  return Vx,Vy,Vz,V_xcoords,V_ycoords,V_zcoords
 
def Concatenate_for_gradient(NX,NY,NZ,V_1,V_coords_1,V_2,V_coords_2,plan):
  
  start_time = time.time()
  print("\n\t Shaping arrays for gradient ...")
  # Make a good shape to pass to np.gradient
  Vx_1,Vy_1,Vz_1,V_xcoords_1,V_ycoords_1,V_zcoords_1 = Reshape_for_gradient(NX,NY,NZ,V_1,V_coords_1)
  Vx_2,Vy_2,Vz_2,V_xcoords_2,V_ycoords_2,V_zcoords_2 = Reshape_for_gradient(NX,NY,NZ,V_2,V_coords_2)
  
  # Concatenate coords in the direction normal to the plan (the others are the same in both plans)
  if plan == 0:
    # Concatenate along the axis normal to the plan (We need 2 plans to compute a gradient in the direction normal to the plan)
    Vx = np.concatenate((Vx_1,Vx_2),axis=1)
    Vy = np.concatenate((Vy_1,Vy_2),axis=1)
    Vz = np.concatenate((Vz_1,Vz_2),axis=1)
    V_xcoords = V_xcoords_1
    V_ycoords = np.concatenate((V_ycoords_1,V_ycoords_2))
    V_zcoords = V_zcoords_1
  elif plan == 1:
    # Concatenate along the axis normal to the plan (We need 2 plans to compute a gradient in the direction normal to the plan)
    Vx = np.concatenate((Vx_1,Vx_2),axis=0)
    Vy = np.concatenate((Vy_1,Vy_2),axis=0)
    Vz = np.concatenate((Vz_1,Vz_2),axis=0)
    V_xcoords = np.concatenate((V_xcoords_1,V_xcoords_2))
    V_ycoords = V_ycoords_1
    V_zcoords = V_zcoords_1
  elif plan == 2:
    # Concatenate along the axis normal to the plan (We need 2 plans to compute a gradient in the direction normal to the plan)
    Vx = np.concatenate((Vx_1,Vx_2),axis=2)
    Vy = np.concatenate((Vy_1,Vy_2),axis=2)
    Vz = np.concatenate((Vz_1,Vz_2),axis=2)
    V_xcoords = V_xcoords_1
    V_ycoords = V_ycoords_1
    V_zcoords = np.concatenate((V_zcoords_1,V_zcoords_2))
  print("\t Array shaped in --- %s seconds ---" % (time.time() - start_time))
  return Vx,Vy,Vz,V_xcoords,V_ycoords,V_zcoords

def Reshape_gradient(NX,NY,NZ,dVxx,dVxy,dVxz,dVyx,dVyy,dVyz,dVzx,dVzy,dVzz):  
  # Make a Gradient matrix of the form:
  #   dV_xx  dV_xy  dV_xz
  #   dV_yx  dV_yy  dV_yz
  #   dV_zx  dV_zy  dV_zz
  # for each point
  Gradient = np.zeros((NX*NY*NZ,3,3))
  for k in np.arange(0,NZ,1,dtype='int32'):
    for j in np.arange(0,NY,1,dtype='int32'):
      for i in np.arange(0,NX,1,dtype='int32'):
        idx = NX*NY*k + NX*j + i
        
        Gradient[idx,0,0] = dVxx[i,j,k]
        Gradient[idx,0,1] = dVxy[i,j,k]
        Gradient[idx,0,2] = dVxz[i,j,k]
        Gradient[idx,1,0] = dVyx[i,j,k]
        Gradient[idx,1,1] = dVyy[i,j,k]
        Gradient[idx,1,2] = dVyz[i,j,k]
        Gradient[idx,2,0] = dVzx[i,j,k]
        Gradient[idx,2,1] = dVzy[i,j,k]
        Gradient[idx,2,2] = dVzz[i,j,k]
  return Gradient

def Compute_StrRateTensor(NX_Q2,NY_Q2,NZ_Q2,Vx,Vy,Vz,V_xcoords,V_ycoords,V_zcoords):
  start_time = time.time()
  print("\n\t Computing Strain Rate Tensor ...")
  # Compute gradient of each component
  dVxx,dVxy,dVxz = np.gradient(Vx,V_xcoords,V_ycoords,V_zcoords)
  dVyx,dVyy,dVyz = np.gradient(Vy,V_xcoords,V_ycoords,V_zcoords)
  dVzx,dVzy,dVzz = np.gradient(Vz,V_xcoords,V_ycoords,V_zcoords)
  
  # Reshape gradient to get [ [dVxx,dVxy,dVxz]
  #                           [dVyx,dVyy,dVyz]
  #                           [dVzx,dVzy,dVzz] ]
  Gradient = Reshape_gradient(NX_Q2,NY_Q2,NZ_Q2,dVxx,dVxy,dVxz,dVyx,dVyy,dVyz,dVzx,dVzy,dVzz)
  # Compute strain rate tensor for each point
  StrRateTensor = 0.5*(Gradient + Gradient.transpose(0,2,1))
  print("\t Computed Strain Rate Tensor in --- %g seconds ---" % (time.time() - start_time))
  return StrRateTensor

def Compute_cell_center(plan,coords,NX,NY,NZ):
  # Compute cell center for MPCell fields coordinates
  if plan == 0:
    Centre = np.zeros(((NX-1)*(NZ-1),2))
    for k in np.arange(0,NZ-1,1,dtype="int32"):
      for i in np.arange(0,NX-1,1,dtype="int32"):
        nc = (NX-1)*k + i
        n  =  NX   *k + i
        Centre[nc,0] = 0.5*(coords[n+1,0]-coords[n,0])  + coords[n,0]
        Centre[nc,1] = 0.5*(coords[n+NX,2]-coords[n,2]) + coords[n,2]  
  elif plan == 1:
    Centre = np.zeros(((NZ-1)*(NY-1),2))
    for k in np.arange(0,NZ-1,1,dtype='int32'):
      for j in np.arange(0,NY-1,1,dtype='int32'):
        nc = (NY-1)*k + j
        n  =  NY   *k + j
        Centre[nc,0] = 0.5*(coords[n+NY,2]-coords[n,2]) + coords[n,2]
        Centre[nc,1] = 0.5*(coords[n+1,1]-coords[n,1])  + coords[n,1]
  elif plan == 2:
    Centre = np.zeros(((NX-1)*(NY-1),2))
    for j in np.arange(0,NY-1,1,dtype='int32'):
      for i in np.arange(0,NX-1,1,dtype='int32'):
        nc = (NX-1)*j + i
        n  =  NX   *j + i
        Centre[nc,0] = 0.5*(coords[n+1,0]-coords[n,0])  + coords[n,0]
        Centre[nc,1] = 0.5*(coords[n+NX,1]-coords[n,1]) + coords[n,1]      
  return Centre

def Create_variables_dictionnary(StrRateTensor):
  dict = {'xx':StrRateTensor[:,0,0],
          'xy':StrRateTensor[:,0,1],
          'xz':StrRateTensor[:,0,2],
          'yx':StrRateTensor[:,1,0],
          'yy':StrRateTensor[:,1,1],
          'yz':StrRateTensor[:,1,2],
          'zx':StrRateTensor[:,2,0],
          'zy':StrRateTensor[:,2,1],
          'zz':StrRateTensor[:,2,2]}
  return dict
  
def DisplayStrainRate():
  """
  Takes command line inputs arguments, use with (as an exemple):
  python pTatin_Extract_sections.py -step 250 -elem 512 64 256 -Xsection 1 -plan 0 -inode 64
  """
  try:
    index = sys.argv.index("-step")
  except:
    raise RuntimeError("A step file number must be given via -step {step}")
  step = sys.argv[index+1]
  
  try:
    index = sys.argv.index("-elem")
  except:
    raise RuntimeError("Provide the number of elements via -elem mx my mz")
  mx = int(sys.argv[index+1])
  my = int(sys.argv[index+2])
  mz = int(sys.argv[index+3])
  
  try:
    index = sys.argv.index("-Xsection")
  except:
    raise RuntimeError("Provide information if Xsection or whole domain should be extracted via -Xsection {0 or 1} or {True or False}")
  ans = sys.argv[index+1]
  
  if ans == "0" or ans == "False":
    Xsection = False
  elif ans == "1" or ans == "True":
    Xsection = True
   
  if Xsection == True:
    try:
      index = sys.argv.index("-plan")
    except:
      raise RuntimeError("-Xsection was given as \"True\" therefore a direction must be given via -plan {0 or 1 or 2}")
    plan = int(sys.argv[index+1])
    
    try:
      index = sys.argv.index("-inode")
    except:
      raise RuntimeError("-Xsection was given as \"True\" therefore a node at which make the cross section must be given via -inode {node_number}")
    inode = int(sys.argv[index+1])
  else:
    plan = None
  
  NX_Q2 = 2*mx + 1
  NY_Q2 = 2*my + 1
  NZ_Q2 = 2*mz + 1
  
  inode_Q2 = 2*inode + 1
  print("\n\t Making cross-section along plan %d at node %d \n" % (plan,inode))
  velocity_file        = 'step'+step+'.dmda-Xu'
  velocity_coords_file = 'step'+step+'.dmda-velocity_dmda_coords.pbvec'
  
  vcoord0   = pTatinReadBinary(velocity_coords_file,NX_Q2,NY_Q2,NZ_Q2,Cross_section=Xsection,dim=3,plan=plan,inode=inode_Q2)
  velocity0 = pTatinReadBinary(velocity_file,NX_Q2,NY_Q2,NZ_Q2,Cross_section=Xsection,dim=3,plan=plan,inode=inode_Q2)
  vcoord1   = pTatinReadBinary(velocity_coords_file,NX_Q2,NY_Q2,NZ_Q2,Cross_section=Xsection,dim=3,plan=plan,inode=inode_Q2-1)
  velocity1 = pTatinReadBinary(velocity_file,NX_Q2,NY_Q2,NZ_Q2,Cross_section=Xsection,dim=3,plan=plan,inode=inode_Q2-1)
  
  if plan == 0:
    Vx,Vy,Vz,V_xcoords,V_ycoords,V_zcoords = Concatenate_for_gradient(NX_Q2,1,NZ_Q2,velocity0,vcoord0,velocity1,vcoord1,plan=plan)
    StrRateTensor = Compute_StrRateTensor(NX_Q2,1,NZ_Q2,Vx,Vy,Vz,V_xcoords,V_ycoords,V_zcoords)
  elif plan == 1:
    Vx,Vy,Vz,V_xcoords,V_ycoords,V_zcoords = Concatenate_for_gradient(1,NY_Q2,NZ_Q2,velocity0,vcoord0,velocity1,vcoord1,plan=plan)
    StrRateTensor = Compute_StrRateTensor(1,NY_Q2,NZ_Q2,Vx,Vy,Vz,V_xcoords,V_ycoords,V_zcoords)
  elif plan == 2:
    Vx,Vy,Vz,V_xcoords,V_ycoords,V_zcoords = Concatenate_for_gradient(NX_Q2,NY_Q2,1,velocity0,vcoord0,velocity1,vcoord1,plan=plan)
    StrRateTensor = Compute_StrRateTensor(NX_Q2,NY_Q2,1,Vx,Vy,Vz,V_xcoords,V_ycoords,V_zcoords)
  
  e = Create_variables_dictionnary(StrRateTensor)
  e2 = e["xx"]**2 + e["yy"]**2 + e["zz"]**2
  e2 += 2*(e["xy"]*e["yx"] + e["yz"]*e["zy"] + e["xz"]*e["zx"])
  e2 = np.sqrt(0.5*e2)
  
  fig,ax = plt.subplots()
  if plan == 0:
    eplot = ax.scatter(vcoord0[:,0],vcoord0[:,2],s=10,c=e2,cmap="jet",norm=colors.LogNorm(vmin=e2.min(), vmax=e2.max()))
    ax.set(aspect="equal",xlabel="x",ylabel="z",title="Strain rate Second Invariant")
  elif plan == 1:
    eplot = ax.scatter(vcoord0[:,2],vcoord0[:,1],s=10,c=e2,cmap="jet",norm=colors.LogNorm(vmin=e2.min(), vmax=e2.max()))
    ax.set(aspect="equal",xlabel="z",ylabel="y",title="Strain rate Second Invariant")
  elif plan == 2:
    eplot = ax.scatter(vcoord0[:,0],vcoord0[:,1],s=10,c=e2,cmap="jet",norm=colors.LogNorm(vmin=e2.min(), vmax=e2.max()))
    ax.set(aspect="equal",xlabel="x",ylabel="y",title="Strain rate Second Invariant")
  cb1 = plt.colorbar(eplot,ax=ax, shrink=0.8, extend='neither',orientation='horizontal',pad=0.08)
  plt.draw()
  
def DisplayCellFields():
  """
  Takes command line inputs arguments, use with (as an exemple):
  python pTatin_Extract_sections.py -step 250 -nfields 1 -fields viscosity region -elem 512 64 256 -Xsection 1 -plan 2 -inode 1
  """
  try:
    index = sys.argv.index("-step")
  except:
    raise RuntimeError("A step file number must be given via -step {step}")
  step = sys.argv[index+1]
    
  try:
    index = sys.argv.index("-nfields")
  except:
    raise RuntimeError("At least 1 cell field is required, use -nfield {number_of_fields}")
  nfields = int(sys.argv[index+1])
  
  try:
    index = sys.argv.index("-fields")
  except:
    raise RuntimeError("At least 1 cell field is required, use -field {name}")
  fields = []
  for n in range(1,nfields+1):
    fields.append(sys.argv[index + n])
  
  try:
    index = sys.argv.index("-elem")
  except:
    raise RuntimeError("Provide the number of elements via -elem mx my mz")
  mx = int(sys.argv[index+1])
  my = int(sys.argv[index+2])
  mz = int(sys.argv[index+3])
  
  try:
    index = sys.argv.index("-Xsection")
  except:
    raise RuntimeError("Provide information if Xsection or whole domain should be extracted via -Xsection {0 or 1} or {True or False}")
  ans = sys.argv[index+1]
  
  if ans == "0" or ans == "False":
    Xsection = False
  elif ans == "1" or ans == "True":
    Xsection = True
   
  if Xsection == True:
    try:
      index = sys.argv.index("-plan")
    except:
      raise RuntimeError("-Xsection was given as \"True\" therefore a direction must be given via -plan {0 or 1 or 2}")
    plan = int(sys.argv[index+1])
    
    try:
      index = sys.argv.index("-inode")
    except:
      raise RuntimeError("-Xsection was given as \"True\" therefore a node at which make the cross section must be given via -inode {node_number}")
    inode = int(sys.argv[index+1])
  else:
    plan = None
  
  NX_Q1 = mx + 1
  NY_Q1 = my + 1
  NZ_Q1 = mz + 1
  inode_Q1 = inode + 1
  
  cell_coords_file = 'step'+step+'.dmda-cell.coords.vec'
  CellCoords = pTatinReadBinary(cell_coords_file,NX_Q1,NY_Q1,NZ_Q1,Cross_section=Xsection,dim=3,plan=plan,inode=inode_Q1)
  Centre = Compute_cell_center(plan,CellCoords,NX_Q1,NY_Q1,NZ_Q1)
  
  CellFields = {}
  for name in fields:
    CellFields[name] = {"file":'step'+step+'.dmda-cell.'+name+'.vec'}
    CellFields[name]["value"] = pTatinReadBinary(CellFields[name]["file"],mx,my,mz,Cross_section=Xsection,dim=1,plan=plan,inode=inode)
  
    fig,ax = plt.subplots()
    fplot = ax.scatter(Centre[:,0],Centre[:,1],s=10,c=CellFields[name]["value"],cmap="jet",norm=colors.LogNorm(vmin=CellFields[name]["value"].min(), vmax=CellFields[name]["value"].max()))
    if plan == 0:
      ax.set(aspect="equal",xlabel="x",ylabel="z",title=name)
    elif plan == 1:
      ax.set(aspect="equal",xlabel="z",ylabel="y",title=name)
    elif plan == 2:
      ax.set(aspect="equal",xlabel="x",ylabel="y",title=name)
    cb1 = plt.colorbar(fplot,ax=ax, shrink=0.8, extend='neither',orientation='horizontal',pad=0.08)
    plt.draw()

if __name__ == "__main__":

  DisplayCellFields()
  DisplayStrainRate()
  plt.show()