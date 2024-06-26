
import os as os
import sys as sys
from pathlib import Path
import json as json

def xmfsplat(mx, gname, aname, use_relative_path):
  c = ''
  c += '<?xml version=\"1.0\" encoding=\"utf-8\"?>' + "\n"
  c += '<Xdmf xmlns:xi=\"http://www.w3.org/2001/XInclude\" Version=\"3.0\">' + "\n"
  c += '<Domain>' + "\n"
  c += '  <Grid Name=\"FVDA\" GridType=\"Uniform\">' + "\n"
  
  # i,j,k are written in reverse order with XDMF
  nx = [str(mx[2]+1), str(mx[1]+1), str(mx[0]+1)]
  nkji = " ".join(list(nx))
  c += '    <Topology TopologyType=\"3DSMesh\" Dimensions=\"' + nkji + '\"/>' + "\n"
  
  msize = mx[0] * mx[1] * mx[2]
  nsize = (mx[0]+1) * (mx[1]+1) * (mx[2]+1)
  
  c += '    <Geometry Type=\"XYZ\">' + "\n"
  c += '      <DataItem DataType=\"Float\" Dimensions=\"' + str(nsize) + ' 3\" Format=\"Binary\" Precision=\"8\" Endian=\"Big\" Seek=\"8\">' + "\n"

  filename = gname
  if use_relative_path == True:
    filename = os.path.split(gname)[1] # strip - only keep the tail
  
  c += '        ' + filename + "\n"
  c += '      </DataItem>' + "\n"
  c += '    </Geometry>' + "\n"

  for item in aname:
    c += '    <Attribute Name=\"' + item[1] + '\" Type=\"Scalar\" Center=\"Cell\">' + "\n"
    c += '      <DataItem DataType=\"Float\" Dimensions=\"' + str(msize) + '\" Format=\"Binary\" Precision=\"8\" Endian=\"Big\" Seek=\"8\">' + "\n"
    
    filename = item[0]
    if use_relative_path == True:
      filename = os.path.split(item[0])[1] # strip - only keep the tail
    c += '        ' + filename + "\n"
    c += '      </DataItem>' + "\n"
    c += '    </Attribute>' + "\n"

  c += '  </Grid>' + "\n"
  c += '</Domain>' + "\n"
  c += '</Xdmf>'

  return c


def petsc_vec_json_parse(jname):
  
  jdata = None
  with open(jname, "r") as fp:
    jdata = json.load(fp)

  try:
    vec = jdata['PETScVec']
  except:
    raise RuntimeError('Not a valid PETScVec JSON file')

  f = vec['fileName']
  path = Path(f)
  return str(path)


def fvda_gen_xdmf(fname, field_fname, field_name):
  print("<",fname)
  fvda_root = os.path.split(fname)[0]
  print("  [fvda-root]",fvda_root)
  
  
  jdata = None
  with open(fname, "r") as fp:
    jdata = json.load(fp)

  try:
    fvda = jdata['FVDA']
  except:
    raise RuntimeError('Not a valid FVDA JSON file')

  mx = [fvda['directions'][0]['M'], fvda['directions'][1]['M'], fvda['directions'][2]['M']]
  nx = [mx[0]+1,mx[1]+1,mx[2]+1]

  suffix = fname.replace("fvda.json","")
  cd = jdata['FVDA']['cellFields']
  celldata = list()
  for item in cd:
    _datafile = suffix + "fvda_cellcoeff_" + item['name'] + ".pbvec"
    datafile = str(Path(_datafile))
    celldata.append((datafile, item['name']))

  celldata.append( (field_fname, field_name))

  # check files actually exist
  celldata_found = list()
  for item in celldata:
    print("<", item[0])
    if os.path.exists(item[0]):
      celldata_found.append(item)
      print("  ... found ...")
    else:
      print("  ... not found ...")
  del celldata

  if os.path.exists(field_fname) is False:
    print("  ... FV solution output not found here:",os.path.split(field_fname)[0])

  jcoor = jdata['FVDA']['dm_geometry_coords_json']
  coorfile = petsc_vec_json_parse(os.path.join(fvda_root,jcoor))
  coorfile = os.path.split(coorfile)[1] # strip - only keep the tail
  coorfile = os.path.join(fvda_root,coorfile)
  print("<", coorfile)
  if os.path.exists(coorfile):
    print("  ... found ...")
  else:
    print("  ... not found ...")

  use_relative_path = True
  if fvda_root not in coorfile:
    use_relative_path = False
  for item in celldata_found:
    if fvda_root not in item[0]:
      use_relative_path = False
  print('[note] Using relative paths to reference data files:',use_relative_path)

  blob = xmfsplat(mx, coorfile, celldata_found, use_relative_path)

  fname_out = os.path.split(fname)[1]
  fname_out = fname_out.replace(".json", ".xmf")
  fname_out = os.path.join(fvda_root,fname_out)
  print(">",fname_out)

  with open(fname_out, "w") as fp:
    fp.write(blob)



if __name__ == '__main__':
  
  fvsolfilename = None
  try:
    index = sys.argv.index('-fvQ')
  except:
    raise RuntimeError('Require command line option -fvQ <name-of-petscvec-output-file.pbvec>')
  fvsolfilename = sys.argv[index+1]

  fvsolfield = "Q"
  try:
    index = sys.argv.index('-fvQ_name')
    fvsolfield = sys.argv[index+1]
  except:
    print('[note] Set name of the field via -fvQ_name <string-you-want-to-see-in-paraview>')


  fvdafilename = None
  try:
    index = sys.argv.index('-fvda')
  except:
    raise RuntimeError('Require command line option -fvda <full/path/to/fvda.json>')
  fvdafilename = sys.argv[index+1]
  fvdafilename = os.path.abspath(fvdafilename)

  dir = "./"
  try:
    index = sys.argv.index('-path')
    dir = sys.argv[index+1]
  except:
    fvda_root = os.path.split(fvdafilename)[0]
    dir = fvda_root
    #raise RuntimeError('Require command line option -path')
    print('[note] Assuming FV solution lives here:',fvda_root)
    print('       If this is incorrect, over-ride with command line option -path <path/to/fv-solution-output>')


  fname = os.path.join(dir, fvsolfilename)
  fname = str(Path(fname))
  fname = os.path.abspath(fname)

  # parse
  #fname = "./jout/stepA_fvda.json"

  # generate
  fvda_gen_xdmf(fvdafilename, fname, fvsolfield)
