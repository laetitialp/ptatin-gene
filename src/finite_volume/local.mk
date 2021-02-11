libptatin3d-y.c += $(call thisdir, \
  dmda_warp.c \
  kdtree.c \
  fvda.c \
  fvda_property.c \
  fvda_compatible_velocity.c \
  fvda_view.c \
  fv_ops_time_dep.c \
  fv_ops_ale.c \
  fvda_bc_utils.c \
  fvda_ale_utils.c \
  fvda_project.c \
  fvda_reconstruction.c \
  ptatin3d_energyfv.c \
  fvda_dimap.c \
)

ptatin-tests-y.c += $(call thisdir, \
  fv-ex1.c \
  fv-ex2.c \
  fv-ex3.c fv-ex3-fas.c \
  fv-ex4.c \
  fv-ex5.c \
  fv-ex6.c \
  fv-ex7a.c \
  fv-ex7b.c \
  fv-ex8.c \
  fv-ex9.c \
  fv-ex10.c \
)

TATIN_INC += -I$(abspath $(call thisdir,.))

