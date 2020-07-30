libptatin3d-y.c += $(call thisdir, \
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
)

TATIN_INC += -I$(abspath $(call thisdir,.))

