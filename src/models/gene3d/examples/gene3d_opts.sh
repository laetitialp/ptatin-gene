########### Gene3d options file ###########
-ptatin_model Gene3D
-output_path GENE_TEST
-model_GENE3D_output_markers
-model_GENE3D_poisson_pressure_active
#-model_GENE3D_bc_debug
########### Bounding Box ###########
-model_GENE3D_Ox 0.0
-model_GENE3D_Oy -250.0e3
-model_GENE3D_Oz 0.0
-model_GENE3D_Lx 600.0e3
-model_GENE3D_Ly 0.0
-model_GENE3D_Lz 300.0e3
########### Mesh ###########
-mx 8
-my 8
-mz 8
###### Mesh refinement ######
-model_GENE3D_apply_mesh_refinement
# number of directions in which the mesh is refined
-model_GENE3D_n_refinement_dir 1 
# array of directions, the number of entries must equal n_refinement_dir
# X: 0, Y: 1, Z: 2
-model_GENE3D_refinement_dir 1 
###### y refinement: 1 ######
-model_GENE3D_refinement_npoints_1 4
-model_GENE3D_refinement_xref_1 0.0,0.28,0.65,1.0
-model_GENE3D_refinement_xnat_1 0.0,0.8,0.935,1.0
########### Surface processes ###########
-model_GENE3D_apply_surface_diffusion
# spm diffusivity
-model_GENE3D_diffusivity_spm 1.0e-6
# spm diffusion boundary conditions minimum 1 must be passed
#-model_GENE3D_spm_diffusion_dirichlet_xmin
#-model_GENE3D_spm_diffusion_dirichlet_xmax
-model_GENE3D_spm_diffusion_dirichlet_zmin
-model_GENE3D_spm_diffusion_dirichlet_zmax
########### Initial geometry ###########
-model_GENE3D_mesh_file src/models/gene3d/examples/test_ptatin_md.bin
-model_GENE3D_regions_file src/models/gene3d/examples/test_ptatin_region_cell.bin
-model_GENE3D_n_regions 4
-model_GENE3D_regions_list 1,2,3,4
# Method to locate material points in gmsh mesh
# Brute force: 0, Partitioned box: 1
-model_GENE3D_mesh_point_location_method 1
########### Initial plastic strain for weak zone ###########
-model_GENE3D_n_weak_zones 2
-model_GENE3D_wz_expression_0 exp(-((0.5*3e-5^2)*(x-600e3)^2+(0.5*3e-5^2)*(z-200e3)^2))
-model_GENE3D_wz_expression_1 exp(-((0.5*3e-5^2)*(x-200e3)^2+(0.5*3e-5^2)*(z-400e3)^2))
########### Markers layout ###########
###### Initial marker layout ######
-lattice_layout_Nx 2
-lattice_layout_Ny 2
-lattice_layout_Nz 2
###### Marker management ######
# min/max marker per cell
-mp_popctrl_np_lower 8
-mp_popctrl_np_upper 64
# marker injection in cells
-mp_popctrl_nxp 2
-mp_popctrl_nyp 2
-mp_popctrl_nzp 2
# Boundary conditions for markers
# Faces numbering:
#  0: east  = xmax = imax = Pxi
#  1: west  = xmin = imin = Nxi
#  2: north = ymax = jmax = Peta
#  3: south = ymin = jmin = Neta
#  4: front = zmax = kmax = Pzeta
#  5: back  = zmin = kmin = Nzeta
-model_GENE3D_bc_marker_n_faces 5
-model_GENE3D_bc_marker_faces_list 0,1,4,5,3
########### Initial velocity field ###########
# if nothing is provided the velocity is initialized to 0
-model_GENE3D_v_init_n_dir 2
-model_GENE3D_v_init_dir 0,2
-model_GENE3D_v_init_expression_0 -1.05699306612549e-15*x-3.94475182602832e-15*z+9.08810693741895e-10
-model_GENE3D_v_init_expression_2 2.8322043847362e-16*x+1.05699306612549e-15*z-2.43515091460909e-10
########### Passive tracers ###########
#-model_GENE3D_apply_passive_markers
########### Boundary conditions ###########
-model_GENE3D_poisson_pressure_surface_p 0.0 # 0.0 is default => not necessary
###### Temperature ######
-model_GENE3D_energy_bc_ymax 0.0
-model_GENE3D_energy_bc_ymin 1450.0
###### Velocity ######
-model_GENE3D_bc_nsubfaces 6
-model_GENE3D_bc_tag_list 1,2,3,4,5,6
###### Boundary conditions types ######
# 0: NONE,                 1: TRACTION
# 2: DEMO,                 3: FSSA
# 4: NITSCHE_DIRICHLET,    5: NITSCHE_NAVIER_SLIP
# 6: NITSCHE_GENERAL_SLIP, 7: DIRICHLET
#######################################
# BC 1: Lithosphere Xmin
-model_GENE3D_sc_name_1 xmin_litho
-model_GENE3D_facet_mesh_file_1 src/models/gene3d/examples/test_ptatin_facet_1_mesh.bin
-model_GENE3D_sc_type_1 6
-model_GENE3D_bc_navier_penalty_1 1.0e3
-model_GENE3D_bc_navier_uO_1 6.34195840e-10,0.0
-model_GENE3D_bc_navier_uL_1 -6.34195840e-10,0.0
-model_GENE3D_bc_navier_variation_dir_1 2
-model_GENE3D_bc_navier_rotation_angle_1 15.0
-model_GENE3D_bc_navier_mathcal_H_1 0,1,0,1,1,1
# BC 2: Lithosphere Xmax
-model_GENE3D_sc_name_2 xmax_litho
-model_GENE3D_facet_mesh_file_2 src/models/gene3d/examples/test_ptatin_facet_2_mesh.bin
-model_GENE3D_sc_type_2 6
-model_GENE3D_bc_navier_penalty_2 1.0e3
-model_GENE3D_bc_navier_uO_2 6.34195840e-10,0.0
-model_GENE3D_bc_navier_uL_2 -6.34195840e-10,0.0
-model_GENE3D_bc_navier_variation_dir_2 2
-model_GENE3D_bc_navier_rotation_angle_2 15.0
-model_GENE3D_bc_navier_mathcal_H_2 0,1,0,1,1,1
# BC 3: Asthenosphere
-model_GENE3D_sc_name_3 Asthenosphere
-model_GENE3D_facet_mesh_file_3 src/models/gene3d/examples/test_ptatin_facet_3_mesh.bin
-model_GENE3D_sc_type_3 1
# BC 4: Bottom
-model_GENE3D_sc_name_4 Bottom
-model_GENE3D_facet_mesh_file_4 src/models/gene3d/examples/test_ptatin_facet_4_mesh.bin
-model_GENE3D_sc_type_4 7
-model_GENE3D_dirichlet_bot_u.n_4
# BC 5: Lithosphere Zmin
-model_GENE3D_sc_name_5 zmin_litho
-model_GENE3D_facet_mesh_file_5 src/models/gene3d/examples/test_ptatin_facet_5_mesh.bin
-model_GENE3D_sc_type_5 7
-model_GENE3D_ux_5 -1.05699306612549e-15*x-3.94475182602832e-15*z+9.08810693741895e-10
-model_GENE3D_uz_5 2.8322043847362e-16*x+1.05699306612549e-15*z-2.43515091460909e-10
# BC 6: Lithosphere Zmax
-model_GENE3D_sc_name_6 zmax_litho
-model_GENE3D_facet_mesh_file_6 src/models/gene3d/examples/test_ptatin_facet_6_mesh.bin
-model_GENE3D_sc_type_6 7
-model_GENE3D_ux_6 -1.05699306612549e-15*x-3.94475182602832e-15*z+9.08810693741895e-10
-model_GENE3D_uz_6 2.8322043847362e-16*x+1.05699306612549e-15*z-2.43515091460909e-10
########### Material parameters ###########
###### Upper crust ######
-model_GENE3D_visc_type_1 0
-model_GENE3D_eta0_1 1.0e+23
-model_GENE3D_plastic_type_1 0
-model_GENE3D_softening_type_1 0
-model_GENE3D_density_type_1 0
-model_GENE3D_density_1 2700.0
###### Asthenosphere ######
-model_GENE3D_visc_type_2 0
-model_GENE3D_eta0_2 1.0e+20
-model_GENE3D_plastic_type_2 0
-model_GENE3D_softening_type_2 0
-model_GENE3D_density_type_2 0
-model_GENE3D_density_2 3300.0
###### Lithosphere ######
-model_GENE3D_visc_type_3 0
-model_GENE3D_eta0_3 1.0e+24
-model_GENE3D_plastic_type_3 0
-model_GENE3D_softening_type_3 0
-model_GENE3D_density_type_3 0
-model_GENE3D_density_3 3200.0
###### Lower crust ######
-model_GENE3D_visc_type_4 0
-model_GENE3D_eta0_4 1.0e+21
-model_GENE3D_plastic_type_4 0
-model_GENE3D_softening_type_4 0
-model_GENE3D_density_type_4 0
-model_GENE3D_density_4 2850.0
###############################
########### SOLVERS ###########
###############################
###### MG
-A11_operator_type 2,2,0
-dau_nlevels 3
-stk_velocity_dm_mat_type aij
-a11_op avx
#---- 4 MG lvl ----#
#| 32 x 32 x 32 |
#| 16 x 16 x 16 |
#|  8 x  8 x  8 |
#|  4 x  4 x  4 |
#------------------#
#-stk_velocity_da_refine_hierarchy_x 2,2
#-stk_velocity_da_refine_hierarchy_y 2,2
#-stk_velocity_da_refine_hierarchy_z 2,2
#---- SNES ----#
-newton_its 0
-snes_atol 1e-6
-snes_max_it 5
-snes_rtol 1e-8
-snes_monitor
-snes_converged_reason
-snes_linesearch_type basic
#-snes_view
#---- KSP ----#
-ksp_max_it 200
-ksp_rtol 1.0e-3
-ksp_type fgmres
-pc_type fieldsplit
-pc_fieldsplit_schur_fact_type upper
-pc_fieldsplit_type schur
  #---- Fieldsplit p ----#
  -fieldsplit_p_ksp_type preonly
  -fieldsplit_p_pc_type bjacobi
  #---- Fieldsplit u ----#
  -fieldsplit_u_ksp_max_it 200
  #-fieldsplit_u_ksp_monitor_true_residual
  -fieldsplit_u_ksp_monitor
  -fieldsplit_u_ksp_converged_reason
  -fieldsplit_u_ksp_rtol 1.0e-2
  -fieldsplit_u_ksp_type fgmres
  -fieldsplit_u_pc_type mg
  -fieldsplit_u_pc_mg_levels 3
  -fieldsplit_u_pc_mg_cycle_type v
  -fieldsplit_u_mg_levels_ksp_type fgmres
  -fieldsplit_u_mg_levels_pc_type bjacobi
  -fieldsplit_u_mg_levels_ksp_max_it 8
  -fieldsplit_u_mg_levels_ksp_norm_type NONE
    #---- Coarse Grid ----#
    # 4 x 4 x 4 #
    -fieldsplit_u_mg_coarse_pc_type cholesky
    #-fieldsplit_u_mg_coarse_pc_type telescope
    #-fieldsplit_u_mg_coarse_telescope_pc_type cholesky
    #-fieldsplit_u_mg_coarse_telescope_pc_reduction_factor 2
####### Poisson Pressure ######
#-LP_snes_atol 1.0e-9
-LP_snes_rtol 1.0e-6
#-LP_snes_max_it 10
-LP_snes_monitor
-LP_snes_converged_reason
-LP_ksp_monitor
-LP_ksp_converged_reason
-LP_ksp_atol 1.0e-10
-LP_ksp_rtol 1.0e-6
-LP_ksp_type fgmres
#-LP_snes_mf_operator
-LP_pc_type bjacobi #gamg #lu not parallel
#-LP_snes_type ksponly
-LP_snes_ksp_ew
-LP_snes_ksp_ew_rtol0 1.0e-2
-LP_snes_ksp_ew_version 1
-LP_snes_lag_preconditioner -2