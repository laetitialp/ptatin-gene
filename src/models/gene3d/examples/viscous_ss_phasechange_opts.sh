########### Gene3d options file ###########
-ptatin_model Gene3D
-output_path GENE_TEST
-activate_energyfv true
-model_GENE3D_output_markers
-model_GENE3D_bc_debug
-view_ic
-model_GENE3D_poisson_pressure_active
-model_GENE3D_temperature_ic_from_file
########### Checkpointing ###########
-checkpoint_every_ncpumins 230
-checkpoint_every_nsteps 10
########### Timestep ###########
-nsteps 2 # Max number of steps
-dt_min 1.0e-6
-dt_max 0.5
-dt_max_surface_displacement 5.0e-4
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
-model_GENE3D_apply_mesh_refinement # activate mesh refinement
# number of directions in which the mesh is refined
-model_GENE3D_refinement_ndir 1 
# array of directions, the number of entries must equal refinement_ndir
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
-model_GENE3D_mesh_file src/models/gene3d/examples/box_ptatin_md.bin
-model_GENE3D_regions_file src/models/gene3d/examples/box_ptatin_region_cell.bin
-model_GENE3D_n_regions 4
-model_GENE3D_regions_list 38,39,40,41
# Method to locate material points in gmsh mesh
# Brute force: 0, Partitioned box: 1
-model_GENE3D_mesh_point_location_method 1
########### Initial plastic strain for weak zone ###########
-model_GENE3D_n_weak_zones 2
-model_GENE3D_wz_expression_0 1.0
-model_GENE3D_wz_expression_1 1.0
########### Markers layout ###########
###### Initial marker layout ######
-lattice_layout_Nx 4
-lattice_layout_Ny 4
-lattice_layout_Nz 4
###### Marker management ######
# min/max marker per cell
-mp_popctrl_np_lower 8
-mp_popctrl_np_upper 128
# marker injection in cells
-mp_popctrl_nxp 2
-mp_popctrl_nyp 2
-mp_popctrl_nzp 2
# Boundary conditions for markers, perform injection and no cleaning of markers on marked faces
# Faces numbering:
#  0: east  = xmax = imax = Pxi
#  1: west  = xmin = imin = Nxi
#  2: north = ymax = jmax = Peta
#  3: south = ymin = jmin = Neta
#  4: front = zmax = kmax = Pzeta
#  5: back  = zmin = kmin = Nzeta
-model_GENE3D_bc_marker_n_faces 4
-model_GENE3D_bc_marker_faces_list 0,1,4,5
########### Initial velocity field ###########
# if nothing is provided the velocity is initialized to 0
-model_GENE3D_v_init_n_dir 2
-model_GENE3D_v_init_dir 0,2
-model_GENE3D_v_init_expression_0 -5.28496533062743e-16*x+1.97237591301416e-15*z-1.37307427033301e-10
-model_GENE3D_v_init_expression_2 -1.4161021923681e-16*x+5.28496533062743e-16*z-3.67914141883684e-11
########### Passive tracers ###########
#-model_GENE3D_apply_passive_markers
########### Boundary conditions ###########
-model_GENE3D_poisson_pressure_surface_p 0.0 # 0.0 is default => not necessary
###### Temperature ######
-model_GENE3D_energy_bc_ymax 0.0
-model_GENE3D_energy_bc_ymin 1450.0
###### Velocity ######
-model_GENE3D_bc_nsubfaces 5
-model_GENE3D_bc_tag_list 14,23,32,33,37
###### Boundary conditions types ######
# 0: NONE,                 1: TRACTION
# 2: DEMO,                 3: FSSA
# 4: NITSCHE_DIRICHLET,    5: NITSCHE_NAVIER_SLIP
# 6: NITSCHE_GENERAL_SLIP, 7: DIRICHLET
#######################################
# BC 14: Xmin
-model_GENE3D_sc_name_14 xmin
-model_GENE3D_facet_mesh_file_14 src/models/gene3d/examples/box_ptatin_facet_14_mesh.bin
-model_GENE3D_sc_type_14 6
-model_GENE3D_bc_navier_penalty_14 1.0e3
-model_GENE3D_bc_navier_duxdx_14 -5.28496533062743e-16
-model_GENE3D_bc_navier_duxdz_14 1.97237591301416e-15
-model_GENE3D_bc_navier_duzdx_14 -1.41610219236810e-16
-model_GENE3D_bc_navier_duzdz_14 5.28496533062743e-16
-model_GENE3D_bc_navier_uL_14 3.0629307023372284e-10,8.207098081637518e-11
-model_GENE3D_bc_navier_mathcal_H_14 0,1,0,1,1,1
# BC 23: Zmax
-model_GENE3D_sc_name_23 zmax
-model_GENE3D_facet_mesh_file_23 src/models/gene3d/examples/box_ptatin_facet_23_mesh.bin
-model_GENE3D_sc_type_23 7
-model_GENE3D_ux_23 -5.28496533062743e-16*x+1.97237591301416e-15*z-1.37307427033301e-10
-model_GENE3D_uz_23 -1.4161021923681e-16*x+5.28496533062743e-16*z-3.67914141883684e-11
# BC 32: Xmax
-model_GENE3D_sc_name_32 xmax_litho
-model_GENE3D_facet_mesh_file_32 src/models/gene3d/examples/box_ptatin_facet_32_mesh.bin
-model_GENE3D_sc_type_32 6
-model_GENE3D_bc_navier_penalty_32 1.0e3
-model_GENE3D_bc_navier_duxdx_32 -5.28496533062743e-16
-model_GENE3D_bc_navier_duxdz_32 1.97237591301416e-15
-model_GENE3D_bc_navier_duzdx_32 -1.41610219236810e-16
-model_GENE3D_bc_navier_duzdz_32 5.28496533062743e-16
-model_GENE3D_bc_navier_uL_32 3.0629307023372284e-10,8.207098081637518e-11
-model_GENE3D_bc_navier_mathcal_H_32 0,1,0,1,1,1
# BC 33: Bottom
-model_GENE3D_sc_name_33 Bottom
-model_GENE3D_facet_mesh_file_33 src/models/gene3d/examples/box_ptatin_facet_33_mesh.bin
-model_GENE3D_sc_type_33 7
-model_GENE3D_dirichlet_bot_u.n_33
# BC 37: Zmin
-model_GENE3D_sc_name_37 zmin_litho
-model_GENE3D_facet_mesh_file_37 src/models/gene3d/examples/box_ptatin_facet_37_mesh.bin
-model_GENE3D_sc_type_37 7
-model_GENE3D_ux_37 -5.28496533062743e-16*x+1.97237591301416e-15*z-1.37307427033301e-10
-model_GENE3D_uz_37 -1.4161021923681e-16*x+5.28496533062743e-16*z-3.67914141883684e-11
########### Material parameters ###########
###### Upper crust ######
# viscosity type
# 0: VISCOUS_CONSTANT,    1: VISCOUS_FRANKK,
# 2: VISCOUS_Z,           3: VISCOUS_ARRHENIUS,
# 4: VISCOUS_ARRHENIUS_2, 5: VISCOUS_ARRHENIUS_DISLDIFF
-model_GENE3D_visc_type_38 0
-model_GENE3D_eta0_38 1.0e+22
# plasticity type
# 0: PLASTIC_NONE, 1: PLASTIC_MISES
# 2: PLASTIC_DP,   3: PLASTIC_MISES_H
# 4: PLASTIC_DP_H
-model_GENE3D_plastic_type_38 0
# softening type
# 0: SOFTENING_NONE,       1: SOFTENING_LINEAR
# 2: SOFTENING_EXPONENTIAL
-model_GENE3D_softening_type_38 0
# density type
# 0:DENSITY_CONSTANT,  1: DENSITY_BOUSSINESQ
# 2:DENSITY_TABLE
-model_GENE3D_density_type_38 0
-model_GENE3D_density_38 3000.0
# energy
-model_GENE3D_heat_source_38 0.0 #1.5e-6
-model_GENE3D_conductivity_38 3.3
###### Lower crust ######
-model_GENE3D_visc_type_39 0
-model_GENE3D_eta0_39 1.0e+22
-model_GENE3D_plastic_type_39 0
-model_GENE3D_softening_type_39 0
-model_GENE3D_density_type_39 0
-model_GENE3D_density_39 3000.0
-model_GENE3D_heat_source_39 0.0
-model_GENE3D_conductivity_39 3.3
###### Lithosphere ######
-model_GENE3D_visc_type_40 0
-model_GENE3D_eta0_40 1.0e+22
-model_GENE3D_plastic_type_40 0
-model_GENE3D_softening_type_40 0
-model_GENE3D_density_type_40 2
-model_GENE3D_map_40 example-optionsfiles/model_geometry
-model_GENE3D_density_40 3000.0
-model_GENE3D_heat_source_40 0.0
-model_GENE3D_conductivity_40 3.3
###### Asthenosphere ######
-model_GENE3D_visc_type_41 0
-model_GENE3D_eta0_41 1.0e+22
-model_GENE3D_plastic_type_41 0
-model_GENE3D_softening_type_41 0
-model_GENE3D_density_type_41 0
-model_GENE3D_density_41 3000.0
-model_GENE3D_heat_source_41 0.0
-model_GENE3D_conductivity_41 3.3
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
  -fieldsplit_u_mg_levels_pc_type jacobi
  -fieldsplit_u_mg_levels_ksp_max_it 8
  -fieldsplit_u_mg_levels_ksp_norm_type NONE
    #---- Coarse Grid ----#
    # 4 x 4 x 4 #
    -fieldsplit_u_mg_coarse_pc_type cholesky
    #-fieldsplit_u_mg_coarse_pc_type telescope
    #-fieldsplit_u_mg_coarse_telescope_pc_type cholesky
    #-fieldsplit_u_mg_coarse_telescope_pc_reduction_factor 2
####### Temperature ######
-ptatin_energyfv_nsub 3,3,3
#-energyfv_ksp_monitor
-energyfv_snes_converged_reason
-energyfv_ksp_type fgmres
-energyfv_snes_ksp_ew
-energyfv_snes_ksp_ew_rtol0 1.0e-2
-energyfv_snes_ksp_ew_version 1
-energyfv_snes_lag_preconditioner -2
-energyfv_snes_monitor
-energyfv_snes_rtol 1.0e-6
-energyfv_mg_levels_esteig_ksp_norm_type none
-energyfv_mg_levels_esteig_ksp_type cg
-energyfv_mg_levels_ksp_chebyshev_esteig 0,0.01,0,1.1
-energyfv_mg_levels_ksp_max_it 8
#-energyfv_mg_levels_ksp_norm_type none
-energyfv_mg_levels_ksp_type gmres #chebyshev
-energyfv_mg_levels_pc_type bjacobi
-energyfv_pc_type gamg
-fvpp_ksp_monitor
-fvpp_ksp_rtol 1.0e-10
-fvpp_ksp_type cg
-fvpp_mg_coarse_pc_type redundant
-fvpp_mg_coarse_redundant_pc_factor_mat_solver_type gamg #mkl_pardiso 
-fvpp_mg_coarse_redundant_pc_type gamg
-fvpp_mg_levels_esteig_ksp_norm_type none
-fvpp_mg_levels_esteig_ksp_type cg
-fvpp_mg_levels_ksp_chebyshev_esteig 0,0.01,0,1.1
-fvpp_mg_levels_ksp_max_it 4
-fvpp_mg_levels_ksp_norm_type none
-fvpp_mg_levels_ksp_type chebyshev
-fvpp_mg_levels_pc_type jacobi
-fvpp_operator_fvspace false
-fvpp_pc_type gamg 
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
