########### Gene3d options file ###########
-ptatin_model Gene3D
-output_path GENE_TEST
-model_GENE3D_output_markers
-model_GENE3D_poisson_pressure_active
########### Bounding Box ###########
-model_GENE3D_Ox 0.0
-model_GENE3D_Oy -300.0e3
-model_GENE3D_Oz 0.0
-model_GENE3D_Lx 1000.0e3
-model_GENE3D_Ly 0.0
-model_GENE3D_Lz 600.0e3
########### Mesh ###########
-mx 32
-my 32
-mz 32
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
-model_GENE3D_spm_diffusion_dirichlet_xmin
-model_GENE3D_spm_diffusion_dirichlet_xmax
#-model_GENE3D_spm_diffusion_dirichlet_zmin
#-model_GENE3D_spm_diffusion_dirichlet_zmax
########### Initial geometry ###########
-model_GENE3D_mesh_file src/models/gene3d/examples/subduction_ptatin_md.bin
-model_GENE3D_regions_file src/models/gene3d/examples/subduction_ptatin_region_cell.bin
-model_GENE3D_n_regions 7
-model_GENE3D_regions_list 76,77,78,79,80,81,82
# Method to locate material points in gmsh mesh
# Brute force: 0, Partitioned box: 1
-model_GENE3D_mesh_point_location_method 1
########### Initial plastic strain for weak zone ###########
-model_GENE3D_n_weak_zones 2
-model_GENE3D_wz_expression_0 exp(-((0.5*3e-5^2)*(x-600e3)^2+(0.5*3e-5^2)*(z-200e3)^2))
-model_GENE3D_wz_expression_1 exp(-((0.5*3e-5^2)*(x-200e3)^2+(0.5*3e-5^2)*(z-400e3)^2))
########### Markers layout ###########
-lattice_layout_Nx 2
-lattice_layout_Ny 2
-lattice_layout_Nz 2
########### Initial velocity field ###########
# if nothing is provided the velocity is initialized to 0
-model_GENE3D_v_init_n_dir 3
-model_GENE3D_v_init_dir 0,1,2
-model_GENE3D_v_init_expression_0 2.0/(600e3)*z*(-2)-(-2)
-model_GENE3D_v_init_expression_1 2.0*y*(-2)*(300e3)/((600e3)*(1000e3))
-model_GENE3D_v_init_expression_2 2.0/(600e3)*z*(-2)-(-2)
########### Passive tracers ###########
#-model_GENE3D_apply_passive_markers
########### Boundary conditions ###########
-model_GENE3D_bc_nsubfaces 5
-model_GENE3D_bc_tag_list 83,84,85,86,87
###### Boundary conditions types ######
# 0: NONE,                 1: TRACTION
# 2: DEMO,                 3: FSSA
# 4: NITSCHE_DIRICHLET,    5: NITSCHE_NAVIER_SLIP
# 6: NITSCHE_GENERAL_SLIP, 7: DIRICHLET
#######################################
# BC 83: Bottom
-model_GENE3D_sc_name_83 Bottom
-model_GENE3D_facet_mesh_file_83 src/models/gene3d/examples/subduction_ptatin_facet_83_mesh.bin
-model_GENE3D_sc_type_83 7
-model_GENE3D_ux_83 1.0*x*t
-model_GENE3D_uy_83 1.0*y
-model_GENE3D_uz_83 1.0*z
# BC 84: Asthenosphere
-model_GENE3D_sc_name_84 Asthenosphere
-model_GENE3D_facet_mesh_file_84 src/models/gene3d/examples/subduction_ptatin_facet_84_mesh.bin
-model_GENE3D_sc_type_84 1
-model_GENE3D_dev_stress_84  2e7*cos(0.5236)+p*sin(0.5236)
#-model_GENE3D_ux_84 -1.0*x
#-model_GENE3D_uy_84 -1.0*y
#-model_GENE3D_uz_84 -1.0*z
# BC 85: Ocean_D
-model_GENE3D_sc_name_85 Ocean_D
-model_GENE3D_facet_mesh_file_85 src/models/gene3d/examples/subduction_ptatin_facet_85_mesh.bin
-model_GENE3D_sc_type_85 7
-model_GENE3D_ux_85 -1.0*x
#-model_GENE3D_uy_85 0.0*y
-model_GENE3D_uz_85 -3.0*z
# BC 86: Zmax
-model_GENE3D_sc_name_86 Zmax
-model_GENE3D_facet_mesh_file_86 src/models/gene3d/examples/subduction_ptatin_facet_86_mesh.bin
-model_GENE3D_sc_type_86 7
-model_GENE3D_ux_86 1.0*x
-model_GENE3D_uy_86 0.0*y
#-model_GENE3D_uz_86 3.0*z
# BC 87: Continent_D
-model_GENE3D_sc_name_87 Continent_D
-model_GENE3D_facet_mesh_file_87 src/models/gene3d/examples/subduction_ptatin_facet_87_mesh.bin
-model_GENE3D_sc_type_87 7
#-model_GENE3D_ux_87 1.0*x
#-model_GENE3D_uy_87 0.0*y
-model_GENE3D_uz_87 -3.0*z
########### Material parameters ###########
###### Asthenosphere ######
-model_GENE3D_visc_type_76 0
-model_GENE3D_eta0_76 1.0e+20
-model_GENE3D_plastic_type_76 0
-model_GENE3D_softening_type_76 0
-model_GENE3D_density_type_76 0
-model_GENE3D_density_76 3350.0
###### Oceanic lithosphere ######
-model_GENE3D_visc_type_77 0
-model_GENE3D_eta0_77 1.0e+22
-model_GENE3D_plastic_type_77 0
-model_GENE3D_softening_type_77 0
-model_GENE3D_density_type_77 0
-model_GENE3D_density_77 3300.0
###### Oceanic crust ######
-model_GENE3D_visc_type_78 0
-model_GENE3D_eta0_78 1.0e+22
-model_GENE3D_plastic_type_78 0
-model_GENE3D_softening_type_78 0
-model_GENE3D_density_type_78 0
-model_GENE3D_density_78 3000.0
###### Continental lithosphere ######
-model_GENE3D_visc_type_79 0
-model_GENE3D_eta0_79 1.0e+22
-model_GENE3D_plastic_type_79 0
-model_GENE3D_softening_type_79 0
-model_GENE3D_density_type_79 0
-model_GENE3D_density_79 3300.0
###### Weak zone ######
-model_GENE3D_visc_type_80 0
-model_GENE3D_eta0_80 1.0e+19
-model_GENE3D_plastic_type_80 0
-model_GENE3D_softening_type_80 0
-model_GENE3D_density_type_80 0
-model_GENE3D_density_80 3300.0
###### Continental lower crust ######
-model_GENE3D_visc_type_81 0
-model_GENE3D_eta0_81 1.0e+20
-model_GENE3D_plastic_type_81 0
-model_GENE3D_softening_type_81 0
-model_GENE3D_density_type_81 0
-model_GENE3D_density_81 3300.0
###### Continental upper crust ######
-model_GENE3D_visc_type_82 0
-model_GENE3D_eta0_82 1.0e+23
-model_GENE3D_plastic_type_82 0
-model_GENE3D_softening_type_82 0
-model_GENE3D_density_type_82 0
-model_GENE3D_density_82 3300.0
###############################
########### SOLVERS ###########
###############################
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