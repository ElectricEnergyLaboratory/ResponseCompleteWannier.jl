using LinearAlgebra
import DFTK
import Wannier
include("schur/schur_operator.jl")
include("adapters/wannier_subspace.jl")
include("adapters/dftk_response.jl")
include("adapters/dftk_dynamic.jl")
include("complement/response_residual.jl")
# Schur reduction and auxiliary linearization
export SchurAuxiliaryModel,
       ResponseComplement,
       FixedShiftOperator,
       auxiliary_component,
       auxiliary_solve,
       build_response_residual,
       compress_response_residual,
       full_block_solve,
       linearized_matrix,
       reduce_schur,
       reduced_schur_solve,
       schur_self_energy,
       schur_solve
       schur_solve,
       shifted_solve
# Wannier / MRWF target subspaces (Bloch space, plane-wave representation)
export WannierSubspaceBlock,
       WannierTargetSpace,
       combine_gauges,
       lowdin_orthonormalize,
       mrwf_subspace,
       orthonormality_defect,
       rotate_subspace,
       subspace_distance,
       subspace_projector,
       target_dimension,
       wannier_block,
       wannier_subspace
# DFTK response adapter
export AbstractResponsePerturbation,
       CustomRHSPerturbation,
       DFTKGroundState,
       DenseSternheimerSolver,
       DynamicResponse,
       ElectricFieldPerturbation,
       PerturbationInfo,
       ResponseOrbitalBlock,
       ScalarPotentialPerturbation,
       SchurGMRESSolver,
       SolverCache,
       dftk_static_response,
       dynamic_response,
       gmres_solve,
       induced_density,
       kplusq_slots,
       nbands,
       nperturbations,
       perturbation_rhs,
       plane_wave_perturbation,
       response_blocks,
       response_view,
       select_bands,
       select_perturbations,
       slot_solutions,
       sternheimer_residual,
       target_index
# Response residual and SVD compression
export ResponseComplement,
       build_response_residual,
       column_weights,
       compress_response_residual,
       naux,
       project_onto_subspace,
       project_out_subspace,
       residual_reconstruction_error,
       retained_capture,
       stack_response_residuals,
       truncation_errors
end # module ResponseCompleteWannier


he geometric-completeness diagnostics (overlap spectrum, lost weight, Ω_I, matrix-log Berry connection).


Create ResponseCompleteWannier.jl/src/diagnostics/geometry.jl


"""
