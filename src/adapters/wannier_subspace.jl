"""
Bloch-space Wannier / MRWF target subspaces.
The projection of a first-order response `δψ` onto the Wannier subspace has to
be done in the *same* plane-wave representation as `δψ`, never in real space.
This file therefore assembles
    W_k = Ψ_k[:, bands] * U_k            (N_G(k) × N_wannier)
from the plane-wave coefficients `Ψ_k` of the DFTK orbitals and the gauge
matrices of Wannier.jl. All Wannierization steps (disentanglement, the
Qiao–Pizzi–Marzari split/remix into target manifolds, parallel transport and
maximal localization) are delegated to Wannier.jl; only the Bloch-space
assembly is implemented here.
Wannier.jl convention (verified against `Wannier.jl` 0.3.6): `Model.U` has
size `n_bands × n_wann × n_kpts` and is the *combined* gauge
`U_k = Udis_k * X_k`, i.e. the semi-unitary disentanglement matrix `Udis_k`
(`n_bands × n_wann`) acts first and the unitary Wannier rotation `X_k`
(`n_wann × n_wann`) is multiplied from the right. The Wannier function is
`|w_{n k}⟩ = Σ_m |u_{m k}⟩ U_k[m, n]`, so `W_k = Ψ_k U_k`.
"""
"""
    WannierSubspaceBlock(ik, data, wannier_indices=1:size(data, 2);
                         orthonormality_tol=1e-8, orthonormalize=false)
Orthonormal Bloch-space basis `W_k = [|w_1k⟩ … |w_Nk⟩]` of the target subspace at
k-point `ik`.
- `size(data) = (N_G(k), N_wannier)`: rows are plane-wave coefficients in the
  same representation as the DFTK orbitals and responses, columns are the
  target functions.
- `data' * data ≈ I`. The projector `P_W(k) = W_k W_k†` only depends on
  `span(W_k)`, so any unitary `W_k → W_k V` leaves it unchanged.
- `wannier_indices` labels the target functions inside the Wannier model.
With `orthonormalize=true` the columns are Löwdin-orthonormalized instead of
being rejected when `‖W†W - I‖_F > orthonormality_tol`.
"""
struct WannierSubspaceBlock{T<:Number,MT<:AbstractMatrix{T}}
    ik::Int
    data::MT
    wannier_indices::Vector{Int}
    function WannierSubspaceBlock{T,MT}(ik::Int, data::MT,
                                        wannier_indices::Vector{Int}) where {T<:Number,MT<:AbstractMatrix{T}}
        ik >= 1 || throw(ArgumentError("k-point index must be positive"))
        size(data, 2) == length(wannier_indices) || throw(DimensionMismatch(
            "data has $(size(data, 2)) columns but $(length(wannier_indices)) " *
            "wannier_indices were given"))
        allunique(wannier_indices) || throw(ArgumentError("wannier_indices must be unique"))
        size(data, 1) >= size(data, 2) || throw(DimensionMismatch(
            "a subspace cannot have more columns than plane-wave rows"))
        return new{T,MT}(ik, data, wannier_indices)
    end
end
"""
    orthonormality_defect(W)
Frobenius norm `‖W†W - I‖_F` of the deviation of the columns of `W` from
orthonormality.
"""
orthonormality_defect(W::AbstractMatrix) = norm(adjoint(W) * W - I)
"""
    lowdin_orthonormalize(X; rank_tol=1e-8)
Closest matrix with orthonormal columns (polar factor of the thin SVD). The
span of `X` is unchanged. Throws if `X` is numerically rank deficient.
"""
function lowdin_orthonormalize(X::AbstractMatrix; rank_tol::Real=1e-8)
    factorization = svd(X; full=false)
    minimum(factorization.S) > rank_tol * maximum(factorization.S) ||
        throw(ArgumentError("columns are numerically linearly dependent"))
    return factorization.U * factorization.Vt
end
function WannierSubspaceBlock(ik::Integer, data::AbstractMatrix,
                              wannier_indices::AbstractVector{<:Integer}=1:size(data, 2);
                              orthonormality_tol::Real=1e-8,
                              orthonormalize::Bool=false)
    matrix = orthonormalize ? lowdin_orthonormalize(data) : data
    defect = orthonormality_defect(matrix)
    defect <= orthonormality_tol || throw(ArgumentError(
        "columns of the Wannier subspace block at ik=$ik are not orthonormal " *
        "(‖W†W - I‖_F = $defect > $orthonormality_tol); pass orthonormalize=true " *
        "to Löwdin-orthonormalize"))
    return WannierSubspaceBlock{eltype(matrix),typeof(matrix)}(
        Int(ik), matrix, collect(Int, wannier_indices))
end
Base.size(block::WannierSubspaceBlock) = size(block.data)
Base.size(block::WannierSubspaceBlock, dim::Integer) = size(block.data, dim)
"""
    subspace_projector(W)
Dense projector `P_W = W W†` (intended for small systems and tests).
"""
subspace_projector(W::AbstractMatrix) = W * adjoint(W)
subspace_projector(block::WannierSubspaceBlock) = subspace_projector(block.data)
"""
    rotate_subspace(block, V; tol=1e-8)
    rotate_subspace(target, V)
Apply the internal gauge rotation `W → W V` with unitary `V`. The projector
`P_W` and every response residual are invariant under this transformation.
For a target space `V` is an `N × N × n_kpts` array indexed by the k-index of
each block.
"""
function rotate_subspace(block::WannierSubspaceBlock, V::AbstractMatrix; tol::Real=1e-8)
    size(V, 1) == size(V, 2) == size(block.data, 2) ||
        throw(DimensionMismatch("V must be a square matrix of size N_wannier"))
    norm(adjoint(V) * V - I) <= tol || throw(ArgumentError("V must be unitary"))
    return WannierSubspaceBlock(block.ik, block.data * V, block.wannier_indices;
                                orthonormality_tol=max(tol, 1e-8))
end
"""
    subspace_distance(A, B)
Frobenius distance `‖P_A - P_B‖_F = sqrt(nA + nB - 2‖A†B‖_F²)` between the
subspaces spanned by orthonormal `A` and `B`, evaluated without forming the
projectors.
"""
function subspace_distance(A::AbstractMatrix, B::AbstractMatrix)
    size(A, 1) == size(B, 1) || throw(DimensionMismatch("A and B must have the same row count"))
    squared = size(A, 2) + size(B, 2) - 2 * norm(adjoint(A) * B)^2
    return sqrt(max(squared, zero(squared)))
end
subspace_distance(A::WannierSubspaceBlock, B::WannierSubspaceBlock) =
    subspace_distance(A.data, B.data)
"""
    WannierTargetSpace
Collection of [`WannierSubspaceBlock`](@ref)s, one per k-point, with
`kind ∈ (:wannier, :mrwf)` and construction metadata. It behaves as a vector
of blocks; use [`wannier_block`](@ref) to look a block up by its k-index.
Both kinds define the target projector `P_W(k) = W_k W_k†` through the same
interface.
"""
struct WannierTargetSpace{T<:Number,MT<:AbstractMatrix{T}} <:
       AbstractVector{WannierSubspaceBlock{T,MT}}
    kind::Symbol
    blocks::Vector{WannierSubspaceBlock{T,MT}}
    metadata::NamedTuple
    lookup::Dict{Int,Int}
    function WannierTargetSpace{T,MT}(kind::Symbol,
                                      blocks::Vector{WannierSubspaceBlock{T,MT}},
                                      metadata::NamedTuple) where {T<:Number,MT<:AbstractMatrix{T}}
        kind in (:wannier, :mrwf) || throw(ArgumentError("kind must be :wannier or :mrwf"))
        isempty(blocks) && throw(ArgumentError("a target space needs at least one block"))
        length(unique(size(block.data, 2) for block in blocks)) == 1 ||
            throw(DimensionMismatch("all blocks must have the same number of functions"))
        indices = [block.ik for block in blocks]
        allunique(indices) || throw(ArgumentError("k-point indices must be unique"))
        return new{T,MT}(kind, blocks, metadata,
                         Dict(index => i for (i, index) in enumerate(indices)))
    end
end
function WannierTargetSpace(kind::Symbol,
                            blocks::Vector{WannierSubspaceBlock{T,MT}};
                            metadata::NamedTuple=(;)) where {T<:Number,MT<:AbstractMatrix{T}}
    return WannierTargetSpace{T,MT}(kind, blocks, metadata)
end
Base.size(target::WannierTargetSpace) = size(target.blocks)
Base.getindex(target::WannierTargetSpace, i::Int) = target.blocks[i]
Base.IndexStyle(::Type{<:WannierTargetSpace}) = IndexLinear()
function Base.show(io::IO, target::WannierTargetSpace)
    print(io, "WannierTargetSpace(:", target.kind, ", ", length(target.blocks),
          " k-points, ", target_dimension(target), " functions)")
end
"""
    target_dimension(target)
Number of target functions `N_wannier` per k-point.
"""
target_dimension(target::WannierTargetSpace) = size(first(target.blocks).data, 2)
"""
    wannier_block(target, ik)
Wannier block of k-point index `ik` (the k index of the plane-wave basis the
block is expressed in).
"""
function wannier_block(target::WannierTargetSpace, ik::Integer)
    position = get(target.lookup, Int(ik), nothing)
    isnothing(position) && throw(ArgumentError("target space has no block for ik=$ik"))
    return target.blocks[position]
end
function rotate_subspace(target::WannierTargetSpace, V::AbstractArray{<:Number,3}; tol::Real=1e-8)
    blocks = [rotate_subspace(block, @view(V[:, :, block.ik]); tol) for block in target.blocks]
    return WannierTargetSpace(target.kind, blocks; metadata=target.metadata)
end
"""
    combine_gauges(Udis, X)
Combined Wannier.jl gauge `U_k = Udis_k * X_k`: the semi-unitary
disentanglement/band-selection matrix `Udis` (`n_bands × n_wann × n_kpts`)
acts first and the unitary rotation `X` (`n_wann × n_wann′ × n_kpts`) is
applied from the right.
"""
function combine_gauges(Udis::AbstractArray{<:Number,3}, X::AbstractArray{<:Number,3})
    size(Udis, 3) == size(X, 3) || throw(DimensionMismatch("different numbers of k-points"))
    size(Udis, 2) == size(X, 1) || throw(DimensionMismatch(
        "Udis has $(size(Udis, 2)) columns but X has $(size(X, 1)) rows"))
    T = promote_type(eltype(Udis), eltype(X))
    U = Array{T}(undef, size(Udis, 1), size(X, 2), size(Udis, 3))
    for ik in axes(U, 3)
        U[:, :, ik] .= Udis[:, :, ik] * X[:, :, ik]
    end
    return U
end
function _check_kpoint_ordering(basis, model::Wannier.Model)
    length(basis.kpoints) == model.n_kpts || throw(DimensionMismatch(
        "basis has $(length(basis.kpoints)) k-points but the Wannier model has $(model.n_kpts)"))
    for ik in 1:model.n_kpts
        difference = collect(basis.kpoints[ik].coordinate) - model.kpoints[:, ik]
        norm(difference - round.(difference)) < 1e-6 || throw(ArgumentError(
            "k-point $ik of the basis $(basis.kpoints[ik].coordinate) does not match " *
            "the Wannier model $(model.kpoints[:, ik]); both must use the same ordering"))
    end
    return nothing
end
"""
    wannier_subspace(ψ, U; bands=1:size(U, 1), indices=1:size(U, 2),
                     kind=:wannier, metadata=(;), orthonormality_tol=1e-8,
                     orthonormalize=false)
    wannier_subspace(ψ, model::Wannier.Model; U=model.U, bands, basis=nothing, …)
Bloch-space target basis `W_k = Ψ_k[:, bands] * U_k[:, indices]` for every
k-point.
- `ψ[ik]` is the `N_G(ik) × n_bands_total` plane-wave coefficient matrix of
  the orbitals at k-point `ik` (DFTK convention). `bands` selects the window
  that was passed to Wannier.jl, i.e. `U` has `length(bands)` rows. DFTK's
  Wannier extension writes `ψ[ik][:, 1:n_bands]`, hence `bands=1:n_bands`.
- `U` is the Wannier.jl gauge, `n_bands × n_wann × n_kpts` (for a
  disentangled model the combined `Udis * X`, see [`combine_gauges`](@ref)).
- `indices` selects which Wannier functions form the target space.
For the `Wannier.Model` method the k-point ordering of `model` and `ψ` must be
identical (true for models built by `Wannier.Model(scfres; …)` from the same
unfolded basis); pass `basis` to verify it.
"""
function wannier_subspace(ψ::AbstractVector{<:AbstractMatrix},
                          U::AbstractArray{<:Number,3};
                          bands::AbstractVector{<:Integer}=1:size(U, 1),
                          indices::AbstractVector{<:Integer}=1:size(U, 2),
                          kind::Symbol=:wannier,
                          metadata::NamedTuple=(;),
                          orthonormality_tol::Real=1e-8,
                          orthonormalize::Bool=false)
    length(ψ) == size(U, 3) || throw(DimensionMismatch(
        "ψ has $(length(ψ)) k-points but U has $(size(U, 3))"))
    length(bands) == size(U, 1) || throw(DimensionMismatch(
        "U has $(size(U, 1)) rows but $(length(bands)) bands were selected"))
    all(i -> 1 <= i <= size(U, 2), indices) ||
        throw(ArgumentError("indices must lie in 1:$(size(U, 2))"))
    blocks = map(eachindex(ψ)) do ik
        size(ψ[ik], 2) >= maximum(bands) || throw(DimensionMismatch(
            "ψ[$ik] has only $(size(ψ[ik], 2)) bands"))
        data = Matrix(ψ[ik][:, bands] * U[:, indices, ik])
        WannierSubspaceBlock(ik, data, collect(indices); orthonormality_tol, orthonormalize)
    end
    defect = maximum(block -> orthonormality_defect(block.data), blocks)
    info = (; kind, n_kpoints=length(blocks), n_wannier=length(indices),
            bands=collect(Int, bands), indices=collect(Int, indices),
            orthonormality_defect=defect)
    return WannierTargetSpace(kind, blocks; metadata=merge(info, metadata))
end
function wannier_subspace(ψ::AbstractVector{<:AbstractMatrix}, model::Wannier.Model;
                          U::AbstractArray{<:Number,3}=model.U,
                          bands::AbstractVector{<:Integer}=1:model.n_bands,
                          basis=nothing, metadata::NamedTuple=(;), kwargs...)
    length(ψ) == model.n_kpts || throw(DimensionMismatch(
        "ψ has $(length(ψ)) k-points but the Wannier model has $(model.n_kpts)"))
    isnothing(basis) || _check_kpoint_ordering(basis, model)
    info = (; construction=:wannier, n_bands_window=model.n_bands,
            n_wann_model=model.n_wann, kgrid=collect(model.kgrid))
    return wannier_subspace(ψ, U; bands, metadata=merge(info, metadata), kwargs...)
end
"""
    mrwf_subspace(ψ, UV, X; kwargs...)
    mrwf_subspace(ψ, (model_group, UV); kwargs...)
    mrwf_subspace(ψ, model::Wannier.Model, groups; group=:conduction,
                  localize=true, max_iter=200, kwargs...)
Target space of manifold-remixed Wannier functions (MRWF, Qiao–Pizzi–Marzari).
The split/remix workflow is *upstream* in Wannier.jl:
1. a combined (e.g. valence + conduction) Wannier manifold with gauge `U`,
2. `Wannier.split_eig`/`split_model` diagonalize the Wannier-gauge Hamiltonian
   `U_k† diag(E_k) U_k = V_k Λ_k V_k†` and select a sub-manifold of `V_k`
   (`UV = rotate_U(U, V_group)`, `n_bands × n_group × n_kpts`),
3. `parallel_transport` smooths the gauge inside the group,
4. `max_localize` maximally localizes it (`X = model_group.U`).
The Bloch-space target basis is `W_k = Ψ_k * UV_k * X_k`. The first method takes
`UV` and `X` directly; the second one takes one `(model, UV)` pair as returned
by `Wannier.split_model`/`split_wannierize`; the third one runs the upstream
workflow on a disentangled `model` (whose `model.U` is the combined gauge) with
`groups` either `n_val` or a vector of index vectors, and selects
`group = :valence`, `:conduction` or an integer. For the response of occupied
states the relevant group is usually `:conduction`.
"""
function mrwf_subspace(ψ::AbstractVector{<:AbstractMatrix},
                       UV::AbstractArray{<:Number,3}, X::AbstractArray{<:Number,3};
                       metadata::NamedTuple=(;), kwargs...)
    info = (; construction=:mrwf)
    return wannier_subspace(ψ, combine_gauges(UV, X); kind=:mrwf,
                            metadata=merge(info, metadata), kwargs...)
end
function mrwf_subspace(ψ::AbstractVector{<:AbstractMatrix},
                       group::Tuple{<:Wannier.Model,<:AbstractArray{<:Number,3}};
                       kwargs...)
    model_group, UV = group
    return mrwf_subspace(ψ, UV, model_group.U; kwargs...)
end
function _group_index(group, ngroups::Integer)
    if group === :valence
        return 1
    elseif group === :conduction
        ngroups == 2 || throw(ArgumentError(
            ":conduction is only defined for two groups; pass an integer group index"))
        return 2
    elseif group isa Integer
        1 <= group <= ngroups || throw(ArgumentError("group must lie in 1:$ngroups"))
        return Int(group)
    end
    throw(ArgumentError("group must be :valence, :conduction or an integer"))
end
function mrwf_subspace(ψ::AbstractVector{<:AbstractMatrix}, model::Wannier.Model,
                       groups::Union{Integer,AbstractVector{<:AbstractVector{<:Integer}}};
                       group::Union{Symbol,Integer}=:conduction,
                       localize::Bool=true, max_iter::Integer=200,
                       f_tol::Real=1e-7, g_tol::Real=1e-5,
                       bands::AbstractVector{<:Integer}=1:model.n_bands,
                       basis=nothing, metadata::NamedTuple=(;), kwargs...)
    length(ψ) == model.n_kpts || throw(DimensionMismatch(
        "ψ has $(length(ψ)) k-points but the Wannier model has $(model.n_kpts)"))
    isnothing(basis) || _check_kpoint_ordering(basis, model)
    eig_groups = groups isa Integer ? [collect(1:groups), collect((groups + 1):model.n_wann)] :
                 [collect(Int, g) for g in groups]
    split_groups = Wannier.split_wannierize(model, eig_groups)
    index = _group_index(group, length(split_groups))
    model_group, UV = split_groups[index]
    if localize
        T = eltype(model_group.E)
        model_group.U .= Wannier.max_localize(model_group; max_iter=Int(max_iter),
                                              f_tol=T(f_tol), g_tol=T(g_tol))
    end
    spread = Wannier.omega(model_group, model_group.U)
    info = (; construction=:mrwf, upstream="Wannier.split_wannierize" *
            (localize ? " + Wannier.max_localize" : ""),
            groups=eig_groups, group=index, localized=localize,
            n_bands_window=model.n_bands, n_wann_model=model.n_wann,
            spread_total=Float64(spread.Ω), spread_invariant=Float64(spread.ΩI),
            kgrid=collect(model.kgrid))
    return mrwf_subspace(ψ, UV, model_group.U; bands, metadata=merge(info, metadata), kwargs...)
end


w the dynamic Sternheimer solvers (dense reference + CHKLS-style Schur/GMRES), residual checks, and the first-order density assembly.


Create ResponseCompleteWannier.jl/src/adapters/dftk_dynamic.jl



"""
