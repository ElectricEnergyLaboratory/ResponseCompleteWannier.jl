"""
Geometric (k-transport) completeness diagnostics.
These diagnostics look at the overlap matrices between neighbouring k-points,
    M(k, b)_{mn} = ⟨u_{m,k} | u_{n,k+b}⟩,
and are inspired by the Berry-connection / basis-incompleteness analysis of
Thümmler et al. They are deliberately kept separate from the *response-space*
incompleteness `C_response ≈ ‖(I - P_W) δψ‖` measured by the response layer:
- `C_response` asks whether the first-order response leaves the target space
  at one k+q point;
- `C_geom` asks whether the target manifold is closed under the k → k+b
  transport used by finite-difference derivatives, i.e. how far the singular
  values of `M(k, b)` (cosines of the principal angles between the subspaces at
  `k` and `k+b`) are from one.
The two are *not* assumed to be equivalent; relating them is an empirical
question (see [`spearman_correlation`](@ref) for a plain statistic).
Wannier.jl convention (verified against 0.3.6): `model.M[m, n, ib, ik]` is the
overlap above for the window bands, `model.bvectors.kpb_k[ib, ik]` the index of
`k + b` (reduced to the first zone), `model.bvectors.bvectors[:, ib]` the
Cartesian b-vector and `model.bvectors.weights[ib]` its weight `w_b`, normalised
by `Σ_b w_b b^α b^β = δ_{αβ}`. For a gauge `U_k` (`n_bands × n_wann`) the
Wannier-gauge overlap is `N(k, b) = U_k† M(k, b) U_{k+b}` and the invariant
spread is `Ω_I = (1/N_k) Σ_k Σ_b w_b (n_wann - ‖N(k, b)‖_F²)`.
"""
"""
    OverlapField(overlaps, kpb_k, bvectors, weights; metadata=(;))
    OverlapField(model::Wannier.Model; U=nothing, metadata=(;))
Neighbour overlap matrices on a regular k-mesh. `overlaps[m, n, ib, ik]` is
`⟨u_{m,k}|u_{n,k+b}⟩` (`n_a × n_b × n_bvecs × n_kpts`), `kpb_k[ib, ik]` the
index of `k + b`, `bvectors[:, ib]` the Cartesian b-vector and `weights[ib]` its
weight.
For a Wannier model, `U = nothing` keeps the overlaps of the window bands
(`n_bands × n_bands`); a gauge `U` (`n_bands × n_wann × n_kpts`, e.g.
`model.U` or the gauge of a [`WannierTargetSpace`](@ref)) gives the
Wannier-gauge overlaps `N = U_k† M U_{k+b}`.
"""
struct OverlapField{T<:Real}
    overlaps::Array{Complex{T},4}
    kpb_k::Matrix{Int}
    bvectors::Matrix{T}
    weights::Vector{T}
    metadata::NamedTuple
    function OverlapField{T}(overlaps::Array{Complex{T},4}, kpb_k::Matrix{Int},
                             bvectors::Matrix{T}, weights::Vector{T},
                             metadata::NamedTuple) where {T<:Real}
        n_bvecs, n_kpts = size(overlaps, 3), size(overlaps, 4)
        size(kpb_k) == (n_bvecs, n_kpts) || throw(DimensionMismatch(
            "kpb_k must have size (n_bvecs, n_kpts) = ($n_bvecs, $n_kpts)"))
        size(bvectors) == (3, n_bvecs) || throw(DimensionMismatch(
            "bvectors must have size (3, n_bvecs)"))
        length(weights) == n_bvecs || throw(DimensionMismatch("one weight per b-vector is required"))
        all(1 .<= kpb_k .<= n_kpts) || throw(ArgumentError("kpb_k contains invalid k-point indices"))
        return new{T}(overlaps, kpb_k, bvectors, weights, metadata)
    end
end
function OverlapField(overlaps::AbstractArray{<:Number,4}, kpb_k::AbstractMatrix{<:Integer},
                      bvectors::AbstractMatrix{<:Real}, weights::AbstractVector{<:Real};
                      metadata::NamedTuple=(;))
    T = float(promote_type(real(eltype(overlaps)), eltype(bvectors), eltype(weights)))
    return OverlapField{T}(Array{Complex{T},4}(overlaps), Matrix{Int}(kpb_k),
                           Matrix{T}(bvectors), Vector{T}(weights), metadata)
end
function OverlapField(model::Wannier.Model; U::Union{Nothing,AbstractArray{<:Number,3}}=nothing,
                      metadata::NamedTuple=(;))
    bvectors = model.bvectors
    M = model.M
    n_bvecs, n_kpts = size(M, 3), size(M, 4)
    if isnothing(U)
        overlaps = Array(M)
        gauge = :window_bands
    else
        size(U, 1) == size(M, 1) && size(U, 3) == n_kpts || throw(DimensionMismatch(
            "U must have size (n_bands, n_wann, n_kpts) matching the overlaps"))
        overlaps = Array{eltype(M),4}(undef, size(U, 2), size(U, 2), n_bvecs, n_kpts)
        for ik in 1:n_kpts, ib in 1:n_bvecs
            ikpb = bvectors.kpb_k[ib, ik]
            overlaps[:, :, ib, ik] = adjoint(U[:, :, ik]) * M[:, :, ib, ik] * U[:, :, ikpb]
        end
        gauge = :wannier
    end
    info = (; source="Wannier.Model", gauge, n_kpts, n_bvecs)
    return OverlapField(overlaps, bvectors.kpb_k, bvectors.bvectors, bvectors.weights;
                        metadata=merge(info, metadata))
end
"""
    OverlapSpectrum
Singular values of the neighbour overlaps and derived completeness measures,
returned by [`overlap_completeness_spectrum`](@ref):
- `singular_values[:, ib, ik]` (descending): cosines of the principal angles
  between the subspaces at `k` and `k+b`; they equal one iff the manifold is
  closed under the transport `k → k+b`;
- `min_singular_value`: smallest of all of them;
- `lost_weight[ib, ik] = Σ_m (1 - σ_m²)`: weight that leaks out of the manifold;
- `lost_weight_per_kpoint[ik] = Σ_b w_b lost_weight[ib, ik]`;
- `omega_invariant = (1/N_k) Σ_k Σ_b w_b lost_weight[ib, ik]`: for Wannier-gauge
  overlaps `N = U†MU` this is the invariant spread `Ω_I` of Wannier.jl (same
  units, Å² for Å⁻¹ b-vectors).
"""
struct OverlapSpectrum{T<:Real}
    singular_values::Array{T,3}
    weights::Vector{T}
    min_singular_value::T
    lost_weight::Matrix{T}
    lost_weight_per_kpoint::Vector{T}
    omega_invariant::T
end
"""
    overlap_completeness_spectrum(field)
Singular values of every `M(k, b)` of an [`OverlapField`](@ref) together with
the lost-weight measures of [`OverlapSpectrum`](@ref). Rounding can make
individual singular values exceed one by `O(ε)`; nothing is clamped.
"""
function overlap_completeness_spectrum(field::OverlapField{T}) where {T<:Real}
    n_a, n_b, n_bvecs, n_kpts = size(field.overlaps)
    n = min(n_a, n_b)
    singular_values = Array{T,3}(undef, n, n_bvecs, n_kpts)
    lost_weight = Matrix{T}(undef, n_bvecs, n_kpts)
    for ik in 1:n_kpts, ib in 1:n_bvecs
        σ = svdvals(@view field.overlaps[:, :, ib, ik])
        singular_values[:, ib, ik] = σ
        lost_weight[ib, ik] = sum(s -> 1 - s^2, σ)
    end
    per_kpoint = [sum(field.weights[ib] * lost_weight[ib, ik] for ib in 1:n_bvecs)
                  for ik in 1:n_kpts]
    return OverlapSpectrum{T}(singular_values, copy(field.weights), minimum(singular_values),
                              lost_weight, per_kpoint, sum(per_kpoint) / n_kpts)
end
"""
    overlap_completeness_error(spectrum_or_field; measure=:omega_invariant)
Scalar geometric-completeness error `C_geom`. `measure` selects
- `:omega_invariant`: `(1/N_k) Σ_k Σ_b w_b Σ_m (1 - σ_m²)` (the invariant spread
  `Ω_I` for Wannier-gauge overlaps);
- `:lost_weight`: plain average over `(k, b)` of `Σ_m (1 - σ_m²)`;
- `:max_deviation`: `max (1 - σ_m)`, the worst singular-value deviation from one.
"""
function overlap_completeness_error(spectrum::OverlapSpectrum; measure::Symbol=:omega_invariant)
    if measure === :omega_invariant
        return spectrum.omega_invariant
    elseif measure === :lost_weight
        return sum(spectrum.lost_weight) / length(spectrum.lost_weight)
    elseif measure === :max_deviation
        return 1 - spectrum.min_singular_value
    end
    throw(ArgumentError("measure must be :omega_invariant, :lost_weight or :max_deviation"))
end
overlap_completeness_error(field::OverlapField; kwargs...) =
    overlap_completeness_error(overlap_completeness_spectrum(field); kwargs...)
"""
    spearman_correlation(x, y)
Spearman rank correlation of two samples (ties receive their average rank). A
plain descriptive statistic for comparing, e.g., `C_geom(k)` with the response
incompleteness per k-point; it carries no claim by itself.
"""
function spearman_correlation(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    length(x) == length(y) || throw(DimensionMismatch("x and y must have the same length"))
    length(x) >= 2 || throw(ArgumentError("at least two samples are required"))
    function ranks(v)
        order = sortperm(v)
        r = zeros(Float64, length(v))
        i = 1
        while i <= length(v)
            j = i
            while j < length(v) && v[order[j + 1]] == v[order[i]]
                j += 1
            end
            r[order[i:j]] .= (i + j) / 2
            i = j + 1
        end
        return r
    end
    rx, ry = ranks(x), ranks(y)
    (iszero(norm(rx .- sum(rx) / length(rx))) || iszero(norm(ry .- sum(ry) / length(ry)))) &&
        return NaN
    return cor(rx, ry)
end
"""
    BerryConnection
Berry connection `A^α_{mn}(k) = i⟨u_{m,k}|∂_{k_α} u_{n,k}⟩` (Cartesian
components `α = 1, 2, 3`, units of the b-vectors' inverse) obtained from
overlap matrices: `A[m, n, α, ik]`. `method` records the algorithm,
`hermiticity_defect = ‖A - A†‖_F / ‖A‖_F` and `max_polar_distance` (the largest
`‖N - polar(N)‖_F` that was discarded when the overlaps were not unitary) are
diagnostics.
"""
struct BerryConnection{T<:Real}
    A::Array{Complex{T},4}
    method::Symbol
    hermiticity_defect::T
    max_polar_distance::T
    metadata::NamedTuple
end
# Principal matrix logarithm of a unitary matrix through its (diagonal) Schur form.
function _unitary_log(U::AbstractMatrix{<:Complex})
    decomposition = schur(Matrix(U))
    phases = angle.(diag(decomposition.T))
    return decomposition.Z * Diagonal(im .* phases) * adjoint(decomposition.Z)
end
function _polar_unitary(N::AbstractMatrix{<:Complex})
    factorization = svd(N)
    unitary = factorization.U * factorization.Vt
    return unitary, norm(N - unitary)
end
"""
    berry_connection_from_overlaps(field; method=:matrix_log)
Berry connection from neighbour overlaps.
`:matrix_log` is the initial (first-order) estimate. For a smooth gauge
`N(k, b) = ⟨u_k|u_{k+b}⟩ ≈ exp(-i b·A(k))`, hence `b·A(k) ≈ i log N(k, b)`, and
with `Σ_b w_b b^α b^β = δ_{αβ}`
    A^α(k) = Σ_b w_b b^α  i log N(k, b).
`log` is the **principal matrix logarithm** of the unitary polar factor of
`N(k, b)` (computed from its Schur form), never an element-wise logarithm: for
non-commuting components of `A` the two differ at first order. The deviation of
`N` from unitarity (lost weight) is discarded here and reported by
[`overlap_completeness_spectrum`](@ref). Valid for `|b·A| < π`.
`:self_consistent_log` is reserved for the Thümmler-style refinement (path-
ordered propagation / Magnus expansion with self-consistent basis-incompleteness
correction) and is not implemented.
"""
function berry_connection_from_overlaps(field::OverlapField{T}; method::Symbol=:matrix_log) where {T<:Real}
    if method === :matrix_log
        return _berry_connection_matrix_log(field)
    elseif method === :self_consistent_log
        throw(ArgumentError(
            "method=:self_consistent_log is reserved for the Thümmler-style self-consistent " *
            "Berry-connection refinement (path ordering / Magnus expansion) and is not " *
            "implemented yet; use method=:matrix_log"))
    end
    throw(ArgumentError("unknown method $method; available: :matrix_log"))
end
function _berry_connection_matrix_log(field::OverlapField{T}) where {T<:Real}
    n_a, n_b, n_bvecs, n_kpts = size(field.overlaps)
    n_a == n_b || throw(DimensionMismatch(
        "the matrix logarithm needs square overlaps, got $n_a × $n_b"))
    A = zeros(Complex{T}, n_a, n_a, 3, n_kpts)
    worst_polar = zero(T)
    for ik in 1:n_kpts, ib in 1:n_bvecs
        unitary, distance = _polar_unitary(@view field.overlaps[:, :, ib, ik])
        worst_polar = max(worst_polar, distance)
        generator = im * _unitary_log(unitary)  # ≈ b·A(k), Hermitian
        for α in 1:3
            A[:, :, α, ik] .+= (field.weights[ib] * field.bvectors[α, ib]) .* generator
        end
    end
    scale = norm(A)
    defect = iszero(scale) ? zero(T) :
             sqrt(sum(abs2, A .- permutedims(conj.(A), (2, 1, 3, 4)))) / scale
    return BerryConnection{T}(A, :matrix_log, defect, worst_polar,
                              (; n_kpts, n_bvecs, source=field.metadata))
end



