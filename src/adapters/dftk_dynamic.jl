"""
Dynamic first-order Sternheimer response for insulators (independent-particle
level, i.e. the response whose density is `χ0(ω) δV`).
Time-dependent first-order perturbation theory for the orbitals of a
perturbation `δV(t) = δV e^{-izt} + δV† e^{+i z̄ t}` with complex frequency
`z = ω + iη` (η > 0 gives the retarded response; Hartree atomic units) gives,
for the orbital `ψ_{n,k}`, with `Q = 1 - P_occ` the projector on the empty states
of the k-point space the response lives in,
    (H_{k+q} - ε_{n,k} - z)    a_{n,k} = -Q δV ψ_{n,k}          [+q branch]
    (H_{k-q} - ε_{n,k} + z̄)    b_{n,k} = -Q δV† ψ_{n,k}         [-q branch]
where `δV` has the wave vector `q` (periodic part stored, `V_full = e^{iq·r} δV`)
and `δV†` the wave vector `-q` with periodic part `conj(δV)`. The `e^{-izt}`
component of the induced density is
    δρ(r) = Σ_{k,n} f_n w_k [ conj(ψ_{n,k}) a_{n,k} + conj(b_{n,k}) ψ_{n,k} ].
For `z = 0` and real `δV` this is DFTK's static `2 f w Re(conj(ψ) δψ)` and the
`+q` branch is exactly DFTK's Sternheimer equation. Occupied–occupied rotations
cancel in `δρ` for insulators, so only the empty-space components are solved
for. The two branches coincide at `q = 0` for real `δV` up to `z → -z̄`.
Two interchangeable solvers are provided: a dense eigen-decomposition (the
reference, for small bases) and the Cancès–Herbst–Kemlin–Levitt–Stamm
Schur-complement splitting of the Sternheimer system, with the orbitals of the
extra bands treated exactly and the remainder solved by a (right-preconditioned)
GMRES, which accepts the complex shifts of the dynamic problem.
"""
abstract type AbstractSternheimerSolver end
"""
    DenseSternheimerSolver()
Reference solver: for every k-point slot the Hamiltonian restricted to the empty
space `Q H Q` is diagonalized once (cached) and the shifted systems are solved
spectrally. Cost `O(N_G³)` per slot; intended for small bases and validation.
"""
struct DenseSternheimerSolver <: AbstractSternheimerSolver end
"""
    SchurGMRESSolver(; tol=1e-10, restart=50, maxiter=500, n_extra=nothing)
Matrix-free solver. The empty space is split into the span of the extra bands
`E` (the first `n_extra` bands beyond the occupied ones in the ground state;
all of them for `nothing`) and its complement `R`. The system on `E` is solved
exactly and the Schur complement on `R`,
    (R H R - s - R H E (E†HE - s)⁻¹ E†H R) x_R = R b - R H E (E†HE - s)⁻¹ E† b,
is solved by restarted GMRES with a kinetic preconditioner. `tol` is the
relative residual tolerance.
"""
Base.@kwdef struct SchurGMRESSolver <: AbstractSternheimerSolver
    tol::Float64 = 1e-10
    restart::Int = 50
    maxiter::Int = 500
    n_extra::Union{Nothing,Int} = nothing
end
struct DenseSlotSpectrum
    energies::Vector{Float64}
    vectors::Matrix{ComplexF64}
end
struct SchurSlotData
    ψocc::Matrix{ComplexF64}
    E::Matrix{ComplexF64}
    HE::Matrix{ComplexF64}
    EHE::Matrix{ComplexF64}
    RHE::Matrix{ComplexF64}
    preconditioner::Vector{Float64}
end
"""
    SolverCache()
Mutable cache of per-slot factorizations (dense spectra and Schur data), keyed
by the k-point slot. It stays valid as long as the ground state is unchanged.
"""
struct SolverCache
    dense::Dict{Int,DenseSlotSpectrum}
    schur::Dict{Tuple{Int,Int},SchurSlotData}
end
SolverCache() = SolverCache(Dict{Int,DenseSlotSpectrum}(), Dict{Tuple{Int,Int},SchurSlotData}())
occupied_block(gs::DFTKGroundState, jk::Integer) = gs.ψ[jk][:, gs.occupied[jk]]
"""
    project_out_occupied(gs, jk, X)
Apply `Q = 1 - P_occ` of slot `jk` to the columns of `X`.
"""
function project_out_occupied(gs::DFTKGroundState, jk::Integer, X::AbstractVecOrMat)
    return project_out_subspace(X, occupied_block(gs, jk))
end
function dense_slot_spectrum(gs::DFTKGroundState, jk::Integer)
    ψocc = occupied_block(gs, jk)
    n, nocc = size(ψocc)
    full = qr(ψocc).Q * Matrix{eltype(ψocc)}(I, n, n)
    Z = full[:, (nocc + 1):end]
    eigenpairs = eigen(Hermitian(adjoint(Z) * (Matrix(gs.ham.blocks[jk]) * Z)))
    return DenseSlotSpectrum(eigenpairs.values, Z * eigenpairs.vectors)
end
function schur_slot_data(gs::DFTKGroundState, solver::SchurGMRESSolver, jk::Integer)
    ψocc = Matrix{ComplexF64}(occupied_block(gs, jk))
    extra_bands = gs.extra[jk]
    n_extra = isnothing(solver.n_extra) ? length(extra_bands) :
              min(solver.n_extra, length(extra_bands))
    raw = Matrix{ComplexF64}(gs.ψ[jk][:, extra_bands[1:n_extra]])
    emptied = project_out_subspace(raw, ψocc)
    E = n_extra == 0 ? emptied : Matrix(qr(emptied).Q)
    block = gs.ham.blocks[jk]
    HE = n_extra == 0 ? zero(E) : Matrix{ComplexF64}(block * E)
    EHE = adjoint(E) * HE
    RHE = HE - ψocc * (adjoint(ψocc) * HE) - E * (adjoint(E) * HE)
    kinetic = [sum(abs2, g) / 2 for g in DFTK.Gplusk_vectors_cart(gs.basis, gs.basis.kpoints[jk])]
    return SchurSlotData(ψocc, E, HE, EHE, RHE, 1 ./ (kinetic .+ 1))
end
"""
    gmres_solve(op, b; tol, restart, maxiter)
Restarted GMRES for the matrix-free operator `op` (a function mapping a vector
to a vector). Returns `(; x, iterations, residual, converged)` with the
relative true residual `‖b - op(x)‖/‖b‖`.
"""
function gmres_solve(op, b::AbstractVector{T}; tol::Real=1e-10, restart::Integer=50,
                     maxiter::Integer=500) where {T<:Number}
    n = length(b)
    x = zeros(T, n)
    bnorm = norm(b)
    iszero(bnorm) && return (; x, iterations=0, residual=0.0, converged=true)
    r = copy(b)
    resnorm = bnorm
    iterations = 0
    while resnorm > tol * bnorm && iterations < maxiter
        m = min(restart, maxiter - iterations)
        V = zeros(T, n, m + 1)
        H = zeros(T, m + 1, m)
        V[:, 1] = r / resnorm
        y = T[]
        steps = 0
        for j in 1:m
            w = op(V[:, j])
            for _ in 1:2  # classical Gram–Schmidt applied twice
                coefficients = adjoint(view(V, :, 1:j)) * w
                H[1:j, j] .+= coefficients
                w -= view(V, :, 1:j) * coefficients
            end
            H[j + 1, j] = norm(w)
            steps = j
            iterations += 1
            breakdown = H[j + 1, j] <= eps(real(T)) * resnorm
            breakdown || (V[:, j + 1] = w / H[j + 1, j])
            e1 = zeros(T, j + 1)
            e1[1] = resnorm
            y = H[1:(j + 1), 1:j] \ e1
            estimate = norm(e1 - H[1:(j + 1), 1:j] * y)
            (estimate <= tol * bnorm || breakdown) && break
        end
        x += view(V, :, 1:steps) * y
        r = b - op(x)
        resnorm = norm(r)
    end
    return (; x, iterations, residual=resnorm / bnorm, converged=resnorm <= tol * bnorm)
end
function _schur_gmres_columns(gs::DFTKGroundState, solver::SchurGMRESSolver, data::SchurSlotData,
                              jk::Integer, rhs::AbstractMatrix, shifts::AbstractVector)
    block = gs.ham.blocks[jk]
    n, ncolumns = size(rhs)
    m = size(data.E, 2)
    x = zeros(ComplexF64, n, ncolumns)
    factorizations = Dict{ComplexF64,Any}()
    total_iterations = 0
    all_converged = true
    remove_R(v) = v - data.ψocc * (adjoint(data.ψocc) * v) - data.E * (adjoint(data.E) * v)
    for c in 1:ncolumns
        s = ComplexF64(shifts[c])
        b = -project_out_subspace(rhs[:, c], data.ψocc)
        solve_E = if m == 0
            identity
        else
            factorization = get!(() -> lu(data.EHE - s * I), factorizations, s)
            v -> factorization \ v
        end
        Eb = adjoint(data.E) * b
        g = remove_R(b) - data.RHE * solve_E(Eb)
        function schur_operator(v)
            Hv = vec(block * reshape(v, :, 1))
            return remove_R(Hv) - s * v - data.RHE * solve_E(adjoint(data.HE) * v)
        end
        preconditioned(w) = schur_operator(remove_R(data.preconditioner .* w))
        result = gmres_solve(preconditioned, g; tol=solver.tol, restart=solver.restart,
                             maxiter=solver.maxiter)
        xR = remove_R(data.preconditioner .* result.x)
        y = solve_E(Eb - adjoint(data.HE) * xR)
        x[:, c] = data.E * y + xR
        total_iterations += result.iterations
        all_converged &= result.converged
    end
    return x, total_iterations, all_converged
end
"""
    sternheimer_residual(gs, jk, x, rhs, shifts)
Largest relative residual `‖Q((H - s_c) x_c + rhs_c)‖/‖Q rhs_c‖` over the columns
`c` of the solution `x` of the Sternheimer system `Q (H - s) Q x = -Q rhs` at
slot `jk`.
"""
function sternheimer_residual(gs::DFTKGroundState, jk::Integer, x::AbstractMatrix,
                              rhs::AbstractMatrix, shifts::AbstractVector)
    Hx = Matrix{ComplexF64}(gs.ham.blocks[jk] * Matrix{ComplexF64}(x))
    residual = project_out_occupied(gs, jk, Hx - x .* transpose(shifts) + rhs)
    scale = [norm(col) for col in eachcol(project_out_occupied(gs, jk, rhs))]
    worst = 0.0
    for c in axes(x, 2)
        iszero(scale[c]) || (worst = max(worst, norm(view(residual, :, c)) / scale[c]))
    end
    return worst
end
function sternheimer_solve(::DenseSternheimerSolver, gs::DFTKGroundState, jk::Integer,
                           rhs::AbstractMatrix, shifts::AbstractVector, cache::SolverCache)
    spectrum = get!(() -> dense_slot_spectrum(gs, jk), cache.dense, jk)
    coefficients = adjoint(spectrum.vectors) * rhs
    coefficients ./= (spectrum.energies .- transpose(shifts))
    return -(spectrum.vectors * coefficients), 0, true
end
function sternheimer_solve(solver::SchurGMRESSolver, gs::DFTKGroundState, jk::Integer,
                           rhs::AbstractMatrix, shifts::AbstractVector, cache::SolverCache)
    key = (jk, something(solver.n_extra, -1))
    data = get!(() -> schur_slot_data(gs, solver, jk), cache.schur, key)
    return _schur_gmres_columns(gs, solver, data, jk, rhs, shifts)
end
"""
    solve_sternheimer_branch(gs, solver, rhs, wavevector, z; branch, cache)
Solve one branch of the dynamic Sternheimer problem for all perturbations.
`rhs[p][jk]` are slot-indexed right-hand sides (see [`perturbation_rhs`](@ref));
`wavevector` is the wave vector carried by the response (`q` for `:plus`, `-q`
for `:minus`), `z` the complex frequency. Returns `(; solutions, residual,
iterations, converged)` where `solutions[p][jk]` is the `N_G(jk) × n_occ`
response of the occupied orbitals of the orbital k-point with `k + wavevector ≡ jk`.
"""
function solve_sternheimer_branch(gs::DFTKGroundState, solver::AbstractSternheimerSolver,
                                  rhs::AbstractVector, wavevector::AbstractVector{<:Real},
                                  z::Number; branch::Symbol=:plus,
                                  cache::SolverCache=SolverCache())
    branch in (:plus, :minus) || throw(ArgumentError("branch must be :plus or :minus"))
    np = length(rhs)
    nk = n_kpoints(gs)
    _, orbital_of_slot = kplusq_slots(gs, wavevector)
    solutions = [Vector{Matrix{ComplexF64}}(undef, nk) for _ in 1:np]
    worst_residual = 0.0
    iterations = 0
    converged = true
    for jk in 1:nk
        ik = orbital_of_slot[jk]
        bands = gs.occupied[ik]
        nb = length(bands)
        ε = gs.eigenvalues[ik][bands]
        shifts = branch === :plus ? ε .+ z : ε .- conj(z)
        rhs_columns = reduce(hcat, (Matrix{ComplexF64}(rhs[p][jk][:, bands]) for p in 1:np))
        all_shifts = repeat(shifts, np)
        x, nit, ok = sternheimer_solve(solver, gs, jk, rhs_columns, all_shifts, cache)
        worst_residual = max(worst_residual,
                             sternheimer_residual(gs, jk, x, rhs_columns, all_shifts))
        iterations += nit
        converged &= ok
        for p in 1:np
            solutions[p][jk] = x[:, ((p - 1) * nb + 1):(p * nb)]
        end
    end
    return (; solutions, residual=worst_residual, iterations, converged)
end
"""
    induced_density(gs, plus, minus, q)
`e^{-izt}` component (periodic part, `(nx, ny, nz, 1)` complex array) of the
first-order density from slot-indexed `+q` responses `plus[jk]` and `-q`
responses `minus[jk]` (or `nothing` to keep only the `+q` term):
    δρ(r) = Σ_{k,n} f_n w_k [ conj(ψ_{n,k}) a_{n,k} + conj(b_{n,k}) ψ_{n,k} ].
The k-point weights and occupations are those of `gs`; no symmetrization and no
time-reversal assumption is made. The responses are transferred to the actual
`k±q` plane-wave bases with `DFTK.transfer_blochwave_equivalent_to_actual`.
"""
function induced_density(gs::DFTKGroundState, plus::AbstractVector,
                         minus::Union{Nothing,AbstractVector}, q::AbstractVector{<:Real})
    basis = gs.basis
    qv = DFTK.Vec3{Float64}(q)
    a_actual = DFTK.transfer_blochwave_equivalent_to_actual(basis, plus, qv)
    b_actual = isnothing(minus) ? nothing :
               DFTK.transfer_blochwave_equivalent_to_actual(basis, minus, -qv)
    ρ = zeros(ComplexF64, basis.fft_size..., basis.model.n_spin_components)
    ψ_real = zeros(ComplexF64, basis.fft_size...)
    response_real = zeros(ComplexF64, basis.fft_size...)
    ifft_normalization = basis.fft_grid.ifft_normalization
    for (ik, kpt) in enumerate(basis.kpoints)
        for (column, n) in enumerate(gs.occupied[ik])
            DFTK.ifft!(ψ_real, basis, kpt, view(gs.ψ[ik], :, n); normalize=false)
            weight = ifft_normalization^2 * gs.occupation[ik][n] * basis.kweights[ik]
            DFTK.ifft!(response_real, basis, a_actual[ik].kpt, view(a_actual[ik].ψk, :, column);
                       normalize=false)
            @views ρ[:, :, :, kpt.spin] .+= weight .* conj.(ψ_real) .* response_real
            if !isnothing(b_actual)
                DFTK.ifft!(response_real, basis, b_actual[ik].kpt,
                           view(b_actual[ik].ψk, :, column); normalize=false)
                @views ρ[:, :, :, kpt.spin] .+= weight .* conj.(response_real) .* ψ_real
            end
        end
    end
    return ρ
end
"""
    DynamicResponse
Result of [`dynamic_response`](@ref): `plus[p][jk]` and `minus[p][jk]` slot
solutions (`minus` is `nothing` when the `-q` branch was not requested),
`density[p]` induced densities (or `nothing`), the complex frequency `z`, the
wave vector `q`, and solver diagnostics `residual`, `iterations`, `converged`.
"""
struct DynamicResponse
    plus::Vector{Vector{Matrix{ComplexF64}}}
    minus::Union{Nothing,Vector{Vector{Matrix{ComplexF64}}}}
    density::Union{Nothing,Vector{Array{ComplexF64,4}}}
    z::ComplexF64
    q::Vector{Float64}
    residual::Float64
    iterations::Int
    converged::Bool
end
"""
    dynamic_response(gs, perturbations, ω; η=0.0, solver=DenseSternheimerSolver(),
                     cache=SolverCache(), branches=(:plus, :minus), density=true)
First-order orbital responses `δψ` (both Sternheimer branches) and induced
densities for all `perturbations` (which must share the wave vector `q`) at the
complex frequency `z = ω + iη` (Hartree). See the file header for the equations.
The density needs both branches; it is only computed when both are requested
and available.
"""
function dynamic_response(gs::DFTKGroundState, perturbations::AbstractVector{<:AbstractResponsePerturbation},
                          ω::Real; η::Real=0.0,
                          solver::AbstractSternheimerSolver=DenseSternheimerSolver(),
                          cache::SolverCache=SolverCache(),
                          branches::Tuple=(:plus, :minus), density::Bool=true)
    isempty(perturbations) && throw(ArgumentError("no perturbations given"))
    q = collect(Float64, perturbation_q(first(perturbations)))
    all(p -> isapprox(perturbation_q(p), q; atol=1e-10), perturbations) || throw(ArgumentError(
        "all perturbations of one response must share the wave vector q"))
    z = ComplexF64(ω, η)
    plus = solve_sternheimer_branch(gs, solver, [perturbation_rhs(gs, p; branch=:plus) for p in perturbations],
                                    q, z; branch=:plus, cache)
    minus = nothing
    if :minus in branches
        minus = solve_sternheimer_branch(gs, solver,
                                         [perturbation_rhs(gs, p; branch=:minus) for p in perturbations],
                                         -q, z; branch=:minus, cache)
    end
    densities = nothing
    if density && !isnothing(minus)
        densities = [induced_density(gs, plus.solutions[p], minus.solutions[p], q)
                     for p in eachindex(perturbations)]
    end
    residual = max(plus.residual, isnothing(minus) ? 0.0 : minus.residual)
    iterations = plus.iterations + (isnothing(minus) ? 0 : minus.iterations)
    converged = plus.converged && (isnothing(minus) || minus.converged)
    return DynamicResponse(plus.solutions, isnothing(minus) ? nothing : minus.solutions,
                           densities, z, q, residual, iterations, converged)
end



Thinking

Edit ResponseCompleteWannier.jl/src/ResponseCompleteWannier.jl


