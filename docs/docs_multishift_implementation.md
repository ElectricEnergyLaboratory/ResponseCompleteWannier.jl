# Multi-shift Krylov restoration: concrete Julia implementation guide

This file is intended to be committed as `docs/multishift_implementation.md` in ResponseCompleteWannier.jl and given to GitHub Copilot as the implementation specification.

## Goal

Implement a real shared-Krylov solver for

```math
(A-z_j I)x_j=b,\qquad j=1,\ldots,N_\omega,
```

such that **one Arnoldi/Lanczos basis is generated and reused for all complex shifts**. Do not implement a fake multi-shift method that calls a full iterative solver independently for each frequency.

The existing Schur and auxiliary-linearization code must remain the source of the frequency-independent operator. For a compressed complement basis `Qaux`, use

```math
T=Q_{aux}^\dagger DQ_{aux},\qquad C=BQ_{aux},
```

and

```math
H_{aux}=\begin{pmatrix}A&C\\C^\dagger&T\end{pmatrix}.
```

Elimination of the auxiliary block must reproduce

```math
S(z)=A-zI-C(T-zI)^{-1}C^\dagger.
```

---

## 1. Add `src/krylov/shared_krylov.jl`

Start with the following implementation. Adapt names to the repository style if necessary, but preserve the algorithm and tests.

```julia
using LinearAlgebra

"""Result of one shared Krylov construction and many shifted projected solves."""
struct SharedKrylovResult{T,RT}
    shifts::Vector{T}
    solutions::Vector{Vector{T}}
    residuals::Vector{RT}
    converged::BitVector
    krylov_dimension::Int
    operator_applications::Int
end

"""Count calls to an expensive matrix-free operator."""
mutable struct CountedOperator{F}
    f::F
    napply::Int
end
CountedOperator(f::F) where {F} = CountedOperator{F}(f, 0)
function (A::CountedOperator)(x)
    A.napply += 1
    A.f(x)
end
reset_count!(A::CountedOperator) = (A.napply = 0; A)

_apply(A::AbstractMatrix, x) = A * x
_apply(A, x) = A(x)

"""
One Arnoldi factorization for a general Hermitian/non-Hermitian linear operator.
Returns V, H, beta. Full reorthogonalization is deliberately used in the first
correctness-oriented implementation.
"""
function arnoldi_basis(A, b::AbstractVector; m::Integer, atol=1e-14, reorthogonalize=true)
    m > 0 || throw(ArgumentError("m must be positive"))
    beta = norm(b)
    beta > atol || throw(ArgumentError("right-hand side is numerically zero"))
    T = promote_type(eltype(b), ComplexF64)
    n = length(b)
    V = zeros(T, n, m + 1)
    H = zeros(T, m + 1, m)
    V[:, 1] .= b ./ beta
    actual_m = 0
    for j in 1:m
        w = T.(_apply(A, view(V, :, j)))
        for i in 1:j
            H[i, j] = dot(view(V, :, i), w)
            w .-= H[i, j] .* view(V, :, i)
        end
        if reorthogonalize
            for i in 1:j
                h2 = dot(view(V, :, i), w)
                H[i, j] += h2
                w .-= h2 .* view(V, :, i)
            end
        end
        H[j + 1, j] = norm(w)
        actual_m = j
        H[j + 1, j] <= atol && break
        V[:, j + 1] .= w ./ H[j + 1, j]
    end
    return V[:, 1:actual_m+1], H[1:actual_m+1, 1:actual_m], beta
end

"""Projected solution for all shifts using ONE already-generated Arnoldi basis."""
function projected_shifted_solutions(V, H, beta, shifts)
    m = size(H, 2)
    Vm = @view V[:, 1:m]
    Hm = Matrix(@view H[1:m, 1:m])
    e1 = zeros(eltype(Hm), m)
    e1[1] = beta
    zs = ComplexF64.(collect(shifts))
    xs = Vector{Vector{ComplexF64}}(undef, length(zs))
    for (j, z) in pairs(zs)
        y = (Hm - z * I) \ e1
        xs[j] = Vector{ComplexF64}(Vm * y)
    end
    xs
end

"""
Solve (A-z_j I)x_j=b for many complex shifts using exactly one Arnoldi basis.
The convergence check always uses the FULL operator residual, not only the
projected residual. Adaptive mode rebuilds/extends the common basis dimension,
never a separate basis per shift.
"""
function solve_multishift(A, b::AbstractVector, shifts; tol=1e-10, maxiter=min(length(b), 200), initial_dim=min(16, maxiter), growth=16, reorthogonalize=true)
    zs = ComplexF64.(collect(shifts))
    isempty(zs) && throw(ArgumentError("shifts must not be empty"))
    nb = norm(b)
    nb > 0 || throw(ArgumentError("right-hand side must be nonzero"))
    m = min(initial_dim, maxiter)
    best_x = Vector{Vector{ComplexF64}}()
    best_r = fill(Inf, length(zs))
    while true
        V, H, beta = arnoldi_basis(A, b; m=m, reorthogonalize=reorthogonalize)
        best_x = projected_shifted_solutions(V, H, beta, zs)
        best_r = [norm(b - (_apply(A, x) - z .* x)) / nb for (x, z) in zip(best_x, zs)]
        if all(best_r .<= tol) || size(H, 2) >= maxiter || size(H, 2) < m
            napp = A isa CountedOperator ? A.napply : -1
            return SharedKrylovResult(zs, best_x, best_r, BitVector(best_r .<= tol), size(H, 2), napp)
        end
        m = min(m + growth, maxiter)
    end
end

"""Reference only: independent dense solves, never the production multi-shift path."""
function solve_shifted_direct(A::AbstractMatrix, b::AbstractVector, shifts)
    n = size(A, 1)
    [Vector{ComplexF64}((ComplexF64.(A) - ComplexF64(z) * I(n)) \ ComplexF64.(b)) for z in shifts]
end
```

### Important optimization after correctness

The adaptive implementation above rebuilds the Arnoldi factorization when `m` grows. This is acceptable only as a first correctness implementation. Then refactor it so an `ArnoldiState` is **extended in place**, preserving earlier vectors and ensuring the expensive basis generation is literally one continuous construction. The final performance test should count approximately `m` basis-building applications, not the sum of repeated rebuilds.

Add:

```julia
mutable struct ArnoldiState{T}
    V::Matrix{T}
    H::Matrix{T}
    beta::Float64
    m::Int
end
```

and implement `init_arnoldi(A,b,maxiter)` plus `extend_arnoldi!(state,A,new_m)`. `solve_multishift` should use this state before declaring the milestone complete.

---

## 2. Add `src/operators/auxiliary_operator.jl` or extend the existing equivalent

If the repository already has an auxiliary operator, extend it rather than duplicate it.

```julia
using LinearAlgebra

struct AuxiliaryOperator{TA,TC,TT}
    A::TA
    C::TC
    T::TT
end

Base.size(H::AuxiliaryOperator) = (size(H.A,1) + size(H.T,1), size(H.A,2) + size(H.T,2))

function (H::AuxiliaryOperator)(x::AbstractVector)
    nL = size(H.A, 1)
    xL = @view x[1:nL]
    xa = @view x[nL+1:end]
    yL = H.A * xL + H.C * xa
    ya = H.C' * xL + H.T * xa
    vcat(yL, ya)
end

function auxiliary_matrix(H::AuxiliaryOperator)
    [Matrix(H.A) Matrix(H.C); Matrix(H.C') Matrix(H.T)]
end

function build_auxiliary_operator(A, B, D, Qaux)
    size(D,1) == size(Qaux,1) || throw(DimensionMismatch("D and Qaux are incompatible"))
    size(B,2) == size(Qaux,1) || throw(DimensionMismatch("B and Qaux are incompatible"))
    norm(Qaux' * Qaux - I, Inf) < 1e-8 || throw(ArgumentError("Qaux must be orthonormal"))
    T = Qaux' * D * Qaux
    C = B * Qaux
    AuxiliaryOperator(A, C, T)
end

function compressed_schur_matrix(A, C, T, z)
    nL = size(A,1)
    na = size(T,1)
    ComplexF64.(A) - z * I(nL) - ComplexF64.(C) * ((ComplexF64.(T) - z * I(na)) \ ComplexF64.(C'))
end
```

The signs must match the existing repository convention. If its block operator uses `-C` instead of `+C`, modify consistently and add an equivalence test rather than forcing this exact sign convention.

---

## 3. Wire includes/exports into `src/ResponseCompleteWannier.jl`

Add equivalents of:

```julia
include("operators/auxiliary_operator.jl")
include("krylov/shared_krylov.jl")

export AuxiliaryOperator, build_auxiliary_operator, auxiliary_matrix
export compressed_schur_matrix
export CountedOperator, SharedKrylovResult, solve_multishift, solve_shifted_direct
```

Do not duplicate includes/functions already present in the package.

---

## 4. Add `test/test_multishift.jl`

```julia
using Test
using LinearAlgebra
using Random
using ResponseCompleteWannier

@testset "Auxiliary Schur equivalence" begin
    Random.seed!(11)
    nL, nH, na = 5, 9, 4
    A = randn(ComplexF64,nL,nL); A = (A+A')/2
    D = Diagonal(collect(range(2.0, 6.0; length=nH))) |> Matrix
    B = 0.2randn(ComplexF64,nL,nH)
    Q,_ = qr(randn(ComplexF64,nH,na)); Qaux = Matrix(Q)[:,1:na]
    Hop = build_auxiliary_operator(A,B,D,Qaux)
    H = auxiliary_matrix(Hop)
    C, T = Hop.C, Hop.T
    b = randn(ComplexF64,nL)
    for z in ComplexF64[0.1+0.05im, 0.8+0.1im, 1.7+0.08im, 3.1+0.12im]
        xa = (H-z*I) \ vcat(b,zeros(ComplexF64,na))
        xs = compressed_schur_matrix(A,C,T,z) \ b
        @test isapprox(xa[1:nL], xs; rtol=1e-10, atol=1e-11)
    end
end

@testset "One Krylov space, many complex shifts" begin
    Random.seed!(12)
    n = 48
    X = randn(ComplexF64,n,n)
    H = (X+X')/2 + 8I
    b = randn(ComplexF64,n)
    shifts = ComplexF64.(range(-1.0,4.0;length=16) .+ 0.15im)
    counted = CountedOperator(x -> H*x)
    result = solve_multishift(counted,b,shifts;tol=1e-9,maxiter=n,initial_dim=n)
    refs = solve_shifted_direct(Matrix(H),b,shifts)
    errs = [norm(x-r)/norm(r) for (x,r) in zip(result.solutions,refs)]
    @test maximum(result.residuals) < 1e-8
    @test maximum(errs) < 1e-8
    @test result.krylov_dimension <= n
end

@testset "Schur == auxiliary == shared Krylov" begin
    Random.seed!(13)
    nL, nH, na = 6, 12, 5
    A = randn(ComplexF64,nL,nL); A=(A+A')/2
    D = Diagonal(collect(range(3.0,10.0;length=nH))) |> Matrix
    B = 0.15randn(ComplexF64,nL,nH)
    Q,_ = qr(randn(ComplexF64,nH,na)); Qaux=Matrix(Q)[:,1:na]
    Hop = build_auxiliary_operator(A,B,D,Qaux)
    H = auxiliary_matrix(Hop)
    bL = randn(ComplexF64,nL)
    baug = vcat(bL,zeros(ComplexF64,na))
    shifts = ComplexF64[0.0+0.1im,0.5+0.1im,1.5+0.1im,2.5+0.1im]
    counted = CountedOperator(Hop)
    ms = solve_multishift(counted,baug,shifts;tol=1e-10,maxiter=size(H,1),initial_dim=size(H,1))
    for (j,z) in pairs(shifts)
        xdir = (H-z*I) \ baug
        xschur = compressed_schur_matrix(A,Hop.C,Hop.T,z) \ bL
        @test isapprox(xdir[1:nL],xschur;rtol=1e-9,atol=1e-10)
        @test isapprox(ms.solutions[j],xdir;rtol=1e-8,atol=1e-9)
    end
end
```

Also include this file from `test/runtests.jl`.

---

## 5. Add a test proving the basis is shared, not one solve per shift

After the extendable Arnoldi state is implemented, add an internal statistic `basis_builds` and require it to equal 1. At minimum expose the number of basis vectors and operator applications.

The test should compare 4 vs 64 shifts using the same fixed requested Krylov dimension:

```julia
@testset "Krylov build independent of number of shifts" begin
    Random.seed!(14)
    n=60
    X=randn(ComplexF64,n,n); H=(X+X')/2+10I
    b=randn(ComplexF64,n)
    function run(nw)
        shifts=ComplexF64.(range(-2,4;length=nw) .+ 0.2im)
        A=CountedOperator(x->H*x)
        result=solve_multishift(A,b,shifts;tol=1e-8,maxiter=n,initial_dim=n)
        return A.napply,result
    end
    n4,r4=run(4)
    n64,r64=run(64)
    # Full residual verification costs one A*x per shift, so raw napply includes
    # validation. Track basis applications separately in the final implementation.
    @test r4.krylov_dimension == r64.krylov_dimension
end
```

**Final implementation requirement:** distinguish `basis_operator_applications` from `residual_check_applications`. The scientific performance metric is the former. It should not scale with `Nω` for a fixed basis dimension.

---

## 6. Add `test/test_naux_convergence.jl`

The purpose is to separate auxiliary truncation error from Krylov error.

```julia
using Test, LinearAlgebra, Random
using ResponseCompleteWannier

@testset "Naux convergence separated from Krylov error" begin
    Random.seed!(15)
    nL,nH = 6,20
    A=randn(ComplexF64,nL,nL); A=(A+A')/2
    d=collect(range(4.0,16.0;length=nH)); D=Diagonal(d)|>Matrix
    B=0.1randn(ComplexF64,nL,nH)
    b=randn(ComplexF64,nL)
    z=1.2+0.15im
    fullS=ComplexF64.(A)-z*I-B*((ComplexF64.(D)-z*I)\B')
    xfull=fullS\b
    # In the synthetic test use the SVD of coupling-active complement directions.
    U,S,V=svd(B')
    errs=Float64[]
    for na in 1:min(nH,nL)
        Qaux=U[:,1:na]
        Hop=build_auxiliary_operator(A,B,D,Qaux)
        xrc=compressed_schur_matrix(A,Hop.C,Hop.T,z)\b
        push!(errs,norm(xrc-xfull)/norm(xfull))
    end
    @test errs[end] <= errs[1] + 1e-10
end
```

Do **not** assert strict monotonicity here unless the chosen reduced-space construction mathematically guarantees it. For physical response-derived Qaux, report the curve and separately test the guaranteed SVD residual monotonicity already present elsewhere.

---

## 7. Add `examples/multishift_response_demo.jl`

```julia
using LinearAlgebra, Random
using ResponseCompleteWannier

Random.seed!(20261031)
nL,nH,na=8,24,8
A=randn(ComplexF64,nL,nL); A=(A+A')/2
D=Diagonal(collect(range(4.0,20.0;length=nH)))|>Matrix
B=0.12randn(ComplexF64,nL,nH)
Q,_=qr(randn(ComplexF64,nH,na)); Qaux=Matrix(Q)[:,1:na]
Hop=build_auxiliary_operator(A,B,D,Qaux)
Hdense=auxiliary_matrix(Hop)
b=vcat(randn(ComplexF64,nL),zeros(ComplexF64,na))
omegas=collect(range(0.0,6.0;length=64)); eta=0.15
shifts=ComplexF64.(omegas .+ im*eta)
Acount=CountedOperator(Hop)
res=solve_multishift(Acount,b,shifts;tol=1e-10,maxiter=size(Hdense,1),initial_dim=size(Hdense,1))
refs=solve_shifted_direct(Hdense,b,shifts)
errs=[norm(x-r)/norm(r) for (x,r) in zip(res.solutions,refs)]
println("Naux = ",na)
println("Nfrequencies = ",length(shifts))
println("Krylov dimension = ",res.krylov_dimension)
println("counted operator applications = ",Acount.napply)
println("max relative residual = ",maximum(res.residuals))
println("max error vs direct = ",maximum(errs))
println("omega, relres, error_vs_direct")
for j in eachindex(shifts)
    println(omegas[j], ", ",res.residuals[j],", ",errs[j])
end
```

Once the physical response adapter is ready, add a second example that replaces random Qaux by

```text
DFTK dynamic δψ
 -> R=(I-P_W)δψ
 -> SVD
 -> Qaux
 -> Haux
 -> shared Krylov frequencies.
```

Do not delete the synthetic example because it is the fast regression test of the mathematics.

---

## 8. Required follow-up refactor: incremental Arnoldi

The code above intentionally prioritizes correctness. Before claiming performance success, implement incremental basis extension. Suggested skeleton:

```julia
mutable struct ArnoldiState{T}
    V::Matrix{T}
    H::Matrix{T}
    beta::Float64
    m::Int
    basis_operator_applications::Int
end

function init_arnoldi(b::AbstractVector,maxiter::Int)
    beta=norm(b)
    T=promote_type(eltype(b),ComplexF64)
    V=zeros(T,length(b),maxiter+1)
    H=zeros(T,maxiter+1,maxiter)
    V[:,1].=b./beta
    ArnoldiState(V,H,beta,0,0)
end

function extend_arnoldi!(st::ArnoldiState,A,target_m;atol=1e-14,reorthogonalize=true)
    target_m <= size(st.H,2) || throw(ArgumentError("target_m exceeds allocation"))
    for j in st.m+1:target_m
        w=eltype(st.V).(_apply(A,view(st.V,:,j)))
        st.basis_operator_applications += 1
        for i in 1:j
            st.H[i,j]=dot(view(st.V,:,i),w)
            w .-= st.H[i,j].*view(st.V,:,i)
        end
        if reorthogonalize
            for i in 1:j
                h2=dot(view(st.V,:,i),w)
                st.H[i,j]+=h2
                w .-= h2.*view(st.V,:,i)
            end
        end
        st.H[j+1,j]=norm(w)
        st.m=j
        st.H[j+1,j] <= atol && return st
        st.V[:,j+1].=w./st.H[j+1,j]
    end
    st
end
```

Refactor projected solves to accept `ArnoldiState`. Adaptive convergence should call `extend_arnoldi!` rather than recomputing earlier vectors.

For a fixed final dimension `m`, require

```julia
state.basis_operator_applications == m
```

(up to an early happy breakdown). This is the clean proof that basis construction is shared.

---

## 9. Future block-Krylov interface

Do not delay the single-RHS milestone for this. However, avoid API choices that prevent future

```julia
solve_multishift(A, B::AbstractMatrix, shifts; ...)
```

for Cartesian electric fields, G channels, Raman perturbations, etc. The future implementation may use block Arnoldi/Lanczos.

---

## 10. Physical integration with response-complete Qaux

If the preceding repository work has introduced `ResponseComplement` or equivalent, add a thin bridge rather than duplicate SVD logic. Conceptually:

```julia
function auxiliary_from_response(A,B,D,complement)
    Qaux = complement.basis   # adapt to actual field name
    build_auxiliary_operator(A,B,D,Qaux)
end
```

Never guess the field name. Inspect current types and adapt this function to the real repository.

For finite q, response residuals must already have been formed using W(k+q). This Krylov module should not redo Wannier projection logic.

---

## 11. Acceptance criteria

Do not report this milestone as complete unless all are true:

1. Existing package tests still pass.
2. `H_aux` is frequency independent.
3. Eliminating auxiliary variables reproduces the compressed Schur solve.
4. One Arnoldi/Krylov basis is shared by all complex shifts.
5. No production path calls a full iterative solver inside a loop over frequencies.
6. Shared-Krylov solutions agree with independent direct solutions.
7. Full residuals are checked for every shift.
8. Basis-building operator application count is explicitly reported and, for fixed `m`, independent of `Nω`.
9. Auxiliary truncation (`Naux`) and Krylov truncation (`m`) errors are reported separately.
10. `examples/multishift_response_demo.jl` runs successfully.
11. `julia --project=. -e 'using Pkg; Pkg.test()'` passes.
12. If physical Qaux is available, an integration test reaches `physical response -> Qaux -> Haux -> shared multi-frequency solve`.

Final numerical identity to demonstrate at several complex frequencies:

```math
x_{Schur}(z) \approx x_{aux,direct}(z) \approx x_{shared-Krylov}(z).
```

---

## 12. Copilot implementation order

Implement in this order and keep fixing code until each stage passes:

1. Inspect existing Schur/auxiliary APIs and adapt names in this document.
2. Add/extend `AuxiliaryOperator` without duplicating existing code.
3. Implement shared Arnoldi projected shifted solves.
4. Implement incremental `ArnoldiState` so the basis is genuinely generated once.
5. Add direct reference solves.
6. Add Schur/auxiliary/shared-Krylov equivalence tests.
7. Add operation counters separating basis construction from residual validation.
8. Add the executable multi-frequency demo.
9. Connect response-derived Qaux if the physical adapter is currently operational.
10. Run the full test suite and fix all failures.

Do not stop with a plan or pseudocode. Modify the repository and run the tests.
