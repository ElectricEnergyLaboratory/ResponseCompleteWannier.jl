# Response Orbitals for Dielectric Function: Definition & Implementation

このドキュメントは `ResponseCompleteWannier.jl` における `responses` 行列の物理的定義を誘電関数計算向けに固定し、DFTK v0.8.1 + Wannier v0.3.6 での具体的 Julia 実装を与える。

## 1. 物理的定義: なぜ phonon ではないか

誘電関数 $\varepsilon$ は電場 $E$ に対する分極応答。必要なのは原子変位 $\partial_{R_I}$ ではなく、電場による波動関数の1次応答:

```
|δ^E_β u_nk> = Σ_{m ∉ occ} |u_mk> <u_mk| ∂_{kβ} (H_k - ε_nk) |u_nk> / (ε_nk - ε_mk)
```

DFPT では Sternheimer 方程式:

```
(H_k - ε_nk) Pc |δ^E_β u_nk> = - Pc ∂_{kβ} H_k |u_nk>
Pc = 1 - Σ_{m∈occ} |u_m><u_m|
```

この |δ^E> を集めたものが `responses`。

phonon応答 |δ^{R_I} u> は局在ポテンシャル微分、電場応答は r_β そのもの = k微分。DFTKでは k 演算子で表現。誘電関数狙いでは responses = k-derivative response を使う。

## 2. DFTK での対応物

- `DFTK.apply_χ0(ham, ψ, ...)` : χ0 V を Sternheimerで解く
- `DFTK` の k微分は `G_vectors_cartesian` + 非局所項微分
- WannierExt では `apply_χ0` を使って ∂_k u を作れる

## 3. 実装コード

### 3.1 `src/wannier/kderivative.jl`

```julia
using DFTK
using LinearAlgebra

"""
    build_k_derivative_operator(basis, kpoint, α::Int)
α = 1,2,3 (x,y,z) 方向の ∂/∂k_α H_k
"""
function build_k_derivative_operator(basis::PlaneWaveBasis, kpoint::Kpoint, α::Int)
    # 運動エネルギー部分: ∂k (1/2 |k+G|^2) = (k+G)_α
    G_cart = DFTK.G_vectors_cartesian(basis, kpoint) # (3, nG)
    # 対角行列として返す (toy版)
    return Diagonal(G_cart[α, :])
end

"""
    sternheimer_k_response(ham, ψ_occ, ε_occ, kder_op)

(H - ε_n) Pc δψ = - Pc (∂k H) ψ を解く
実コードでは DFTK.Sternheimer.solve_systems を使うこと。ここは概念実装。
"""
function sternheimer_k_response(ham::DFTK.HamiltonianBlock,
                                ψ_occ::AbstractMatrix,
                                ε_occ::AbstractVector,
                                kder_op::AbstractMatrix; tol=1e-10)
    n_occ = size(ψ_occ,2)
    Pc = I - ψ_occ * ψ_occ'  # projector to conduction
    rhs = - Pc * (kder_op * ψ_occ)

    δψ = similar(rhs, ComplexF64)
    Hmat = Matrix(ham.blocks[1]) # v0.8.1 では ham.blocks
    for n in 1:n_occ
        H_minus_ε = Hmat - ε_occ[n]*I
        # Pc空間で正則化して解く
        A = Pc * H_minus_ε * Pc + (I - Pc)
        δψ[:, n] = A \ rhs[:, n]
        δψ[:, n] = Pc * δψ[:, n]
    end
    return δψ
end
```

### 3.2 `src/wannier/response_orbitals.jl` - メインAPI

```julia
using DFTK
using Wannier
using LinearAlgebra
import ResponseCompleteWannier: build_response_residual, compress_response_residual

struct ResponseOrbitals{T}
    kpoints::Vector{DFTK.Kpoint}
    responses::Vector{Matrix{T}} # kごとに (n_G, n_occ*3)
    wannier_U::Vector{Matrix{ComplexF64}}
end

"""
    build_dielectric_responses(basis, ψ, ε, wannier_model)

電場3方向の応答 |δ^E_x,y,z u_nk> を作り、Wannierゲージに回転
"""
function build_dielectric_responses(basis::PlaneWaveBasis,
                                    ψ::Vector{<:AbstractMatrix},
                                    ε::Vector{<:AbstractVector},
                                    wannier_model::Wannier.Model;
                                    directions=1:3)
    n_k = length(ψ)
    all_responses_k = Vector{Matrix{ComplexF64}}(undef, n_k)

    for (ik, kpt) in enumerate(basis.kpoints)
        ham = basis.ham_blocks[ik]
        ψk = ψ[ik]
        εk = ε[ik]
        occ_idx = findall(e -> e < basis.model.εF, εk)
        ψocc = ψk[:, occ_idx]
        εocc = εk[occ_idx]

        δψ_dir = Matrix{ComplexF64}[]
        for α in directions
            kder = build_k_derivative_operator(basis, kpt, α)
            δψ_α = sternheimer_k_response(ham, ψocc, εocc, kder)
            push!(δψ_dir, δψ_α)
        end
        δψ_all = hcat(δψ_dir...) # (n_G, n_occ*3)
        Uk = wannier_model.U[ik]
        # occ部分だけゲージ変換
        δψ_gauged = δψ_all * Uk[occ_idx, :]
        all_responses_k[ik] = δψ_gauged
    end
    return ResponseOrbitals(basis.kpoints, all_responses_k, wannier_model.U)
end

"""
    assemble_response_matrix(ro, W)

roを build_response_residual が食べられる Matrix に変換
WはWannier基底 (n_G, n_wann)
"""
function assemble_response_matrix(ro::ResponseOrbitals, W::AbstractMatrix)
    # 開発用: Γ点のみ
    R = ro.responses[1] # (n_G, n_wann*3)
    residual = build_response_residual(R, W)
    return residual
end

# --- エンドツーエンド ---
# basis, ψ, ε = run_scf(silicon)
# wann_model = Wannier.run_wannier(basis, ψ)
# W = ψ[1] * wann_model.U[1]  # Wannier.jl v0.3.6 での作り方要確認
# ro = build_dielectric_responses(basis, ψ, ε, wann_model)
# R = assemble_response_matrix(ro, W)
# comp = compress_response_residual(R; naux=8)
# choose_naux(comp, 0.99)
```

### 3.3 `src/complement/naux_selector.jl` 追加提案

```julia
function choose_naux(comp::ResponseComplement, threshold::Real=0.99)
    idx = findfirst(>=(threshold), comp.cumulative_capture)
    return isnothing(idx) ? length(comp.cumulative_capture) : idx
end
```

## 4. なぜ誘電関数に効くか

誘電関数の収束は高エネルギー伝導帯 c の和で決まる:

```
χ0 ~ Σ_{v,c,k} <v|e^{-i(q+G)r}|c><c|e^{i(q+G')r}|v> / (ε_v - ε_c)
```

Wannier Wだけでは c 成分が足りない。上記 δ^E u_vk は

```
|δ^E u_vk> = Σ_c |u_ck> <u_ck|∂_k|u_vk> / (ε_v - ε_c)
```

なので高エネルギーcの重み付き和を低ランクで保持。これをSVDで圧縮した basisが補完基底。結果 Schur補完 T = Q† D Q, V = B Q が誘電関数の極を再現する補助空間になる。

## 5. 次のアクション

1. `src/wannier/kderivative.jl` と `response_orbitals.jl` を上記で作成
2. `src/ResponseCompleteWannier.jl` に `include("wannier/kderivative.jl")` と `include("wannier/response_orbitals.jl")` を追加
3. `test/runtests.jl` に toyテスト追加: `W' * residual ≈ 0`
4. silicon Γ点だけで統合テスト: SCFは重いので最初は乱数ψで代用

このファイルを `docs/response_definition.md` として保存すれば論文Methodsにそのまま貼れる。
