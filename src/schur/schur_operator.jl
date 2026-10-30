struct SchurAuxiliaryModel{MT<:AbstractMatrix,MC<:AbstractMatrix}
    T::MT
    coupling::MC

    function SchurAuxiliaryModel(T::AbstractMatrix, coupling::AbstractMatrix)
        size(T, 1) == size(T, 2) ||
            throw(DimensionMismatch("T must be square"))
        size(coupling, 2) == size(T, 1) ||
            throw(DimensionMismatch("coupling and T dimensions do not match"))
        new{typeof(T),typeof(coupling)}(T, coupling)
    end
end

function _validate_schur_dimensions(A, B, D, b)
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("A must be square"))
    size(D, 1) == size(D, 2) || throw(DimensionMismatch("D must be square"))
    size(B) == (size(A, 1), size(D, 1)) ||
        throw(DimensionMismatch("B must have size (size(A, 1), size(D, 1))"))
    length(b) == size(A, 1) || throw(DimensionMismatch("b has the wrong length"))
    return nothing
end

"""
    schur_self_energy(B, D, z)

Evaluate `B * (D - zI)^(-1) * B'` without forming an inverse.
"""
function schur_self_energy(B::AbstractMatrix, D::AbstractMatrix, z::Number)
    size(D, 1) == size(D, 2) || throw(DimensionMismatch("D must be square"))
    size(B, 2) == size(D, 1) ||
        throw(DimensionMismatch("B and D dimensions do not match"))
    return B * ((D - z * I) \ adjoint(B))
end

"""
    schur_solve(A, B, D, z, b)

Solve the exact Schur-reduced system
`(A - zI - B * (D - zI)^(-1) * B')x = b`.
"""
function schur_solve(A::AbstractMatrix, B::AbstractMatrix, D::AbstractMatrix,
                     z::Number, b::AbstractVector)
    _validate_schur_dimensions(A, B, D, b)
    S = A - z * I - schur_self_energy(B, D, z)
    return S \ b
end

"""
    full_block_solve(A, B, D, z, b)

Solve the full shifted block system with right-hand side `[b; 0]`.
"""
function full_block_solve(A::AbstractMatrix, B::AbstractMatrix, D::AbstractMatrix,
                          z::Number, b::AbstractVector)
    _validate_schur_dimensions(A, B, D, b)
    H = [A B; adjoint(B) D]
    rhs = vcat(b, zeros(promote_type(eltype(H), eltype(b), typeof(z)), size(D, 1)))
    return (H - z * I) \ rhs
end

"""
    reduce_schur(B, D, Q)

Project the eliminated space onto orthonormal columns of `Q`, returning the
frequency-independent `SchurAuxiliaryModel` for that reduced resolvent.
"""
function reduce_schur(B::AbstractMatrix, D::AbstractMatrix, Q::AbstractMatrix)
    size(D, 1) == size(D, 2) || throw(DimensionMismatch("D must be square"))
    size(B, 2) == size(D, 1) ||
        throw(DimensionMismatch("B and D dimensions do not match"))
    size(Q, 1) == size(D, 1) ||
        throw(DimensionMismatch("Q and D dimensions do not match"))
    gram = adjoint(Q) * Q
    identity = Matrix{eltype(gram)}(I, size(Q, 2), size(Q, 2))
    isapprox(gram, identity; atol=1e-10, rtol=1e-10) ||
        throw(ArgumentError("columns of Q must be orthonormal"))
    return SchurAuxiliaryModel(adjoint(Q) * D * Q, B * Q)
end

"""
    reduced_schur_solve(A, model, z, b)

Solve using the rational self-energy represented by `model`.
"""
function reduced_schur_solve(A::AbstractMatrix, model::SchurAuxiliaryModel,
                             z::Number, b::AbstractVector)
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("A must be square"))
    size(model.coupling, 1) == size(A, 1) ||
        throw(DimensionMismatch("model and A dimensions do not match"))
    length(b) == size(A, 1) || throw(DimensionMismatch("b has the wrong length"))
    S = A - z * I - model.coupling *
        ((model.T - z * I) \ adjoint(model.coupling))
    return S \ b
end

"""
    linearized_matrix(A, model)

Return the fixed block matrix whose shifted Schur complement is represented by
`model`.
"""
function linearized_matrix(A::AbstractMatrix, model::SchurAuxiliaryModel)
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("A must be square"))
    size(model.coupling, 1) == size(A, 1) ||
        throw(DimensionMismatch("model and A dimensions do not match"))
    return [A model.coupling; adjoint(model.coupling) model.T]
end

"""
    auxiliary_solve(A, model, z, b)

Solve the auxiliary-linearized shifted block system with right-hand side
`[b; 0]`.
"""
function auxiliary_solve(A::AbstractMatrix, model::SchurAuxiliaryModel,
                         z::Number, b::AbstractVector)
    H = linearized_matrix(A, model)
    naux = size(model.T, 1)
    length(b) == size(A, 1) || throw(DimensionMismatch("b has the wrong length"))
    rhs = vcat(b, zeros(promote_type(eltype(H), eltype(b), typeof(z)), naux))
    return (H - z * I) \ rhs
end
