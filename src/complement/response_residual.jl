struct ResponseComplement{T<:Number,S<:Real}
    basis::Matrix{T}
    singular_values::Vector{S}
    cumulative_capture::Vector{S}
end

"""
    build_response_residual(responses, W)

Project response-orbital columns out of the orthonormal subspace spanned by
the columns of `W`.
"""
function build_response_residual(responses::AbstractMatrix, W::AbstractMatrix)
    size(responses, 1) == size(W, 1) ||
        throw(DimensionMismatch("responses and W must have the same row count"))
    gram = adjoint(W) * W
    identity = Matrix{eltype(gram)}(I, size(W, 2), size(W, 2))
    isapprox(gram, identity; atol=1e-10, rtol=1e-10) ||
        throw(ArgumentError("columns of W must be orthonormal"))
    return responses - W * (adjoint(W) * responses)
end

"""
    compress_response_residual(residuals; naux=nothing)

Compute an ordinary SVD of response residuals and retain the leading
`naux` left singular vectors. `cumulative_capture[m]` is the fraction of
residual Frobenius norm captured by the first `m` vectors.
"""
function compress_response_residual(residuals::AbstractMatrix;
                                    naux::Union{Nothing,Integer}=nothing)
    factorization = svd(residuals; full=false)
    rank_bound = length(factorization.S)
    retained = isnothing(naux) ? rank_bound : naux
    0 <= retained <= rank_bound ||
        throw(ArgumentError("naux must be between zero and the residual rank bound"))

    squared_singular_values = abs2.(factorization.S)
    total = sum(squared_singular_values)
    cumulative_capture = iszero(total) ?
        zeros(eltype(factorization.S), rank_bound) :
        cumsum(squared_singular_values) ./ total
    return ResponseComplement(
        Matrix(factorization.U[:, 1:retained]),
        collect(factorization.S),
        collect(cumulative_capture),
    )
end
