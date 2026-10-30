using LinearAlgebra
using Test
using ResponseCompleteWannier

@testset "Schur reduction and auxiliary linearization" begin
    A = ComplexF64[2.5 0.2im; -0.2im 3.0]
    D = Diagonal([1.3, 2.1, 4.2]) |> Matrix
    B = ComplexF64[0.1 0.2im 0.005; -0.1im 0.15 0.008]
    b = ComplexF64[1.0, 0.2im]
    Q = Matrix{Float64}(I, 3, 3)[:, 1:2]
    reduced_model = reduce_schur(B, D, Q)

    @testset "full block equals exact Schur" begin
        for z in (0.2 + 0.3im, 2.8 + 0.4im)
            full_solution = full_block_solve(A, B, D, z, b)
            schur_solution = schur_solve(A, B, D, z, b)
            @test full_solution[1:size(A, 1)] ≈ schur_solution atol=1e-12
            @test (A - z * I - schur_self_energy(B, D, z)) *
                  schur_solution ≈ b atol=1e-12
        end
    end

    @testset "reduced Schur and auxiliary system agree" begin
        for z in (0.2 + 0.3im, 2.8 + 0.4im)
            reduced_solution = reduced_schur_solve(A, reduced_model, z, b)
            auxiliary_solution = auxiliary_solve(A, reduced_model, z, b)
            @test auxiliary_solution[1:size(A, 1)] ≈ reduced_solution atol=1e-12
            @test norm(reduced_solution - schur_solve(A, B, D, z, b)) < 1e-3
        end

        full_model = reduce_schur(B, D, Matrix{Float64}(I, 3, 3))
        @test linearized_matrix(A, full_model) ≈ [A B; B' D]
        for z in (0.2 + 0.3im, 2.8 + 0.4im)
            exact = schur_solve(A, B, D, z, b)
            linearized = auxiliary_solve(A, full_model, z, b)
            @test linearized[1:size(A, 1)] ≈ exact atol=1e-12
        end
    end
end

@testset "Response residual projection and SVD" begin
    W = Matrix{Float64}(I, 8, 8)[:, 1:2]
    responses = zeros(8, 4)
    responses[1:2, :] .= [1.0 0.0 2.0 1.0; 0.0 1.0 0.0 -1.0]
    responses[3, 2] = 4.0
    responses[4, 3] = 2.0
    responses[5, 4] = 1.0

    residuals = build_response_residual(responses, W)
    @test W' * residuals ≈ zeros(2, 4) atol=1e-14
    @test W * (W' * responses) + residuals ≈ responses atol=1e-14

    complement = compress_response_residual(residuals)
    @test complement.singular_values ≈ [4.0, 2.0, 1.0, 0.0]
    @test complement.cumulative_capture[1:3] ≈ [16 / 21, 20 / 21, 1.0]

    heldout = zeros(8, 1)
    heldout[3:5, 1] .= 1.0
    heldout_residual = build_response_residual(heldout, W)
    errors = [
        let basis = compress_response_residual(residuals; naux=n).basis
            norm(heldout_residual - basis * (basis' * heldout_residual))
        end
        for n in 0:3
    ]
    @test all(diff(errors) .< 0)
    @test errors[end] < 1e-14
end
