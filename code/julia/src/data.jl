
using LinearAlgebra
using Plots
using Statistics

include("utils.jl")

function generate_dataset(L::Int, W_star::AbstractMatrix; M=1)
    x = zeros(Int8, L, M)
    for m in 1:M
        for i in 1:L
            h_i = sum(W_star[i, 1:(i-1)] .* x[1:(i-1), m]) / sqrt(L)
            x[i, m] = rand() < σ(h_i) ? 1 : -1
        end
    end

    return M == 1 ? vec(x) : x
end

