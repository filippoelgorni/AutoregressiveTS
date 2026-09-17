
using LinearAlgebra
using Plots

function σ(x)
    return 1 / (1 + exp(-x))
end

function generate_datum(L::Int, W_star::AbstractMatrix; P=1)
    x = zeros(Int8, L, P)
    for p in 1:P
        for i in 1:L
            h_i = sum(W_star[i, 1:(i-1)] .* x[1:(i-1), p]) / sqrt(L)
            x[i, p] = rand() < σ(h_i) ? 1 : -1
        end
    end

    return P == 1 ? vec(x) : x
end

function plot_gram_spectrum(X::AbstractMatrix, t::Real)
    L = size(X, 1)
    nrows = floor(Int, t * L)

    X_tilde = Float64.(@view X[1:nrows, :])
    eigenvalues = eigvals(Symmetric(X_tilde * X_tilde')/L)

    bins::Int = 200

    return histogram(
        eigenvalues;
        bins=bins,
        xlabel="Eigenvalue",
        ylabel="Count",
        title="Spectrum of X̃X̃ᵀ (first $nrows of $L rows)",
        label=false,
    )
end
