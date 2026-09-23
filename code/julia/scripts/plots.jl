using Revise

includet("../src/utils.jl")

function plot_gram_spectrum(X::AbstractMatrix, t::Real)
    L = size(X, 1)
    M = size(X, 2)
    nrows = floor(Int, t * L)

    X_tilde = Float64.(@view X[1:nrows, :])
    eigenvalues = eigvals(Symmetric(X_tilde * X_tilde')/M)

    bins::Int = 200
    p = plot(
        xlabel="Eigenvalue",
        ylabel="Count",
        title="Spectrum of X̃X̃ᵀ/M (first $nrows of $L rows)",
        label=false)
    histogram!(p,
        eigenvalues;
        bins=bins,
        normalize=true,
    )
    xrange = range(0, stop=maximum(eigenvalues), length=1000)
    α = M/nrows
    plot!(p,
        xrange,
        MP_bulk.(xrange; λ=1/α);
        color=:tomato,
        lw=2,
        label="MP law",
    )
    plot!(p,
        xrange,
        finite_gamma_bulk.(xrange; λ=1/α);
        color=:orange,
        lw=2,
        label="Asymptotic spectrum",
    )

    m1 = mean(eigenvalues)
    m2 = mean(abs2, eigenvalues)
    m3 = mean(x -> x^3, eigenvalues)

    mp_m2 = 1 + 1/α
    mp_m3 = 1 + 3/α + (1/α^2)

    vC = (M * (m2 - 1) - (nrows - 1)) / (M - 1)

    @show m1 m2 mp_m2 m3 mp_m3 vC
    return p
end


