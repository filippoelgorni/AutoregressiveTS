using .FiniteGammaSpectrum

include("finite_gamma_spectrum.jl")

function σ(x::Real)
    return 1 / (1 + exp(-x))
end

function MP_bulk(x::Real; λ::Real)
    if x < (1 - √(λ))^2 || x > (1 + √(λ))^2
        return 0.0
    end

    return √((x - (1 - √(λ))^2) * ((1 + √(λ))^2 - x)) / (2 * π * λ * x)
end


const _spectrum_cache = Dict{NTuple{3,Float64},SpectrumClosure}()

function finite_gamma_bulk(x::Real;
    λ::Real,
    gamma::Real=1.0,
    t::Real=1.0,
    eta::Real=0.01)
    key = (Float64(gamma), Float64(λ), Float64(t))
    model = get!(_spectrum_cache, key) do
        SpectrumClosure(gamma=gamma, alpha=t/λ, t=t)
    end
    return imag(stieltjes(model, complex(x, eta))) / π
end