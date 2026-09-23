module FiniteGammaSpectrum

using LinearAlgebra

export SpectrumClosure, stieltjes, spectral_density, spectral_moments

"""
    SpectrumClosure(; gamma=1.0, alpha=Inf, t=1.0,
                      nsteps=1600, gh_order=120)

Numerical characterization of the proposed large-L covariance spectrum for
P(x_j | x_<j) = logistic(x_j * sum(W[j,k]*x_k)/sqrt(L)),
with W[j,k] iid N(0,gamma^2) for k<j.

alpha=Inf computes the population prefix covariance C.
alpha=M/L computes the sample covariance X[1:floor(t*L),:] * X[...]' / M,
where X has L positions in rows and M independent sequences in columns.

The nonlinear response is kept at its full gamma dependence. The spectral
equation is a research prediction: the complete limiting-law proof remains
to be written. No claim of learning/Gibbs-measure universality is made.
Only Julia's LinearAlgebra standard library is needed.
"""
struct SpectrumClosure
    gamma::Float64
    alpha::Float64
    t::Float64
    nsteps::Int
    kappas::Vector{Float64}
end

function SpectrumClosure(; gamma=1.0, alpha=Inf, t=1.0,
                         nsteps=1600, gh_order=120)
    isfinite(gamma) && gamma >= 0 || throw(ArgumentError("gamma must be finite and nonnegative"))
    alpha > 0 || throw(ArgumentError("alpha must be positive, or Inf for population covariance"))
    0 < t <= 1 || throw(ArgumentError("t must lie in (0,1]"))
    nsteps >= 20 || throw(ArgumentError("nsteps must be at least 20"))
    gh_order >= 8 || throw(ArgumentError("gh_order must be at least 8"))

    # Gauss-Hermite quadrature for a STANDARD NORMAL, not exp(-x^2).
    eig = eigen(SymTridiagonal(zeros(gh_order), sqrt.(Float64.(1:gh_order-1))))
    nodes = eig.values
    weights = abs2.(eig.vectors[1, :])
    kappas = Float64[]
    for s in range(0.0, t; length=2*nsteps+1)
        # sech(x)^2 = 4 exp(-2|x|)/(1+exp(-2|x|))^2 avoids overflow.
        e = exp.(-gamma * sqrt(s) .* abs.(nodes))
        mean_sech2 = sum(weights .* (4 .* e ./ (1 .+ e).^2))
        push!(kappas, gamma^2 * mean_sech2^2 / 4)
    end
    return SpectrumClosure(Float64(gamma), Float64(alpha), Float64(t),
                           Int(nsteps), kappas)
end

function _rhs(s, y, k, inv_alpha)
    z, g, jz, jg = y
    a = 1-k*s
    b = a*inv_alpha+k*z
    d = a-z-b*(s+z*g)
    u = -b*z
    v = 1+k*s+(a*inv_alpha+2*k*z)*g

    # Analytic sensitivity of the characteristic to its initial z.
    aa = (-(a*inv_alpha+2*k*z)*d+u*v)/d^2
    bb = -u^2/d^2
    cc = (2*k*g*d+v^2)/d^2
    return (u/d, v/d, aa*jz+bb*jg, cc*jz-aa*jg)
end

_shift(y, dy, h) = ntuple(k -> y[k]+h*dy[k], 4)

function _forward(c::SpectrumClosure, z0)
    h = c.t/c.nsteps
    inv_alpha = 1/c.alpha
    y = (ComplexF64(z0), 0.0+0.0im, 1.0+0.0im, 0.0+0.0im)
    for r in 1:c.nsteps
        s = (r-1)*h
        k1 = _rhs(s, y, c.kappas[2*r-1], inv_alpha)
        k2 = _rhs(s+h/2, _shift(y,k1,h/2), c.kappas[2*r], inv_alpha)
        k3 = _rhs(s+h/2, _shift(y,k2,h/2), c.kappas[2*r], inv_alpha)
        k4 = _rhs(s+h, _shift(y,k3,h), c.kappas[2*r+1], inv_alpha)
        y = ntuple(k -> y[k]+h*(k1[k]+2*k2[k]+2*k3[k]+k4[k])/6, 4)
    end
    return y
end

function _mp_stieltjes(z, q)
    lo, hi = (1-sqrt(q))^2, (1+sqrt(q))^2
    # This square-root branch has the correct asymptotic m(z) ~ -1/z.
    s = sqrt(z-lo)*sqrt(z-hi)
    return 2/(1-q-z-s)
end

function _shoot(c::SpectrumClosure, target; initial=nothing,
                tol=1e-9, maxiter=30)
    z = ComplexF64(target)
    imag(z) > 0 || throw(ArgumentError("The spectral argument must have positive imaginary part"))
    q = c.t/c.alpha
    z0 = isnothing(initial) ? z*(1+q*_mp_stieltjes(z,q)) : ComplexF64(initial)
    y = _forward(c,z0)
    for iteration in 1:maxiter
        residual = y[1]-z
        all(isfinite,y) || error("Nonfinite characteristic; increase eta or nsteps")
        if abs(residual) <= tol*(1+abs(z))
            imag(y[2]) >= -100*tol || error("Unphysical Stieltjes branch")
            return y[2]/c.t, z0
        end
        abs(y[3]) > eps(Float64) || error("Singular characteristic map; increase eta")
        delta = residual/y[3]
        accepted = false
        for backtrack in 0:14
            trial = z0-delta*2.0^(-backtrack)
            imag(trial) > 0 || continue
            yt = _forward(c,trial)
            if all(isfinite,yt) && abs(yt[1]-z) < abs(residual)
                z0, y = trial, yt
                accepted = true
                break
            end
        end
        accepted || error("Characteristic shooting failed; increase eta or nsteps")
    end
    error("Characteristic shooting did not converge; increase eta or nsteps")
end

"""
    stieltjes(c, z)

Predicted normalized transform int nu(dc)/(c-z), for imag(z)>0.
For the population spectrum use c.alpha=Inf. For a sample spectrum use M/L.
"""
stieltjes(c::SpectrumClosure,z; kwargs...) = first(_shoot(c,z; kwargs...))

"""
    spectral_density(c, xs; eta=0.02)

Return imag(stieltjes(c,x+i*eta))/pi on xs. This is the density convolved
with a Cauchy kernel of width eta, not a sharp eta=0 boundary value.
Atoms are consequently broadened. Use smaller eta together with a larger
nsteps to check convergence near spectral edges or a narrow bulk.
"""
function spectral_density(c::SpectrumClosure, xs; eta=0.02)
    eta > 0 || throw(ArgumentError("eta must be positive"))
    xvec = Float64.(collect(xs))
    values = similar(xvec)
    previous = nothing
    for index in sortperm(xvec)
        z = complex(xvec[index],eta)
        solution = try
            _shoot(c,z; initial=previous)
        catch err
            if err isa ErrorException && !isnothing(previous)
                _shoot(c,z)
            else
                rethrow()
            end
        end
        m, previous = solution
        values[index] = imag(m)/pi
    end
    return values
end

function _moment_rhs(s, y, k, inv_alpha)
    u2,u3 = y
    a = 1-k*s
    return (1+2*a*inv_alpha*s+2*k*u2,
            1+3*a*inv_alpha*(s+u2)+3*k*(u2+u3))
end

"""Return the first three predicted normalized spectral moments."""
function spectral_moments(c::SpectrumClosure)
    h = c.t/c.nsteps
    ia = 1/c.alpha
    y = (0.0,0.0)
    shift(v,d,h) = (v[1]+h*d[1],v[2]+h*d[2])
    for r in 1:c.nsteps
        s = (r-1)*h
        k1 = _moment_rhs(s,y,c.kappas[2*r-1],ia)
        k2 = _moment_rhs(s+h/2,shift(y,k1,h/2),c.kappas[2*r],ia)
        k3 = _moment_rhs(s+h/2,shift(y,k2,h/2),c.kappas[2*r],ia)
        k4 = _moment_rhs(s+h,shift(y,k3,h),c.kappas[2*r+1],ia)
        y = ntuple(k -> y[k]+h*(k1[k]+2*k2[k]+2*k3[k]+k4[k])/6,2)
    end
    return (m1=1.0,m2=y[1]/c.t,m3=y[2]/c.t)
end

end # module
