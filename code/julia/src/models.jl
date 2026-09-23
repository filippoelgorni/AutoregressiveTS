"""
    LSM(D => N, act)
    LSM(Z, act)

The Linear Score Model.
"""
struct LSM{M<:AbstractMatrix,F<:Function}
    W::M
    act::F
end

Flux.@functor LSM
Flux.trainable(m::LSM) = (; m.W)

LSM(W::AbstractMatrix; act) = LSM(W, act)

function LSM(W::AbstractMatrix, act)
    return LSM(W, act)
end

function (m::LSM)(x)
    L = size(m.W, 1)
    return act(1/Float32(√L) * m.W' * x)
end

Base.show(io::IO, m::LSM) = print(io, "LSM(", size(m.W, 1))
