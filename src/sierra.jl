using Distributions

𝕀(b) = b ? 1 : 0 # indicator function

function sierra_mixture(; 𝛍=[0,0], σ=3, 𝐬=[+σ,-σ], δ=2, η=6, decay=true)
    𝚺 = [σ 0.0; 0.0 σ]
    decay = 𝕀(decay)
    origin = [0.0, 0.0]
    𝐒 = MvNormal[MvNormal(𝛍, 𝚺/(σ*η))]

    for g in [[+δ, +δ], [+δ, -δ], [-δ, +δ], [-δ, -δ]]
        for (i,𝐩) in enumerate([origin, [+1, +1], [+2, 0], [+3, +1], [0, +2], [+1, +3]])
            for s in 𝐬
                push!(𝐒, MvNormal(g + s*𝐩 + 𝛍, i^decay/η * 𝚺))
            end
        end
    end

    return MixtureModel(𝐒)
end



global MMCACHE = Dict() # hyperparameters => MixtureModel

function sierra(x; minimize=true, kwargs...)
    global MMCACHE

    hpk = kwargs
    if haskey(MMCACHE, hpk)
        𝐒 = MMCACHE[hpk]
    else
        𝐒 = sierra_mixture(; kwargs...)
    end

    # mixture model with equal weights
    sgn = minimize ? -1 : +1
    return sgn*pdf(𝐒, x)
end