"""
Cross-entropy method variants for optimization.
    - Cross-entropy method (standard)
    - Cross-entropy surrogate method
    - Cross-entropy mixture method

See paper for full description:
    http://web.stanford.edu/~mossr/pdf/cem_variants.pdf
"""
module CrossEntropyVariants

export ce_surrogate,
       ce_mixture,
       cross_entropy_method,
       sierra,
       branin,
       ackley,
       paraboloid,
       PlotSettings


using Distributions
using DataStructures
using GaussianMixtures
using GaussianProcesses
using Parameters
using Optim
using StatsBase
using Random
using Colors
using ColorSchemes
using PGFPlots
using PyPlot
using LinearAlgebra
using Suppressor

const ℝ = Float64
const ℤ = Int64

include("utils.jl")
include("sierra.jl")
include("test_objective_functions.jl")
include("surrogate_models.jl")
include("evaluation_schedules.jl")
include("plotting.jl")

dist_manhattan(𝐯, 𝐯′) = norm(𝐯 - 𝐯′, 1)
dist_euclidean(𝐯, 𝐯′) = norm(𝐯 - 𝐯′, 2)
dist_supremum(𝐯, 𝐯′) = norm(𝐯 - 𝐯′, Inf)

function nearest_neighbors(x′, 𝒟, dist)
    𝒟[argmin([dist(x, x′) for (x,y) ∈ 𝒟])][2]
end

import Distributions.MvNormal
function MvNormal(mu::Vector{Int}, Sigma::Matrix{Int})
    # Fix PDMat errors when Sigma is an integer matrix
    # Temporarty fix: https://github.com/JuliaStats/PDMats.jl/pull/117
    return MvNormal(mu, convert(Matrix{Float64}, Sigma))
end

"""
    The cross-entropy surrogate method.
"""
function ce_surrogate(f, 𝐌; # objective function `f` and initial distribution `𝐌`
                      m=10, # Number of samples per iteration
                      m_elite=5, # Number of elite samples
                      α=1, # Covariance scale for model-elites
                      k_max=10, # Maximum number of iterations
                      ϵ=1e-9, # Change in objective stopping condition (unused)
                      plot=false, # Plotting indicator
                      redraw=false, # Redraw plot every time
                      p_geo=NaN, # Evaluation scheduling parameter for geometric distribution
                      plot_settings=PlotSettings(), # Plotting keywork arguments (see `plotting`)
                      surrogate_kwargs...) # Surrogate model parameters (see `surrogate_model`)
    plot ? initial_plot(f) : nothing
    𝒟 = Queue{Vector}() # population data

    bᵥ = Inf
    bₓ = missing
    beᵥ = Inf # Best estimate value (from surrogate model).
    beₓ = missing # Best estimate x-value (from surrogate model).
    Σ = deepcopy(isa(𝐌, MixtureModel) ? 𝐌.components[1].Σ : 𝐌.Σ)

    if isnan(p_geo)
        use_evaluation_schedule = false
    else
        use_evaluation_schedule = true
        Pₛ = truncated(Geometric(p_geo), 0, k_max) # for evaluation schedule.
        Int(Random.GLOBAL_RNG.seed[1]) == 1 ? @info("Evaluation schedule: $(map(k->evaluation_schedule(Pₛ, k, k_max, m, m_elite)[1], 1:k_max))") : nothing
    end

    for k in 1:k_max
        if use_evaluation_schedule
            mₑ, m_elite = evaluation_schedule(Pₛ, k, k_max, m, m_elite)
        else
            mₑ = m
        end

        samples = rand(𝐌, mₑ)
        Y = [f(samples[:,i]) for i in 1:mₑ]
        order = sortperm(Y)
        elite = samples[:,order[1:m_elite]]

        𝐄, 𝐌, 𝒮, model_elite = model_elite_set!(𝒟, samples, elite, Y, 𝐌, Σ, m, m_elite, α, k; surrogate_kwargs...)

        plot ? plotting(k, missing, 𝐌, 𝒮, missing, m_elite, samples, model_elite, elite, true; settings=plot_settings) : nothing

        𝐌 = fit(𝐌, 𝐄)

        # Monitoring. Non-algorithmic.
        if length(order) > 1 # For the cases where evaluation_schedule returned 0
            yₜ = Y[order[1]] # Top elite y value.
            if yₜ < bᵥ
                bᵥ = yₜ # Better value.
                bₓ = samples[:,order[1]]
            end
        end

    end

    return 𝐌, bₓ, bᵥ
end


"""
Model sub-component elite set.
    ℓ: short-term memory length
"""
function model_elite_set!(𝒟, samples, elite, Y, 𝐌, Σ, m, m_elite, α, k; ℓ=3, surrogate_kwargs...)
    # Surrogate model: Gaussian process.
    𝒟ₚ = map(tuple, columns(samples), Y) # fit to entire population

    # Use short-term memory
    enqueue!(𝒟, 𝒟ₚ)
    while length(𝒟) > ℓ
        dequeue!(𝒟)
    end
    𝒟 = reduce(vcat, 𝒟) # Re-cast, abuse of variable notation
    𝒮hat = surrogate_model(𝒟; surrogate_kwargs...)

    m_model = 10m
    m_model_elite = 10m_elite
    model_samples = rand(𝐌, m_model)
    model_Y = [𝒮hat(model_samples[:,i]) for i in 1:(m_model)]
    model_order = sortperm(model_Y)
    model_elite = model_samples[:,model_order[1:(m_model_elite)]]

    # Elites form their own distributions based on the surrogate model.
    cov = Σ/α
    𝐦 = map(e->MvNormal(e, cov), columns(elite))

    𝐄 = Matrix{ℝ}(undef, 2, 0) # Elite set.

    # For each elite in 𝐦, run CE-method to completion to choose new "sub-elites" (use μ = eₓ and Σ = M.Σ)
    sub_elite = Matrix{ℝ}(undef, 2, 0) # Sub-elite set.
    for i in 1:length(𝐦)
        # Note. If this k_max is too large, then we could converge/overfit to the surrogate model
        # (especially when the outer `k` is low, the surrogate model has not matured yet)
        𝐦[i], cem_bₓ, cem_bᵥ = cross_entropy_method(𝒮hat, 𝐦[i]; m=100,
                                                                 m_elite=10,
                                                                 k_max=2)
        # Add top elite, not the mean.
        sub_elite = hcat(sub_elite, cem_bₓ)
    end

    if isa(𝐌, MixtureModel)
        # Mix 𝐦 into 𝐌 (mixture)
        𝐌 = MixtureModel(𝐦)
    end

    𝐄 = hcat(𝐄, elite, model_elite, sub_elite)
    return 𝐄, 𝐌, 𝒮hat, model_elite
end


"""
Cross-entropy mixture method. Same as `ce_surrogate` but using mixture models.
"""
function ce_mixture(f, 𝐌; kwargs...)
    𝐌 = MixtureModel([𝐌])
    @show 𝐌
    return ce_surrogate(f, 𝐌; kwargs...)
end



import Distributions.fit
fit(𝐌::MixtureModel, 𝐄; kwargs...) = fit(GMM(𝐌), 𝐄; kwargs...) # Cast input 𝐌 to GMM
"""
Gaussian mixture model fitting using the Expectation Maximization algorithm.
"""
function fit(𝐌::GMM, 𝐄)
    try
        # Expectation maximization to fit mixture model 𝐌 to elite set (data) 𝐄
        @suppress em!(𝐌, permutedims(𝐄))
    catch err
        @warn err
    end

    return MixtureModel(𝐌) # re-cast
end

"""
Fit the multivariate Normal distribution 𝐌 to data set 𝐄.
"""
function fit(𝐌::MvNormal, 𝐄)
    try
        𝐌 = fit(typeof(𝐌), 𝐄)
    catch err
        @warn err
    end

    return 𝐌
end



# Kochenderfer and Wheeler, Algorithms for Optimization, Algorithm 8.7, Page 135
"""
    The cross-entropy method, which takes an objective
    function `f` to be minimized, a proposal distribution
    `P`, an iteration count `k_max`, a sample size `m`, and
    the number of samples to use when refitting the
    distribution `m_elite`. It returns the updated distribution
    over where the global minimum is likely to exist.
"""
function cross_entropy_method(f, P; m=10, m_elite=5, k_max=10, ϵ=1e-9, plot=false, redraw=false, p_geo=nothing)
    bᵥ = Inf
    bₓ = missing
    plot ? initial_plot(f) : nothing
    μ = mean(P)

    for k in 1:k_max
        samples = rand(P, m)
        Y = [f(samples[:,i]) for i in 1:m]
        order = sortperm(Y)
        elite = samples[:,order[1:m_elite]]

        # Top elite.
        yₜ = Y[order[1]] # Top elite y value
        if yₜ < bᵥ
            bᵥ = yₜ # Found better. (Monitoring only)
            bₓ = samples[:,order[1]]
        end

        plot ? plot_cem(P, samples, order, m_elite) : nothing

        try
            P = fit(typeof(P), elite)
        catch err
            @warn err
            @info k
        end

        μ = mean(P)
    end

    return (P, bₓ, bᵥ)
end


end # module CrossEntropyVariants