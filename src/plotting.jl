function initial_plot(f; plot=true, redraw=true, newfig=true, kwargs...)
    global SURROGATE_PLOT

    if plot
        if newfig
            figure()
        end
        if redraw
            clf()
            SURROGATE_PLOT = nothing # imshow can now be called again (instead of using set_data)
            plot_objective_function(f)
        end
    end
end

global LIMITS = (X=[-15, 15], Y=[-15, 15]) # To be filled.
function set_limits(X, Y)
    global LIMITS
    LIMITS = (X=X, Y=Y)
end

function plot_objective_function(f, limits = LIMITS; bins=100, kwargs...)
    rx = range(limits.X[1], stop=limits.X[2], length=bins)
    ry = range(limits.Y[1], stop=limits.Y[2], length=bins)

    # note reverse and x, y switch (imshow has a weird mapping)
    imshow([f([x,y]; kwargs...) for y in reverse(ry), x in rx],
        extent=[limits.X[1], limits.X[2], limits.Y[1], limits.Y[2]],
        cmap="viridis_r")
end


function plot_cem(P, samples, order, m_elite, limits=LIMITS)
    gca().collections = [] # update figure instead of redrawing
    scatter(samples[1,:], samples[2,:], s=7, c="black", edgecolors="white", linewidths=1/2)
    scatter(samples[1,order[1:m_elite]], samples[2,order[1:m_elite]], s=7, c="red", edgecolors="white", linewidths=1/2)
    xlim(LIMITS.X)
    ylim(LIMITS.Y)

    # Plot covariance.
    X = range(LIMITS.X[1], stop=LIMITS.X[2], length=100)
    Y = range(LIMITS.Y[1], stop=LIMITS.Y[2], length=100)
    Z = [pdf(P, [x,y]) for y in Y, x in X] # Note: reverse in X, Y.
    contour(Z, extent=[LIMITS.X[1], LIMITS.X[2], LIMITS.Y[1], LIMITS.Y[2]], cmap="hot", alpha=0.3)
    gcf() # |> display # tell Juno to display figure
    sleep(1/2)
end


function plot_mixture(mm, limits=LIMITS; bins=100, newfig=true)
    if newfig
        figure()
    end
    rx = range(limits.X[1], stop=limits.X[2], length=bins)
    ry = range(limits.Y[1], stop=limits.Y[2], length=bins)
    # note reverse and x, y switch (imshow has a weird mapping)
    imshow([pdf(mm, [x,y]) for y in reverse(ry), x in rx], extent=[limits.X[1], limits.X[2], limits.Y[1], limits.Y[2]], cmap="viridis_r")
end


function plot_voronoi(𝒟, φ=x->x, limits=(X=[1,10], Y=[1,10]); newfig=false)
    # Voronoi diagram (Euclidean)
    f = (x1,x2)->nearest_neighbors(φ([x1,x2]), 𝒟, dist_euclidean)

    if newfig
        figure()
    end

    rx = range(limits.X[1], stop=limits.X[2], length=100)
    ry = range(limits.Y[1], stop=limits.Y[2], length=100)

    # note reverse and x, y switch (imshow has a weird mapping)
    imshow([f(x,y) for y in reverse(ry), x in rx], extent=[limits.X[1], limits.X[2], limits.Y[1], limits.Y[2]], cmap="viridis_r")

    xlim(limits.X)
    ylim(limits.Y)

end


global SURROGATE_PLOT = nothing
function plot_surrogate_model(f)
    global SURROGATE_PLOT

    # note reverse and x, y switch (imshow has a weird mapping)
    rx = range(LIMITS.X[1], stop=LIMITS.Y[2], length=100)
    ry = range(LIMITS.X[1], stop=LIMITS.Y[2], length=100)
    𝒟 = [f([x,y]) for y in reverse(ry), x in rx]
    if isnothing(SURROGATE_PLOT)
        SURROGATE_PLOT = imshow(𝒟, extent=[rx[1], rx[end], ry[1], ry[end]], cmap="viridis_r") #, alpha=0.33) # Greys
    else
        SURROGATE_PLOT.set_data(𝒟)
    end
end


function plot_covariance(𝐦, 𝐌; show_sub_layer_distributions=true)
    global LIMITS
    varX = range(LIMITS.X[1], stop=LIMITS.X[2], length=100)
    varY = range(LIMITS.Y[1], stop=LIMITS.Y[2], length=100)

    if show_sub_layer_distributions && !ismissing(𝐦)
        for 𝐦ᵢ in 𝐦
            Z = [pdf(𝐦ᵢ, [x,y]) for y in varY, x in varX] # Note: reverse in X, Y.
            contour(Z, extent=[LIMITS.X[1], LIMITS.X[2], LIMITS.Y[1], LIMITS.Y[2]], cmap="hot", alpha=0.3)
        end
    end

    Z = [pdf(𝐌, [x,y]) for y in varY, x in varX] # Note: reverse in X, Y.
    contour(Z, extent=[LIMITS.X[1], LIMITS.X[2], LIMITS.Y[1], LIMITS.Y[2]], cmap="hot", alpha=0.3)
end


"""
Handle settings for the `plotting` function
"""
@with_kw mutable struct PlotSettings
    show_non_elites=true
    highlight_sub_elites=true
    highlight_elites=true
    show_covariance_plots=true
    show_surrogate_model=false
    show_mixture=false
    show_voronoi=false
end


function plotting(k, 𝐦, 𝐌, 𝒮, 𝒟, m_elite, samples, sub_elites, elite, nextcycle; settings=PlotSettings())
    clear_scatter()

    viridis_r = ColorMaps.RGBArrayMap(ColorSchemes.viridis, interpolation_levels=m_elite, invert=true)

    # Color according to elite-class
    colors = fill(:black, m_elite) # [:green, :blue, :black, :cyan, :yellow, :gray]

    if settings.show_non_elites
        nonelite = columns(samples)
        for ne in nonelite
            scatter(ne[1], ne[2], s=7, c="black", edgecolors="white", linewidths=1/2)
        end
    end


    # Plot new sub-elites.
    if nextcycle
        if settings.highlight_sub_elites
            scatter(map(first, columns(sub_elites)), map(last, columns(sub_elites)), s=7, c="white", edgecolors="black", linewidths=1/2)
        else
            for (i,e) in enumerate(columns(sub_elites))
                scatter(e[1], e[2], s=7, c=colors[i], edgecolors="white", linewidths=1/2)
            end
        end
    end


    # Highlight elite samples, on top.
    if settings.highlight_elites
        scatter(map(first, columns(elite)), map(last, columns(elite)), s=7, c="red", edgecolors="white", linewidths=1/2)
    else
        for (i,e) in enumerate(columns(elite))
            scatter(e[1], e[2], s=7, c=colors[i], edgecolors="white", linewidths=1/2)
        end
    end


    title("CE-mixture method ($k)")
    xlabel(L"x_1")
    ylabel(L"x_2")


    if settings.show_covariance_plots
        plot_covariance(𝐦, 𝐌)
    end


    if settings.show_surrogate_model
        plot_surrogate_model(𝒮)
    end

    if settings.show_mixture
        plot_mixture(𝐌; newfig=false)
    end

    if settings.show_voronoi
        figure()
        plot_voronoi(𝒟, LIMITS)
    end

    xlim(LIMITS.X)
    ylim(LIMITS.Y)
    gcf() # |> display # tell Juno to display figure.
    sleep(1/20)
end


sample_components(𝐌) = map(rand, 𝐌.components)
component_means(𝐌) = map(m->m.μ, 𝐌.components)

plot_component_means(𝐌) = plot_samples(component_means(𝐌))
plot_sampled_components(𝐌) = plot_samples(sample_components(𝐌))

function plot_samples(𝐦)
    scatter(map(first, 𝐦), map(last, 𝐦); s=7, c="black", edgecolors="white", linewidths=1/2)
end

function clear_scatter()
    ax = gca()
    ax.collections = [] # update figure instead of redrawing
    plt.draw()
end