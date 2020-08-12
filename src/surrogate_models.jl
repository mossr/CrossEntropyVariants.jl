using LinearAlgebra
using PyPlot


"""
Surrogate model. Defaults to Gaussian process.

Available radial basis function for regression include:
	:gaussian, :linear, :squared, :multiquadratic
    (:none indicates to use Gaussian process)
"""
function surrogate_model(𝒟; basis=:none, kwargs...)
    # Data
    X = map(first, 𝒟)
	y = map(last, 𝒟)

    if basis == :none # Use Gaussian processes
        kern = SE(0., 0.) # squared exponential covairance kernel
		X = cmat(X)
        gp = GP(X, y, MeanZero(), kern)
        optimize!(gp, NelderMead())
        f = x->predict_f(gp, reshape(x', (2,1)))[1][1]
    else
        # Using regression with radial basis
        if basis == :gaussian
            ψ = r->gaussian_basis(r)
        elseif basis == :linear
            ψ = r->r
        elseif basis == :squared
            ψ = r->r^2
        elseif basis == :multiquadratic
            ψ = (r;σ=1)->(r^2 + σ^2)^(1/2)
        else
            error("No regression matching $basis")
        end

        bases = radial_bases(ψ, X) # use x-values as centers

        # Predictor
        f = regression(X, y, bases)
    end

    return f::Function
end


# Kochenderfer, Mykel J.. Algorithms for Optimization (The MIT Press)

function design_matrix(X)
	n, m = length(X[1]), length(X)
	return [j==0 ? 1.0 : X[i][j] for i in 1:m, j in 0:n]
end

function linear_regression(X, y)
	θ = pinv(design_matrix(X))*y
	return x -> θ⋅[1; x]
end


function example_linear_regression()
	# Data
	D = [(1,1), (2,3), (3,3), (4,4)]
	X = map(first, D)
	y = map(last, D)

	# Predictor
	f = linear_regression(X,y)

	scatter(X, y)
	xn = 0:0.1:4
	PyPlot.plot(xn, f.(xn))
	ylim([0,5])
	xlim([0,5])
end




function regression(X, y, bases)
	B = [b(x) for x in X, b in bases]
	θ = pinv(B)*y
	return x -> sum(θ[i] * bases[i](x) for i in 1:length(θ))
end


function example_basis_regression(k=4)
	# Data
	D = [(1,1), (2,3), (3,3), (4,4)]
	X = map(first, D)
	y = map(last, D)

	bases = polynomial_bases(1, k) # 1D polynomial of degree k

	# Predictor
	f = regression(X, y, bases)

	clf()
	scatter(X, y)
	xn = 0:0.1:4
	PyPlot.plot(xn, f.(xn))
	ylim([0,5])
	xlim([0,5])
end


function example_radial_basis_regression()
	# Data
	D = [(1,1), (2,3), (3,3), (4,4)]
	X = map(first, D)
	y = map(last, D)

	ψ = r->gaussian_basis(r)
	bases = radial_bases(ψ, X) # use x-values as centers

	# Predictor
	f = regression(X, y, bases)

	clf()
	scatter(X, y)
	xn = 0:0.1:4
	PyPlot.plot(xn, f.(xn))
	ylim([0,5])
	xlim([0,5])
end



function example_radial_basis_regression_2d()
	# Data
	D = [([1,1],1), ([2,2],3), ([3,3],3), ([2,2],4)]
	X = map(first, D)
	y = map(last, D)

	ψ = r->gaussian_basis(r)
	bases = radial_bases(ψ, X) # use x-values as centers

	# Predictor
	f = regression(X, y, bases)

	clf()
	scatter(map(first, X), map(last, X), s=7, c="black", edgecolors="white", linewidths=1/2)

    r = range(0, stop=4, length=100)
    # note reverse and x, y switch (imshow has a weird mapping)
    imshow([f([x,y]) for y in reverse(r), x in r], extent=[r[1], r[end], r[1], r[end]], cmap="viridis_r")
end




function plot_regression()
	X = 0:0.1:10

	for i in 1:7
		Y = [i*x + randn() for x in X]
		PyPlot.plot(X, Y, marker=".", linestyle="none")

		f = linear_regression(X, Y)

		PyPlot.plot(X, f.(X))

		f4 = regression(X, Y, [x->x+0.8cos(5x)])
		PyPlot.plot(X, f4.(X))
	end
end


# Kochenderfer, Mykel J.. Algorithms for Optimization (The MIT Press)
function sinusoidal_bases_1d(j, k, a, b)
	T = b[j] - a[j]
	bases = Function[x->1/2]
	for i in 1:k
		push!(bases, x->sin(2π*i*x[j]/T))
		push!(bases, x->cos(2π*i*x[j]/T))
	end
	return bases
end

function sinusoidal_bases(k, a, b)
	n = length(a)
	bases = [sinusoidal_bases_1d(i, k, a, b) for i in 1:n]
	terms = Function[]
	for ks in Iterators.product([0:2k for i in 1:n]...)
		powers = [div(k+1,2) for k in ks]
		if sum(powers) ≤ k
			push!(terms,  x->prod(b[j+1](x) for (j,b) in zip(ks,bases)))
		end
	end
	return terms
end



# Kochenderfer, Mykel J.. Algorithms for Optimization (The MIT Press)
radial_bases(ψ, C, p=2) = [x->ψ(norm(x - c, p)) for c in C]

gaussian_basis(r, σ=1) = exp(-r^2 / (2σ^2))

polynomial_bases_1d(i, k) = [x->x[i]^p for p in 0:k]
function polynomial_bases(n, k)
	bases = [polynomial_bases_1d(i, k) for i in 1:n]
	terms = Function[]
	for ks in Iterators.product([0:k for i in 1:n]...)
		if sum(ks) ≤ k
			push!(terms,  x->prod(b[j+1](x) for (j,b) in zip(ks,bases)))
		end
	end
	return terms
end


bases = [(x)->x^2 * log(x)]

function regression(X, y, bases, λ)
	B = [b(x) for x in X, b in bases]
	θ = (B'B + λ*I)\B'y
	return x -> sum(θ[i] * bases[i](x) for i in 1: length(θ))
end