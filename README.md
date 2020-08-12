# CrossEntropyVariants.jl


Cross-entropy method variants for optimization. Each method takes an objective function `f` and proposal distribution `𝐌`. 
* Cross-entropy method (standard): [`cross_entropy_method(f, 𝐌)`](https://github.com/mossr/CrossEntropyVariants.jl/blob/master/src/CrossEntropyVariants.jl#L228)
* Cross-entropy surrogate method: [`ce_surrogate(f, 𝐌)`](https://github.com/mossr/CrossEntropyVariants.jl/blob/master/src/CrossEntropyVariants.jl#L65)
* Cross-entropy mixture method: [`ce_mixture(f, 𝐌)`](https://github.com/mossr/CrossEntropyVariants.jl/blob/master/src/CrossEntropyVariants.jl#L180)

See paper for full explanation of each method: [`Cross-Entropy Method Variants for Optimization`](http://web.stanford.edu/~mossr/pdf/cem_variants.pdf)

![Contour plots of CEM variants](https://github.com/mossr/CrossEntropyVariants.jl/blob/master/figures/cem-variants.png)

# Usage
### Installation
Install the package with:
```julia
] add https://github.com/mossr/CrossEntropyVariants.jl
```

### Example Optimization Problem
Then run a simple optimization problem:
```julia
using CrossEntropyVariants
using Distributions

f = sierra # objective function
𝛍 = [0, 0]
𝚺 = [200 0; 0 200]
𝐌 = MvNormal(𝛍, 𝚺) # proposal distribution

(𝐌, bestₓ, bestᵥ) = ce_surrogate(f, 𝐌)
```


### Surrogate Models
The performance of the surrogate models may be dependent on the underlying objective function. The default surrogate model is a Gaussian process with the squared exponential kernel function (see [`surrogate_models.jl`](https://github.com/mossr/CrossEntropyVariants.jl/blob/master/src/surrogate_models.jl)). To use radial basis functions instead of a Gaussian process, you can specify a `basis` keyword input:
```julia
f = paraboloid # objective function
𝐌 = MvNormal([0, 0], [200 0; 0 200]) # proposal distribution

(𝐌, bestₓ, bestᵥ) = ce_surrogate(f, 𝐌; basis=:squared)
```


# Test Objective Function

Test objective functions such as [`branin`](https://github.com/mossr/CrossEntropyVariants.jl/blob/master/src/test_objective_functions.jl#L3), [`ackley`](https://github.com/mossr/CrossEntropyVariants.jl/blob/master/src/test_objective_functions.jl#L7), and [`paraboloid`](https://github.com/mossr/CrossEntropyVariants.jl/blob/master/src/test_objective_functions.jl#L12) are included. We recommend using [BenchmarkFunctions.jl](https://github.com/rbalexan/BenchmarkFunctions.jl) for a more comprehensive set of test objective functions.

### Sierra
Included is a new parameterized test objective function called `sierra` with many local minima and a single global minimum. Refer to the [paper](http://web.stanford.edu/~mossr/pdf/cem_variants.pdf) for a full description (also see [`sierra.jl`](https://github.com/mossr/CrossEntropyVariants.jl/blob/master/src/sierra.jl)).

![Sierra test function](https://github.com/mossr/CrossEntropyVariants.jl/blob/master/figures/sierra-function.png)
