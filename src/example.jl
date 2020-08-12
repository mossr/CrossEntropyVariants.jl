using CrossEntropyVariants
using Distributions


## Gaussian process surrogate model example

f = sierra # objective function
𝛍 = [0, 0]
𝚺 = [200 0; 0 200]
𝐌 = MvNormal(𝛍, 𝚺) # proposal distribution

(𝐌, bestₓ, bestᵥ) = ce_surrogate(f, 𝐌; plot=true)



## Radial basis function example

f = paraboloid # objective function
𝐌 = MvNormal([0, 0], [200 0; 0 200]) # proposal distribution

(𝐌, bestₓ, bestᵥ) = ce_surrogate(f, 𝐌; plot=true, basis=:squared)