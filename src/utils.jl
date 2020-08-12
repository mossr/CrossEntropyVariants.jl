# Use LaTeX fonts in PyPlot
matplotlib.rc("font", family=["serif"]) # sans-serif keyword not supported
matplotlib.rc("font", serif=["Helvetica"]) # matplotlib.rc("font", serif=["Palatino"])
matplotlib.rc("text", usetex=true)


# Note: rename so it's not confused with "only get 'columns'" (e.g. columnize, rowize?)
rows(X) = [X[i,:] for i in 1:size(X,1)]
columns(X) = [X[:,i] for i in 1:size(X,2)]

# column matrix (undoes columns)
cmat(V) = [V[i][j] for j in 1:length(V[1]), i in 1:length(V)]

# norm approx
closeto(a,b,ϵ=1e-5) = norm(a - b) < ϵ

# (:)([-2, 2]...) # works....
colon(a,b) = a:b
colon(x) = x[1]:x[2]

# numerical local minima checking
isminima(f, x, ϵ=1e-15) = (f(x) <= f(x .+ ϵ) && f(x) <= f(x .- ϵ))