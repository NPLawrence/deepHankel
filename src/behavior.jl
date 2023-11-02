

"""
Creates excitation signal of order `L`. Additional inputs `num_data::Int` and `m::Int` ensure a minimum length signal is obtained.
"""
function excite(L::Int; num_data::Int=100, m::Int=1)
    num_data = max((m+1)*L , num_data)
    u = randn(num_data)
end

function excite(num_data::Int=100)
    u = randn(num_data)
end

"""
Hankel(u::Vector, L::Int)

Creates Hankel matrix of depth `L` from signal `u`
"""
function Hankel(u::Vector, L::Int)
    N = length(u)
    H = vcat([u[i:N-L+i]' for i in 1:L]...)
end

# Hankel(u,y,L) = Hankel(u,L), Hankel(y,L)
Hankel_shift(u,y,L) = Hankel.([u[1:end-1], y[1:end-1], u[2:end], y[2:end]], L)


# TODO: create a type that carries input-output signals, noise profile, and true system
# TODO: use LinearSolve.jl and methods from Krylov.jl -- pass solver as argument


"""
Hankel_solver(H_u::Matrix{Float64}, H_y::Matrix{Float64}, u_init::Vector{Float64}, y_init::Vector{Float64}; solver="")

Solve the standard linear system in Willems' fundamental lemma.
"""
function Hankel_solver(H_u::Matrix{Float64}, H_y::Matrix{Float64}, u_init::Vector{Float64}, y_init::Vector{Float64}; solver="")
    A = vcat([H_u,H_y]...)
    b = vcat([u_init,y_init]...)
    α = pinv(A)*b
end

"""
step_traj(H_u::Matrix{Float64}, H_y::Matrix{Float64}, α::Vector{Float64})

Multiply Hankel matrices by solution vector ``\alpha``.
Intended uses:
1. Advance a trajectory forward in time using a time-shifted Hankel matrix
2. Verify a previously computed solution
"""
function step_traj(H_u::Matrix{Float64}, H_y::Matrix{Float64}, α::Vector{Float64})
    H_y*α, H_u*α
end

# TODO: make this a method on the underlying dynamics type
function rollout(input::Vector{Float64}, u_init::Vector{Float64}, y_init::Vector{Float64}, u_data::Vector{Float64}, y_data::Vector{Float64}, L::Int)
    H_u, H_y, H_u_shift, H_y_shift = Hankel_shift(u_data,y_data,L)
    output = zeros(size(input))
    for (i,u) in enumerate(input)
        α = Hankel_solver(H_u, H_y, u_init, y_init)
        y_init,_ = step_traj(H_u_shift, H_y_shift, α)
        u_init = vcat([u_init,u]...)[2:end]  
        output[i] = y_init[end]
    end
    output
end