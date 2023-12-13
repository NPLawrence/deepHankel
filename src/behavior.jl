mutable struct behavior{T}
    P
    u_clean::Vector{T}
    u::Vector{T}
    y_clean::Vector{T}
    y::Vector{T}
    t::StepRangeLen
    noise::Vector{T}
    σ²::T
    L::Int

end


function preprocess_data(u, y, L; method="Noisy", leave_u=false)

    if method == "Noisy"
        y = y
    elseif method == "SSA"
        mn = mean(y)
        # y = y .- mn
        # y, ys = analyze(y, L, robust=true)
        y = ssa(y, L)
        # y = y .+ mn
    elseif method == "Smooth"
        K =  Int(floor(sqrt(L)))
        leave_u ? u=u : u = rollmean(u, K)
        y = rollmean(y, K) 
    end
    vec(u), vec(y)
end

function behavior(P;L::Int64=10,N::Int64=200,σ²::Float64=0.1, u=[], method="Noisy")
    isempty(u) ? u_clean = excite(N) : u_clean=u
    y_clean, t, _, _ = lsim(P, u_clean')
    y_clean = vec(y_clean)
    noise = σ²*randn(N)
    y = y_clean .+ noise
    u, y = preprocess_data(u_clean, y, L, method=method)
    # if preprocess == "ssa"
    #     y, ys = analyze(y, L, robust=true)
    # elseif preprocess == "smooth"
    #     nothing
    # end
    behavior(P,u_clean,u,y_clean,y,t,noise,σ²,L)
end


"""
A bit confusingly, L in `preprocess` is d in this function.
d corresponds to how many singular values we use, while L in `ssa` is some fixed and large-ish depth parameter.
"""
function ssa(y, d; L=30)

    # L = Int(floor(length(y)/4))
    d = min(d+1,L) # I'm not sure why but tricking the algorithm into adding 1 extra singular value helps stability ALOT

    # USV = hsvd(y,d, robust=true)
    # trend, seasonal_groupings = autogroup(USV,0.999)
    # reconstruct(USV, trend, seasonal_groupings)[1]

    # ynew = analyze(y,d,robust=false)[1]
    
    H = Hankel(y,L)
    # H = rpca(H)[1]
    USV = svd(H)
    Happrox = USV.U[:,1:d] * Diagonal(USV.S[1:d]) * USV.Vt[1:d,:]

    ynew = unhankel(Happrox)

end



"""
Creates excitation signal of order `L`. Additional inputs `num_data::Int` and `m::Int` ensure a minimum length signal is obtained.
"""
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


"""
Hankel_solver(H_u::Matrix{Float64}, H_y::Matrix{Float64}, u_init::Vector{Float64}, y_init::Vector{Float64}; solver="")

Solve the standard linear system in Willems' fundamental lemma.
"""
function Hankel_solver(H_u::Matrix{Float64}, H_y::Matrix{Float64}, u_init::Vector{Float64}, y_init::Vector{Float64}; solver=nothing, sol=nothing)
    A = vcat([H_u,H_y]...)
    b = vcat([u_init,y_init]...)
    # prob = LinearProblem(A,b)
    if isnothing(sol)
        α = A\b, pinv(A)
    else
        α = sol*b, sol
    end

    # sol = solve(prob, solver)
    # α = sol.u
    
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

"""
Hankel_SS(u_data::Vector{Float64}, y_data::Vector{Float64}, L::Int)

Construct a state-space model describing the ``\alpha`` dynamics, assuming the Moore-Penrose inverse solution.
"""
function Hankel_SS(u_data::Vector{Float64}, y_data::Vector{Float64}, L::Int; Ts::Float64=0.1)

    H_u, H_y, H_u_shift, H_y_shift = Hankel_shift(u_data,y_data,L)

    H = vcat([H_u, H_y]...)
    H_pinv = pinv(H)
    Ã = vcat([H_u_shift[1:end-1,:],zeros(1,size(H_u_shift,2)),H_y_shift]...)
    B̃ = vcat([zeros(L-1),ones(1),zeros(L)]...)

    A = H_pinv*Ã
    B = H_pinv*B̃
    C = H_y_shift[end,:]'

    ss(A,B,C,0,Ts)

end

function Hankel_SS(B::behavior)

    Hankel_SS(B.u, B.y, B.L, Ts=B.P.Ts), Hankel_SS(B.u, B.y_clean, B.L, Ts=B.P.Ts)

end

"""
Hankel_LQR()

Computes a gain matrix ``K`` via a state-space realization of the ``\alpha`` dynamics --- see `Hankel_SS()`.
Features an option to compute an integral gain.
"""
function Hankel_LQR(u_data, y_data, L; Q::Union{UniformScaling,Diagonal}=I, R::Union{UniformScaling,Diagonal}=I, integrator::Bool=false)

    sys = Hankel_SS(u_data, y_data, L)
    Â, B̂, Ĉ = sys.A, sys.B, sys.C
    
    R̂ = R

    if integrator
        A = [Â zeros(size(Â,1),size(Ĉ,1)); -Ĉ I]
        B = vcat([B̂, 0]...)
        Q̂ = [zeros(size(Â,1)); 1.0]*Q*[zeros(size(Â,1)); 1.0]'
        lqr(Discrete,A,B,Q̂,R̂)
    else
        Q̂ = Ĉ'*Q*Ĉ # cost on the output, not the "state" α
        lqr(Discrete,Â,B̂,Q̂,R̂)
    end
end


function Hankel_LQR(B::behavior; Q::Union{UniformScaling,Diagonal}=I, R::Union{UniformScaling,Diagonal}=I, integrator::Bool=false)

    Hankel_LQR(B.u, B.y, B.L, Q=Q, R=R, integrator=integrator), Hankel_LQR(B.u, B.y_clean, B.L, Q=Q, R=R, integrator=integrator)

end

function step_LQR(P, u, y, L; tfinal = nothing, Q::Union{UniformScaling,Diagonal}=I, R::Union{UniformScaling,Diagonal}=I)

    Ts = P.Ts
    z = tf("z", Ts)

    K = Hankel_LQR(u,y,L,Q=Q,R=R,integrator=true)
    H_u, H_y, H_u_shift, H_y_shift = Hankel_shift(u,y,L)
    H_pinv = pinv(vcat([H_u, H_y]...))
    A_fifo, B_fifo, C_fifo = fifo_LQR(K[1:end-1]', H_pinv, L)
    u_x = ss(A_fifo, B_fifo, C_fifo, 0, Ts)

    sys = feedback(feedback(P,-u_x)*(-K[end]*Ts*z/(z-1)))
    
    return isnothing(tfinal) ? vec(step(sys).y) : vec(step(sys, tfinal).y)

end

# TODO: make this a method on the underlying dynamics type
function Hankel_rollout(input::Vector{Float64}, u_init::Vector{Float64}, y_init::Vector{Float64}, u_data::Vector{Float64}, y_data::Vector{Float64}, L::Int; solver=nothing)
    H_u, H_y, H_u_shift, H_y_shift = Hankel_shift(u_data,y_data,L)
    output = zeros(size(input))
    α_norm = zeros(size(input))
    pinv=nothing
    for (i,u) in enumerate(input)
        α, pinv = Hankel_solver(H_u, H_y, u_init, y_init, solver=solver, sol=pinv)
        # if i == 1
        #     α, pinv = Hankel_solver(H_u, H_y, u_init, y_init, solver=solver)
        # else
        #     α, pinv = Hankel_solver(H_u, H_y, u_init, y_init, solver=solver, sol=pinv)
        # end
        y_init,_ = step_traj(H_u_shift, H_y_shift, α)
        u_init = vcat([u_init,u]...)[2:end]  
        output[i] = y_init[end]
        α_norm[i] = norm(α)
    end
    output, α_norm
end


function Hankel_rollout(B::behavior; noisy=true, solver=nothing)
    noisy ? Hankel_rollout(B.u_clean, zeros(B.L), zeros(B.L), B.u, B.y, B.L, solver=solver) : Hankel_rollout(B.u, zeros(B.L), zeros(B.L), B.u, B.y_clean, B.L, solver=solver)
end


"""
fifo_LQR(K,H)

Takes the LQR controller ``u=-K\alpha`` and re-writes it in state-space form.
Namely, we want ``u=-K H^{+}_{L} [ū ȳ]^T`` instead, which requires maintaining a moving history of past input-output data.
"""
function fifo_LQR(K,H_pinv,L)

    C = -K*H_pinv
    A = [zeros(L-1,1) I zeros(L-1,L); C; zeros(L-1,L+1) I; zeros(1,2L)]
    B = [zeros(2L-1,1); 1]

    A, B, C
end

function behavior_lsim(P,u,t,x,L)

    ȳ = zeros(L)
    ū = zeros(L)

    for _ in t
        x = P.A*x + P.B*u(x,t)
        y = P.C*x
    end
end
