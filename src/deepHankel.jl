module deepHankel

using ControlSystems
using LinearAlgebra
using Statistics
using Plots
using LaTeXStrings
using BSON
# using LinearSolve
using SingularSpectrumAnalysis
using RollingFunctions

# using JLD
# using RollingFunctions
# using SingularSpectrumAnalysis
# using OrdinaryDiffEq


export excite, Hankel, Hankel_shift, Hankel_rollout, Hankel_SS, fifo_LQR, Hankel_LQR, step_LQR, behavior, preprocess_data

include("behavior.jl")

end # module deepHankel