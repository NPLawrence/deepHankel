"""
code to generate figures from paper...see approx_error.ipynb for more illuminating/understandable use cases. 
"""

using Revise
using deepHankel
using ControlSystems
using LinearAlgebra
# using LinearSolve
using Statistics
using Plots
using Plots.PlotMeasures
using LaTeXStrings
using DelimitedFiles
using BSON
using SingularSpectrumAnalysis


plot_font = "Computer Modern"
default(fontfamily=plot_font,
        linewidth=4, framestyle=:axes, grid=true)




function depth_singular(; L = 10, samples=50)

    traj_len = 10000
    N_vals = 100:100:traj_len
    median_max_min = []
    for N in N_vals
        singular_values = []

        for i in 1:samples

            z = randn(N)
            H = Hankel(z, L)
            σ² = minimum(svd(H).S)
            push!(singular_values, σ²)

        end
        p = 0.75
        push!(median_max_min, [median(singular_values), quantile(singular_values,p), quantile(singular_values,1-p)])

    end 
    push!(median_max_min, N_vals)

    median_max_min




end


function plot_singular(mean_std; L=10, save_path="./figures/singular_values")

    gr()
    colors = palette(:default)

    plt = plot( 
        xlabel  = L"N",
        ylabel = L"\frac{1}{\sigma}",
        xlims = (mean_std[end][1], mean_std[end][end]),
        yticks = 0:0.1:0.2,
        xaxis=:log,
        guidefont = (20, plot_font),
        legendfontsize = 16,
        tickfontsize=20,
        legend = true,
        right_margin=16px,
        left_margin=16px,
        bottom_margin=8px,
    )


    yvals = [1/ms[1] for ms in mean_std[1:end-1]]
    ribbon_lower = [1/ms[2] for ms in mean_std[1:end-1]]
    ribbon_upper = [1/ms[3] for ms in mean_std[1:end-1]]


    plot!(plt, mean_std[end], yvals, ribbon=(ribbon_lower, ribbon_upper), label=L"\frac{1}{\sigma_\min}")

    f(N) = ((L+1)/L) / sqrt(N)

    plot!(plt, mean_std[end], f, label=L"\frac{1}{\sqrt{N}}\frac{L+1}{L}", linestyle=:dash)
    # for (i,ms) in enumerate(mean_std[1:end-1])

    #     println(ms[1])

    #     plot!(plt, mean_std[end], ms[1], ribbon=ms[2], label="") 
    #     # ax.plot(B.t, ms[1])

    # end
    plot!(legend=:topright)
    savefig(save_path*".pdf")
    savefig(save_path)

    plt

end
