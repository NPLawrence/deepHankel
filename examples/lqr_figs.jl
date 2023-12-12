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
# scalefontsizes(1.3)

Ts = 1.0
s = tf("s")
z = tf("z", Ts)
# P_c = 1/(s^2 + 0.5s + 1)
# P = c2d(P_c,  Ts)
# P = (0.01z^4 + 0.0074z^3 + 0.000924z^2 − 0.000017642z) / (z^5 − 2.14z^4 + 1.5549z^3 − 0.4387z^2 + 0.042025z)
P = 0.1159(z^3 + 0.5z) / (z^4 - 2.2z^3 + 2.42z^2 - 1.87z + 0.7225)

plot(impulse(P))

L = 5
N = 400
σ² = 1.0
# u = randn(N)
B = behavior(P,L=L,N=N,σ²=σ²,u=u,method="Noisy")


function depth_lqr(B::behavior; L_list = [5 10 20], rollouts=10, solver=nothing, method="Noisy")

    
    # baseline = step_LQR(P, B.u, B.y_clean, B.L)
    T = 100
    mean_std = []
    û = deepcopy(B.u) 
    for L in L_list
        outputs = []
        u = û
        for i in 1:rollouts
            B.noise = B.σ²*randn(length(B.noise))
            y = B.y_clean .+ B.noise
            u, y = preprocess_data(u, y, L, method=method, leave_u = i > 1)
            step_response = step_LQR(P, u, y, L, tfinal=T)
            ground_truth  = step_LQR(P, B.u, B.y_clean, L, tfinal=T)
        push!(outputs, ground_truth .- step_response)
        # push!(outputs, step_response)
        end
        # println(size(outputs))
        push!(mean_std, [mean(outputs), std(outputs)])
    end
    # push!(mean_std, baseline)
    mean_std, L_list
end




function lqr_plot(B::behavior, mean_std, L_list; save_path="./figures/lqr_plot_delete")
    gr()
    colors = palette(:default)
    # L2D = PyPlot.matplotlib.lines.Line2D
    # plt = PyPlot.axes()
    # fig, ax = PyPlot.subplots()
    plt = plot( 
        xlabel  = "Time",
        ylabel = "Output",
        # ylabel = "Deviation",
        # ylims = (-0.18, 0.18),
        xlims = (0, 100),
        guidefont = (20, plot_font),
        legendfontsize = 16,
        tickfontsize=14,
        legend = true,
        right_margin=12px,
    )
    # annotate!(4.15,0.15, L"\uparrow L")

    linestyles=[:dashdot :solid :dot]
    c = [colors[2] colors[1] colors[3]]
    # for (i,ms) in enumerate(mean_std)
    #     plot!(plt, ms[1], ribbon=ms[2], label="", linestyle = linestyles[i], color=c[i]) 
    #     # ax.plot(B.t, ms[1])
    #     hline!([2], lw=1.25, label=LaTeXString("\$L = $(L_list[i])\$"), linestyle = linestyles[i], color=c[i])
    # end
    # plot!(plt, mean_std[end], lw=4, linestyle=:dash, color="grey", label="")
    # plot!(legend=:topright)
    # hline!([2], linestyle=:dash, lw=1)
    # hline!([1], color="grey", linestyle=:dash, lw=2, label="")
    
    plot!(B.y, lw=1.5, linestyle=linestyles[2], color=c[2], label="Noisy", linetype=:step, linealpha=0.75)
    plot!(B.y_clean, linestyle=linestyles[3], color=c[1], label="", linetype=:step)
    plot!(ylims = (1.01*minimum(B.y), 1.05*maximum(B.y)))
    hline!([10], lw=1.5, linestyle=linestyles[3], color=c[1], label="Clean")

    hlabel1 = 0.70
    hlabel2 = 0.30
    n = 4
    xvals = [0.05;0.25]
    yvals = vcat([hlabel1 - ((n-1)*0.075)/2:0.075:hlabel1 + ((n-1)*0.075)/2]...)
    # plot!(xvals,[yvals'; yvals'],frame=:box, xlims=(0,1), ylims=(0,1),
    #        palette = [colors[1:4];"grey"],
    #        yticks=false,
    #        xticks=false,
    #        legend=false,
    #        inset=bbox(0.10,0.05,0.29,0.150,:right),
    #        annotations=[(0.30,hlabel1,"Predictions"), (0.30,hlabel2,"Ground truth")],
    #        annotationhalign=:left,
    #        annotationfontsize=12,
    #        subplot=2)
    # plot!(xvals, [hlabel2;hlabel2], ls=:dash, color="grey", subplot=2)
    savefig(save_path*".pdf")
    savefig(save_path)

    plt
end



