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
        linewidth=3, framestyle=:axes, grid=true)

Ts = 0.1
s = tf("s")
z = tf("z", Ts)
P_c = 1/(s^2 + 0.5s + 1)
P = c2d(P_c,  Ts)
# P = 0.1159(z^3 + 0.5z) / (z^4 - 2.2z^3 + 2.42z^2 - 1.87z + 0.7225)

L = 5
N = 200
σ² = 0.05

u = -1*vec(readdlm("./examples/data/input.txt"))
N = length(u)

# u = randn(N)
B = behavior(P,L=L,N=N,σ²=σ²,u=u,method="Noisy")

function depth_predict(B::behavior; L_list = [2 5 10 20], rollouts=50, solver=nothing, method="Noisy")

    mean_std = []
    û = deepcopy(B.u) 
    for L in L_list
        outputs = []
        u = û
        for i in 1:rollouts
            B.noise = B.σ²*randn(length(B.noise))
            y = B.y_clean .+ B.noise
            # u = B.u
            u, y = preprocess_data(u, y, L, method=method, leave_u = i > 1)
            # B.u = u
            # B.y = y
            # B.y_clean = B.y_clean
            # B.L = L
            # push!(outputs, Hankel_rollout(B, solver=solver)[1])
            push!(outputs, Hankel_rollout(û, zeros(L), zeros(L), u, y, L, solver=solver)[1])
        end
        push!(mean_std, [mean(outputs), std(outputs)])
    end
    push!(mean_std, B.y_clean)
    mean_std, L_list
end




function depth_plot(B::behavior, mean_std, L_list; save_path="./figures/depth_plot_delete")
    gr()
    colors = palette(:default)

    plt = plot( 
        xlabel  = "Time",
        ylabel = "Output",
        ylims = (-0.05, 0.425),
        xlims = (B.t[1], B.t[41]),
        formatter=Returns(""),
        guidefont = (16, plot_font),
        legendfontsize = 12,
        legend = false,
        right_margin=32px,
    )
    annotate!(4.15,0.15, L"\uparrow L")

    
    # lines = ax.plot(data)
    # ax.legend(custom_lines, ["'Cold'", "'Medium'", "'Hot'"])

    for (i,ms) in enumerate(mean_std[end-1:-1:1])

        plot!(plt, B.t, ms[1], ribbon=ms[2], label="") 
        # ax.plot(B.t, ms[1])

    end
    plot!(plt, B.t, mean_std[end], lw=4, linestyle=:dash, color="grey", label="")
    # plot!(legend=:outerright)

    # plot!(legend = (custom_lines, ["'Cold'", "'Medium'", "'Hot'"]))
    # ax = PyPlot.gca()
    # ax.legend(custom_lines, ["'Cold'", "'Medium'", "'Hot'"])   
    
    # plt_legend = plot(legend=false)


    hlabel1 = 0.70
    hlabel2 = 0.30
    n = 4
    xvals = [0.05;0.25]
    yvals = vcat([hlabel1 - ((n-1)*0.075)/2:0.075:hlabel1 + ((n-1)*0.075)/2]...)
    plot!(xvals,[yvals'; yvals'],frame=:box, xlims=(0,1), ylims=(0,1),
           palette = [colors[1:4];"grey"],
           yticks=false,
           xticks=false,
           legend=false,
           inset=bbox(0.10,0.05,0.29,0.150,:right),
           annotations=[(0.30,hlabel1,"Predictions"), (0.30,hlabel2,"Ground truth")],
           annotationhalign=:left,
           annotationfontsize=12,
           subplot=2)
    plot!(xvals, [hlabel2;hlabel2], ls=:dash, color="grey", subplot=2)
    savefig(save_path*".pdf")
    savefig(save_path)

    plt
end




function depth_gif(B::behavior, mean_std, L_list; save_path="./figures/depth_gif_delete")
    gr()
    colors = palette(:default)

    plt = plot( 
        xlabel  = "Time",
        ylabel = "Output",
        title = "Predictions vs ground truth",
        ylims = (-0.05, 0.425),
        xlims = (B.t[1], B.t[41]),
        formatter=Returns(""),
        guidefont = (16, plot_font),
        legendfontsize = 12,
        legend = false,
        right_margin=80px,
    )


    plot!(plt, B.t, mean_std[end], lw=4, color="grey", label="")
    anim = Animation()
    frame(anim, plt)


    plot!(plt, B.t, B.y, seriestype=:scatter, markersize=3.0, color="red")
    frame(anim, plt)

    
    # lines = ax.plot(data)
    # ax.legend(custom_lines, ["'Cold'", "'Medium'", "'Hot'"])


    for (i,ms) in enumerate(mean_std[1:end-1])

        if i == 1
            annotate!(4.35,0.05, L"\uparrow"*"depth")
        end

        plot!(plt, B.t, ms[1], ribbon=ms[2], label="", color=colors[length(L_list)+1-i])
        frame(anim, plt)

    end



    # plot!(legend=:outerright)

    # plot!(legend = (custom_lines, ["'Cold'", "'Medium'", "'Hot'"]))
    # ax = PyPlot.gca()
    # ax.legend(custom_lines, ["'Cold'", "'Medium'", "'Hot'"])   
    
    # plt_legend = plot(legend=false)


    # hlabel1 = 0.70
    # hlabel2 = 0.30
    # n = 4
    # xvals = [0.05;0.25]
    # yvals = vcat([hlabel1 - ((n-1)*0.075)/2:0.075:hlabel1 + ((n-1)*0.075)/2]...)
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
    # plot!(xvals, [hlabel2;hlabel2], ls=:solid, color="grey", subplot=2)

    for i in 1:5
        frame(anim, plt)
    end

    # savefig(save_path*".gif")
    # savefig(save_path)
    gif(anim, save_path*".gif", fps=0.80)
end



"""
Each value in N_list implies one curve, where the x axis will be L and the y axis the mse.
We collect the mean and std over rollouts for each value in L_list x N_list.
"""
function depth_mse(P, σ²; methods=["Noisy" "Smooth" "SSA"], L_list = 2:20, N_list = [150 200 250], rollouts=10, save_path="./examples/data/mse_delete.bson")
    # 2:2:40, [1500 2500 5000] 1.0 var

    mse = []
    for method in methods
        @info  "Prediction strategy: "  method
        mse_L = Array{Float64}(undef, 0, 6)
        for L in L_list
            @info  "Depth: "  L
            errors = []
            for N in N_list
                for _ in 1:rollouts
                    B = behavior(P,L=L,N=N,σ²=σ²,method=method)
                    data_length = 1:min(length(B.y_clean),500)
                    loss = sqrt(mean((Hankel_rollout(B)[1][data_length] .- B.y_clean[data_length]).^2))
                    push!(errors, loss)
                    # push!(errors, mean(abs.(Hankel_rollout(B)[1] .- B.y_clean)))
                    # mse += mean((Hankel_rollout(B)[1] .- B.y_clean).^2)
                end
            end
            mse_L = vcat([mse_L, [L mean(errors) median(errors) std(errors) minimum(errors) maximum(errors)]]...)
        end
        push!(mse, (method, mse_L))
    end
    BSON.@save save_path mse
    mse
end


function mse_plot(mse; save_path="./figures/mse_plot_mean_delete")
    gr()
    colors = palette(:default)
    # L2D = PyPlot.matplotlib.lines.Line2D
    # plt = PyPlot.axes()
    # fig, ax = PyPlot.subplots()
    plt = plot( 
        title = "Large variance",
        xlabel  = L"L",
        ylabel = "RMSE",
        # ylims = (0.0, 0.3),
        # formatter=Returns(""),
        guidefont = (16, plot_font),
        legendfontsize = 12,
        legend = true
        # right_margin=32px,
    )

    linestyles=[:solid :dash :dot]
    
    for (i,method) in enumerate(mse)

        # println(method[2])
        data = method[2]
        # plot!(data[:,1], data[:,3], ribbon=data[:,4])
        # abs.(stats["median"].-stats["min"])
        label = method[1]
        # plot!(data[:,1], data[:,3], ribbon=(abs.(data[:,3] .- data[:,5]),abs.(data[:,3] .- data[:,6])), linestyle = linestyles[i], label=label)
        plot!(data[:,1], data[:,2], ribbon=ribbon=data[:,4], label=label, linestyle = linestyles[i])
        plot!(xlims=(data[1,1],data[end,1]))
        # plot!(ylims=(0,0.5))


    end
    plot!(legend=:topright)
    savefig(save_path*".pdf")
    savefig(save_path)
    plt
end