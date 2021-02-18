#!/usr/bin/env julia

module PINningDevIdealBump
import Dates
import LinearAlgebra
import Random

import MAT
import NPZ
import GR

#function wrap(;bottled_noise=false, learn=true, run=true, save=false, plot=false)
    bottled_noise = true
    learn = true
    run = true
    save = false
    plot = false

    if bottled_noise === true
        loaded = MAT.matread("shared_randoms.mat")
    end

    g = 1.5
    nRunTot = 125
    nFree = 5
    dtData = 0.0641
    dt = 0.001  # 1 ms integration step
    tau = 0.01  # 10ms time constant
    P0 = 1
    tauWN = 1
    ampIn = 1
    N = 500
    nLearn = 50
    epochs = 165

    JR = nothing
    err = nothing
    PJ = nothing

    if bottled_noise === true
        learnList = convert(Array{Int32}, loaded["learnList"])
    else
        learnList = Random.randperm(N)
    end
    cL = learnList[1:nLearn]
    @assert length(cL) == nLearn
    @assert size(cL) == (nLearn,)
    nCL = learnList[nLearn+1:end]

    tData = [0:dtData:epochs*dtData;]
    t=[0:dt:tData[end];]

    xBump = zeros(N, length(tData))
    sig = 0.0343*N  # % scaled correctly in neuron space!!!

    decay = 2*sig^2
    for unit = 1:N
        xBump[unit, :] = exp.(-((float(unit) .- N*tData/tData[end]) .^ 2.0) / decay)
    end
    hBump = log.((xBump .+ 0.01)./(1 .- xBump .+ 0.01))  # current from rate

    ampWN = sqrt(tauWN/dt)

    if bottled_noise === true
        iWN = ampWN*loaded["wn_rand"]
    else
        iWN = ampWN*randn(N, length(t))
    end
    input = ones(N, length(t))
    decay = exp(- (dt / tauWN))
    for tt = 2:length(t)
        input[:, tt] = iWN[:, tt] + (input[:, tt - 1] - iWN[:, tt]) * decay
    end
    input = ampIn*input

    noiseLevel = 0.5
    sigN = noiseLevel * sqrt(tau / dt)

    if bottled_noise === true
        J = g * loaded["weights_rand"] / sqrt(N)
    else
        J = g * randn(N, N) / sqrt(N)
    end

    J0 = copy(J)
    R = zeros(N, length(t))
    JR = zeros(N, 1)


    if run === true
        if learn === true
            PJ = P0 * LinearAlgebra.I(nLearn)
            err = 0
        end
        rightTrial = false

        for nRun = 1:nRunTot
            global JR
            global err
            global PJ

            println(nRun)
            H = xBump[:, 1]
            tLearn = 0
            iLearn = 2
            for tt = 2:length(t)
                tLearn += dt
                R[:, tt] = 1 ./ (1 .+ exp.(-H))
                if bottled_noise === true
                    noise = loaded["stupid_rand"]
                else
                    noise = randn(N, 1)
                end
                JR = input[:, tt] + sigN * noise + J * R[:, tt]
                H = H + dt * (-H + JR) / tau
                if learn === true && tLearn >= dtData && nRun < (nRunTot - nFree + 1)
                    tLearn = 0
                    err = JR[1:N, :] - hBump[1:N, iLearn]
                    iLearn = iLearn + 1
                    r_slice = R[cL, tt]
                    k = PJ * r_slice
                    rPr = (transpose(r_slice) * k)[1]
                    c = 1.0/(1.0 + rPr)
                    PJ = PJ - c*(k * transpose(k))
                    J[1:N, cL] = J[:, cL] - c* err * k'
                end
            end
            if plot === true
                p = GR.heatmap(t, [1:500.0;], R[1:end-1, 1:end-1]/maximum(R))
            end
        end
        if save === true
           timestamp = Dates.format(Dates.now(), "yyyymmddTHHMM")
           outfile = "finals_$timestamp.npz"
           NPZ.npzwrite(outfile, Dict("J" => J,
                                      "err" => err,
                                      "R" => R))
        end
    end
    return
#end

#if abspath("PINningDevIdealBump.jl") == @__FILE__
#    wrap(run=true, learn=true, save=true, bottled_noise=true,
#         plot=false)
#    wrap(run=true, learn=true, save=true, bottled_noise=true,
#         plot=true)
end
end
