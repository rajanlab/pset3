% tic
first = 1;
learn = 1;
run = 1;
if (first)
%    load('shared_randoms.mat') % learnList, wn_rand, weights_rand;
    g = 1.5;
    nRunTot = 125;
    nFree = 5;
    dtData = 0.0641;
    dt = 0.001; % integration step
    tau = 0.01; % 10ms time constant
    P0 = 1;
    tauWN = 1;
    ampIn = 1;
    N = 500;
    nLearn = 50;
    epochs = 165;
    learnList = randperm(N);
    cL = learnList(1:nLearn);
    nCL = learnList(nLearn:end);
    
    tData = 0:dtData:epochs*dtData;
    t = 0:dt:tData(end);
    xBump = zeros(N, length(tData));
    
    sig = 0.0343*N; % scaled correctly in neuron space!!!
    xBump = zeros(N, length(tData));
    for i=1:N
        xBump(i, :) = exp(-(i-N*tData/tData(end)).^2/(2*sig^2));
    end
    %             xBump = xBump/max(max(xBump));
    hBump = log((xBump+0.01)./(1-xBump+0.01)); % current from rate
    %         xBump = hBump;
    
    ampWN = sqrt(tauWN/dt);
    iWN = ampWN*randn(N, length(t));
%    iWN = ampWN*wn_rand;
    input = ones(N, length(t));
    for tt = 2: length(t)
        input(:, tt) = iWN(:, tt) + (input(:, tt - 1) - iWN(:, tt))*exp(-(dt/tauWN));
    end
    input = ampIn*input;
    
    noiseLevel = 0.5;
    sigN = noiseLevel*sqrt(tau/dt);
    
    J = g*randn(N,N)/sqrt(N);
%    J = g*weights_rand/sqrt(N);
    J0 = J;
    R = zeros(N, length(t));
    JR = zeros(N, 1);
    %                 figure(1)
    %                 imagesc(tData, 1:N, xBump)
    %            colorbar;
end

%disp "learnList"
%size(learnList)
%disp "cL"
%size(cL)
%disp "xBump"
%size(xBump)
%disp "hBump"
%size(hBump)
%
%disp "input"
%size(input)
%
%disp "J"
%size(J)
%disp "JR"
%size(JR)
%disp "R"
%size(R)
%
%learnList(1, 1:10)
%cL(1, 1:10)


if (run)
    if (learn)
        PJ = P0*eye(nLearn, nLearn);
    end
    rightTrial = false;
    for nRun=1:nRunTot
                    nRun
        %         if (run==30)
        %             J(nCL, nCL) = 0;
        %         end
        %             chi2(nRun) = 0;
        H = xBump(:, 1);
        tLearn = 0;
        iLearn = 2;
        %             frac = floor(nL(pp)*N);
        for tt=2:length(t)
            tLearn = tLearn + dt;
            R(:, tt) = 1./(1+exp(-H));
            JR = J*R(:, tt) + input(:, tt)+ sigN*randn(N, 1);
%            p1  = input(:, tt)+ sigN*stupid_rand;
%            p2  = J*R(:, tt);
%            JR = p1 + p2 ;
            H = H + dt*(-H + JR)/tau;
            if ((learn)&(tLearn>=dtData)&(nRun<nRunTot-nFree))
                tLearn = 0;
                err(1:N, :) = JR(1:N, :) - hBump(1:N, iLearn);
                %       chi2(nRun) = chi2(nRun) + mean(err.^2);
                %       err = R(:, tt) - xBump(:, iLearn);
                iLearn = iLearn + 1;
%                fprintf("size(%s): %s", 'R', disp(size(R(cL, tt))))
%                fprintf("%s: %s", 'R', disp(R(1:5,1)))

                k = PJ*R(cL, tt);
                rPr = R(cL, tt)'*k;
                c = 1.0/(1.0 + rPr);
                PJ = PJ - c*(k*k');
                p1 = J(1:N, cL);
                p2 = c*err(1:N, :)*k';
                J(1:N, cL) = p1 - p2;
            end
        end
        figure(3)
        imagesc(R/max(max(R)));
        axis square; colorbar;
        pause(0.01);
    end
end
%     cv(ll) = std(std(J(cL,cL)))/mean(mean(J(cL,cL)))
% end
% toc
%     MagnL(ll) = sum(sum(abs(J0 - J)))./sum(sum(abs(J0)))
% chi2 = chi2/floor(length(t)/dt);
% rData = hBump;
% varData = var(reshape(rData, N*length(tData), 1));
% chi2 = chi2/(sqrt(N*length(tData))*varData);
% figure(14)
% hold on
% plot(chi2, '.-');
% xlabel('learning steps');
% ylabel('chi^2 error during learning')
% end
%     rData = xBump;%randn(N, length(tData));
%     rModel = R/max(max(R));%randn(N, length(tModel));
%     iModelSample = zeros(length(tData), 1);
%     for i=1:length(tData)
%         [tMdodelSample, iModelSample(i)] = min(abs(tData(i)-t));
%     end
%     rModelSample = rModel(:, iModelSample);
%     stdData = std(reshape(rData, N*length(tData), 1))
%     pVarIdSeqnL(ll) = 1 - (norm(rData - rModelSample, 'fro')/(sqrt(N*length(tData))*stdData)).^2
% end
%     G = schur(J, 'complex');
%     GD = diag(G);
%     GN = G - diag(GD);
%     GVec = reshape(GN, N.^2, 1);
%     GSum = GVec'*GVec;
%     DSum = GD'*GD;
%     QOneSeqNoInputnL(ll)= GSum/(DSum+GSum)


P = R/max(max(R));

%         P = hBump;
%     [ppMax, pMax] = max(P');
%     PShift = zeros(size(P));
%     iPeak = pMax;
%     for i=1:size(P, 1)
%         j = ceil(size(P, 2)/2)-iPeak(i);
%         PShift(i, :) = circshift(P(i, :), [0 j]);
%     end
%     for i=1:size(P, 1)
%         iPeak(i) = round(mean(PShift(i, :).*(1:size(P, 2)))/mean(PShift(i, :)));
%     end
%     for i=1:size(P, 1)
%         j = ceil(size(P, 2)/2)-iPeak(i);
%         PShift(i, :) = circshift(PShift(i, :), [0 j]);
%     end
%     PVec = reshape(P, size(P, 1)*size(P, 2), 1);
%     PVarN(pp) = 100*(1-mean(var(PShift))/var(PVec))
rData = xBump;%randn(N, length(tData));
rModel = P;%R/max(max(R));%randn(N, length(tModel));
iModelSample = zeros(length(tData), 1);
for i=1:length(tData)
    [tMdodelSample, iModelSample(i)] = min(abs(tData(i)-t));
end
rModelSample = rModel(:, iModelSample);
stdData = std(reshape(rData, N*length(tData), 1))
pVarN = 1 - (norm(rData - rModelSample, 'fro')/(sqrt(N*length(tData))*stdData)).^2
% end

%     rData = xBump;%randn(N, length(tData));
%     rModel = P;%R/max(max(R));%randn(N, length(tModel));
%     iModelSample = zeros(length(tData), 1);
%     for i=1:length(tData)
%         [tMdodelSample, iModelSample(i)] = min(abs(tData(i)-t));
%     end
%     rModelSample = rModel(:, iModelSample);
%     stdData = std(reshape(rData, N*length(tData), 1))
%     pVar2(pp) = 1 - (norm(rData - rModelSample, 'fro')/(sqrt(N*length(tData))*stdData)).^2
% end

%     eigP = eig(cov(P'));
%     eigP = eigP/sum(eigP);
%     eigP = sort(eigP, 'descend');
%     [m dim(pp)] = min(cumsum(eigP)<0.95)
% end
% save J40500.mat J0 J R PVar;

