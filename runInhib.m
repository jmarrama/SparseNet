
%%%%%% beginning stuff - if data is loaded

excitatoryTrained = 0;
inhibitoryTrained = 0;
alltrained = 0;

if(alltrained ==0)

    %%%%%% load data and train  

    loadImages

%     [Evishid eOut] = chlRBM(trainSet, 500);
    [Evishid eOut] = bpRBMf(trainSet, 500);

    [inhibOut, Ivishid Ihidout IhidB IoutB] = bpAutoEncF(trainSet, eOut, 125);


else
   
    load 'inhibModel/300-75-run.mat'
    
end

%%%%%% run data through, calc overall output - TODO - add into bpAutoEncF
% batchSize = size(trainSet, 2);
% inhibOut = Ivishid*trainSet + repmat(IhidB,1,batchSize);    
% inhibOut = Ihidout*inhibOut + repmat(IoutB,1,batchSize);

totalOut = eOut - inhibOut;

%%%%%% see what this looks like!

[overallWeights, nld] = chlRBMtarget(trainSet, totalOut);
