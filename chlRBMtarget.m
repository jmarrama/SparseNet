function [vishid, nextLayerData] = chlRBMtarget(trainSet, targetAct)

%set initial params
%%%%%%%% learning params %%%%%%%%%%%%%%%%
inputSize = size(trainSet,1);
outputSize = size(targetAct,1);
momentum = 0.0;
weightdecay = 0.01;

anSched = [0.5 0.5 0.1];
anStart = [1 20 4000];
anPos = 1;
learnrate = anSched(1);

%%%%%%%%% algorithm params %%%%%%%%%%%%%
numepochs = 1000;
numEx = size(trainSet,2);
batchSize = numEx;
numBatches = floor(size(trainSet,2)/batchSize);

vishidinc = 0;
%visbiasinc = 0;
%hidbiasinc = 0;

%%%%%%%%% initialize weights %%%%%%%%%%%%%%
vishid = 0.05*randn(outputSize, inputSize);
%visbiases = zeros(inputSize,1);
%hidbiases = zeros(hiddenLayerSize,1);


for epoch=1:numepochs
   
    errsum = 0;
    tic;
    
    %%%%%%%% simulated annealing with momentum and learning rate %%%%%%%%
%     if epoch>5,
%         momentum = 0.9;
%     else
%         momentum = 0.5;
%     end;
    
    if (anPos < length(anStart))
        if (epoch >= anStart(anPos+1))
            anPos = anPos +1;
            learnrate = anSched(anPos);
        end
    end
    
    for ex=1:numBatches
        
        if (mod(ex, 1000) == 0)
           fprintf('ex = %d\n', ex); 
        end
        
        data = trainSet(:,((ex-1)*batchSize+1):(ex*batchSize));
        
        
        %%%%%%% run data through network %%%%%%%%%%%%%%%%%%%%%
        hidact = vishid*data ;% + repmat(hidbiases,1,batchSize);
        output = vishid'*hidact ;% + repmat(visbiases,1,batchSize);
        neghidact = vishid*output;
        %hidprobs = hidact;
        
        
        %%%%%%% calc error and update scores %%%%%%%%%%%%%%%%%%
%         delta = targetAct - hidact;
        delta = data - output;
        error = sum(sum(delta.^2));
        errsum = errsum + error;
        
        vishidinc = momentum*vishidinc + ...
            learnrate*(hidact*data' - neghidact*output' - weightdecay*vishid)./batchSize;
        
        vishid = vishid + vishidinc;
        
    end
    
    % Output Statistics
    fprintf('Epoch   %d\t Error %f\t W-Norm %f\t Time %f\n', ...
            epoch, errsum, norm(vishid(:)), toc);
    plotrf(vishid', floor((size(trainSet,1))^.5), 'temp');
    
    
end

nextLayerData = vishid*trainSet;


end