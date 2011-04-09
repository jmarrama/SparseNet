%function [vishid, nextLayerData] = genrecRBM(trainSet, hiddenLayerSize)

%set initial params
%%%%%%%% learning params %%%%%%%%%%%%%%%%
inputSize = size(trainSet,1);
learnrate = 0.01;
momentum = 0.9;
weightdecay = 0.0;

anSched = [0.01 0.01 0.005 0.001];
anStart = [1 5 40 135];
anPos = 1;

%%%%%%%%% algorithm params %%%%%%%%%%%%%
numepochs = 60;
batchSize = 10;
numBatches = floor(size(trainSet,2)/batchSize);

vishidinc = 0;
%visbiasinc = 0;
%hidbiasinc = 0;

%%%%%%%%% initialize weights %%%%%%%%%%%%%%
vishid = 0.01*randn(hiddenLayerSize, inputSize);
%visbiases = zeros(inputSize,1);
%hidbiases = zeros(hiddenLayerSize,1);


for epoch=1:numepochs
   
    errsum = 0;
    tic;
    
    %%%%%%%% simulated annealing with momentum and learning rate %%%%%%%%
    if epoch>5,
        momentum = 0.9;
    else
        momentum = 0.5;
    end;
    
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
        delta = data - output;
        error = sum(sum(delta.^2));
        errsum = errsum + error;
        
        vishidinc = momentum*vishidinc + ...
            learnrate*((hidact - neghidact)*output' - weightdecay*vishid);
        
        vishid = vishid + vishidinc;
        
    end
    
    % Output Statistics
    fprintf('Epoch   %d\t Error %f\t W-Norm %f\t Time %f\n', ...
            epoch, errsum, norm(vishid(:)), toc);
    plotrf(vishid', floor((size(trainSet,1))^.5), 'temp');
    
    
end

nextLayerData = vishid*trainSet;


%end