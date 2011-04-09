%function [vishid, visbiases, hidbiases] = bpRBM(trainSet, hiddenLayerSize)

%set initial params
%%%%%%%% learning params %%%%%%%%%%%%%%%%
inputSize = size(trainSet,1);
learnrate = 0.01;
momentum = 0.9;
weightdecay = 0.001;

%%%%%%%%% algorithm params %%%%%%%%%%%%%
numepochs = 50;
batchSize = 50;
numBatches = floor(size(trainSet,2)/batchSize);

vishidinc = 0;
visbiasinc = 0;
hidbiasinc = 0;

%%%%%%%%% initialize weights %%%%%%%%%%%%%%
vishid = 0.01*randn(hiddenLayerSize, inputSize);
visbiases = zeros(inputSize,1);
hidbiases = zeros(hiddenLayerSize,1);

%%%%%%%%%% maybe make trainSet between 0 and 1??? %%%%%
% trainSet = trainSet - min(min(trainSet));
% trainSet = trainSet./max(max(trainSet));


for epoch=1:numepochs
   
    errsum = 0;
    tic;
    
    %%%%%%%% simulated annealing with momentum and learning rate %%%%%%%%
    if epoch>5,
        momentum = 0.9;
    else
        momentum = 0.5;
    end;
    
    if epoch > 25
        learnrate = 0.001;
    elseif epoch > 100
        learnrate = 0.005;
    end
    
    
    for ex=1:numBatches
        
        if (mod(ex, 1000) == 0)
           fprintf('ex = %d\n', ex); 
        end
        
        data = trainSet(:,((ex-1)*batchSize+1):(ex*batchSize));
        
        
        %%%%%%% run data through network %%%%%%%%%%%%%%%%%%%%%
        hidact = vishid*data ;% + repmat(hidbiases,1,batchSize);
%         output = vishid'*hidact ;% + repmat(visbiases,1,batchSize);
        %hidprobs = hidact;
        
        %%% old code for stochastic update %%%
        hidact  = 1./(1 + exp( -hidact ));
        %hidstates = hidprobs > rand(size(hidprobs));
        output = vishid'*hidact ;% + repmat(visbiases,1,batchSize);
        
        
        %%%%%%% calc error and update scores %%%%%%%%%%%%%%%%%%
        delta = data - output;
        error = sum(sum(delta.^2));
        errsum = errsum + error;
        
        vishidinc = momentum*vishidinc + ...
            learnrate*(hidact*(delta') - weightdecay*vishid);
        
        vishid = vishid + vishidinc;
        
        %%%%% normalize weights? hmm... %%%%%%%%%%%%%%%
        %vishid = vishid./(max(max(vishid)));
        
    end
    
    % Output Statistics
    fprintf('Epoch   %d\t Error %f\t W-Norm %f\t Time %f\n', ...
            epoch, errsum, norm(vishid(:)), toc);
    plotrf(vishid', floor(inputSize^.5), 'temp');
    
    
end







%end