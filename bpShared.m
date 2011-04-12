%function [vishid, visbiases, hidbiases] = bpRBM(trainSet, hiddenLayerSize)

%% set initial params
%%%%%%%% learning params %%%%%%%%%%%%%%%%
inputSize = size(trainSet,1);
momentum = 1;
weightdecay = 0.001;
targetActivation = 0.01;
beta = 3; %sparsity term

%%%%% learn rate stuff %%%%%%%%%%%

anSched = [0.05 0.05 0.05 0.05 0.05];
anStart = [1 5 10 30 60];
anPos = 1;
learnrate = anSched(1);

%%%%%%%%% algorithm params %%%%%%%%%%%%%
numepochs = 100;
numEx = size(trainSet,2);
batchSize = numEx;
numBatches = floor(numEx/batchSize);


%%%%%%%%% initialize weights %%%%%%%%%%%%%%
% we'll choose weights uniformly from the interval [-r, r]
r  = sqrt(6) / sqrt(hiddenLayerSize + inputSize +1);
vishid = rand(hiddenLayerSize, inputSize) * 2 * r - r;
% vishid = 0.1*randn(hiddenLayerSize, inputSize);
% vishid = vishidback;

outbiases = zeros(inputSize,1);
hidbiases = zeros(hiddenLayerSize,1);

%%%%%%%%%% maybe make trainSet between 0 and 1??? %%%%%
% trainSet = trainSet - min(min(trainSet));
% trainSet = trainSet./max(max(trainSet));

%% train rbm
for epoch=1:numepochs
   
    errsum = 0;
    tic;
    
%     %%%%%%%% simulated annealing with learning rate %%%%%%%%
    if (anPos < length(anStart))
        if (epoch >= anStart(anPos+1))
            anPos = anPos +1;
            learnrate = anSched(anPos);
        end
    end
    
 
    vishidinc = 0;
    outbiasinc = 0;
    hidbiasinc = 0;
    
    
    for ex=1:numBatches
        %% run batch gradient descent
        
        
        if (mod(ex, 1000) == 0)
           fprintf('ex = %d\n', ex); 
        end
        
        data = trainSet(:,((ex-1)*batchSize+1):(ex*batchSize));
        
        
        %% run data through network %%%%%%%%%%%%%%%%%%%%%
        hidact = vishid*data + repmat(hidbiases,1,batchSize);
        hidact  = 1./(1 +  exp( -hidact ));
        
        output = vishid'*hidact + repmat(outbiases,1,batchSize);
%         output = 1./(1 + exp( -output ));
        
        
      
        %% update sparsity
        % THIS ONLY WORKS IF batchSize IS THE SIZE OF THE WHOLE TRAIN SET!
        
        pHat = sum(hidact,2) ./ numEx;
        sparse_grad = ((1-targetActivation) ./ (1-pHat)) - ...
            (targetActivation ./ pHat);
        sparse_grad_toAdd = repmat(sparse_grad,1,numEx);
        
        
        %% calc error and update scores %%%%%%%%%%%%%%%%%%
        err = data - output;
        delta = -(err) ;%  .* (output.*(1-output));
        hid_delta = (vishid*delta + beta*sparse_grad_toAdd) ...
            ;%.*(hidact.*(1-hidact));
        
        error = sum(sum(err.^2));
        errsum = errsum + error;
        

        vishidinc = vishidinc + (hid_delta*data' + hidact*delta')./2;
        hidbiasinc = hidbiasinc + sum(hid_delta,2);
        outbiasinc = outbiasinc + sum(delta,2);
        

    end

    
    %% update weights
    vishid = vishid - learnrate*(vishidinc/numEx ...
            + weightdecay*vishid);
    hidbiases = hidbiases - learnrate*hidbiasinc/numEx;
    outbiases = outbiases - learnrate*outbiasinc/numEx;
        
    
    %% Output Statistics
    fprintf('Epoch   %d\t Error %f\t Avg Sparsity  %f\t W-Norm %f\t Time %f\n', ...
            epoch, errsum, mean(pHat), norm(vishid(:)), toc);
%     
%     toplot = vishid - min(min(vishid));
%     toplot = toplot./(max(max(toplot)));
%     toplot = 2*toplot - 1;
    plotrf(vishid', floor(inputSize^.5), 'temp');
    
    errsum = 0;
    
    
end







%end