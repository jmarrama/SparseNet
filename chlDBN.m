%function [vishid, nextLayerData] = chlRBM(trainSet, hiddenLayerSize)

%set initial params
%%%%%%%% learning params %%%%%%%%%%%%%%%%
inputSize = size(trainSet,1);
hidSize = 1000;
hid2Size = 500;
hid3Size = 300;

anSched = [0.01 0.01 0.005 0.001];
anStart = [1 5 40 80 100];
anPos = 1;
learnrate = 0.01;
momentum = 0.9;
weightdecay = 0.0;

%%%%%%%%% algorithm params %%%%%%%%%%%%%
numepochs = 60;
batchSize = 10;
numBatches = floor(size(trainSet,2)/batchSize);

vishidinc = 0;
hidhid2inc = 0;
hid2hid3inc = 0;

%visbiasinc = 0;
%hidbiasinc = 0;

%%%%%%%%% initialize weights %%%%%%%%%%%%%%
vishid = 0.01*randn(hidSize, inputSize);
hidhid2 = 0.01*randn(hid2Size, hidSize);
hid2hid3 = 0.01*randn(hid3Size, hid2Size);
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
        hidact = vishid*data;
        hid2act = hidhid2*hidact;
        hid3act = hid2hid3*hid2act;
        hid2out = hid2hid3'*hid3act;
        hidout = hidhid2'*hid2out;
        output = vishid'*hidout;
        neghidact = vishid*output;
        neghid2act = hidhid2*neghidact;
        neghid3act = hid2hid3*neghid2act;
        
        
        %%%%%%% calc error and update scores %%%%%%%%%%%%%%%%%%
        delta = data - output;
        error = sum(sum(delta.^2));
        errsum = errsum + error;
        
        vishidinc = momentum*vishidinc + ...
            learnrate*(hidact*data' - neghidact*output' - weightdecay*vishid);
        hidhid2inc = momentum*hidhid2inc + ...
            learnrate*(hid2act*hidact' - neghid2act*neghidact' - weightdecay*hidhid2);
        hid2hid3inc = momentum*hid2hid3inc + ...
            learnrate*(hid3act*hid2act' - neghid3act*neghid2act' - weightdecay*hid2hid3);
        
        
        vishid = vishid + vishidinc;
        hidhid2 = hidhid2 + hidhid2inc;
        hid2hid3 = hid2hid3 + hid2hid3inc;

    end
    
    % Output Statistics
    fprintf('Epoch   %d\t Error %f\t W-Norm %f\t Time %f\n', ...
            epoch, errsum, norm(vishid(:)), toc);
    plotrf(vishid', floor((size(trainSet,1))^.5), 'temp');
    
    
end

nextLayerData = vishid*trainSet;


%end