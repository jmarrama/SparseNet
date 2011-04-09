%set initial params
%%%%%%%% learning params %%%%%%%%%%%%%%%%
inputSize = size(trainSet,1);
hidSize = 500;
hid2Size = 300;
hid3Size = 200;

anSched = [0.01 0.05 0.01 0.005 0.001];
anStart = [1 5 10 20 30];
anPos = 1;
learnrate = 0.01;
momentum = 0.9;
weightdecay = 0.001;

%%%%%%%%% algorithm params %%%%%%%%%%%%%
numepochs = 50;
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

%%%%%%%%%% maybe make trainSet between 0 and 1??? %%%%%
trainSet = trainSet - min(min(trainSet));
trainSet = trainSet./max(max(trainSet));


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
        
        if (mod(ex, 500) == 0)
           fprintf('ex = %d\n', ex); 
        end
        
        data = trainSet(:,((ex-1)*batchSize+1):(ex*batchSize));
        
        
        %%%%%%% run data through network %%%%%%%%%%%%%%%%%%%%%
        hidact = vishid*data;
        hidact  = 1./(1 + exp( -hidact ));
        
        hid2act = hidhid2*hidact;
        hid2act  = 1./(1 + exp( -hid2act ));
        
        hid3act = hid2hid3*hid2act;
        hid3act  = 1./(1 + exp( -hid3act ));

        hid2out = hid2hid3'*hid3act;
        hid2out = 1./(1 + exp( -hid2out ));
        
        hidout = hidhid2'*hid2out;
        hidout = 1./(1 + exp( -hidout ));
        
        output = vishid'*hidout;
        
        %%% old code for stochastic update %%%
        %hidact  = 1./(1 + exp( -hidact ));
        %hidstates = hidprobs > rand(size(hidprobs));
        %output = vishid'*hidstates ;% + repmat(visbiases,1,batchSize);
        
        
        %%%%%%% calc error and update scores %%%%%%%%%%%%%%%%%%
        delta = data - output;
        
        deltaHid = (vishid*delta);
        deltaHid = 1./(1 + exp( -deltaHid));
        
        deltaHid2 = (hidhid2*deltaHid);
        deltaHid2 = 1./(1 + exp( -deltaHid2));
        
        
        error = sum(sum(delta.^2));
        errsum = errsum + error;
        
        %todo - hidout or hidact?
        vishidinc = momentum*vishidinc + ...
            learnrate*(hidout*(delta') - weightdecay*vishid);
        hidhid2inc = momentum*hidhid2inc + ...
            learnrate*(hid2out*(deltaHid') - weightdecay*hidhid2);
        hid2hid3inc = momentum*hid2hid3inc + ...
            learnrate*(hid3act*(deltaHid2') - weightdecay*hid2hid3);
        
        
        vishid = vishid + vishidinc;
        hidhid2 = hidhid2 + hidhid2inc;
        hid2hid3 = hid2hid3 + hid2hid3inc;
        
    end
    
    % Output Statistics
    fprintf('Epoch   %d\t Error %f\t W-Norm %f\t Time %f\n', ...
            epoch, errsum, norm(vishid(:)), toc);
    plotrf(vishid', floor(inputSize^.5), 'temp');
    
    
end