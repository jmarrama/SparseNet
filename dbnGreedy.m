%% set intial stuff
do_all_layer_BP = 0;


%% train DBN layer by layer %%%%%%%%%%%%%%%%%%%%%%%%
[vishid, nld] = bpRBMf(trainSet, 500);
[hidhid2, nld2] = bpRBMf(nld, 300);
[hid2hid3, nld3] = bpRBMf(nld2, 200);

%% run data through, compute error %%%%%%%%%%%%%%%%%%%%
hidact = vishid*trainSet;
hid2act = hidhid2*hidact;
hid3act = hid2hid3*hid2act;
hid2out = hid2hid3'*hid3act;
hidout = hidhid2'*hid2out;
output = vishid'*hidout;

delta = trainSet - output;
error = sum(sum(delta.^2));
fprintf('overall error after layerwise training: %f\n',error);
plotrf(vishid', floor((size(trainSet,1))^.5), 'temp');


%% do backprop on all layers (if set to do so) %%%%%%%%%%%%%%%%%%%%%%%
if(do_all_layer_BP)

%%%%%%%%%% initialize stuff %%%%%%%%%%%%%%%%
anSched = [0.05 0.05 0.01 0.005 0.001];
anStart = [1 5 10 50 100];
anPos = 1;
learnrate = 0.1;
momentum = 0.9;
weightdecay = 0.001;

%%%%%%%%% algorithm params %%%%%%%%%%%%%
numepochs = 50;
batchSize = 10;
numBatches = floor(size(trainSet,2)/batchSize);

vishidinc = 0;
hidhid2inc = 0;
hid2hid3inc = 0;


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
        
        if (mod(ex, 50) == 0)
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
        
        %%% old code for stochastic update %%%
        %hidprobs  = 1./(1 + exp( -hidact ));
        %hidstates = hidprobs > rand(size(hidprobs));
        %output = vishid'*hidstates ;% + repmat(visbiases,1,batchSize);
        
        
        %%%%%%% calc error and update scores %%%%%%%%%%%%%%%%%%
        delta = data - output;
        deltaHid = (vishid*delta);
        deltaHid2 = (hidhid2*deltaHid);
        
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
    plotrf(vishid', floor((size(trainSet,1))^.5), 'temp');
    
    
end

end

%% run input through one final time %%%%%%%%%%%%%%%%%%%%%%%%%%%%

% hidact = vishid*trainSet;
% hid2act = hidhid2*hidact;
% hid3act = hid2hid3*hid2act;