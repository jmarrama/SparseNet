%% Start Script
%function [nextLayerData, nextLayerTestData, vishid, visbiases, hidbiases] = sparseRBM(trainSet, testSet)
function [nextLayerData, vishid, visbiases, hidbiases] = sparseRBM(trainSet, hiddenLayerSize)
    
%normalize training set
minval = min(min(trainSet));
maxval = max(max(trainSet));
if(minval < 0 || maxval > 1)
    data = (trainSet - minval)/(maxval-minval);
else
    data = trainSet;
end

inputSize = size(data, 2);
numcases = size(data, 1);
%hiddenLayerSize = 20; %need to find a value that doesn't overload memory

weightcost = 0.001;
epsilon = 0.01;

activationAveragingConstant = 0.99;
targetActivation = 0.1;
learningRateSparsity = 3;          % <- Set to 0 to disable target sparsity

sigma = 1; % For Gaussian-Binary RBM

vishidinc = 0;
visbiasinc = 0;
hidbiasinc = 0;

errsum = 0;
reconerr = 0;
momentum = 0;
averageActivations = 0;

%vishid = 0.05 * randn(inputSize, hiddenLayerSize);
vishid = 0.001 * randn(inputSize, hiddenLayerSize);
visbiases = zeros(1, inputSize);
hidbiases = zeros(1, hiddenLayerSize);

scl = 1/(sigma^2);

%end initialization
fprintf('finished initilization');

%% Run Sparse RBM
for epoch = 1:100

    errsum = 0;
    tic;
    if epoch>5,
        momentum = 0.9;
    else
        momentum = 0.5;
    end;


    % Positive Phase
    poshidprobs  = 1./(1 + exp( scl*(  -data*vishid - repmat(hidbiases, numcases, 1)))  );
    posprods     = data' * poshidprobs;
    poshidact    = sum(poshidprobs, 1);
    posvisact    = sum(data, 1);
    poshidstates = poshidprobs > rand(size(poshidprobs));
    %clear poshidprobs;

    % Negative Phase
    % Following is a gaussian-binary rbm.
    negdata     = (poshidstates * vishid') + repmat(visbiases, numcases, 1);
    %clear poshidstates;
    neghidprobs = 1./(1 + exp( scl*(  -negdata*vishid - repmat(hidbiases, numcases, 1)))  );
    negprods    = negdata' * neghidprobs;
    neghidact   = sum(neghidprobs, 1);
    negvisact   = sum(negdata, 1);
    %clear neghidprobs;

    %Reconstruction Error
    err = mean(sum( (data-negdata).^2 ));
    errsum = err + errsum;

    % Sparsity
    averageActivations = (1 - activationAveragingConstant) * averageActivations + ...
                             activationAveragingConstant * ...
                             mean(poshidprobs, 1);

    % Update Sparsity
    sparsityError  = averageActivations - targetActivation;
    sparsityinc    = - sparsityError * learningRateSparsity * epsilon;

    % Update Steps with Momentum
    vishidinc = momentum*vishidinc + ...
                    epsilon*( (posprods-negprods)/numcases - weightcost*vishid);

    visbiasinc = momentum*visbiasinc + (epsilon/numcases)*(posvisact-negvisact);
    hidbiasinc = momentum*hidbiasinc + sparsityinc + (epsilon/numcases)*(poshidact-neghidact);

    % Actual Updates
    vishid = vishid + vishidinc;
    visbiases = visbiases + visbiasinc;
    hidbiases = hidbiases + hidbiasinc;

    
    
    % Output Statistics
    fprintf('Epoch   %d\tError %f\tAverage Sparsity %f\tW-Norm %f\tTime %f\n', ...
            epoch, errsum, mean(averageActivations), norm(vishid(:)), toc);
    
    % Visualize using plotrf (from niscode)
    %plotrf(vishid', 15, []);
    
end

%% generate next layer data:

nextLayerData =1./(1 + exp( -data*vishid - repmat(hidbiases, numcases, 1)));
%minT = min(min(testSet));
%maxT = max(max(testSet));
%nextLayerTestData = ((testSet - minT)/maxT)*vishid + repmat(hidbiases, size(testSet, 1), 1);

%clear weights?
%clear vishid;


end