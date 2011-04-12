%% Parameters
inputSize = size(trainSet,1);
hiddenLayerSize = 500;

weightcost = 0.001;
epsilon = 0.01;

activationAveragingConstant = 0.99;
targetActivation = 0.1;
learningRateSparsity = 3;          % <- Set to 0 to disable target sparsity

sigma = 1; % For Gaussian-Binary RBM

%%%%%%%%% algorithm params %%%%%%%%%%%%%
numepochs = 100;
numEx = size(trainSet,2);
batchSize = 100;
numBatches = floor(numEx/batchSize);

%% Init
vishidinc = 0;
visbiasinc = 0;
hidbiasinc = 0;

errsum = 0;
reconerr = 0;
momentum = 0;
averageActivations = 0;

vishid = 0.05 * randn(hiddenLayerSize, inputSize);
visbiases = zeros(inputSize, 1);
hidbiases = zeros(hiddenLayerSize, 1);

scl = 1/(sigma^2);

% Put into batches of 100
% data = reshape(data, size(data, 1), 100, numel(data)/(100*size(data,1)));
% batchSize = size(data, 2);

%% Run Sparse RBM
for epoch = 1:numepochs

    errsum = 0;
    tic;
    if epoch>5,
        momentum = 0.9;
    else
        momentum = 0.5;
    end;

    for ex=1:numBatches

        % Pick out some data
        X = trainSet(:,((ex-1)*batchSize+1):(ex*batchSize));

        % Positive Phase
        poshidprobs  = 1./(1 + exp( scl*(  -vishid*X - repmat(hidbiases, 1, batchSize)))  );
        posprods     = poshidprobs * X';
        poshidact    = sum(poshidprobs, 2);
        posvisact    = sum(X, 2);
        poshidstates = poshidprobs > rand(size(poshidprobs));

        % Negative Phase
        % Following is a gaussian-binary rbm.
        % If you want binary-binary, do a sigmoid on negdata
        negdata     = (vishid'*poshidstates) + repmat(visbiases,1, batchSize);
        neghidprobs = 1./(1 + exp( scl*(  -vishid*negdata - repmat(hidbiases,1, batchSize)))  );
        negprods    = neghidprobs * negdata';
        neghidact   = sum(neghidprobs, 2);
        negvisact   = sum(negdata, 2);

        % Stochastic Reconstruction Error
        err = mean(sum( (X-negdata).^2 ));
        errsum = err + errsum;

        % Sparsity
        averageActivations = (1 - activationAveragingConstant) * averageActivations + ...
                             activationAveragingConstant * ...
                             mean(poshidprobs, 2);

        % Update Sparsity
        sparsityError  = averageActivations - targetActivation;
        sparsityinc    = - sparsityError * learningRateSparsity * epsilon;

        % Update Steps with Momentum
        vishidinc = momentum*vishidinc + ...
                    epsilon*( (posprods-negprods)/batchSize - weightcost*vishid);

        visbiasinc = momentum*visbiasinc + (epsilon/batchSize)*(posvisact-negvisact);
        hidbiasinc = momentum*hidbiasinc + sparsityinc + (epsilon/batchSize)*(poshidact-neghidact);

        % Actual Updates
        vishid = vishid + vishidinc;
        visbiases = visbiases + visbiasinc;
        hidbiases = hidbiases + hidbiasinc;

    end
    
    % Output Statistics
    fprintf('Epoch   %d\tError %f\tAverage Sparsity %f\tW-Norm %f\tTime %f\n', ...
            epoch, errsum, mean(averageActivations), norm(vishid(:)), toc);
    
    % Visualize using plotrf (from niscode)
%     plotrf(vishid', 15, []);
    plotrf(vishid', floor(inputSize^.5), []);
    
end

%% Save vishid, visbiases, hidbiases ...

