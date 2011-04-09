%% load mnist data %%%%%%%%%%%

trainSet = [];
testSet = [];
trainLabels = [];
testLabels = [];

for d = 0:9
    X = load(['mnist/digit' num2str(d) '.mat']);
    trainSet = [trainSet; X.D];
    newTL = d*ones(size(X.D,1), 1);
    trainLabels = [trainLabels; newTL];
    
    Y = load(['mnist/test' num2str(d) '.mat']);
    testSet = [testSet; Y.D];
    newTsL = d*ones(size(Y.D,1), 1);
    testLabels = [testLabels; newTsL];
end

clear X;
clear Y;
clear newTL;
clear newTsL;
%%%%%%%% randomly permute the data %%%%%%%%%%%%%%%%%%

trainOrder = randperm(size(trainSet,1));
testOrder = randperm(size(testSet,1));

trainSet = trainSet(trainOrder(1:size(trainSet,1)),:);
trainLabels = trainLabels(trainOrder(1:size(trainSet,1)));
testSet = testSet(testOrder(1:size(testSet,1)),:);
testLabels = testLabels(testOrder(1:size(testSet,1)));

%% train the dbn %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dbnGreedy;
%bpDBN

%% run data through the network, compute outputs 
%%%%% assume weights have just been trained! %%%%%%%%%%

hidact = vishid*trainSet;
hid2act = hidhid2*hidact;
hid3act = hid2hid3*hid2act;

%% train svm on output data %%%%%%%%%%%%%%%%%%%%


%% train svm on normal mnist %%%%%%%%%%%%%%%%%%%%
