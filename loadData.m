load novar20.mat
tsize = 40000
rp = randperm(size(X, 2));
trainSet = X(:,rp(1:tsize));
hiddenLayerSize = 500;