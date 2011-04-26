load IMAGES.mat;

%params

imsize = 20
numpatches = 10000

%initialize stuff
trainSet = zeros(imsize*imsize, numpatches);
samplePerImage = floor(numpatches/size(IMAGES,3));
sizex = size(IMAGES,2);
sizey = size(IMAGES,1);
sampleNum = 1;


for im = 1:size(IMAGES, 3)
    
  posx = floor( rand(1,samplePerImage) *(sizex-imsize-2))+1;
  posy = floor( rand(1,samplePerImage) *(sizey-imsize-1))+1;
  
  for j=1:samplePerImage
    trainSet(:,sampleNum) = reshape( IMAGES(posy(1,j):posy(1,j)+imsize-1, ...
			posx(1,j):posx(1,j)+imsize-1),[imsize^2 1]);
    sampleNum=sampleNum+1;
  end 

end

% clear all except training data
clearvars -except trainSet

hiddenLayerSize = 300