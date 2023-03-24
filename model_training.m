% training 
% if no training data store created, run training data creation script
clear;
run C:\HL\Github\matlab_maskrcnn\create_training_dataset.m
%%
classNames = {'cell'};
numClasses = numel(classNames);
% Add a background class
classNames = [classNames {'background'}];
%{ 
%%
AnchorBoxes = [[32 16];
    [64 32];
    [128 64];    
    [32 32];
    [64 64];
    [128 128];   
    [16 32];
    [32 64];
    [64 128];
   ];
%% 
% detector_df = maskrcnn("resnet50-coco");

%% load ? 
detector = maskrcnn("resnet50-coco",classNames,...
    'InputSize', [128 128 3]);
 
%% train using pre-trained network

%% 
options = trainingOptions("sgdm", ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropFactor=0.2, ...
    LearnRateDropPeriod=5, ...
    MaxEpochs=1000, ...
    MiniBatchSize=4, ...
    BatchNormalizationStatistics ='moving', ...
    ResetInputNormalization = false); % , ...    Plots="training-progress"
%%

trainedDetector = trainMaskRCNN(ds, detector, options);
%}

%% 
%{%
%% 
%%
params = createMaskRCNNConfig(imageSize, numClasses, classNames);
disp(params);
%% 
dlnet = createMaskRCNN(numClasses, params, 'resnet101');
%%
if canUseGPU
    executionEnvironment = "gpu";
else
    executionEnvironment = "cpu";
end
executionEnvironment = "cpu";
%% SGDM learning parameters
initialLearnRate = 0.01;
momemtum = 0.9;
decay = 0.0001;
velocity = [];
maxEpochs = 30;

minibatchSize = 2;
%%
% Configure the data dispatcher

% Create the batching function. The images are concatenated along the 4th
% dimension to get a HxWxCxminiBatchSize shaped batch. The other ground truth data is
% configured a cell array of length = minibatchSize.

myMiniBatchFcn = @(img, boxes, labels, masks) deal(cat(4, img{:}), boxes, labels, masks);

mb = minibatchqueue(ds, 4, "MiniBatchFormat", ["SSCB", "", "", ""],...
                            "MiniBatchSize", minibatchSize,...
                            "OutputCast", ["single","","",""],...
                            "OutputAsDlArray", [true, false, false, false],...
                            "MiniBatchFcn", myMiniBatchFcn,...
                            "OutputEnvironment", [executionEnvironment,"cpu","cpu","cpu"]);
%%
numEpoch = 1;
numIteration = 1; 

start = tic;
doTraining = true;
if doTraining
    
     % Create subplots for the learning rate and mini-batch loss.
    fig = figure;
    [lossPlotter] = helper.configureTrainingProgressPlotter(fig);
    
    % Initialize verbose output
    helper.initializeVerboseOutput([]);
    
    % Custom training loop.
    while numEpoch < maxEpochs
    mb.reset();
    mb.shuffle();
    
        while mb.hasdata()
            % get next batch from minibatchqueue
            [X, gtBox, gtClass, gtMask] = mb.next();
        
            % Evaluate the model gradients and loss using dlfeval
            [gradients, loss, state] = dlfeval(@networkGradients, X, gtBox, gtClass, gtMask, dlnet, params);
            dlnet.State = state;
            
            % compute the learning rate for current iteration
            learnRate = initialLearnRate/(1 + decay*numIteration);
            
            if(~isempty(gradients) && ~isempty(loss))
    
                [dlnet.Learnables, velocity] = sgdmupdate(dlnet.Learnables, gradients, velocity, learnRate, momemtum);
            else
                continue;
            end
            helper.displayVerboseOutputEveryEpoch(start,learnRate,numEpoch,numIteration,loss);
                
            % Plot loss/ accuracy metric
             D = duration(0,0,toc(start),'Format','hh:mm:ss');
            addpoints(lossPlotter,numIteration,double(gather(extractdata(loss))))
            
            subplot(2,1,2)

            title("Epoch: " + numEpoch + ", Elapsed: " + string(D))
           
            drawnow
            
            numIteration = numIteration + 1;
    
        end
    numEpoch = numEpoch + 1;
    
    end
end
%%
%}
%% 
net = dlnet;

%% save model
save('C:\HL\Github\matlab_maskrcnn\model.mat', 'net');
%% use model to do detection/inference

%%
pretrained = load('C:\HL\Github\matlab_maskrcnn\model.mat');
net = pretrained.net;

% Extract Mask segmentation sub-network
maskSubnet = helper.extractMaskNetwork(net);
%%
% Define the target size of the image for inference
targetSize = [512 512 3]; % imageSize
test_image_fn = 'C:\HL\Github\matlab_maskrcnn\test_images\syth_image_001.tiff';
img = imread(test_image_fn);
% Resize the image maintaining the aspect ratio and scaling the largest
% dimension to the target size.
imgSize = size(img);
[~, maxDim] = max(imgSize);
resizeSize = [NaN NaN]; 
resizeSize(maxDim) = targetSize(maxDim);

img = imresize(img, resizeSize);
% img = repmat(img, [1 1 3]);
cmap = jet(2^16);
img = ind2rgb(img, cmap);

% detect the objects and their masks
[boxes, scores, labels, masks] = detectMaskRCNN(net, maskSubnet, img, params, executionEnvironment);


%% Visualize Predictions

% Overlay the detected masks on the image using the insertObjectMask
% function.
if(isempty(masks))
    overlayedImage = img;
else
    overlayedImage = insertObjectMask(img, masks);
end
figure, imshow(overlayedImage)

% Show the bounding boxes and labels on the objects
showShape("rectangle", gather(boxes), "Label", labels, "LineColor",'r')