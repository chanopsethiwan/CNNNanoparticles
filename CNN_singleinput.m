%% Section1: Clear workspace + Add path to 4D Array Gen function
clear all; %#ok<CLALL>
clc;
close all;
addpath('C:\Users\micha\Documents\University\Year 4 (CET IIB)\Research project\rMATLAB coding + library\5. CNN codes');
%% Section2: Import Transformed Images from Output Folder + Make into 4D-Array

selected_training_data = load('C:\Users\micha\Documents\University\Year 4 (CET IIB)\Research project\rMATLAB coding + library\3. Single input image datastores\Stdev\NaBH4 mole fraction\selected_training_data.mat');
selected_validation_data = load('C:\Users\micha\Documents\University\Year 4 (CET IIB)\Research project\rMATLAB coding + library\3. Single input image datastores\Stdev\NaBH4 mole fraction\selected_validation_data.mat');
for i = 1:size(selected_training_data.filename)
    FilePathsTrain.name(i) = strcat('C:\Users\micha\Documents\University\Year 4 (CET IIB)\Research project\rMATLAB coding + library\2. Grayscale - transformed CFD images\Helical geometry\NaBH4 mole fraction\Outputs_Grayscale_Labelled_Images_STdevs\trimmed_folder','\',selected_training_data.filename(i));
end
imds_Train = imageDatastore(FilePathsTrain.name);
[XTrain, ~] = imds2array_gray(imds_Train);
% X - 4-D Arrays of # sample Images: Input data as an H-by-W-by-C-by-N array
% ~ - Categorical vector containing the labels for each observation.

for i = 1:size(selected_validation_data.filename)
    FilePathsValid.name(i) = strcat('C:\Users\micha\Documents\University\Year 4 (CET IIB)\Research project\rMATLAB coding + library\2. Grayscale - transformed CFD images\Helical geometry\NaBH4 mole fraction\Outputs_Grayscale_Labelled_Images_STdevs\trimmed_folder','\',selected_validation_data.filename(i));
end
imds_Valid = imageDatastore(FilePathsValid.name);
[XVali, ~] = imds2array_gray(imds_Valid);
    
%% Section3: Import Particle Data
%TrainingData = load('training set.mat');       %if vars already in workspace
%ValidationData = load('validation set.mat');   %if vars already in workspace
YTrain      = selected_training_data.particle_size;     %no rounding %note that it says size but its st.dev
YVali = selected_validation_data.particle_size;   %no rounding %note that it says size but its st.dev
XValidation = XVali(:,:,:,1:1000);
YValidation = YVali(1:1000);
XTest = XVali(:,:,:,1001:1350);
YTest = YVali(1001:1350);

% Y is a single column of correct answer sets containing angle of rotation for each image

%% Section4: Define Architecture Layers for CNN
layers = [
    imageInputLayer([100 100 1])
    %new grayscale reformatting code by Bruno produces images of H 500 W 500
    %first layer defines the size and type of the input data. The input images are H-by-W-by-C. (C for channel size: 1 for Grayscale, 3 for RGB)
    
    convolution2dLayer(10,8,'Padding','same')
    %%convolution2dLayer(a,b,'Padding','same')
    % creates a 2-D conv layer with b filters of size [a a] and 'same' padding
    
    batchNormalizationLayer
    % normalizes each input channel across a mini-batch
    % to speed up training of CNN + reduce the sensitivity to initialization
    
    reluLayer
    % performs a threshold operation to each element of the input
    % any value less than zero is set to zero
    
    averagePooling2dLayer(10,'Stride',10)
    %%averagePooling2dLayer(2,'Stride',2)
    % creates an average pooling layer with pool size [2 2] and stride [2 2]
    
    convolution2dLayer(10,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    averagePooling2dLayer(2,'Stride',2)
    convolution2dLayer(10,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(10,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.2)
    % randomly sets input elements to zero with a given probability
    % changes the underlying architecture between iterations + prevents overfitting
    
    fullyConnectedLayer(1)
    % FC layer defines the size and type of output data
    % multiplies the input by a weight matrix W and adds a bias vector b
    
    regressionLayer];
    % Regression layer essentially provides predictive ability

%% Set Up Training Parameters
miniBatchSize  = 10;
% batch size represents the # of subset data used before solver updates parameters

validationFrequency = floor(numel(YTrain)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize,'Momentum',0.91, ...
    'MaxEpochs',8, 'L2Regularization',0.0011,...
    'InitialLearnRate',1e-4, 'ValidationData',{XValidation,YValidation},...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',20, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false); % Option to display training progress in cmd window

net = trainNetwork(XTrain,YTrain,layers,options); % Executing the training
disp('CNN Training Completed')


YPredicted = predict(net,XTest);
% Define error
predictionError = YTest - YPredicted;
squares = predictionError.^2;
rmse = sqrt(mean(squares)) 



