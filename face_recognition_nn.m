clear;
addpath('simpleNN')

load("data/face/YaleB_32x32.mat");
fea = fea';
gnd = gnd';
% fea stores face images in 1024 x 2414 vector
% gnd stores labels in 1 x 2414 vector

num_images = size(gnd, 2);      % number of images = 2414
rand_list = randperm(num_images, 100);    % list of 100 indices choosen randomly to be used as testing data
rand_list = sort(rand_list, 'descend');

train_data = fea; test_data = [];
train_label = gnd; test_label = [];

for i = 1:100
    train_data(:, rand_list(i)) = [];
    train_label(:, rand_list(i)) = [];
    test_data = [fea(:, i) test_data];
    test_label = [gnd(:, i) test_label];
end

num_images = num_images - 100;

m = mean(train_data')';
train_data = train_data - m;

cv = (train_data * train_data') / (num_images - 1);   % covariance matrix
[u, s, v] = svd(cv);        % singular value decomposition. u contains eigenvectors and s is a diagonal matrix of eigenvalues

k = 50;
vk = u(:, 4:k+3);             % First k eigenvectors

proj = vk' * train_data;    % Projection of training data on eigenvectors

%% Setting up

% using the default options
nnOptions = {};

%% Alternative options
% nnOptions = {'lambda', 0.1,...
%             'maxIter', 50,...
%             'hiddenLayers', [40 20],...
%             'activationFn', 'tanh',...
%             'validPercent', 30,...
%             'doNormalize', 1};

%% Learning
modelNN = learnNN(proj', train_label', nnOptions);
% plotting the confusion matrix for the validation set
figure(1); cla(gca);
plotConfMat(modelNN.confusion_valid);
%%

% Now testing the remaining images
proj2 = vk' * (test_data - m);          % Projection of test data on eigenvectors
correct = 0;

for i = 1:100
    p = predictNN(proj2(:, i)', modelNN);
    if p == test_label(i)
        correct = correct + 1;
    end
end

fprintf('The accuracy of the test set is %d %% \n', correct);
