clear
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

coeff = zeros(k, 38);       % Matrix to store coefficients or weights
tot_images = zeros(38);     % Array to store how many times each image appears in training data

for i = 1:num_images
    ind = train_label(i);
    tot_images(ind) = tot_images(ind) + 1;
    coeff(:, ind) = coeff(:, ind) + proj(:, i);
end

for i = 1:38
    coeff(:, i) = coeff(:, i) / tot_images(i);
end

% Now testing the remaining images
proj2 = vk' * (test_data - m);
correct = 0;

for i = 1:100
    df = coeff - proj2(:, i);   % taking difference
    df = df .* df;              % difference squared
    on = ones(1, k);
    [val, ans] = min(on * df);  % sexy way to find sum of elements of df
    if ans == test_label(i)
        correct = correct + 1;
    end
end

fprintf('The accuracy of the test set is %d %% \n', correct);
