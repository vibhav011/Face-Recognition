clear;
S = load("data/mnist.mat");
dtrn = S.digits_train;
ltrn = S.labels_train;
dtest = S.digits_test;
ltest = S.labels_test;

dtest = double(reshape(dtest, 784, 10000));
dtrn = double(reshape(dtrn, 784, 60000));
m = mean(dtrn')';
dnew = dtrn - m;

cv = (dnew * dnew') / 59999;
[u, s, v] = svd(cv);

k = 50;

vk = u(:, 1:k);
proj = vk' * dnew;

coeff = zeros(k, 10);

for i = 1:60000
    if ltrn(i, 1) == 0
        coeff(:, 10) = coeff(:, 10) + proj(:, i);
    else
        coeff(:, ltrn(i, 1)) = coeff(:, ltrn(i, 1)) + proj(:, i);
    end
end

coeff = coeff / 6000;

proj2 = vk' * (dtest - m);

correct = 0;

for i = 1:10000
    df = coeff - proj2(:, i);
    df = df .* df;
    on = ones(1, k);
    [val, ans] = min(on * df);
    if ans == ltest(i, 1)
        correct = correct + 1;
    end
end

disp(correct);
    





