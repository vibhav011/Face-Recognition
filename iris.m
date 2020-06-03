format short
file = fopen("data/iris.data", "r");
data_ar = split(fscanf(file,"%c"), newline);
data_ar = data_ar(1:150);
data_mat = split(data_ar, ",");
data_mat = str2double(data_mat(:, 1:4));
 
sub_mat = cat(3, data_mat(1:50, :), data_mat(51:100, :), data_mat(101:150, :));

final_data = (data_mat - mean(data_mat)) ./ std(data_mat);
final_sub = (sub_mat - mean(sub_mat)) ./ std(sub_mat);

cv = cov(final_data);
sub_cv = cat(3, cov(final_sub(:, :, 1)), cov(final_sub(:, :, 2)), cov(final_sub(:, :, 3)));
[u, s, v] = svd(cv);
[u1, s1, v1] = svd(sub_cv(:, :, 1));
[u2, s2, v2] = svd(sub_cv(:, :, 2));
[u3, s3, v3] = svd(sub_cv(:, :, 3));

u = u(:, 1:2); u1 = u1(:, 1:2); u2 = u2(:, 1:2); u3 = u3(:, 1:2);

proj = u' * final_data';
proj1 = u1' * final_sub(:, :, 1)';
proj2 = u2' * final_sub(:, :, 2)';
proj3 = u3' * final_sub(:, :, 3)';

scatter(proj(1,1:50), proj(2,1:50), 20, "RED")
hold on
scatter(proj(1,51:100), proj(2,51:100), 20, "GREEN")
hold on
scatter(proj(1,101:150), proj(2,101:150), 20, "BLUE")
hold off
legend('Iris-setosa', 'Iris-versicolor', 'Iris-virginica')

% scatter(proj1(1,:), proj1(2,:))
% legend('Iris-setosa')

% scatter(proj2(1,:), proj2(2,:))
% legend('Iris-versicolor')

% scatter(proj3(1,:), proj3(2,:))
% legend('Iris-virginica')

