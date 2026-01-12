% Step 1: Compute Class Means
classLabels = unique(I);
numClasses = length(classLabels);
numFeatures = size(X, 2);
classMeans = zeros(numClasses, numFeatures);

for i = 1:numClasses
    classMeans(i, :) = mean(X(I == classLabels(i), :), 1);
end

% Step 2: Compute Overall Mean
overallMean = mean(X, 1);

% Step 3: Within-class Scatter Matrix
Sw = zeros(numFeatures, numFeatures);
for i = 1:numClasses
    Xi = X(I == classLabels(i), :);
    Si = cov(Xi);
    Sw = Sw + (size(Xi, 1) - 1) * Si;
end

% Step 4: Between-class Scatter Matrix
Sb = zeros(numFeatures, numFeatures);
for i = 1:numClasses
    Ni = sum(I == classLabels(i));
    meanDiff = (classMeans(i, :) - overallMean)' * (classMeans(i, :) - overallMean);
    Sb = Sb + Ni * meanDiff;
end

% Step 5: Solve Eigenvalue Problem
[eigenVectors, eigenValues] = eig(pinv(Sw) * Sb);

% Step 6: Sort Eigenvalues and Eigenvectors
[~, sortIdx] = sort(diag(eigenValues), 'descend');
eigenVectors = eigenVectors(:, sortIdx);

% Step 7: Project Data
X_lda = X * eigenVectors(:, 1:2);

% Step 8: Plot the LDA components
figure;
gscatter(X_lda(:, 1), X_lda(:, 2), I, 'rgb', 'osd');
title('LDA on Iris Data');
xlabel('First LDA Component');
ylabel('Second LDA Component');
legend('Iris setosa', 'Iris versicolor', 'Iris virginica');
grid on;
set(gca, 'FontSize', 12);
