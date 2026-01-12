% Step 1: Center the Data
X_centered = X - mean(X, 1);

% Step 2: Compute Covariance Matrix
covMatrix = cov(X_centered);

% Step 3: Eigen Decomposition
[eigenVectors, eigenValues] = eig(covMatrix);

% Step 4: Sort Eigenvalues and Eigenvectors
[~, sortIdx] = sort(diag(eigenValues), 'descend');
eigenVectors = eigenVectors(:, sortIdx);
eigenValues = eigenValues(sortIdx, sortIdx);

% Step 5: Project Data
score = X_centered * eigenVectors;

% Step 6: Plot the first two principal components
figure;
gscatter(score(:, 1), score(:, 2), I, 'rgb', 'osd');
title('PCA on Iris Data');
xlabel('First Principal Component');
ylabel('Second Principal Component');
grid on;
set(gca, 'FontSize', 12);
