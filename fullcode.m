% Center the data
X_centered = X - mean(X, 1);

% Perform PCA
[coeff, score, latent] = pca(X_centered);

% Plot the first two principal components
figure;
gscatter(score(:, 1), score(:, 2), I, 'rgb', 'osd');
title('PCA on Iris Data');
xlabel('First Principal Component');
ylabel('Second Principal Component');
grid on;
set(gca, 'FontSize', 12);

% Perform LDA
lda = fitcdiscr(X, I);

% Project data onto the LDA components
X_lda = X * lda.Coeffs(1,2).Linear;
if size(lda.Coeffs, 1) > 1
    X_lda(:, 2) = X * lda.Coeffs(2,3).Linear;
end

% Plot the LDA components
figure;
gscatter(X_lda(:, 1), X_lda(:, 2), I, 'rgb', 'osd');
title('LDA on Iris Data');
xlabel('First LDA Component');
ylabel('Second LDA Component');
legend('Iris setosa', 'Iris versicolor', 'Iris virginica');
grid on;
set(gca, 'FontSize', 12);

% Center the data
X_centered = X - mean(X, 1);

% Perform NMF
options = statset('MaxIter', 100, 'Display', 'final');
[W, H, D] = nnmf(X_centered, 2, 'replicates', 10, 'options', options);

% Plot the first two NMF components
figure;
gscatter(W(:, 1), W(:, 2), I, 'rgb', 'osd');
title('NMF on Iris Data');
xlabel('First NMF Component');
ylabel('Second NMF Component');
grid on;
set(gca, 'FontSize', 12);

% Center the data
X_centered = X - mean(X, 1);

% Perform k-means clustering
k = 3; % Number of clusters
[idx, C] = kmeans(X_centered, k, 'Replicates', 5);

% Plot the k-means clustering results
figure;
gscatter(X_centered(:, 1), X_centered(:, 2), idx, 'rgb', 'osd');
hold on;
plot(C(:, 1), C(:, 2), 'kx', 'MarkerSize', 15, 'LineWidth', 3);
title('K-means Clustering on Iris Data');
xlabel('First Feature');
ylabel('Second Feature');
legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Centroids', 'Location', 'Best');
grid on;
set(gca, 'FontSize', 12);

% Perform PCA
[coeff, score, latent] = pca(X_centered);

% Plot singular values
figure;
plot(latent, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
title('Singular Values from PCA');
xlabel('Principal Component');
ylabel('Singular Value');
grid on;
set(gca, 'FontSize', 12);

figure;
subplot(1,2,1);
gscatter(score(:, 1), score(:, 2), I, 'rgb', 'osd');
title('PCA on Iris Data');
xlabel('First Principal Component');
ylabel('Second Principal Component');
grid on;
set(gca, 'FontSize', 12);

subplot(1,2,2);
gscatter(X_lda(:, 1), X_lda(:, 2), I, 'rgb', 'osd');
title('LDA on Iris Data');
xlabel('First LDA Component');
ylabel('Second LDA Component');
legend('Iris setosa', 'Iris versicolor', 'Iris virginica');
grid on;
set(gca, 'FontSize', 12);

figure;
subplot(1,2,1);
gscatter(score(:, 1), score(:, 2), I, 'rgb', 'osd');
title('PCA on Iris Data');
xlabel('First Principal Component');
ylabel('Second Principal Component');
grid on;
set(gca, 'FontSize', 12);

subplot(1,2,2);
gscatter(W(:, 1), W(:, 2), I, 'rgb', 'osd');
title('NMF on Iris Data');
xlabel('First NMF Component');
ylabel('Second NMF Component');
grid on;
set(gca, 'FontSize', 12);

figure;
subplot(1,2,1);
gscatter(score(:, 1), score(:, 2), I, 'rgb', 'osd');
title('PCA on Iris Data');
xlabel('First Principal Component');
ylabel('Second Principal Component');
grid on;
set(gca, 'FontSize', 12);

subplot(1,2,2);
gscatter(X_centered(:, 1), X_centered(:, 2), idx, 'rgb', 'osd');
hold on;
plot(C(:, 1), C(:, 2), 'kx', 'MarkerSize', 15, 'LineWidth', 3);
title('K-means Clustering on Iris Data');
xlabel('First Feature');
ylabel('Second Feature');
legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Centroids', 'Location', 'Best');
grid on;
set(gca, 'FontSize', 12);

