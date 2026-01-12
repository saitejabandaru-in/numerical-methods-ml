% Step 1: Center the Data
X_centered = X - mean(X, 1);

% Step 2: Perform k-means clustering
k = 3; % Number of clusters
[idx, C] = kmeans(X_centered, k, 'Replicates', 5);

% Step 3: Plot the k-means clustering results
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
