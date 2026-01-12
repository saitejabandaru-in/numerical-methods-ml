% Step 1: Center the Data
X_centered = X - mean(X, 1);

% Step 2: Perform NMF
options = statset('MaxIter', 100, 'Display', 'final');
[W, H, D] = nnmf(X_centered, 2, 'replicates', 10, 'options', options);

% Step 3: Plot the first two NMF components
figure;
gscatter(W(:, 1), W(:, 2), I, 'rgb', 'osd');
title('NMF on Iris Data');
xlabel('First NMF Component');
ylabel('Second NMF Component');
grid on;
set(gca, 'FontSize', 12);
