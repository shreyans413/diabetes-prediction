% SHREYANS JAIN 24/B06/040
% Diabetes Prediction Using Machine Learning

% Initialization
clear
clc
close all

% Import Data Set
data = readtable('diabetes.csv');

% Dataset Summary
disp('Dataset Summary:');
summary(data)

% Check for missing values
disp('Missing Values Check:');
disp(any(ismissing(data)));  % Check if any column has missing values
disp('Count of Missing Values in each column:');
disp(sum(ismissing(data)));  % Get count of missing values

% Replace 0s with NaNs
dataCopy = data;
zeroCols = {'Glucose','BloodPressure','SkinThickness','Insulin','BMI'};
for i = 1:length(zeroCols)
    col = zeroCols{i};
    dataCopy.(col)(dataCopy.(col) == 0) = NaN;
end

% Count missing values after replacement
disp('Missing Values Count After Replacement:');
disp(sum(ismissing(dataCopy)));

% Plotting histograms of data before imputation
figure;
tiledlayout(3, 3);  % Adjust layout based on the number of columns
for i = 1:width(data)
    nexttile;
    histogram(data{:, i}, 'FaceColor', 'b');
    title(data.Properties.VariableNames{i});
    xlabel(data.Properties.VariableNames{i});
    ylabel('Frequency');
end

% Impute missing values
dataCopy.Glucose = fillmissing(dataCopy.Glucose, 'constant', mean(dataCopy.Glucose, 'omitnan'));
dataCopy.BloodPressure = fillmissing(dataCopy.BloodPressure, 'constant', mean(dataCopy.BloodPressure, 'omitnan'));
dataCopy.SkinThickness = fillmissing(dataCopy.SkinThickness, 'constant', median(dataCopy.SkinThickness, 'omitnan'));
dataCopy.Insulin = fillmissing(dataCopy.Insulin, 'constant', median(dataCopy.Insulin, 'omitnan'));
dataCopy.BMI = fillmissing(dataCopy.BMI, 'constant', median(dataCopy.BMI, 'omitnan'));

% Plotting histograms of data after imputation
figure;
tiledlayout(3, 3);
for i = 1:width(dataCopy)
    nexttile;
    histogram(dataCopy{:, i}, 'FaceColor', 'g');
    title(dataCopy.Properties.VariableNames{i});
    xlabel(dataCopy.Properties.VariableNames{i});
    ylabel('Frequency');
end

% Visualize the missing value count per column
missingValues = sum(ismissing(data));  % Count missing values for each column
figure;
bar(missingValues);
title('Missing Values Count');
xlabel('Columns');
ylabel('Count');
xticklabels(data.Properties.VariableNames);
xtickangle(45);

% Count the number of instances in each Outcome class (0 and 1)
outcomeCat = categorical(data.Outcome);
outcomeCounts = countcats(outcomeCat);
outcomeLabels = categories(outcomeCat);
disp('Outcome Distribution:');
for i = 1:length(outcomeLabels)
    fprintf('%s    %d\n', outcomeLabels{i}, outcomeCounts(i));
end

% Plot a bar chart showing the number of samples in each Outcome class
figure;
bar(outcomeCounts);
title('Outcome Distribution');
xlabel('Outcome');
ylabel('Count');
xticklabels({'0', '1'});

% Boxplot and distribution of Insulin
figure;
subplot(1, 2, 1);
histogram(data.Insulin, 'FaceColor', 'b');
title('Insulin Distribution');
xlabel('Insulin');
ylabel('Frequency');

subplot(1, 2, 2);
boxplot(data.Insulin);
title('Insulin Boxplot');
xticklabels({'Insulin'});

% Correlation Heatmap between features
corrMatrix = corr(table2array(data(:,1:end-1)), 'Rows', 'complete');  % Exclude Outcome column
figure;
heatmap(data.Properties.VariableNames(1:end-1), data.Properties.VariableNames(1:end-1), corrMatrix, 'Colormap', hot, 'ColorLimits', [-1, 1]);
title('Correlation Heatmap');

% Feature matrix and label vector
X = dataCopy{:, 1:end-1};
y = dataCopy.Outcome;

% Feature scaling
X = normalize(X);

% Display the scaled data
disp(newline);
disp('Scaled Features:');
disp(X(1:5, :));  % Display first 5 rows of scaled features

% Display first few entries of y
disp('Outcome Variable:');
disp(y(1:5));

% Train/test split
cv = cvpartition(y, 'HoldOut', 0.33);
X_train = X(training(cv), :);
X_test = X(test(cv), :);
y_train = y(training(cv));
y_test = y(test(cv));

% Convert to tables and back for fillmissing consistency
X_train = table2array(fillmissing(array2table(X_train), 'constant', mean(X_train)));
X_test = table2array(fillmissing(array2table(X_test), 'constant', mean(X_test)));

% ---------------- Train Base Models ----------------

% Random Forest
rng(7);
rfc = TreeBagger(200, X_train, y_train, ...
    'Method', 'classification', ...
    'OOBPrediction', 'on', ...
    'OOBPredictorImportance', 'on');
pred_rfc = str2double(predict(rfc, X_test));

% XGBoost (LogitBoost)
MdlXGB = fitcensemble(X_train, y_train, ...
    'Method', 'LogitBoost', ...
    'NumLearningCycles', 100, ...
    'Learners', templateTree('MaxNumSplits', 10));
pred_xgb = predict(MdlXGB, X_test);

% SVM
svmModel = fitcsvm(X_train, y_train);
pred_svm = predict(svmModel, X_test);

% ---------------- Create Stacked Features ----------------
stacked_X = [pred_rfc, pred_xgb, pred_svm];

% ---------------- Train Meta-Model ----------------
metaModel = fitcensemble(stacked_X, y_test, ...
    'Method', 'LogitBoost', ...
    'NumLearningCycles', 100, ...
    'Learners', templateTree('MaxNumSplits', 5));

% Predict using the stacked model
final_pred = predict(metaModel, stacked_X);

% ---------------- Evaluate Stacked Model ----------------
acc_stacked = sum(final_pred == y_test) / numel(y_test);
fprintf("\nStacked Model Accuracy: %.2f%%\n", acc_stacked * 100);
disp(newline);
classificationReport(y_test, final_pred);

% ---------------- Compare All Accuracies ----------------
fprintf("\nIndividual Model Accuracies:\n");
fprintf("Random Forest Accuracy: %.2f%%\n", sum(pred_rfc == y_test) / numel(y_test) * 100);
fprintf("XGBoost Accuracy: %.2f%%\n", sum(pred_xgb == y_test) / numel(y_test) * 100);
fprintf("SVM Accuracy: %.2f%%\n", sum(pred_svm == y_test) / numel(y_test) * 100);

% ---------------- ROC Curve for Stacked Model ----------------
% Get prediction scores (probabilities) from the meta-model
[~, scores] = predict(metaModel, stacked_X);

% Compute ROC curve and AUC
[rocX, rocY, ~, AUC] = perfcurve(y_test, scores(:,2), 1);

% Plot ROC Curve
figure;
plot(rocX, rocY, 'b-', 'LineWidth', 2);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(['ROC Curve - Stacked Model (AUC = ' num2str(AUC, '%.2f') ')']);
grid on;

% HELPER FUNCTION
function report = classificationReport(yTrue, yPred)
    % Convert to categorical if not already
    yTrue = categorical(yTrue);
    yPred = categorical(yPred);

    % Confusion matrix and derived metrics
    [C, order] = confusionmat(yTrue, yPred);

    precision = diag(C) ./ sum(C, 2);
    recall = diag(C) ./ sum(C, 1)';
    f1 = 2 * (precision .* recall) ./ (precision + recall);

    % Calculate the weighted F1 score (weighted by support)
    support = sum(C, 1)';  % Number of instances in each class
    weightedF1 = sum(f1 .* support) / sum(support);

    % Display the class-wise metrics and the overall weighted F1 score
    report = table(order, precision, recall, f1, 'VariableNames', ...
        {'Class', 'Precision', 'Recall', 'F1_Score'});

    disp(report);
    fprintf('Overall Weighted F1 Score: %.4f\n', weightedF1);
end
