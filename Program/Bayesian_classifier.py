import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import prepare_train_test as ptt

# Custom standard scaler for data normalization
class CustomStandardScaler:
    def __init__(self):
        # Initialize mean_ and scale_ to None
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """
        Calculate the mean and standard deviation of the data X.

        Parameters:
        - X: Data to be normalized.
        """
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)

    def transform(self, X):
        """
        Standardize the data X using the previously calculated mean and standard deviation.

        Parameters:
        - X: Data to be normalized.

        Returns:
        - Standardized data.
        """
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        """
        Combine the fit and transform steps.

        Parameters:
        - X: Data to be normalized.

        Returns:
        - Standardized data.
        """
        self.fit(X)
        return self.transform(X)

# Custom KMeans clustering algorithm
class CustomKMeans:
    def __init__(self, n_clusters=4, max_iter=300, tol=1e-4):
        """
        Initialize KMeans parameters.

        Parameters:
        - n_clusters: Number of clusters.
        - max_iter: Maximum number of iterations.
        - tol: Tolerance for convergence.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.cluster_centers_ = None

    def fit(self, X):
        """
        Fit the KMeans model to the data X.

        Parameters:
        - X: Data to be clustered.
        """
        n_samples, n_features = X.shape
        if n_samples == 0:
            raise ValueError("No samples to fit.")
        rng = np.random.default_rng()
        # Randomly initialize cluster centers
        initial_indices = rng.choice(n_samples, self.n_clusters, replace=False)
        self.cluster_centers_ = X[initial_indices]

        for _ in range(self.max_iter):
            # Assign labels to each sample based on closest cluster center
            labels = self.predict(X)
            # Calculate new cluster centers
            new_centers = np.array([X[labels == j].mean(axis=0) for j in range(self.n_clusters)])
            # Check for convergence
            if np.linalg.norm(new_centers - self.cluster_centers_) < self.tol:
                break
            self.cluster_centers_ = new_centers

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters:
        - X: Data to be clustered.

        Returns:
        - Array of cluster labels.
        """
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
        return np.argmin(distances, axis=1)

# Custom Gaussian Mixture Model (GMM) algorithm
class CustomGMM:
    def __init__(self, n_components=1, tol=1e-3, max_iter=300, reg_covar=1e-6):
        """
        Initialize GMM parameters.

        Parameters:
        - n_components: Number of mixture components.
        - tol: Tolerance for convergence.
        - max_iter: Maximum number of iterations.
        - reg_covar: Regularization parameter for covariance.
        """
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.reg_covar = reg_covar
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.converged_ = False

    def fit(self, X):
        """
        Fit the GMM model to the data X.

        Parameters:
        - X: Data to be modeled.
        """
        n_samples, n_features = X.shape
        if n_samples == 0:
            raise ValueError("No samples to fit.")
        resp = np.random.rand(n_samples, self.n_components)
        resp /= resp.sum(axis=1, keepdims=True)
        resp += 1e-10  # Add a small value to avoid zero division

        # Initialize means using KMeans
        kmeans = CustomKMeans(n_clusters=self.n_components)
        kmeans.fit(X)
        self.means_ = kmeans.cluster_centers_

        log_likelihood = 0
        self.weights_ = np.ones(self.n_components) / self.n_components
        self.covariances_ = np.array([np.eye(n_features) for _ in range(self.n_components)])

        for iteration in range(self.max_iter):
            prev_log_likelihood = log_likelihood

            # E-step: Compute responsibilities
            log_prob_norm = self._e_step(X, resp)

            # M-step: Update model parameters
            self._m_step(X, resp)

            # Compute log likelihood
            log_likelihood = np.sum(log_prob_norm)
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                self.converged_ = True
                break

    def _e_step(self, X, resp):
        """
        E-step: Calculate responsibilities.

        Parameters:
        - X: Data to be modeled.
        - resp: Responsibilities array.

        Returns:
        - log_prob_norm: Normalized log probabilities.
        """
        weighted_log_prob = self._estimate_log_prob(X) + np.log(self.weights_)
        max_log_prob = np.max(weighted_log_prob, axis=1, keepdims=True)
        log_prob_norm = max_log_prob + np.log(np.sum(np.exp(weighted_log_prob - max_log_prob), axis=1, keepdims=True))
        resp[:] = np.exp(weighted_log_prob - log_prob_norm)
        return log_prob_norm

    def _m_step(self, X, resp):
        """
        M-step: Update weights, means, and covariances.

        Parameters:
        - X: Data to be modeled.
        - resp: Responsibilities array.
        """
        n_samples = X.shape[0]
        resp_sum = resp.sum(axis=0) + 1e-10  # Add a small value to avoid zero division
        self.weights_ = resp_sum / n_samples
        self.means_ = np.dot(resp.T, X) / resp_sum[:, np.newaxis]

        for k in range(self.n_components):
            diff = X - self.means_[k]
            self.covariances_[k] = np.dot(resp[:, k] * diff.T, diff) / resp_sum[k]
            self.covariances_[k].flat[::X.shape[1] + 1] += self.reg_covar  # Regularization
            self.covariances_[k] = np.nan_to_num(self.covariances_[k], nan=1e-6)

    def _estimate_log_prob(self, X):
        """
        Estimate log probabilities of the data X.

        Parameters:
        - X: Data to be modeled.

        Returns:
        - log_probs: Array of log probabilities.
        """
        log_probs = []
        for k in range(self.n_components):
            try:
                log_prob = multivariate_normal(mean=self.means_[k], cov=self.covariances_[k]).logpdf(X)
            except np.linalg.LinAlgError:
                log_prob = np.full(X.shape[0], -np.inf)
            log_probs.append(log_prob)
        return np.array(log_probs).T

# Function to get GMM parameters for each class
def prob_parameter_gmm(data, n_components=1):
    """
    Get GMM parameters for each class.

    Parameters:
    - data: Input data with labels.
    - n_components: Number of GMM components.

    Returns:
    - prob_para: List of GMM parameters for each class.
    - num_classes: Number of unique classes.
    - class_labels: List of class labels.
    """
    unique_labels = data['Label'].unique()
    class_labels = sorted(unique_labels)
    num_classes = len(class_labels)
    prob_para = []

    for label in class_labels:
        x = data.loc[data['Label'] == label].iloc[:, 3:]
        if x.empty:
            continue
        gmm = CustomGMM(n_components=n_components)
        gmm.fit(x.values)
        prob_para.append({
            'weights': gmm.weights_,
            'means': gmm.means_,
            'covariances': gmm.covariances_,
            'prior': len(x) / len(data)
        })
    return prob_para, num_classes, class_labels

# Function to classify a sample using GMM parameters
def prob_class_gmm(x, prob_para, num_classes):
    """
    Classify a sample using GMM parameters.

    Parameters:
    - x: Sample to be classified.
    - prob_para: List of GMM parameters for each class.
    - num_classes: Number of unique classes.

    Returns:
    - Predicted class label.
    """
    probs = np.zeros(num_classes)
    for i in range(num_classes):
        gmm_params = prob_para[i]
        for j in range(len(gmm_params['weights'])):
            weight = gmm_params['weights'][j]
            mean = gmm_params['means'][j]
            cov = gmm_params['covariances'][j]
            pdf = multivariate_normal.pdf(x, mean=mean, cov=cov)
            probs[i] += weight * pdf
        probs[i] *= gmm_params['prior']
    return np.argmax(probs)

# Function to display prediction statistics
def pred_statistics(data, option=True, class_labels=None):
    """
    Display prediction statistics.

    Parameters:
    - data: Data containing true labels and predictions.
    - option: Boolean flag to indicate validation (True) or test (False) results.
    - class_labels: List of class labels.

    Returns:
    - result: DataFrame containing prediction statistics.
    """
    if option:
        print('\nValidation result')
        result = pd.DataFrame(index=class_labels, columns=['label', 'pred', 'correct', 'wrong'])
        for label in class_labels:
            label_prop = (data['Label'] == label).sum() / len(data)
            pred_prop = (data['pred'] == label).sum() / len(data)
            correct = ((data['Label'] == label) & (data['pred'] == label)).sum() / len(data[data['pred'] == label])
            wrong = ((data['Label'] != label) & (data['pred'] == label)).sum() / len(data[data['pred'] == label])
            result.loc[label] = [round(label_prop, 2), round(pred_prop, 2), round(correct, 2), round(wrong, 2)]
    else:
        print('\nTest result')
        result = pd.DataFrame(index=class_labels, columns=['pred'])
        for label in class_labels:
            pred_prop = (data['pred'] == label).sum() / len(data)
            result.loc[label] = [round(pred_prop, 2)]
    print('=' * 30)
    print(result)
    print('=' * 30)
    return result

# Function to calculate accuracy
def Accuracy(data):
    """
    Calculate accuracy.

    Parameters:
    - data: Data containing true labels and predictions.

    Returns:
    - accuracy: Accuracy value.
    """
    accuracy = (data['Label'] == data['pred']).sum() / len(data)
    print(f'\nAccuracy: {accuracy:.2f}')
    return accuracy

# Function to normalize the data
def normalize_data(data, features):
    """
    Normalize the data.

    Parameters:
    - data: Data to be normalized.
    - features: List of features to normalize.

    Returns:
    - Normalized data.
    """
    scaler = CustomStandardScaler()
    data[features] = scaler.fit_transform(data[features])
    return data

# Function to oversample the data to handle class imbalance
def oversample_data(data):
    """
    Oversample the data to handle class imbalance.

    Parameters:
    - data: Data to be oversampled.

    Returns:
    - Oversampled data.
    """
    counts = data['Label'].value_counts()
    max_count = counts.max()
    data_resampled = [data[data['Label'] == label].sample(max_count, replace=True, random_state=0) for label in counts.index]
    return pd.concat(data_resampled)

# Function to perform cross-validation
def cross_validate(data, select_feature, n_components=4, k_folds=3):
    """
    Perform cross-validation.

    Parameters:
    - data: Data to be cross-validated.
    - select_feature: List of selected features.
    - n_components: Number of GMM components.
    - k_folds: Number of folds for cross-validation.

    Returns:
    - None
    """
    # Shuffle the data to ensure random distribution across folds
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    fold_size = len(data) // k_folds
    accuracies = []

    for k in range(k_folds):
        # Split the data into training and validation sets
        validation_data = data.iloc[k*fold_size:(k+1)*fold_size].copy()
        training_data = pd.concat([data.iloc[:k*fold_size], data.iloc[(k+1)*fold_size:]]).copy()

        prob_para, num_classes, class_labels = prob_parameter_gmm(training_data, n_components=n_components)

        for i in range(len(validation_data)):
            x = validation_data.iloc[i, 3:].values
            predicted_label = prob_class_gmm(x, prob_para, num_classes)
            validation_data.loc[validation_data.index[i], 'pred'] = class_labels[predicted_label]

        accuracy = Accuracy(validation_data)
        accuracies.append(accuracy)
        pred_statistics(validation_data, option=True, class_labels=class_labels)

    print(f'Cross-Validation Accuracy: {np.mean(accuracies):.2f}')

# Function to validate the model and perform cross-validation
def predict_valid(select_feature, n_components=4, option=True):
    """
    Validate the model and perform cross-validation.

    Parameters:
    - select_feature: List of selected features.
    - n_components: Number of GMM components.
    - option: Boolean flag to indicate whether to expand data.

    Returns:
    - data: Training data.
    - prob_para: List of GMM parameters for each class.
    - num_classes: Number of unique classes.
    - class_labels: List of class labels.
    """
    feature_file_path_train = './training_features_total357.csv'
    data = ptt.read_data(feature_file_path_train, 'ML_Train.csv', select_feature)
    if option:
        data = ptt.expand_data(data)
    
    data = oversample_data(data)  # Oversample data to handle class imbalance

    cross_validate(data, select_feature, n_components=n_components, k_folds=3)

    # Train on the entire dataset to get final model parameters
    prob_para, num_classes, class_labels = prob_parameter_gmm(data, n_components=n_components)
    return data, prob_para, num_classes, class_labels

# Function to predict on test data and save results
def predict_test(prob_para, select_feature, num_classes, class_labels):
    """
    Predict on test data and save results.

    Parameters:
    - prob_para: List of GMM parameters for each class.
    - select_feature: List of selected features.
    - num_classes: Number of unique classes.
    - class_labels: List of class labels.

    Returns:
    - None
    """
    feature_file_path_test = './testing_features_total357.csv'
    pred_file_path = './Bayesian_Classifier.csv'
    data_test = ptt.read_data(feature_file_path_test, 'ML_Test.csv', select_feature, option=False)
    
    for i in range(len(data_test)):
        x = data_test.iloc[i, 2:].values
        predicted_label = prob_class_gmm(x, prob_para, num_classes)
        data_test.loc[data_test.index[i], 'pred'] = class_labels[predicted_label]
    
    pred_statistics(data_test, option=False, class_labels=class_labels)
    
    data_test_result = data_test[['SubjectId', 'pred']].copy()
    data_test_result['Label'] = data_test_result['pred'].astype(int)
    data_test_result.drop(columns='pred', inplace=True)
    
    data_test_result.to_csv(pred_file_path, index=False)

# Example usage
select_feature = ['dPT_y_11', "dTT'_y_1", 'dPT_y_10', "dTT'_y_0", 'dST_y_11', 'Power_Ratio_0', 'Spectral_Entropy_0', 'dPT_y_0', "dRS'_x_11", "dRS'_x_0", 'dPT_y_9', 'Power_Ratio_11', 'Power_Ratio_4', 'Power_Ratio_1', "dST'_y_11", 'dPT_y_1', 'Spectral_Entropy_1', 'dST_y_10', "dRS'_x_10", "dRS'_x_1", 'Power_Ratio_5', 'Spectral_Entropy_9', 'Spectral_Entropy_11', "dS'T'_y_0", 'Spectral_Entropy_4', "dS'T'_x_10", 'dST_y_1', 'Spectral_Entropy_10', "dS'T'_x_11", 'flourish_4', 'dRT_y_10', 'dRT_y_11', "dRS'_x_9", 'Power_Ratio_10', 'dRT_y_1']
data, prob_para, num_classes, class_labels = predict_valid(select_feature, n_components=4, option=True)
predict_test(prob_para, select_feature, num_classes, class_labels)
