import numpy as np
from tqdm.auto import tqdm
from matplotlib import pyplot as plt


def get_accuracy_on_samples(instantiable_model, X, y, model_params_dict=None,
                            fit_params_dict=None, n_iterations=100, sample_ratio=0.7,
                            train_ratio=0.8, random_seed=None, sample_with_replacement=True,
                            is_one_hot=False, verbose=True, progress_bar=True):
    
    """Evaluates the performances of a model over different data samples. For each iteration, a train and test
    set are sampled at random from the provided data; then a new model instance is created and trained on the
    train set. The trained instance is then tested on the held-out test data for the particular iteration. The
    method returns a dict with the accuracies on test data for each iteration. 
    
    Parameters:
    
    instantiable_model: a function that returns a new model instance. The model must provide fit and predict methods.
    
    X: the model input.
    
    y: the model target.
    
    model_params_dict: a dictionary with parameters for the instantiation of the model. (optional)
    
    fit_params_dict: a dictionary with parameters to be passed on calls of the fit method of the model. (optional)
    
    n_iterations: the number of times the data is resampled and a new model evaluated.
    
    sample_ratio: what percentage of the dataset (X, y) should be resampled at each iteration. 0<sample_ratio<=1
    
    train_ratio: what percentage of the resampled data should be used for trainining. 0<train_ratio<1
    
    random_seed: sets the random seed for the resampling in order to have reproducible results. (optional)
    
    sample_with_replacement: if True, at each iteration (X, y) is sampled with replacement. 
    
    is_one_hot: must be set to True if the labels are one hot encoded.
    
    verbose: if True details on the samples and train/test size will be printed.
    
    progress_bar: if True a progress bar will be displayed.
    
    
    Returns: a dictionary of the form {iteration_number_i: accuracy_on_test_for_iteration_i}
    """
    
    assert 0<sample_ratio and sample_ratio<=1
    assert 0<train_ratio and train_ratio<1
    assert n_iterations>1
    assert random_seed is None or isinstance(random_seed, int)
    
    if not isinstance(X, np.ndarray): X = np.array(X)
    if not isinstance(y, np.ndarray): y = np.array(y)
    
    if model_params_dict is None: model_params_dict = {}
    if fit_params_dict is None: fit_params_dict = {}
    
    accuracy_dict = {}
    
    dataset_size = len(X)
    samples_size = int(sample_ratio*dataset_size)
    train_size = int(train_ratio*samples_size)
    test_size = samples_size-train_size
    
    assert train_size > 0 and test_size > 0
    
    if verbose:
        print(f"Evaluating model on {n_iterations} samples of (X, y).")
        print(f"Each sample will contain {samples_size} elements from (X, y), out of which {train_size} elements")
        print(f"will be used to train the model, while the remaining {test_size} elements to test its accuracy.")
    
    for idx in tqdm(range(n_iterations), disable=not progress_bar):
        
        if random_seed: np.random.seed(idx + random_seed)
        model = instantiable_model(**model_params_dict)
        
        sample_idxs = np.random.choice(dataset_size, samples_size, replace=sample_with_replacement)
        sampled_X, sampled_y = X[sample_idxs], y[sample_idxs]
        
        train_sample_idxs = np.random.choice(samples_size, train_size, replace=False)
        test_sample_idxs = [i for i in range(samples_size) if i not in train_sample_idxs]
        
        # checks that train and test set are disjoint
        assert len(list(set(train_sample_idxs) & set(test_sample_idxs))) == 0
        
        train_X, train_y = sampled_X[train_sample_idxs], sampled_y[train_sample_idxs]
        test_X, test_y = sampled_X[test_sample_idxs], sampled_y[test_sample_idxs]
                
        model.fit(train_X, train_y, **fit_params_dict)
        predictions = model.predict(test_X)
        
        if not isinstance(predictions, np.ndarray): predictions = np.array(predictions)
            
        if is_one_hot:
            predictions = np.argmax(predictions, axis=-1)
            test_y = np.argmax(test_y, axis=-1)
        
        assert len(predictions) == len(test_y)
        correct_preds = [np.array_equal(p, t) for (p, t) in zip(predictions, test_y)]
        n_correct_preds = sum(correct_preds)
        
        accuracy = n_correct_preds/test_size
        accuracy_dict[idx]=accuracy
        
    return accuracy_dict
        

def get_confidence_interval(acc_dict, alpha, verbose=True):
    """Computes the confidence interval on the provided data for the given alpha.
    
    Parameters:
    
    acc_dict: dictionary of accuracies output of the method get_accuracy_on_samples from this repo.
    
    alpha: specification of the confidence interval. The confidence is given by (100-alpha)%.
    
    Returns: dictionary containing lower bound, upper bound and mode for the desired confidence degree.
    """
    assert isinstance(acc_dict, dict)
    assert 0 < alpha and alpha < 15
    
    sorted_accuracies = sorted(acc_dict.values())
    lower_percentile = alpha/2
    upper_percentile = 100-lower_percentile
    
    lower_bound = np.percentile(sorted_accuracies, lower_percentile)
    upper_bound = np.percentile(sorted_accuracies, upper_percentile)
    
    median = np.percentile(sorted_accuracies, 50)
    
    if verbose:
        confidence_deg = 100-alpha
        
        est_mean = (lower_bound + upper_bound)/2        
        est_var = est_mean - lower_bound
        
        est_mean = round(est_mean*100, 2)
        est_var = round(est_var*100, 2)
        
        str_accuracy = f"{est_mean}% +/-{est_var}"
        
        print(f"From the given data, with {confidence_deg}% probability the accuracy of the model is {str_accuracy}.")
        
    results = {}
    
    results[f'lower_bound'] = lower_bound
    results['median'] = median
    results[f'upper_bound'] = upper_bound
    
    return results


def plot_confidence(lower_bounds, median, upper_bounds,
                    x_axis=None,color='b', y_bottom=None,
                    y_top=None, xlabel=None, ylabel="accuracy"):
    
    assert len(lower_bounds) == len(median) and len(median) == len(upper_bounds)
    assert x_axis is None or len(x_axis) == len(median)
    
    if y_bottom is None and y_top is not None: y_bottom=0
    if y_bottom is not None and y_top is None: y_top=1
    
    fig, ax = plt.subplots()
    
    if x_axis is None: x_axis = np.arange(len(median))
    if y_top is not None and y_bottom is not None: plt.ylim((y_bottom, y_top))
        
    ax.plot(x_axis, median)
    ax.fill_between(x_axis, lower_bounds, upper_bounds, color='b', alpha=.1)
    
    if xlabel is not None: plt.xlabel(xlabel)
    if ylabel is not None: plt.ylabel(ylabel)
    
    plt.show()
