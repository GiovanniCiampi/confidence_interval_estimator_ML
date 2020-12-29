import numpy as np
from tqdm.auto import tqdm


def get_accuracy_on_samples(instantiable_model, X, y, model_params_dict=None,
                            fit_params_dict=None, n_iterations=100, sample_ratio=0.7,
                            train_ratio=0.8, random_seed=None, sample_with_replacement=True,
                            is_one_hot=False, verbose=True):
    
    """Evaluates the performances of a model over different data samples. For each iteration, a train and test
    set are sampled at random from the provided data; then a new model instance is created and trained on the
    train set. The trained instance is then tested on the held-out test data for the particular iteration. The
    method returns a dict with the accuracies on test data for each iteration. 
    Parameters
    ----------
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
    
    
    returns: a dictionary of the form {iteration_number_i: accuracy_on_test_for_iteration_i}
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
        print(f"Each sample will contain {samples_size} elements from X, out of which {train_size} elements")
        print(f"will be used to train the model, while the remaining {test_size} elements to test its accuracy.")
    
    for idx in tqdm(range(n_iterations)):
        
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
    
    assert isinstance(acc_dict, dict)
    assert 0 < alpha and alpha < 15
    
    sorted_accuracies = sorted(acc_dict.values())
    lower_percentile = alpha/2
    upper_percentile = 100-lower_percentile
    
    lower_bound = np.percentile(sorted_accuracies, lower_percentile)
    upper_bound = np.percentile(sorted_accuracies, upper_percentile)
    
    if verbose:
        confidence_deg = 100-alpha
        est_mean = (lower_bound + upper_bound)/2
        est_var = est_mean - lower_bound
        print(f"From the given data, with {confidence_deg}% probability,")
        print(f"the accuracy of the model is {round(est_mean*100, 2)}% +/- {round(est_var*100, 2)}")

    return lower_bound, upper_bound