import numpy as np
from tqdm.auto import tqdm
import scipy.stats as st
import matplotlib.pyplot as plt

def parametric_confidence_interval(confidence_level, mean, sample_size, verbose=True):
    
    """Computes the confidence interval on the provided data for the given confidence_level. If the sample
    size is >=30, it will be assumed that the sample followed a standard normal distribution (according to
    the central limit theorem); otherwise it will be assumed that the data follow a t-distribution with 
    degree of freedom = sample_size-1.
    
    Parameters:
    
    acc_dict: dictionary of accuracies output of the method get_accuracy_on_samples from this repo.
    
    confidence_level: A float in [0.8, 1).
    
    verbose: if True additional information is printed.
    
    Returns: mean and estimated range of the confidence interval.
    """
        
    assert .8 <= confidence_level and confidence_level < 1
    assert sample_size > 0
    assert 0 <= mean and mean <=1
        
    const = None
    if sample_size >= 30:
        if verbose: print("Computing C.I. assuming the data follow a standard normal distribution")
        const=st.norm.ppf(confidence_level)
    else:
        if verbose: print("Computing C.I. assuming the data follow a t-distribution (sample_size < 30)")
        const=st.t.ppf(confidence_level, df=sample_size-1)
        
    r = const * ((mean * (1-mean))/sample_size)**.5
    
    # If necessary, the interval is clipper to be contained in [0, 1]
    upper_bound = min(1, r+mean)
    lower_bound = max(0, mean-r)
        
    mean = (upper_bound+lower_bound)/2.0
    r = mean-lower_bound
    
    if verbose: 
        rounded_confidence_level = round(confidence_level, 3)
        rounded_r = round(r, 3)
        rounded_mean = round(mean*100, 3)
        print(f"With {rounded_confidence_level}% confidence the accuracy of the model is in {rounded_mean}% +/-{rounded_r}.")
    return mean, r

def get_accuracy_samples(instantiable_model, X, y, model_params_dict=None,
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
        

def nonparametric_confidence_interval(acc_dict, confidence_level, verbose=True, return_median=False):
    """Computes the confidence interval on the provided data for the given alpha.
    
    Parameters:
    
    acc_dict: dictionary of accuracies output of the method get_accuracy_on_samples from this repo.
    
    confidence_level: A float in [0.8, 1).
    
    verbose: if True additional information is printed.
    
    return_median: if True a dict containing lower bound, upper bound median and mean is returned.
    
    Returns: mean and estimated range of the confidence interval, if return_median=False.
    """
    
    assert isinstance(acc_dict, dict)
    assert .8 <= confidence_level and confidence_level < 1
    alpha = (1 - confidence_level) * 100
    
    sorted_accuracies = sorted(acc_dict.values())
    lower_percentile = alpha/2
    upper_percentile = 100-lower_percentile
    
    lower_bound = np.percentile(sorted_accuracies, lower_percentile)
    upper_bound = np.percentile(sorted_accuracies, upper_percentile)
    
    mean = (lower_bound + upper_bound)/2
    r = mean - lower_bound
    
    if verbose:
        rounded_mean = round(mean*100, 3)
        rounded_r = round(r*100, 3)
        rounded_confidence = round(confidence_level*100, 3)
        
        str_accuracy = f"{rounded_mean}% +/-{rounded_r}"
        print(f"From the given data, with {rounded_confidence}% confidence the accuracy of the model is in {str_accuracy}.")
    
    if return_median:
        median = np.percentile(sorted_accuracies, 50)
        
        results = {}
        results['lower_bound'] = lower_bound
        results['median'] = median
        results['mean'] = mean
        results['upper_bound'] = upper_bound
        return results
    
    return mean, r


x_ticks = ("90% Confidence", "95% Confidence", "99% Confidence")
def plot_util(intervals_A, intervals_B=None, x_ticks=x_ticks, label_A=None, label_B=None,
              ylim=None, xlabel=None, ylabel=None):
    
    x_axis = np.arange(1, len(intervals_A)+1)
    
    mean_s = [i[0] for i in intervals_A]
    range_s = [i[1] for i in intervals_A]

    plt.errorbar(x=x_axis, y=mean_s, yerr=range_s, color="black", capsize=3,
                 linestyle="None", marker="s", markersize=7, mfc="black", mec="black", label=label_A)
    
    if intervals_B is not None:
        mean_s = [i[0] for i in intervals_B]
        range_s = [i[1] for i in intervals_B]
        plt.errorbar(x=x_axis+0.2, y=mean_s, yerr=range_s, color="grey", capsize=3,
                 linestyle="None", marker="s", markersize=7, mfc="grey", mec="gray", label=label_B)

    plt.xticks(x_axis, x_ticks, rotation=45)
    
    if label_A is not None or label_B is not None: plt.legend()
    if ylim is not None: plt.ylim(ylim)
    if xlabel is not None: plt.xlabel(xlabel)
    if ylabel is not None: plt.ylabel(ylabel)
        
    plt.tight_layout()
    plt.show()