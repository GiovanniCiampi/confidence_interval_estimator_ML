# Boostrap_Confidence_Intervals_ML
This repo contains code to perform Bootstrap Confidence Intervals estimation (a.k.a. Monte Carlo Confidence Interval or Empirical Confidence Interval estimation) for Machine Learing models.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wBWNSMH64q8uDk9vr15uT-_QQqFz752X)


The code is very straightforward to use. You just need to:

1. Copy the bootstrap_confindence_intervals.py file.
2. Import the functions defined in the file: <br/> ```from bootstrap_confindence_intervals import get_accuracy_on_samples, get_confidence_interval```
3. Call the ```get_accuracy_on_samples``` function specifying model, data and sampling you intend to use: ```accs=get_accuracy_on_samples(RandomForestClassifier, X, y, n_iterations=100)```. You can find all the details on this function by looking at the documentation or by looking at the examples provided in the Notebook contained in this repo.
4. Call the ```get_confidence_interval``` function with the accuracies obtained at point 3 and the desired alpha: 
```
>>>get_confidence_interval(accs, alpha=10, verbose=True)
>>> From the given data, with 90% probability, the accuracy of the model is 95.24% +/- 4.76
>>> (0.9047619047619048, 1.0)
```

For further details, please look at the code documentation or the examples provided in the Notebook.ipynb.
