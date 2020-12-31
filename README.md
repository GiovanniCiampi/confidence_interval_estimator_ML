# Bootstrap_Confidence_Intervals_ML
This repo contains code to perform Bootstrap Confidence Intervals estimation (a.k.a. Monte Carlo Confidence Interval or Empirical Confidence Interval estimation) for Machine Learing models.

# Colab Demo 
You can easily try this tool in Google Colab by clicking on the following badge: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wBWNSMH64q8uDk9vr15uT-_QQqFz752X?usp=sharing)

# Quickstart 
```
from bootstrap_confindence_intervals import get_accuracy_on_samples, get_confidence_interval

test_accuracies = get_accuracy_on_samples(get_classifier_instance, X, y, n_iterations=100, sample_ratio=0.7,
                             train_ratio=0.8, random_seed=None, sample_with_replacement=True, verbose=False)
                             
get_confidence_interval(test_accuracies, alpha=5, verbose=True)
>>> From the given data, with 95% probability,
>>> the accuracy of the model is 95.24% +/- 4.76
>>> (0.90476, 1.0)
```

# Further Details
You can find further details on the usage of the tool in the code documentation or in the examples provided in the Notebook.ipynb file. If you need help, please don't hesitate to open a github issue or getting in touch.





