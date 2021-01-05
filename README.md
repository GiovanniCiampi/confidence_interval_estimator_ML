# Confidence_Interval_Estimator_ML
This repo contains code to perform estimation of Confidence Intervals both parametric and non-parametric (a.k.a. Monte Carlo Confidence Interval or Empirical Confidence Interval estimation) for Machine Learing models.

# Colab Demo 
You can easily try this tool in Google Colab by clicking on the following badge: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16EwUmq9NBiytpS6JoN22l7zDqzqTb_-w?usp=sharing)

# Quickstart 
```
>>> import confidence_interval_estimator_ML.utils as cie
>>> # Parametric Estimation:
>>> cie.parametric_confidence_interval(confidence_level=.95, mean=model_accuracy, sample_size=sample_size, verbose=True)
 Computing C.I. assuming the data follow a standard normal distribution
 With 95.0% confidence the accuracy of the model is in 0.977 +/-0.023.
 (0.9773132867658845, 0.02268671323411553)
 
>>> # Non-Parametric Estimation:
>>> test_accuracies = cie.get_accuracy_on_samples(get_classifier_instance, X, y, n_iterations=100, sample_ratio=1,
                             train_ratio=0.8, random_seed=None, sample_with_replacement=True, verbose=False) 
                             
>>> cie.nonparametric_confidence_interval(accs, .9)
 From the given data, with 90.0% confidence the accuracy of the model is in 0.959 +/-4.05.
 (0.9595, 0.04050000000000009)
```

# Further Details
You can find further details on the usage of the tool in the code documentation or in the examples provided in the Notebook.ipynb file. If you need help, please don't hesitate to open a github issue or getting in touch.





