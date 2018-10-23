# Unsupervised Domain Translation with Alignment Guarantees

Code for 'Unsupervised Domain Translation with Alignment Guarantees'

## Dependencies

```
tensorflow
skimage
matplotlib
python 3.5
```

## Handwriting-to-Label Translation

### Running the code

Here are commands to reproduce the results of full model / without shuffling / without pairwise discriminator

```
python main_mnist.py --data zipcode
python main_mnist.py --data zipcode --model nodop
python main_mnist.py --data zipcode --model nopd
```

### Visualizing the Training Process

Here is the visualization of samples of `G: Z -> X` (sampled handwritten digits given the label, left) and confusion matrix of `F: X -> Z` (`C_[i,j]` is the probability that our model will predict digit `j-1` when the true digit is `i-1`, diagonal is perfect, right) every 100 iters. There is a discrete optimization every 1000 iters.

![](assets/logs.gif)

The results is obtained using CPU (accuracy is `98.6%` at iteration 12,500), the results of GPU will be a little different due to the randomness GPU introduced in.

## State-to-Image Translation for RL Environments

To be coming

