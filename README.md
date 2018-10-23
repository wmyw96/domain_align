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

The code will generate serveral files in a folder to tack the logs, here are the details

- `mnist_d_op.log`: log summary.
- `img[x]c.png`: sample images after `x` iters, the images in the `i`-th column are all digit `x-1`.
- `final_img_d[x]_[y]c.png`: final sample images of digit `x`.
- `mnist_d_op.log.e[x].pred.png`: the estimated confusion matrix after `x` iters.
- `mnist_d_op.log.e[x].trans.png`: the estimated transition matrix of our model after `x` iters.

### Visualizing the Training Process

Here is the visualization of samples of `G: Z -> X` (sampled handwritten digits given the label, left) and confusion matrix of `F: X -> Z` (`C_[i,j]` is the probability that our model will predict digit `j-1` when the true digit is `i-1`, diagonal is perfect, right) every 100 iters. There is a discrete optimization every 1000 iters.

![](assets/logs.gif)

The results is obtained using CPU (accuracy is `98.6%` at iteration 12,500), the results of GPU will be a little different due to the randomness GPU introduced in.

## State-to-Image Translation for RL Environments

To be coming

