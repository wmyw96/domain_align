# Unsupervised Domain Translation with Domain Alignment

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

Here are codes to reproduce the results of full model / without shuffling / without pairwise discriminator

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

## State-to-Image Translation for RL Environments

To be coming

