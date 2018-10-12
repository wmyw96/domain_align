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

## State-to-Image Translation for RL Environments

To be coming

