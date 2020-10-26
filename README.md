# Readable Capsule Networks

Capsule network in < 80 lines.

Research-friendly implementation of capsule networks from [Dynamic Routing Between Capsules](https://papers.nips.cc/paper/6975-dynamic-routing-between-capsules.pdf)
in pytorch.



### What are capsules?

Capsules are groups of neurons each describing an entity with positional characteristics. 

Capsules are meant to form a 'soft parsing tree' for a scene with strength of confidence encoded by norm of capsule's activations.


### Intuition behind CapsNets

<a href='https://www.youtube.com/watch?v=x5Vxk9twXlE'>
<img src='http://arogozhnikov.github.io/images/etc/hinton_explains_capsnets.png' width=500 />
</a>

Video of lecture: <https://www.youtube.com/watch?v=x5Vxk9twXlE>


## What is different about this implementation

- completely readable and very compact
- capsule layers are perfectly stackable
- auto inference of number of capsules after convolutional stem
- **memory efficiency:** <br />
  almost 3x less GPU memory foorprint compared to other [implementation](https://github.com/cedrickchee/capsule-net-pytorch) (2Gb vs 5.9Gb for batch size of 256)
- **blazingly fast:** <br /> 
  15 sec/epoch vs 747 sec/epoch on single V100 vs other [implementation](https://github.com/cedrickchee/capsule-net-pytorch)

and, well, I didn't even use `torch.jit.trace` and did not use `fp16`, which will provide an additional boost in efficiency.


Two most important changes that made this possible are:

- all convolutional capsules are packed into one fat convolution instead of splitting into several convolutions and then reshaping and concatenating, 
  `einops` takes care of making that efficiently. Additionally `einops` resolves weight management for capsules.
- ugly routing implementation made efficient - all split/concat and all unnecessary repeats are eliminated by proper usage of `einsum` and `einops`


## Project origins:

There is a ton of implementations for CapsNets. 

Most of them are hardly readable, almost all are inefficient and a large fraction is simply wrong.
All of them require to make some computations specific for each dataset and disallow easy tweaking of parts.   

@michaelklachko [suggested](https://github.com/arogozhnikov/einops/issues/53)
to rewrite 
his [implementation](https://github.com/michaelklachko/CapsNet/blob/master/capsnet_cifar.py#L68-L91) 
of routing algorithm to pytorch using ein-notation.

This turned out to be a very nice exercise. 
Additionally to improvements in routing done by Michael, 
I've minified all tricky places in encoder/decoder with `einops` layers.   
