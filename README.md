# Readable Capsule Networks

Capsule network in < 80 lines.

Research-friendly implementation of capsule networks from [Dynamic Routing Between Capsules](https://papers.nips.cc/paper/6975-dynamic-routing-between-capsules.pdf)
in pytorch.


### Hinton explaining CapsNets

<a href='https://www.youtube.com/watch?v=x5Vxk9twXlE'>
<img src='http://arogozhnikov.github.io/images/etc/hinton_explains_capsnets.png' width=500 />
</a>

Video of lecture: <https://www.youtube.com/watch?v=x5Vxk9twXlE>

### Project origins:

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

