# Readable Capsule Networks

Capsule network in < 80 lines.

Research-friendly implementation of capsule networks from [Dynamic Routing Between Capsules](https://papers.nips.cc/paper/6975-dynamic-routing-between-capsules.pdf)
in pytorch.


### Project origins:

@michaelklachko [suggested](https://github.com/arogozhnikov/einops/issues/53)
to rewrite 
his [implementation](https://github.com/michaelklachko/CapsNet/blob/master/capsnet_cifar.py#L68-L91) 
of routing algorithm to pytorch using ein-notation.

This turned out to be a very nice exercise. 
Additionally to improvements in routing done by Michael, 
I've minified all tricky places in a model with `einops` layers.   

