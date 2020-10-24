
As an exercise to deepen my understanding of reinforcement learning I decided to implement a few common algorithms in JAX. The current JAX ecosystem, while thriving, is still bare-bones compared to frameworks such as TensorFlow and PyTorch. This means that many of the common conveniences that I take for granted were removed. Engineering decisions and considerations are sometimes as important as the theory, and almost always the ease of implementation plays into the mainstream success of an algorithm or method. The reparameterization trick was one such consideration which came up over the course of this projecte. I had read about it previously, and digested the general idea, but I believe that most articles covering the topic don't cover a simple intuition I'd like to discuss.

### The Problem

The setting in which we'd come across the reparameterization trick is when we're implementing a **parametric model** using a neural network. A parametric model defines a probability distribution, or set of probability distributions, using their descriptive parameters. For example, if we want to model a Gaussian distribution we can do so by trying to learn the values of its mean and variance.

When does this come up in reinforcement learing? All over the place! Let's think of the most simple example in the continuous setting -- cartpole. In cartpole we have two **actuators**, which we can think of as motors which control movement of one part of the object in one dimension, in this case the movement of the cart and the movement of the pole. The input we want to give the system is the directional force applied to each of these. You might be thinking: "Isn't this a deterministic setting? Why don't we just learn a single value for each state?" While the former is true, we have to remember that the state space can be massive in reinforcement learning, and we want to create the best sample efficiency possible, i.e. be able to generalize well with the fewest observations. A Gaussian policy is useful, because it encodes the fact that many of the regions of the state space that we will encounter during evaulation won't have been observed during training, and thus we have a certain degree of uncertainty.

This might seem simple enough; we can build a network with two outputs, one for each parameter, and train that, right? Well, at each training step we have to pass our observed state into the network and get the two parameters of our Gaussian distribution, but that's not an action to pass into our environment. We still have to sample from our distribution! Here's where reparameterization comes into play, because sampling from a distribution is not a differentiable operation.

### The Trick

So, how do we restore our ability to do backpropogation? Well, if we're using TensorFlow or PyTorch each have great modules for probability distributions which take care of this for us. But we're using JAX! As many answers online suggest, we need to change where the randomness is coming from, but what does this mean in practice? In the case of a Gaussian distribution, it's astonishingly simple. Instead of sampling from the distribution paramerized by our network outputs, we can instead sample a "z-score" from a separate N(0, 1) and construct our desired sample from this value.

How we would naively think to do things:

```python
action = np.random.normal(mu, sigma)
def test():
    pass
```

How to reparameterize:

```python
action = mu + np.random.normal() * sigma 
```



