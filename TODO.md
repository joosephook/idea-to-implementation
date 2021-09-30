# Task Description
1. In this task, you will implement a visual attention mechanism called Spatial Transformer
   Networks (STN for short). You can read more about the spatial transformer networks in this
   DeepMind paper. STNs allows a neural network to perform spatial manipulation on the input data
   within the network to enhance the geometric invariance of the model.
   * It can be a useful
     mechanism because CNNs are not invariant to rotation and scale and more general affine
     transformations. STNs can be simply inserted into existing convolutional architectures without
     any extra training supervision or modification to the optimization process.
   * [This PyTorch tutorial might help you kick off this task. Please use this implementation as a baseline.](https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html)

2. As the next step, let’s investigate if using `CoordConv` layers instead of standard `Conv` will help
   to improve the performance of our baseline.
   * `CoordConv` works by giving
     convolution operation access to its input coordinates through the use of extra coordinate
     channels. [You can read more about the `CoordConv` solution in this Uber AI paper.](https://arxiv.org/pdf/1807.03247v2.pdf)
   * Compare the performance of the new model in your
     preferred evaluation metrics
   * Motivate the choice of metrics.

* STN is an emerging topic in the vision and learning communities.
  Explore the latest advancements and new ideas that might achieve better
  performance than conventional STNs.
 

* Your objective should be to help us understand
how you approach converting an idea to an experiment – not achieving a SOTA model. You will
later have the opportunity to discuss your work in more detail with your future team.
* You should respond to the email with a URL to a publicly accessible GitHub repository of your
solution. Please note that we will evaluate your solution also based on code quality, readability,
and implementation of best practices. Reproducibility and documentation of your solution are of
equivalent importance as the solution itself. Note that all external resources should be properly
cited.

# Tasklist
1. Reproducibility: torch implementation as reproducible as possible
2. Implement CoordConv for torch based on `CoordConv.py`
3. Think experiments