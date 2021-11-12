1. i.
  We store images in this network as a function of x, y and gamma, where gamma is the specific image
  we desire to view.
  Generalization in this task means the ability to store more than one image successfully. It's
  limited by the capacity of the network, as well as the difference required by adjacent images in the
  gamma space: the more differences between adjacent images, the more difficult it is for the network to
  to store the images.
  A full expression of the concept of generalization therefore has two components:
  a. One component is based on hyperparameters such as the number of images and the difference between adjacent
  images in the gamma space. We could try to express this component as the sum of L2 deltas between adjacent
  imaged in the gamma space. This component is constant per number of images and layout in gamma space.
  b. The second component is network-specific, and concerns the capability of the network itself given a task.
  We can get a sense of the performance of a network by taking the average L2 loss over all images the network
  is trying to memorize. Subtracting this loss from the generalization constant gives us an idea of network
  generalization.

2.
  
  
  

   
   
