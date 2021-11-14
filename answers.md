1. a.i.
  We store images in this network as a function of x, y and gamma, where gamma is the specific image
  we desire to view.
  Generalization in this task means the ability to store more than one image successfully. It's
  limited by the capacity of the network, as well as the difference required by adjacent images in the
  gamma space: the more differences between adjacent images at a given x-y location,
  the more difficult it should be for the network to store the images.
  A full expression of the concept of generalization therefore has two components:
  a. One component is based on hyperparameters such as the number of images and the difference between adjacent
  images in the gamma space. We could try to express this component as the sum of RMS errors between adjacent
  images in the gamma space. This component is constant per number of images and layout in gamma space.
  b. The second component is network-specific, and concerns the capability of the network itself given a task.
  We can get a sense of the performance of a network by taking the average L2 loss over all images the network
  is trying to memorize. Subtracting this loss from the generalization constant gives us an idea of network
  generalization.
  To run the program, use `python3 train_ims.py`. We print only average total loss, as the task-based component
  is constant.

3. a. Upsampling the images works by supplying the --upsample N argument, where N=the desired width.
      An example can be viewed in `imgs/48_n1_up256/`. To focus on one image, one can use the `--num_images`
      command.

   b.
      i. RMS error would suggest the two best candidates for interpolation. SSIM would be another 
         measure. In general, it would make sense to interpolate two similar-looking images.
         To interpolate 2 images, we use the `--img-list X Y` command to choose index numbers between
         0 and 99 to refer to the images. To get the full list of images, use the `--create-list` argument.
         Finally, add `--interpolate` to cause interpolation images to be created.
      ii. I chose 3 pairs:
        shopping cart and shopping_car_loaded,
        price_tag_usd and price_tag_euro
        cathedral and bureau
      iii. Demonstration of the results can be viewed in the `imgs/48_l29-35_up256_i`,
        `imgs/48_l30-95_up256_i` and `imgs/48_l79-94_up256_i` directories.
        In all instances, we interpolate over 1000 epochs between the first image (0) and the second image
        (1)
      iv. The results are unsatisfactory. As soon as we move in gamma space away from one image,
        we get noise rather than getting a mix between the images.
        The reason is probably that even though semantically we think the images should mix in a way that
        makes sense from an image-based perspective (having one image fade out and the other fade in),
        the sine activation function-based neurons don't necessarily mix in that way.
        Instead, they can rearrange themselves in unintuitive ways before reaching their final state.
        I thought this may have to do with the sine-based gradient allowing movement in multiple directions,
        and therefore created `test_simultaneous.py` which modifies the algorithm to train the 3d-space
        at once. I saw no improvement from this approach though.
        The big mystery from my perspective is why interpolating in the x-y plane works, while interpolating
        in the gamma plane does not. I *think* some of this has to do with the smoothness between adjacent
        pixels and the regular pattern of the pixels. 

5. a. An improved algorithm would force the network to retain a similarity to the images even when
      it's between two images in the gamma space.
   b-c. We would train the images with the same L2 loss, but we would also add a second term
      when training between images. This L2 term would consist of a mix of d(out, image1) and d(out, image2).
      The two terms would be mixed in proprotion to the distance along the gamma dimension. This would
      make the network stay true to the images when in-between them on the gamma space.
        
        
        
        






