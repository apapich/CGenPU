CGenPU - Conditional Generative PU framework
========

|License Badge|

The conditional Generative PU framework (CGenPU) is a novel approach to learning the positive and negative data distributions from the positive and unlabeled dataset. The framework is based on the Auxiliary Classifier GAN (AC-GAN) [1]_, generating class-conditional examples. CGenPU reuses AC-GAN's framework and training scheme. However, to discriminate positive and negative examples, a novel auxiliary objective function is implemented. The experimental evaluation has shown that CGenPU requires fewer labeled training examples than existing methods. On top of that, CGenPU is the first single-stage generative framework for positive and unlabeled learning. For more details, please, refer to the article Papič et al. [2]_.

.. figure:: docs/images/arch.png
   :alt: Conditional Generative Positive and Unlabeled Learning
   :align: center

   The scheme of the CGenPU learning framework, with a class-conditional generator and an auxiliary classifier (seen in dotted boxes on the right-hand side). Solid arrows indicate the flow of data within the framework, while the red dashed lines refer to the flow of labels. The framework's output is i) a generator capable of generating artificial positive and negative data, and ii) a binary classifier capable of classifying in positive or negative class.
   
The article is available under open access at the following [URL](https://www.sciencedirect.com/science/article/pii/S0957417423005481).

Results CIFAR-10
--------

Generated artificial positive (car) and negative (ship) examples using the CIFAR-10 dataset [3]_. During the experimental evaluation, we varied the number of labeled examples. Images in both left and right columns at matching positions have the same latent vectors but different input labels. Red boxes indicate out-of-distribution examples.

.. figure:: docs/images/cifar10_50.png
   :alt: CIFAR10 50 labeled examples
   :align: center

   50 labeled positive examples.

.. figure:: docs/images/cifar10_25.png
   :alt: CIFAR10 25 labeled examples
   :align: center

   25 labeled positive examples.

.. figure:: docs/images/cifar10_10.png
   :alt: CIFAR10 10 labeled examples
   :align: center

   10 labeled positive examples.


Installation
--------

The project requires Python 3.8 or higher in order to run.

.. code::

    matplotlib==3.3.1
    numpy==1.19.5
    tensorflow-gpu==2.4.0
    tqdm==4.58.0
    wandb==0.10.21

The CGenPU framework can be installed from the source.

.. code::

    python setup.py install

Usage
--------

Getting started with CGenPU is very simple. Follow the template below.

.. code:: python

    from pu.CGenPU import CGenPU

    # Load data
    xp, xu = <put your code>

    # Parameters
    input_dim = <put your code>
    latent_dim = <put your code>

    # Load models
    D = <put your code>
    A = <put your code>
    G = <put your code>

    # Initialize framework
    g = CGenPU(input_dim, latent_dim, (D, A, G))
    # Start training process
    g.fit(xp, xu, epochs=<put your code>)

Benchmark example on synthetic moon dataset is available `HERE <./examples/basic.py>`_.


Citation
--------

If you use CGenPU for your work, we would appreciate it if you would cite the paper.

.. code::

   @article{PAPIC2023120046,
      title = {Conditional generative positive and unlabeled learning},
      journal = {Expert Systems with Applications},
      pages = {120046},
      year = {2023},
      issn = {0957-4174},
      doi = {https://doi.org/10.1016/j.eswa.2023.120046},
      url = {https://www.sciencedirect.com/science/article/pii/S0957417423005481},
      author = {Aleš Papič and Igor Kononenko and Zoran Bosnić},
   }

References
--------

.. [1] Augustus Odena, C. Olah, and J. Shlens, "Conditional image synthesis with auxiliary classifier gans," in ICML, 2017, pp. 2642–2651.
.. [2] Papič, Aleš, Igor Kononenko, and Zoran Bosnić. "Conditional generative positive and unlabeled learning." Expert Systems with Applications (2023): 120046.
.. [3] Alex Krizhevsky, "Learning multiple layers of features from tiny images," University of Toronto, Tech. Rep., 2009.

.. |License Badge| image:: https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg
   :target: https://creativecommons.org/licenses/by/4.0/
