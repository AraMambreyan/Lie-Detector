# Lie Detection 

This repository shows that modern lie detection algorithms claiming accuracies as high as
99% are biased.

## Background and Gender Bias

Recently, many papers were written claiming almost perfect (>95%) accuracies for lie detection.
Some of these papers have gotten significant press coverage. The [GitHub repository](https://github.com/Doubaibai/DARE)
of [[2]](#2), which claims around 92% AUC metrics, has 94 stars and 26 forks.

At first glance, it may seem that we are living in a Black Mirror-esque world and, after a couple of years, we
will all be wearing Google glasses which will make your friend think twice before lying that "you look good".

The reality is that these algorithms, at least the ones that we have seen, do not learn to distinguish dishonesty. 
Instead, they learn incidental properties of the dataset.

These papers all used the popular real-life trial dataset [[1]](#1) which, at the moment, has 112 citations. 
However, the dataset is small in size (121 data points) and there is significant gender bias as females
in the dataset lie more frequently than males.

## Experiments
 
To show that these algorithms are biased, we train the classifiers to learn gender. Then, for
a test data point, if the algorithm predicts "female", we assign the label "lie" to it and vice-versa.

Using IDT features similarly to [[2]](#2), we achieve comparable results with this ad-hoc method. The notebook
for this experiment is `IDT.ipynb`. 

Using manually annotated micro-expressions, we achieve 65% using gender labels whereas training on lie/truth labels
78% is achieved which shows that even manually annotated micro-expressions are implicitly correlated with gender.
The experiment for this is in `micro-expressions.ipynb`.

The `helpers.py` contains useful functions for visualizing and running experiments. We needed fine-grained control
on the way cross-validation was run so we implemented  our own functions instead of using `sklearn`'s.

## Paper

To support our points with references, provide more detailed background and explain the way we conducted the 
experiments, we will upload a paper/report to this repository in the coming months once it is ready.

## Running

You need `numpy`, `sklearn`, `matplotlib` and `pandas` libraries
to run the code. Go to a directory where you 
want to clone the repository and run:

`git clone https://github.com/AraMambreyan/LieDetector-IDT.git`

Open `IDT.ipynb` or `micro-expressions.ipynb` notebooks. Change the `run_gender` variable to the experiment
you'd like to run. Then click `run all` and our results will be reproduced.

## References
<a id="1">[1]</a> 
R. Mihalcea, V. P´erez-Rosas, M. Abouelenien and
M. Burzo. Deception detection using real-life trial data.
In *Proceedings of the 2015 ACM on International Conference on Multimodal Interaction*, page 59–66, 2015.

<a id="2">[2]</a> 
 Z. Wu, B. Singh, L. S. Davis, and V. S. Subrahmanian. Deception Detection in Videos.
In *AAAI*, 2018.