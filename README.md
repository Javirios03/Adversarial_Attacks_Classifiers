# Adversarial Attacks on Image Classifiers

Repo for all codes related to my Bachelor Thesis in the major _Mathematical Engineering and Artificial Intelligence_, which I studied in Universidad Pontificia Comillas ICAI. Its purpose is to study the current knowledge shared by the ML community about adversarial attacks on image classifiers. Specifically, it focuses on the findings by Su et al. ([[1]](#1)) on One-Pixel Attacks towards DNNs, whose success poses a dangerous situation, given the vulnerability of _SOTA_ classifiers.

After replicating the results stated in the original paper, the main intention is to assess whether this, certainly, limited attack may be extrapolated to other architectures, as well as if the use of bigger sample sizes' (this is, using images with a higher resolution) diminishes, radically, the power of these attacks.

## Getting Started

First of all, clone the repo:

    git clone https://github.com/Javirios03/Adversarial_Attacks_Classifiers.git

### Prerequisites

Even though, for storage's sake, the data isn't explicitly provided, a _data_ folder must be present in the root directory, in which PyTorch will install the Cifar-10 dataset. Added to this, Python and Pip are required

### Installing

Having cloned the repository, we must go on and install the dependencies:

    pip install -r requirements.txt

## Deployment

## Built With

## Contributing

## Versioning

## Authors

- **Francisco Javier Ríos**

## License

## Acknowledgments

## References

<a id="1">[1]</a>
Su et al. (2017).
One Pixel Attack for Fooling Deep Neural Networks
_IEEE Transactions on Evolutionary Computation_
https://arxiv.org/abs/1710.08864
