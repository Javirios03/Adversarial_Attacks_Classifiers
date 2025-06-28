# One-Pixel Attack on CIFAR-10 – Robustness of CNN Classifiers

> **Bachelor thesis project – Francisco Javier Ríos Montes (ICAI, 2025)**

This repository contains all the code, pretrained weights and supporting material used in the thesis *“Robustness of Classifiers against Targeted Attacks with One-Pixel Attack”*. The project reproduces – and extends – the original One-Pixel Attack (OPA) and analyses how model accuracy and receptive-field size affect adversarial robustness on CIFAR-10 CNN classifiers.

---

## 1  Repository layout

| Path                                         | Purpose                                                        |
| -------------------------------------------- | -------------------------------------------------------------- |
| `/src/`                                    | Training & attack source code (Python)                         |
| `/models/`                                 | Pre-trained weights tracked with Git LFS                       |
| `/notebooks/`                              | Jupyter notebooks for exploratory work / visualisation         |
| `/figures/`                                | Graphs exported from the thesis for quick reference            |
| `/docs/`                                   | Additional documentation – incl. this README & the thesis PDF |
| `TFG - Ríos Montes, Francisco Javier.pdf` | Full thesis (root for citation convenience)                    |

*The original folder structure is preserved; the new folders above are **additions** only.*

---

## 2  Quick-start guide

```bash
# 1  Clone & install (Python ≥3.10 recommended)
$ git clone https://github.com/Javirios03/Adversarial_Attacks_Classifiers.git
$ cd Adversarial_Attacks_Classifiers
$ pip install -r requirements.txt            # will be uploaded separately

# 2  Initialise Git LFS (needed for >100 MB weights)
$ git lfs install
$ git lfs pull                                # fetches *.pth files in /models/

# 3  Download CIFAR-10 automatically on first use

# 4  Run an example attack (untargeted AllConv baseline)
$ python src/attack.py \
        --model allconv_baseline \
        --weights models/allconv_baseline.pth \
        --mode untargeted
```

### 2.1  Training from scratch

```bash
$ python src/train.py --model allconv --epochs 150 --save models/allconv_custom.pth
```

Training logs and checkpoints are saved under `/results/` by default.

### 2.2  Reproducing thesis figures

All plotting scripts live in `src/visualisation/` and read experiment logs created by `train.py` / `attack.py`. Run e.g.

```bash
$ python src/visualisation/plot_success_rates.py --logdir results/allconv_baseline
```

---

## 3  Pre-trained weights (Git LFS)

The 130 MB VGG‑16 checkpoint exceeds GitHub’s 100 MB hard limit; therefore **all** `.pth` files are tracked with [Git LFS](https://git-lfs.github.com/).

```
git lfs track "*.pth"
```

If Git LFS is not an option for you, each weight file can also be downloaded from the **Releases** page or regenerated with the training scripts.

| File                     | Size    | Top‑1 Acc. (%) |
| ------------------------ | ------- | --------------- |
| `allconv_baseline.pth` | 10 MB  | 85.6            |
| `nin_baseline.pth`     | 10 MB  | 87.2            |
| `vgg16_baseline.pth`   | 130 MB | 83.3            |
| `allconv_highacc.pth`  | 11 MB  | 89.96           |
| `vgg16_highacc.pth`    | 132 MB | 89.66           |

---

## 4  Experimental results (summary)

| Model   | Variant       | Acc. (%)        | Targeted OPA ↓   | Untargeted OPA ↓ |
| ------- | ------------- | --------------- | ----------------- | ----------------- |
| AllConv | baseline      | 85.6            | **4.44 %** | **28 %**   |
| NiN     | baseline      | 87.2            | 6.56 %           | 38.67 %          |
| VGG‑16 | baseline      | 83.3            | 7.48 %           | 41 %             |
| AllConv | + accuracy    | **89.96** | 2.83 %           | 20 %             |
| VGG‑16 | + accuracy    | **89.66** | 6.22 %           | 33.5 %           |
| AllConv | k = 3 (REF) | –              | 3.06 %           | 22 %             |
| AllConv | k = 5       | –              | 3.78 %           | 24.5 %           |
| AllConv | k = 7       | –              | 5.00 %           | 31 %             |

### Key take‑aways

* Single‑pixel perturbations suffice to fool deep CNNs.
* Improving clean‑accuracy **can** mildly improve robustness provided over‑fitting is controlled.
* Enlarging the theoretical receptive field alone **decreases** robustness unless the effective receptive field grows accordingly.

See Chapter 5 of the thesis for full tables and per‑class breakdowns.

---

## 5  Key concepts

* **One‑Pixel Attack** – Black‑box, Differential‑Evolution‑based optimiser that tweaks exactly one pixel.
* **Targeted vs. Untargeted** – Push image to a chosen label vs. any label except the true one.
* **TRF vs. ERF** – Theoretical vs. Effective Receptive Field; only the latter truly matters for robustness.

---

## 6  Thesis & further reading

The complete thesis can be found at the root of this repo as `TFG - Ríos Montes, Francisco Javier.pdf`. It contains detailed methodology, the additional studies and discussion.

---

## 7  Acknowledgements

> *I would like to thank my supervisor, **Emanuel**, for his patience and guidance throughout these months, as well as my parents for their continuous support. This project is dedicated to my late grandfather, **Agustín**, for his advice and counsel during these four years, without which I wouldn’t have been able to complete this thesis.*

---

## 8  License

This work is released under the **MIT License** – see `LICENSE` for full text.

---

## 9  Contributing

Contributions are welcome! Please open an issue first to discuss your proposal. When submitting a PR:

1. Follow PEP‑8.
2. Include unit tests where applicable.
3. Update documentation and examples.

---

### Contact

For questions, open an issue or reach me at **javier.riosmontes@gmail.com**
