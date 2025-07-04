**Notice:** This repo is under development

# VICCA
VICCA: Visual Interpretation and Comprehension of Chest X-ray Anomalies in Generated Report Without Human Feedback [\[paper\]](https://doi.org/10.1016/j.mlwa.2025.100684)



## Authors

- [Sayeh GHOLIPOUR PICHA](https://www.github.com/sayeh1994)
- Dawood Al Chanti
- Alice Caplier
## Acknowledgements

 This repo build upon the awesome work of the following repos:
 - [Grounding Dino](https://github.com/IDEA-Research/GroundingDINO.git)
 - [Open-GroundingDino](https://github.com/longzw1997/Open-GroundingDino.git)
 - [ControlNet](https://github.com/lllyasviel/ControlNet.git)
 - [DETR](https://github.com/facebookresearch/detr.git)
 - [Lung Segmentation](https://github.com/IlliaOvcharenko/lung-segmentation.git)

Please cite them as well if you found this code useful.
## Installation

[Under Development]

```bash
  python 
```
    
## Badges

<!--- Add badges from somewhere like: [shields.io](https://shields.io/) -->

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)


## Run Locally

Clone the project

```bash
  git clone https://github.com/sayeh1994/vicca.git
```
Download the weights and checkpoints from [here](https://drive.google.com/file/d/1BvtTQG9gn_9PlLjS9p2EQJJuFoBWKawf/view?usp=sharing).

Please merge the folders from the downloaded file to the project folder.

Go to the project directory and install the new environment from the yml file:

```bash
  conda env create -f environment.yml
```

Run the inference.py

```bash
  python inference.py \
    --image_path VG/38708899-5132e206-88cb58cf-d55a7065-6cbc983d.jpg \
    --text_prompt "Cardiomegaly with mild pulmonary vascular congestion."
```


## Training
In VICCA, the models were trained separately and later integrated for the final assessment. Joint training of all components is currently under development. If you wish to train the models independently for your own objectives, please follow the instructions in the **Training** section of each Model.
