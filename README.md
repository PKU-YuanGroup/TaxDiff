<h2 align="center"> <a href="hhttps://arxiv.org/abs/2402.16445"> TaxDiff: Taxonomic-Guided Diffusion Model for Protein Sequence Generation </a></h2>
<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2310.01852-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2401.03403)
[![License](https://img.shields.io/badge/Code%20License-MIT-yellow)](https://github.com/HowardLi1984/ECDFormer/blob/main/LICENSE)
[![HuggingFace](https://img.shields.io/badge/Hugging%20Face-TaxDiff%20-blue)](https://github.com/HowardLi1984/ECDFormer/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Dataset%20license-CC--BY--NC%204.0-orange)](https://github.com/HowardLi1984/ECDFormer/blob/main/DATASET_LICENSE) <br>

</h5>
If you like our project, please give us a star ⭐ on GitHub for latest update.
<p align="center">

    <img src="imgs/motivation.png" width="400" style="margin-bottom: 0.2;"/>
<p>
<h5 align="center"> The official code for "TaxDiff: Taxonomic-Guided Diffusion Model for Protein Sequence Generation",submitted to ICML2024. Here we publish the inference code of TaxDiff. The training code & Protein sequence with Taxonomic lables dataset will be released after our paper is accepted.  </h2>
<p align="center">
    <img src="imgs/archrtecture.png" width="700" style="margin-bottom: 0.2;"/>
<p>

<details open><summary>💡 I also have other video-language projects that may interest you ✨. </summary><p>
<!--  may -->

> [**ProLLaMA: A Protein Large Language Model for Multi-Task Protein Language Processing**](https://github.com/Lyu6PosHao/ProLLaMA) <br>
>Liuzhenghao Lv, Zongying Lin, Li Hao, Yuyang Liu, Jiaxi Cui, Calvin Yu-Chian Chen, Li Yuan, Yonghong Tian<br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/Lyu6PosHao/ProLLaMA)  
[![arXiv](https://img.shields.io/badge/Arxiv-2401.15947-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2402.16445) <br>



</p></details>


## 😮 Highlights



### 💡 Protein sequences Generation Model
- To the best of our knowledge, our TaxDiff is **the first controllable protein generation model** utilizing guidance from taxonomies.
### 🔥 Diffusion-based Framework
- TaxDiff proposes a **taxonomic-guided framework** that fits all diffusion-based protein design models. We also propose the patchify attention mechanism for better protein design.
### ⭐ Excellent performance
- Experiments demonstrate that our TaxDiff achieves state-of-the-art results in both taxonomic-guided controllable and unconditional protein sequence generation, excelling in structural modeling scores and sequence consistency.

## 🚀 Main Results

### Unconditional Generation
<p align="left">
<img src="imgs/unconditional.png" width=80%>
</p>

### Controllable Generation
<p align="left">
<img src="imgs/controllable.png" width=80%>
</p>

## 📖 Data Preparation

For inference, please download from [HuggingFace](https://huggingface.co/linzy19/TaxDiff/tree/main)  and put the [ckpt](https://huggingface.co/linzy19/TaxDiff/tree/main) into the folder ckpt/
```bash
ckpt/0012802.ckpt
```
We will release protein sequences with taxonmic labels for training procedure once our paper is accepted.

If you want to select a specific protein taxonomic for your research, you need to first find his corresponding  tax-id in the [data_reader/Taxonnmic_classfication.xlsx](https://github.com/Linzy19/TaxDiff/blob/main/data_reader/Taxonnmic_classfication.xlsx), and then modify protein class lables in the [sample_protein.py](https://github.com/Linzy19/TaxDiff/blob/main/sample_protein.py).
```bash
class_lables = torch.randint(low=1, high=int(23427), size=(1,num))
```

## 🛠️ Requirements and Installation
* Python == 3.10
* Pytorch == 2.2.0
* Torchvision == 0.17.0
* CUDA Version == 12.0
* Install required packages:
```bash
git clone git@[github.com/Linzy19/TaxDiff.git]
cd TaxDiff
pip install -r requirements.txt
```

## 🗝️ Inferencing
The inferencing instruction is in [sample_protein.py](sample_protein.py).
```bash
python sample_protein.py --model DiT-pro-12-h6-L16 --cuda-num cuda:0 --num 500
```

## ✏️ Citation
If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil:.

```BibTeX
@article{li2024deep,
  title={Deep peak property learning for efficient chiral molecules ECD spectra prediction},
  author={Li, Hao and Long, Da and Yuan, Li and Tian, Yonghong and Wang, Xinchang and Mo, Fanyang},
  journal={arXiv preprint arXiv:2401.03403},
  year={2024}
}
```
