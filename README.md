# Federated Clinical Concept and Disease Semantic Learning for Congenital Heart Disease Diagnosis

> Wenke Huang, Yangxu Liao, Wenjia Lei, Guancheng Wan, Xuankun Rong, Chi Wen, He Li, Mang Ye, Qingqing Wu, Bo Du

> *npj Digital Medicine, 2026*  

## 🙌 Abstract
Effective first-trimester screening for congenital heart disease (CHD) remains an unmet clinical need, hindered by technical constraints and the lack of validated diagnostic tools. While artificial intelligence (AI) offers promise, its progress is restricted by data scarcity and privacy concerns surrounding data sharing. Federated learning (FL) offers a promising paradigm for collaborative model training without exposing sensitive patient data. In this study, we establish a Federated Congenital Heart Disease Learning to enable cross-hospital collaboration in early CHD diagnosis. A major challenge arises from inter-hospital heterogeneity, where variations in ultrasound devices, scanning protocols, and patient demographics lead to significant feature distribution shifts, resulting in poor performance. To address this, we introduce federated prototypes that align both clinical concept and disease subtype representations across participating sites, effectively calibrating local updates and enhancing global consistency. Experiments conducted across four tertiary hospitals demonstrate that our method achieves a 10.3\% improvement in F1 score, 5.1\% increase in sensitivity, and 1.0\% improvement in specificity over state-of-the-art federated approaches. These results highlight our effectiveness in improving generalization under real-world clinical heterogeneity.

## 🥳 Citation

If you find this repository helpful for your research, we would greatly appreciate it if you could cite our papers. ✨

```bibtex
@article{FLCHD_npjDM26,
  title={Learn from Downstream and Be Yourself in Multimodal Large Language Models Fine-Tuning},
  author = {Huang, Wenke and Liao, Yangxu and Lei, Wenjia and Wan, Guancheng and Rong, Xuankun and Wen, Chi and Li, He and Ye, Mang and Wu, Qingqing and Du, Bo},
  journal={npj Digital Medicine},
  year={2026}
}
@article{FLSurveyandBenchmarkforGenRobFair_TPAMI24,
    title={Federated Learning for Generalization, Robustness, Fairness: A Survey and Benchmark},
    author={Wenke Huang and Mang Ye and Zekun Shi and Guancheng Wan and He Li and Bo Du and Qiang Yang},
    journal={TPAMI},
    year={2024}
}
@article{FCCLPlus_TPAMI23,
  title={Generalizable Heterogeneous Federated Cross-Correlation and Instance Similarity Learning}, 
  author={Wenke Huang and Mang Ye and Zekun Shi and Bo Du},
  year={2023},
  journal={TPAMI}
}
@inproceedings{FPL_CVPR2023,
    author    = {Huang, Wenke and Mang, Ye and Shi, Zekun and Li, He and Bo, Du},
    title     = {Rethinking Federated Learning with Domain Shift: A Prototype View},
    booktitle = {CVPR},
    year      = {2023}
}
@inproceedings{FCCL_CVPR22,
    title={Learn from others and be yourself in heterogeneous federated learning},
    author={Huang, Wenke and Ye, Mang and Du, Bo},
    booktitle={CVPR},
    year={2022}
}
```

## 💼 Relevant Projects 

[3 Federated Learning for Generalization, Robustness, Fairness: A Survey and Benchmark - TPAMI 2024 [[Link](https://ieeexplore.ieee.org/document/10571602)] [[Code](https://github.com/WenkeHuang/MarsFL)]

[2] Rethinking Federated Learning with Domain Shift: A Prototype View - CVPR 2023 [[Link](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Rethinking_Federated_Learning_With_Domain_Shift_A_Prototype_View_CVPR_2023_paper.pdf)] [[Code](https://github.com/WenkeHuang/RethinkFL)]

[1] Learn from Others and Be Yourself in Heterogeneous Federated Learning - CVPR 2022 [[Link](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Learn_From_Others_and_Be_Yourself_in_Heterogeneous_Federated_Learning_CVPR_2022_paper.pdf)][[Code](https://github.com/WenkeHuang/FCCL)]
