# Revisiting Domain Randomization via Relaxed State-Adversarial Policy Optimization 
[**Video**](https://recorder-v3.slideslive.com/?share=83063&s=ed2dd1ed-f20d-4558-bdd9-0edbaf19a071)


This is an official implementation of [**Revisiting Domain Randomization via Relaxed State-Adversarial
Policy Optimization (ICML 2023)**](https://proceedings.mlr.press/v202/lien23a/lien23a.pdf). 

**Authors:** Yun Hsuan Lien, Ping Chun Hsieh, Yu Shuen Wang

## Abstract

Domain randomization (DR) is widely used in reinforcement learning (RL) to bridge the gap between simulation and reality by maximizing its **average returns** under the perturbation of environmental parameters. However, even the most complex simulators cannot capture all details in reality due to finite domain parameters and simplified physical models. Additionally, the existing methods often assume that the distribution of domain parameters \pch{belongs to} a specific family of probability functions, such as normal distributions, which may not be correct. To overcome these limitations, we propose a new approach to DR by rethinking it from the perspective of **adversarial state perturbation**, without the need for reconfiguring the simulator or relying on prior knowledge about the environment. We also address the issue of over-conservatism that can occur when perturbing agents to the worst states during training by introducing a **Relaxed State-Adversarial Algorithm** that simultaneously maximizes the average-case and worst-case returns. We evaluate our method by comparing it to state-of-the-art methods, providing experimental results and theoretical proofs to verify its effectiveness.

<p align="center">
  <img src="https://github.com/sophialien/RAPPO/blob/main/RAPPO.png" width="800" />
</p>

## Citation
If you find this work useful in your research, please consider citing:
```
@inproceedings{lien2023RAPPO,
 author={Yun-Hsuan Lien, Ping-Chun Hsieh, Yu-Shuen Wang},
 booktitle={International Conference on Machine Learning(ICML)},
 year={2023}
 }
```

## Contact Information
If you have any questions, please contact Sophia Yun-Hsuan Lien: sophia.yh.lien@gmail.com
