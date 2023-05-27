# RAPPO
This is an official implementation of **Revisiting Domain Randomization via Relaxed State-Adversarial
Policy Optimization (ICML 2023)**. 

**Abstract**
Domain randomization (DR) is widely used in reinforcement learning (RL) to bridge the gap between simulation and reality by maximizing its \emph{average returns} under the perturbation of environmental parameters. However, even the most complex simulators cannot capture all details in reality due to finite domain parameters and simplified physical models. Additionally, \pch{the existing} methods often assume that the distribution of domain parameters \pch{belongs to} a specific family of probability functions, such as normal distributions, which may not be correct. To overcome these limitations, we propose a new approach to DR by rethinking it from the perspective of \emph{adversarial state perturbation}, without the need for reconfiguring the simulator or relying on prior knowledge about the environment. We also address the issue of over-conservatism that can occur when perturbing agents to the worst states during training by introducing a \emph{Relaxed State-Adversarial Algorithm} that simultaneously maximizes the average-case and worst-case \yushuen{returns}. We evaluate our method by comparing it to state-of-the-art methods, providing experimental results and theoretical proofs to verify its effectiveness.

