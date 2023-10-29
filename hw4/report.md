# Problem 1: Train dynamics model

## n5_arch2x250
![](./data/hw4_q1_cheetah_n5_arch2x250_cheetah-cs285-v0_28-10-2023_19-23-27/itr_0_losses.png)
![](./data/hw4_q1_cheetah_n5_arch2x250_cheetah-cs285-v0_28-10-2023_19-23-27/itr_0_predictions.png)

## n500_arch2x250
![](./data/hw4_q1_cheetah_n500_arch2x250_cheetah-cs285-v0_28-10-2023_19-23-09/itr_0_losses.png)
![](./data/hw4_q1_cheetah_n500_arch2x250_cheetah-cs285-v0_28-10-2023_19-23-09/itr_0_predictions.png)

## n500_arch1x32
![](./data/hw4_q1_cheetah_n500_arch1x32_cheetah-cs285-v0_28-10-2023_19-22-53/itr_0_losses.png)
![](./data/hw4_q1_cheetah_n500_arch1x32_cheetah-cs285-v0_28-10-2023_19-22-53/itr_0_predictions.png)

## Discussion
The one with the larger num of train steps per iter and larger NN size performs the best. 


# Problem 2: Action selection using learned dynamics model and a given reward function

![](./imgs/q2.png)

# Problem 3: MBRL algorithm with on-policy data collection and iterative model training.
![](./imgs/q3_obstacles.png)
![](./imgs/q3_reacher.png)
![](./imgs/q3_cheetah.png)

early stop cheetah as reward reached

# Problem 4: MBRL Ablation studies
## Horizon
![](./imgs/q4_horizon.png)

Longer horizon seems to lead to poorer result. Doesnt seem intuitive, not v confident on this

## Numseq
![](./imgs/q4_numseq.png)

Higher numseq leads to higher performance, makes sense

## Ensemble
![](./imgs/q4_ensemble.png)

Larger ensemble leads to higher performance, makes sense

# Problem 5: Comparison against CEM


![](./imgs/q5a.png)
![](./imgs/q5b.png)

how does CEM affects

results for different numbers of sampling iterations (2 vs. 4).

# Problem 6: MBPO + SAC

TODO