# Q-learning-cartpole

### Training a cartpole to balance itself in [OpenAI's Gym](http://gym.openai.com/) environment using two Q-learning approaches plus one genetic approach:
1. The first approach involves using only one neural net that is both used for predicting the target value and prediction value (*created without machine learning library*)
2. The second approach uses two networks in which one predicts the target and the second  one predicts next move (*created using tensorflow for simplicity*)

Both of them incorporates reinforcement learning theory to the structure. Instead of traditional Q-learning that uses a Q-table to map state and action to Q value, a deep Q-network will be used to map state to Q value and action at the same time.

For genetic algorithm, I found out that the number of generation to solve the puzzle is highly unstable due to the mutation of genes. One time I was really lucky and the algorithms solves it using only 2 generations. In this approach, I used pytorch to create a neural net, and then using the `PyGad` library to convert Pytorch weights to 1D array, and mutate it by adding `np.random.rand()` to each weight by mutation rate. Then convert the 1D array back to Pytorch weights and change the network weight. In this problem, the genetic algorithm outperforms Q-Learning algorithm in both training time and model accuracy. 

---
### Simplified Training Steps:
1. Initialize weights in the network
2. Choose an action by forward propagation (prediction)
3. Choose an action (target)
4. Update the network's output using the Bellman equation ![](https://miro.medium.com/max/700/1*CLBIXdpk8ft0-1MFH8FwUg.png)
5. Retrain network by backward propagation

### Hyperparameters for Genetic Solution:
```
POPULATION_SIZE = 20
MUTATION_RATE = 0.2
```

### Demonstration:
![Genetic Solution](https://media.giphy.com/media/rmDKU2ql5obG9eqonj/giphy.gif)

*Solution using genetic solution around 30 generations*

![Single network trained around 150 episodes](https://media.giphy.com/media/6kIuZ8Mjnb4o2i7ro8/source.gif?cid=790b7611ccbcbe855f1511b0fcc7ba7e3fe8f5fc4d301cd0&rid=source.gif&ct=g)

*Single network trained around 150 episodes*

![Double network trained around 200 episodes](https://media.giphy.com/media/R30MAxt76V7cHCFona/source.gif?cid=790b761155f7a6263af6b86d24f3dff38dc5b56e57c7e702&rid=source.gif&ct=g)

*Double network trained around 200 episodes*
