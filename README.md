# Q-learning-cartpole

### Training a acrtpole to balance itself in [OpenAI's Gym](http://gym.openai.com/) environment using two approaches:
1. The first approach involves using only one neural net that is both used for predicting the target value and prediction value (*created without machine learning library*)
2. The second approach uses two networks in which one predicts the target and the second  one predicts next move (*created using tensorflow for simplicity*)

Both of them incorporates reinforcement learning theory to the structure. Instead of traditional Q-learning that uses a Q-table to map state and action to Q value, a deep Q-network will be used to map state to Q value and action at the same time.

---
### Simplified Training Steps:
1. Initialize weights in the network
2. Choose an action by forward propagation (prediction)
3. Choose an action (target)
4. Update the network's output using the Bellman equation
5. Retrain network by backward propagation

### Demonstration:
![Single network trained around 150 episodes](https://media.giphy.com/media/6kIuZ8Mjnb4o2i7ro8/source.gif?cid=790b7611ccbcbe855f1511b0fcc7ba7e3fe8f5fc4d301cd0&rid=source.gif&ct=g)

*Single network trained around 150 episodes*

![Double network trained around 200 episodes](https://media.giphy.com/media/R30MAxt76V7cHCFona/source.gif?cid=790b761155f7a6263af6b86d24f3dff38dc5b56e57c7e702&rid=source.gif&ct=g)

*Double network trained around 200 episodes*

