# Implemetation of Q_learning algorithm
## Solve cart-pole problem in gym environment using Q_learning algorithm
### Description of the problem

    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.
 
    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf
    
    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right
 
    Reward:
        Reward is 1 for every step taken, including the termination step
    
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 500.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.

### Soution
    To solve this problem we will use reinforecement learning model (Q_learning algorithm).
  
  #### Reinforcement learning
      Reinforcement learning is ML model that help the agent to make a decission just by given him a reward, like in our project
      we give the agent +1 for every good move and -300 for a bad move so the agent will make just the  moves that give him a good reward.

### Q_learning algorithm
      this algorithm helps us to update our q_learning table using belman equation.
      we make a table where we save our move that we make in every situation and we give it a value (TABLE[situation][action_chosed] = value) describe how         mutch this value is good using BELMAN equation then in the end in every situation we chose action that have more value to reach the gool. And this
      our formula Q_table[state][action] = Q_table[state][action] + learning_rate * (reward + gamma * max(Q_table[next_state][:] -Q_table[state][action]).

## run the project
      To run the projet you should install gym try pip install gym
