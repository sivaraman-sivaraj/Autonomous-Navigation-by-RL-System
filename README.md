# Autonomous-Navigation-by-RL-System
#### Agents ####
This folder contains agents(different ship models) and it's evaluation.
  1. Nomoto Model - one degree (Analytical and Numerical)
  2. Mariner ship and Container Ship
  3. MMG Model ship  (Full scale Model and L7 Model)
#### DQN - Deep Q Network ####
DQN FFNN version 0.1 - It is having fully designed model (episode terminating creteria part in environment needs to be fine tuned before the start of the training) and the file Google_Colab_Codes.py is written based on training with GPU.

DQN FFNN version 0.2 - changes :  Mean Square Error - Memory size of latest 2000 - decaying epsilon - 9 different actions(-35,-20,-10,-5,0,..) - Reward based on CTE and HE 

#### LOS Evaluation ####
This Contains evaluation of designed LOS by PD and PID Controller
  1. PD controller
  2. PID controller version 2.0 (works for all quadrants and all trajectory)
  
 #### Matching Model with Yasukawa's MMG paper ####
 Yasukawa's MMG Model Image was digitized and compared with designed L7 Model.
 
 #### Tabular Solution Methos ####
 Conventional grid world approach with Nomoto one degreee, environment has land, greeen water.,etc 
 
 
