# PsyTrack_Analysis
PsyTrack_Analysis main goal is inferring the trajectory of sensory decision-making strategies.  
The analysis provides a unique point of view of mice learning process, by visualize changes in behavioral strategies.  
Unlike standard psychophysical models, which assume that weights are constant across time, this model assume that the weights evolve gradually over time.
The optimization process is baesd on a unique varitaion of the logistic regression model developed and intorduced by [Roy, N. A., Bak, J. H., Laboratory, T. I. B., Akrami, A., Brody, C. D., & Pillow, J. W. (2021). Extracting the dynamics of behavior in sensory decision-making experiments](https://www.sciencedirect.com/science/article/pii/S0896627320309636)

The analysis includes preprocessing and modeling raw behavioural data of mice learning process. Followed by inferring weight trajectories using PsyTrack Python library and visualize the results.
In addition, this projects provides advanced analsis on the weight trajectories output, such as weights aligments per level for patterns recognition, and correlation test between different weights using linear reggresion.  

Capturing trial-to-trial fluctuations in the weighting of sensory stimuli, bias, and stimulus history, provides a delicate observation that allows the researchers to examine the brain in specific time slots corresponds to the analysis results.


**Analysis Results:**
- Psytrack weights   
![image](https://user-images.githubusercontent.com/83977654/127745467-4d9e0a95-311b-468d-ba50-33b056a5ecea.png)  
  

  
**Additional analysis done on the Psytrack weights output:**  
- Weights alignment per level  
![image](https://user-images.githubusercontent.com/83977654/127771357-1a30733d-1ded-4d73-88ea-0360211b9b05.png)
- Weights correlation per level, using linear regression:  
![image](https://user-images.githubusercontent.com/83977654/127769639-52342705-664b-49fe-9a1a-e03294a51760.png)  
- Mouse Lick rate  
![image](https://user-images.githubusercontent.com/83977654/127771083-dabd59b1-1809-4f96-a888-e2b820242129.png)




