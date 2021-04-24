# streamlit-active-learning-dashboard

* A simple proof of concept of implementing Active Learning with Streamlit and modAL.

# Installation
* Create a Python 3.6.8 environment ( for example with Conda ).
* Activate the environment.
* Install dependencies into the environment:
    * pip install -r requirements.txt

# Execution
* Clone the repo.
* Complete the installation steps above, then from an activated environment, run:
    * streamlit run streamlit_active_learning_poc.py

# Dashboard Phases

## Phase-1 : Unsupervised Learning
* The first phase of the process involves clustering the dataset to identify any underlying structure prior to collecting any labels.
* If there is any structure to the data, then this process should hopefully identify that and diversify the pool of data.

![image info](/images/img1_clustering.png)

## Phase-2 : Initial Labeling
* The first phase of the process involves collecting labels actively from the user. 
* The purpose of this stage is to collect enough data for:
    * Seeding the active learning process, and
    * Continuously evaluating the quality of active learning on a test dataset.

![image info](/images/img2_initial_labeling.png)

## Phase-3.1 : Active Learning
* The first phase of the process involves collecting labels actively from the user. 

![image info](/images/img3_act_learning_eval.png)

## Phase-3.2 : Continuous Evaluation

![image info](/images/img3_act_learning.png)

# Learnings so far:
* <b>Streamlit </b>
    * State management in Streamlit needs additional work, and is in a state of flux currently. 
    * I had to search a lot to get the state management aspects of the Streamlit code to work. 
* <b>modAL </b>
    * Working with modAL has been a very pleasurable experience so far. 
    * I really enjoyed:
        * How easy it was to get a basic PoC going.
        * How extensible the package.
