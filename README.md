Author: Jakob Horvath, u1092049
Date: 12/11/2020

Final Project

** = slow runtime
Algorithm sub-folders:
    1. decision_tree    **
    2. perceptron       **
    3. svm_bagging      **
    4. logistic_regression
    5. log_reg_bagging
    6. random_forests

Due to the runtime of some classifiers,
each sub-folder has its own shell script.

Also, each algorithm expects the project
data folder to be at the first level of the
'project' directory, adjacent to the algorithm
folders.

Instructions to run classifiers:

   - navigate to the specific algorithm's base directory, 
        containing the '.py' program files
   - if 'run.sh' does not have executable privileges:
      - execute the command: 'chmod u+x run.sh'
   - run the program using: './run.sh'
