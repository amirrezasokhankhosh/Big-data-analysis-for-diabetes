# Big Data Analysis on BRFSS data for Diabetes

This implementation consists of two parts: The VIPER algorithm and the DNN model.

The BRFSS_Binary notebook includes all the codes and outputs for the DNN model. There is no need for any modification for running it. If you are in need of just the trained model, my_keras_model.h5 is the the file you are looking for.

The split notebook is another DNN model that uses under sampling for illustration. This code illustrates that over sampling outperforms under sampling

For the VIPER algorithm, just the run each cell one after another to get the results. By running this notebook, the confidence.txt file will be generated which contains frequent pattern of sizes less than or equal to two. To get patterns of greater sizes, change the value of `k` to the desired length.

We have also included the studied data in this repository.