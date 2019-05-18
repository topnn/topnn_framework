Framework description and usage (Beta) 
--------------------------------

![topology change](https://github.com/topnn/topnn_framework/blob/master/pics/topology_change.png)


# Requirements
Python 3.5,
Tensorflow,
Julia 0.64, 
[Eirene](https://github.com/Eetion/Eirene.jl) topological data analysis package (update `.julia_include/include_eirene_run.jl` with the link to local Eirene installation)

# Training a model
* Use `compute.py` running  `train_2_excel_2D` pipeline for two dimensiona data set and 'train_2_excel_3D' for three dimensional datasets.
* Specifies which dataset (D-I "circles_type_8", D-II "rings_9", D-III "spheres_9") and what architecture to use (size of the network and the activation type)
* Set the number of training epochs, learning rate
* Set the overall number of trials. 

For each successful training attempt (perfect classification of the dataset) log file (`/data/<dataset-type>/<architecture>/good_results.txt`) is update. And the resulting model and output of each layer of the network is saved in a new folder tyled by the current time (`/data/<dataset-type>/<architecture>/<data>`). Betti numbers calculation used log file and acessed the saved model to run Betti numbers calculation.   

## Example: 
Train on D-II dataset, using 10 layers with 25  neurons, with `LeakyRelu` activation. With 0.2 learing rate and 12000 training epochs, run 80 trials and report progress every 1000 epochs.  

`computer.py --pipeline_name train_2_excel_3D          --output ./data/rings_9/ --input-tf-dataset ./data/rings_9/rings_9.tfrecords --model 10_by_25 --activation_type LeakyRelu --trials 80 --learning_rate 0.02 --training_epochs 12000 --summary_freq 1000`

# Calculate Betti numbers

* Use `compute.py` running  `texcel_2_betti` pipeline
* Calculated Betti numbers for succesfuly trained neural networks with given architecture and activation type. Split the calculation in parallen on 10 cores. Ecah core limited to 10 Gb memory.  

* Run `computer.py` with appropriately set parametrs: number of neighbors for nearest neighbor graph construction  ( the scale at wich to build Vietoris-Rips complex is fixed in Eirene call at `./julia_include/julia_aux2`.  

## Example: 
Calculate Betti numbers for networks of size 10 layers with 25 neurons, with `LeakyRelu` activation, trained on D-II dataset. The calucaltion proceed on 1/4 subsample of the data, and compute Betti numbers zero and one. 

`computer.py --pipeline_name excel_2_betti --output ./data/rings_9/ --input-tf-dataset ./data/rings_9/rings_9.tfrecords --model 10_by_25 --activation_type LeakyRelu --trials 80 --cat2  --divisor 4 --neighbors 35 --betti_max 1 --read_excel_from LeakyRelu`

# Visualize Dataset
* Run `computer.py` with appropriate pipeline (e.g. `visualize_tfrecords.py`)
* Check generated `.html` file with plotly dataset plot. 


### Trials: 

Simulation results are large (100Gb) will be uploaded soon. 
