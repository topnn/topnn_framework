# Framework description and usage (Beta) 

## ![#f03c15](https://placehold.it/15/f03c15/000000?text=+) ![#1589F0](https://placehold.it/15/1589F0/000000?text=+) ![#f03c15](https://placehold.it/15/f03c15/000000?text=+) Anonymized version for NeurIPS review ![#f03c15](https://placehold.it/15/f03c15/000000?text=+) ![#1589F0](https://placehold.it/15/1589F0/000000?text=+) ![#f03c15](https://placehold.it/15/f03c15/000000?text=+)

--------------------------------------------

![topology change](https://github.com/topnn/topnn_framework/blob/master/pics/topology_change.png)    
        
        
# Requirements
Python 3.5,
Tensorflow,
Julia 0.64,
[Eirene](https://github.com/Eetion/Eirene.jl) topological data analysis package
(update `./julia_include/include_eirene_run.jl` with the link to local Eirene installation, as well as `./transformers/excel_to_betti_feeder_parallel.py` with a link to Julia executable)

# Training a model
* Use `compute.py` running  `train_2_excel_2D` pipeline for two dimensional data sets and `train_2_excel_3D` for three dimensional data sets.
* Specify which data set to use (D-I `circles_type_8`, D-II `rings_9`, D-III `spheres_9`) and what architecture to train (ize of the network and the activation type e.g. `10_by_15`, `Relu`)
* Set the number of training epochs and learning rate
* Set the overall number of trials and frequency of log reports

For each successful training attempt (perfect classification of the data set) log file (`/data/<dataset-type>/<architecture>/good_results.txt`) is updated and the resulting model and output of each layer are saved in a new folder whose name is the current date and time (`/data/<dataset-type>/<architecture>/<activation-type>/<current-data-time>`) and whose location reflects the selected data set and the selected architecture. Betti numbers calculation pipeline will use `good_results.txt` log file to accesse stored models to run betti numbers calculation on their outputs. 

## Example:
Train on D-II data set. Using network of size 10 (layers) by 25 (neurons each), and `LeakyRelu` activation. Set 0.2 learning rate and run 12000 training epochs, with 80 trials, report progress every 1000 epochs:

`computer.py --pipeline_name train_2_excel_3D          --output ./data/rings_9/ --input-tf-dataset ./data/rings_9/rings_9.tfrecords --model 10_by_25 --activation_type LeakyRelu --trials 80 --learning_rate 0.02 --training_epochs 12000 --summary_freq 1000`

# Calculate Betti numbers
* Once a few well trained neural networks have been accumulated use `compute.py` running  `texcel_2_betti` pipeline to compute Betti numbers
* Calculated Betti numbers for successfully trained neural networks with given architecture and activation type. Split the calculation in parallel on 10 cores. Each core limited to 10 Gb memory.  

* Run `computer.py` with appropriately set parameters: number of neighbors for nearest neighbor graph construction  ( the scale at wich to build Vietoris-Rips complex is fixed in Eirene call at `./julia_include/julia_aux2`.  

## Example:
Calculate Betti numbers for networks trained on D-II dataset of size 10 (layers) by 25 (neurons each) with Leaky Relu activation. The calculations proceed on a subsample of the data set, which is one fouth of the original data set size. Compute Betti numbers zero and one for class A (`cat2`):

`computer.py --pipeline_name excel_2_betti --output ./data/rings_9/ --input-tf-dataset ./data/rings_9/rings_9.tfrecords --model 10_by_25 --activation_type LeakyRelu --trials 80 --cat2  --divisor 4 --neighbors 35 --betti_max 1 --read_excel_from LeakyRelu`

# Visualize Data set
* Run `computer.py` with appropriate pipeline (e.g. `visualize_tfrecords.py`)
* Check generated `.html` file with `plotly` data set plot.


### Trials:

Simulation results are large (>100Gb),  will be uploaded soon. 
