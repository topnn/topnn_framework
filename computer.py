import os.path
import time
import pipelines
from simulators.simulator import Simulator
import glob
import tensorflow as tf
import datetime


class ComputerSimulator(Simulator):
    """
    Class for running various pipelines
    """

    def __init__(self, **kwargs):
        Simulator.__init__(self, **kwargs)
        self.input_folder = kwargs.get('input_folder', None)
        self.output_folder = kwargs.get('output_folder', None)
        self.source = kwargs.get('source', 0)
        self.pipe_action = kwargs.get('pipe_action', 'predict')
        

    def run(self, **fit_params):

        content = None
        if self.output_folder is not None and not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        if self.pipe_action == 'predict':
            content = self.pipeline.predict(None)

        if self.pipe_action == 'fit':
            self.pipeline.fit(X=None, training_labels=None, **fit_params)
            
        if self.pipe_action == 'fit_transform':
            self.pipeline.fit_transform(None, None, **fit_params)
        
        if self.pipe_action == 'transform':
            content = self.pipeline.transform(X=None)
        
        if self.output_folder is not None:
            pass
        return content

if __name__ == "__main__":
    import argparse
    from importlib import import_module

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true', help="")
    parser.add_argument('--plot', action='store_true', help="")
    parser.add_argument('--cat1', action='store_true', help="")
    parser.add_argument('--cat2', action='store_true', help="")
    parser.add_argument('--input-tf-folder', default='./data/default',
                        help="Input folder with tfrecords dataset records")
    parser.add_argument('--input-tf-dataset', default='default.tfrecords', help="tfrecords data set name")

    parser.add_argument('--output-tf-dataset', default='./data/default/', help="Generated tfrecords dataset name")
    parser.add_argument('--output-tf-folder', default='./data/default/',
                        help="Output folder with tfrecords dataset records")

    parser.add_argument('--input-pickle-folder', default=None, help="Input folder with model predictins")
    parser.add_argument('--input-pickle-predictions', default=None, help="pickle file name with model predictions")

    parser.add_argument('--output-pickle-predictions', default='simple',
                        help="Generated pickle file name with model predictions")
    parser.add_argument('--output-pickle-folder', default='./data/simple/', help="Output folder with model predictions")

    parser.add_argument('--output-plots-folder', default='./data/simple/',
                        help="Folder name to store visualization results")

    parser.add_argument('--pipe-action', default='transform',
                        choices=('fit', 'predict', 'fit_transform', 'fit_predict', 'transform'),
                        help="Which method of final pipeline stage is to apply.")

    parser.add_argument('--pipeline_name', default='visualize_simple_nn_prediction', choices=pipelines.__all__,
                        help="Pipeline to load")
    parser.add_argument('--output',
                        help='Output folder where model is saved')
    parser.add_argument('--save_freq',
                        default=1,
                        type=int,
                        help='frequency of epochs to save model snapshot.')
    parser.add_argument('--divisor',
                        default=3,
                        type=int,
                        help='when computing betti numbers reduces the number of smaples by divisor')
    parser.add_argument('--betti_max',
                        default=3,
                        type=int,
                        help='when computing betti numbers reduces the number of smaples by divisor')
    parser.add_argument('--trials',
                        default=1,
                        type=int,
                        help='how many time run simulation')
    parser.add_argument('--trial',
                        default=0,
                        type=int,
                        help='initialize simulation with trial having tis value.')
    parser.add_argument('--id',
                        default=0,
                        type=int,
                        help='id for cross referencing')
    parser.add_argument('--neighbors',
                        default=14,
                        type=int,
                        help='number of nearest neighbors for betti numbers calculation')
    parser.add_argument('--validation_freq',
                        default=1,
                        type=int,
                        help='frequency of validation of the model in number of iterations.')
    parser.add_argument('--summary_freq',
                        type=int,
                        default=50,
                        help='frequency of validation of the model in number of iterations.')
    parser.add_argument('--learning_rate', type=float, default=0.00001)
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use.')
    parser.add_argument('--training_epochs', type=int, default=1000, help='Number of GPUs to use.')
    parser.add_argument('--use_cpu', action='store_true', help='Use cpu')
    parser.add_argument('--activation_type', default='LeakyRelu', choices=['Relu', 'LeakyRelu', 'Tanh', 'Custom'],
                        help='Specify  actiavation function')
    parser.add_argument('--read_excel_from', default='list', choices=['list', 'Relu', 'LeakyRelu', 'Tanh', 'Custom'],
                        help='Specify  where to take excel files from')
    parser.add_argument('--gpu', default=-1, type=int, help='gpu id (-1 to disable GPU)'),
    parser.add_argument('--pretrained-model',
                        default=None,
                        help='Set pretrained model path if exists. Otherwises start from random')
    parser.add_argument('--model',
                        default='base',
                        choices=['base', 'arch_type_1', '4_by_50', '10_by_5', '10_by_6', '10_by_10', '10_by_15', '8_by_15', '2_by_15',
                                 '3_by_15',
                                 '4_by_15', '10_by_25','10_by_25', '10_by_50','5_by_15','6_by_15','7_by_15',
                                 '3_by_10_2_by_1_3_by_10',
                                 '3_by_10_3_by_1_3_by_10',
                                 '3_by_25_2_by_1_3_by_25',
                                 '3_by_25_3_by_1_3_by_25', 'nn_simple', 'nn_9_by_6_simple', 'nn_hourglass_simple',
                                 'nn_5_by_50_simple', 'nn_9_by_30_simple', 'nn_9_by_10_simple', 'nn_9_by_15_simple',
                                 'nn_9_by_20_simple', 'nn_9_by_50_simple', 'nn_9_by_3_simple', 'nn_12_by_3_simple',
                                 'nn_16_by_3_simple'],
                        help='Specifies which model to use.')
    parser.add_argument('--save2pkl', action='store_true'),
    parser.add_argument('--pretrained', default='', help='pretrained model to load')
    parser.add_argument('--restore_meta_graph', action='store_true', help='')
    parser.add_argument('--freeze_pretrained', action='store_true', help='')

    args = parser.parse_args()
    j = ''
    initial_path = args.output
    total_good_results = 0

    def main(trial=0):
        # analyze simulation results
        if args.pipeline_name in ['excel_2_betti_3D_parallel', 'excel_2_betti']:
            base = os.path.join(initial_path, args.model)
            # read 100% accuracy results.

            with open(os.path.join(base, args.read_excel_from, 'good_results.txt')) as f:
                content = f.readlines()

                # get rid of empty lines and empty characters
                content = [item.rstrip() for item in content if item.strip()]
                if len(content) < trial - 1:
                    return -1
                else:
                    try:
                        args.pretrained = os.path.join(base, os.path.basename(content[trial]))
                    except:
                        print("content is shorter than trial", trial, len(content))
                        return -1
        print(args.pretrained)
        if args.pipeline_name in ['train_2_betti', 'excel_2_betti', 'train_2_excel_2D',  'train_2_excel_3D', 'excel_2_betti_3D_parallel']:


            all_subdirs = [d for d in glob.glob(args.output + '/*') if os.path.isdir(d)]

            if args.pretrained:
                args.output = os.path.join(initial_path, args.model, os.path.basename(args.pretrained))
            else:
                if args.pretrained_model and not len(all_subdirs) == 0:

                    timestr = time.strftime("%Y-%m-%d-%H-%M-%S")


                    args.output = os.path.join(initial_path, args.model,
                                               timestr + "-pretrained-on-" + os.path.basename(args.pretrained_model))
                    args.output = os.path.join(initial_path, args.model, os.path.basename(args.pretrained_model))
                else:
                    timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
                    args.output = os.path.join(initial_path, args.model, timestr)
                    print('Created new output directory: ' + args.output)

        fit_params = vars(args)
        fit_params.update({'julia': j})
        args.pipeline = import_module('pipelines.{}'.format(args.pipeline_name)).load(args)
        simulator = ComputerSimulator(**vars(args))
        good_result = 0
        try:
            # used in conjecture with "visualize_nn_prediction.py" pipeline
            # and  pipe action predict.
            content = simulator.run(**fit_params)

            if content == '':
                good_result = 0
            else:
                good_result = 1

        except (KeyboardInterrupt, FileNotFoundError):
            print('Stopping simulation')

        del simulator
        return good_result

    GOOD_RESULTS_LIMIT = 15
    GOOD_RESULTS_LIMIT = 3

    now = datetime.datetime.now()
    for trial in range(args.trial, args.trials):
        print("\n" * 4)
        print("*" * 100)
        #"Stating trial:",  trial, "out of total:", args.trials, "| pipeline", args.pipeline_name, "| model", args.model, args.activation_type, "| dataset", os.path.basename(args.input_tf_dataset), "| sofar good results:", total_good_results)
        trial_str = str(args.id) + " | Stating trial: " + str(trial) + " out of total: " + str(args.trials) +  " | pipeline " +  args.pipeline_name + " | model " +  args.model + ', ' + args.activation_type + " | dataset " +  os.path.basename(args.input_tf_dataset) + " | sofar good results: " + str(total_good_results)
        print(trial_str)

        # record of all simulation from the big bang until now.
        with open('./logs/simulation_runs.txt', 'a') as f:
            timestr = time.strftime("%Y-%m-%d-%H-%M-%S")

            f.write(timestr + ' -- ')
            f.write(trial_str)
            f.write("\n")

        # look at the name of this file to see what was the last simulation
        flag_file = os.path.join('./logs/', str(args.id) + '-' + args.model + '-' + str(trial) + '-' + args.activation_type + '-' + os.path.basename(args.input_tf_dataset))
        with  open(flag_file, 'w') as f:
            f.write('look at my name')

        print("*" * 100)
        print(datetime.datetime.now())

        res = main(trial)

        # -1 is returned when no more excel files are avaliable to clculate betti numbers
        if res == -1:
            print("reached end of good_results.txt")
            break

        total_good_results += res
        tf.reset_default_graph()
        if total_good_results == GOOD_RESULTS_LIMIT:
            print("Maximum number of good results reached")
            break
        print("trial", trial, "is over", ":" * 90)
        print("\n" * 4)
        os.remove(flag_file)

    print("Done running configuration, took", datetime.datetime.now() - now)
    print("!" * 100)
