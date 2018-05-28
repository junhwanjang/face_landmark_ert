import dlib

# Set Options
# option list
'''
cascade_depth
tree_depth
num_trees_per_cascade_level
nu
oversampling_amount
feature_pool_size
feature_pool_region_padding
lambda_param

default vlaues
_cascade_depth = 10;
_tree_depth = 4;
_num_trees_per_cascade_level = 500;
_nu = 0.1;
_oversampling_amount = 20;
_feature_pool_size = 400;
_lambda = 0.1;
_num_test_splits = 20;
_feature_pool_region_padding = 0;
_verbose = false;
_num_threads = 0;

'''

options = dlib.shape_predictor_training_options()
options.cascade_depth = 15 # default:10
options.tree_depth = 4 # default:4
options.oversampling_amount = 20 # default:20

# Train landmark model
train_xml_path = 'Your Indir/Name.xml'
data_file_path = 'Your Outdir/Name.dat'
dlib.train_shape_predictor(train_xml_path, data_file_path, options)