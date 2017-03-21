import tensorflow as tf 
from typing import Tuple, List, Union
from tensorflow import Tensor, Variable
from tensorflow import nn


lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
TensVar = Union[Tensor, Variable] #??
ImgShape = Tuple[int, int, int, int] #?? #(batch, height, width, channels)

def rnn_layer(x:TensVar, 			#?? why not x just be tensor?
			  ninp_channels: int,
			  nout_channels: int,
			  sfx: str,
			  nl: tf.nn.elu,
			  W_init=None,
			  b_init=None,
			  batch_norm: bool=True,
			  reuse=False) -> TensVar:
	with tf.name_scope("rnn_layer"):
		with tf.variable_scope(sfx) as scope:
			pass #TODO


	return op


def batch_flatten(t: Tensor) -> Tensor:
    """Reshape tensor so that it is flat vector (except for batch dimension)"""
    return tf.reshape(t, [tf.shape(t)[0], -1])


def rnn_template(inputs: List[TensVar], 
				 inp_shapes: List[ImgShape],
				 out_shapes: List[ImgShape], 
				 **kwargs) -> Tuple[TensVar, List]:
	"""
	Recurrent Neural Network / LSTM
	Args:
		inputs: [TensVar / tf.Variable]
		inp_shapes: []
		out_shapes: ()
	"""
	# Parameters
	layer_width = kwargs['layer_width']
	layer_height = kwargs['layer_height']


	# Reshaping
	print("Input Shapes to RNN", inp_shapes)
    print("Output Shapes to RNN", out_shapes)
	assert consistent_batch_size(inp_shapes + out_shapes), "Batch sizes differ"

	input_width = width(inp_shapes)
    output_width = width(out_shapes)


    # Num inputs and outputs
    flat_inputs = [batch_flatten(inp) for inp in inputs]
    prev_layer = tf.concat(flat_inputs, 1)

    # Layers
    if layer_width != input_width:
 		print("Input projection, layer_width: %s input_width: %s" % (layer_width, input_width))
        wx_sfx = 'wxinpproj'
        wx = rnn_layer(prev_layer, 
        			   input_width, 
        			   layer_width, 
        			   wx_sfx)
    else:
    	print("Skipping input weight projection, layer_width: %s input_width: %s" % (layer_width, input_width))
    	wx = prev_layer

    outputs = 
    params = [] #type: List[Variable]
    return output, params



