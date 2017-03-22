from wacacore.util.misc import stringy_dict
import tensortemplates.res_net as res_net
import tensortemplates.conv_res_net as conv_res_net
import tensortemplates.recurr_net as recurr_net
import tensortemplates as tt
import tensorflow as tf

template_module = {'res_net': res_net,
                   'conv_res_net': conv_res_net
                   }

nl_module = {'elu': tf.nn.elu,
             'relu': tf.nn.relu}
