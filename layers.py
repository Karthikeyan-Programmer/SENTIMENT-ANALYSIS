import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Flatten, Activation, Permute
from tensorflow.keras.layers import Multiply, Lambda, Reshape, Dot, Concatenate, RepeatVector, \
    TimeDistributed, Permute, Bidirectional


class Attention(Layer):

    def __init__(self, context='many-to-many', alignment_type='global', window_width=None,
                 score_function='general', model_api='functional', **kwargs):
        if context not in ['many-to-many', 'many-to-one']:
            raise ValueError("Argument for param @context is not recognized")
        if alignment_type not in ['global', 'local-m', 'local-p', 'local-p*']:
            raise ValueError("Argument for param @alignment_type is not recognized")
        if alignment_type == 'global' and window_width is not None:
            raise ValueError("Can't use windowed approach with global attention")
        if context == 'many-to-many' and alignment_type == 'local-p*':
            raise ValueError("Can't use local-p* approach in many-to-many scenarios")
        if score_function not in ['dot', 'general', 'location', 'concat', 'scaled_dot']:
            raise ValueError("Argument for param @score_function is not recognized")
        if model_api not in ['sequential', 'functional']:
            raise ValueError("Argument for param @model_api is not recognized")
        super(Attention, self).__init__(**kwargs)
        self.context = context
        self.alignment_type = alignment_type
        self.window_width = window_width  
        self.score_function = score_function
        self.model_api = model_api

    def get_config(self):
        base_config = super(Attention, self).get_config()
        base_config['alignment_type'] = self.alignment_type
        base_config['window_width'] = self.window_width
        base_config['score_function'] = self.score_function
        base_config['model_api'] = self.model_api
        return base_config

    def build(self, input_shape):        
        if self.context == 'many-to-many':
            self.input_sequence_length, self.hidden_dim = input_shape[0][1], input_shape[0][2]
            self.target_sequence_length = input_shape[1][1]
        elif self.context == 'many-to-one':
            self.input_sequence_length, self.hidden_dim = input_shape[0][1], input_shape[0][2]
        
        if 'local-p' in self.alignment_type:
            self.W_p = Dense(units=self.hidden_dim, use_bias=False)
            self.W_p.build(input_shape=(None, None, self.hidden_dim))                               # (B, 1, H)
            self._trainable_weights += self.W_p.trainable_weights

            self.v_p = Dense(units=1, use_bias=False)
            self.v_p.build(input_shape=(None, None, self.hidden_dim))                               # (B, 1, H)
            self._trainable_weights += self.v_p.trainable_weights

        if 'dot' not in self.score_function:  
            self.W_a = Dense(units=self.hidden_dim, use_bias=False)
            self.W_a.build(input_shape=(None, None, self.hidden_dim))                               # (B, S*, H)
            self._trainable_weights += self.W_a.trainable_weights

        if self.score_function == 'concat':  
            self.U_a = Dense(units=self.hidden_dim, use_bias=False)
            self.U_a.build(input_shape=(None, None, self.hidden_dim))                               # (B, 1, H)
            self._trainable_weights += self.U_a.trainable_weights

            self.v_a = Dense(units=1, use_bias=False)
            self.v_a.build(input_shape=(None, None, self.hidden_dim))                               # (B, S*, H)
            self._trainable_weights += self.v_a.trainable_weights

        super(Attention, self).build(input_shape)

    def call(self, inputs):        
        if not isinstance(inputs, list):
            raise ValueError("Pass a list=[encoder_out (Tensor), decoder_out (Tensor)," +
                             "current_timestep (int)] for all scenarios")
        
        if self.context == 'many-to-one':
            target_hidden_state = inputs[1]                                                         # (B, H)
            source_hidden_states = inputs[0]                                                        # (B, S, H)
        elif self.context == 'many-to-many':
            target_hidden_state = inputs[1]                                                         # (B, H)
            current_timestep = inputs[2]
            source_hidden_states = inputs[0]                                                        # (B, S, H)
        
        target_hidden_state = tf.expand_dims(input=target_hidden_state, axis=1)                     # (B, 1, H)

        if self.alignment_type == 'global':                                                         # Global Approach
            source_hidden_states = source_hidden_states                                             # (B, S, H)

        elif 'local' in self.alignment_type:                                                        # Local Approach
            self.window_width = 8 if self.window_width is None else self.window_width
           
            if self.alignment_type == 'local-m':                                                    # Monotonic Alignment
                if self.context == 'many-to-one':
                    aligned_position = self.input_sequence_length
                elif self.context == 'many-to-many':
                    aligned_position = current_timestep

                left = int(aligned_position - self.window_width
                           if aligned_position - self.window_width >= 0
                           else 0)
                right = int(aligned_position + self.window_width
                            if aligned_position + self.window_width <= self.input_sequence_length
                            else self.input_sequence_length)
                source_hidden_states = Lambda(lambda x: x[:, left:right, :])(source_hidden_states)  # (B, S*=(D, 2xD), H)

            elif self.alignment_type == 'local-p':                                                  # Predictive Alignment
                aligned_position = self.W_p(target_hidden_state)                                    # (B, 1, H)
                aligned_position = Activation('tanh')(aligned_position)                             # (B, 1, H)
                aligned_position = self.v_p(aligned_position)                                       # (B, 1, 1)
                aligned_position = Activation('sigmoid')(aligned_position)                          # (B, 1, 1)
                aligned_position = aligned_position * self.input_sequence_length                    # (B, 1, 1)

            elif self.alignment_type == 'local-p*':                                                 # Completely Predictive Alignment
                aligned_position = self.W_p(source_hidden_states)                                   # (B, S, H)
                aligned_position = Activation('tanh')(aligned_position)                             # (B, S, H)
                aligned_position = self.v_p(aligned_position)                                       # (B, S, 1)
                aligned_position = Activation('sigmoid')(aligned_position)                          # (B, S, 1)
                aligned_position = tf.squeeze(aligned_position, axis=-1)                            # (B, S)
                top_probabilities = tf.nn.top_k(input=aligned_position,                             # (values:(B, D), indices:(B, D))
                                                k=self.window_width,
                                                sorted=False)
                onehot_vector = tf.one_hot(indices=top_probabilities.indices,
                                           depth=self.input_sequence_length)                        # (B, D, S)
                onehot_vector = tf.reduce_sum(onehot_vector, axis=1)                                # (B, S)
                aligned_position = Multiply()([aligned_position, onehot_vector])                    # (B, S)
                aligned_position = tf.expand_dims(aligned_position, axis=-1)                        # (B, S, 1)
                initial_source_hidden_states = source_hidden_states                                 # (B, S, 1)
                source_hidden_states = Multiply()([source_hidden_states, aligned_position])         # (B, S*=S(D), H)
                aligned_position += tf.keras.backend.epsilon()                                      # (B, S, 1)
                source_hidden_states /= aligned_position                                            # (B, S*=S(D), H)
                source_hidden_states = initial_source_hidden_states + source_hidden_states          # (B, S, H)

        if 'dot' in self.score_function:                                                            # Dot Score Function
            attention_score = Dot(axes=[2, 2])([source_hidden_states, target_hidden_state])         # (B, S*, 1)
            if self.score_function == 'scaled_dot':
                attention_score *= 1 / np.sqrt(float(source_hidden_states.shape[2]))                # (B, S*, 1)

        elif self.score_function == 'general':                                                      # General Score Function
            weighted_hidden_states = self.W_a(source_hidden_states)                                 # (B, S*, H)
            attention_score = Dot(axes=[2, 2])([weighted_hidden_states, target_hidden_state])       # (B, S*, 1)

        elif self.score_function == 'location':                                                     # Location-based Score Function
            weighted_target_state = self.W_a(target_hidden_state)                                   # (B, 1, H)
            attention_score = Activation('softmax')(weighted_target_state)                          # (B, 1, H)
            attention_score = RepeatVector(source_hidden_states.shape[1])(attention_score)          # (B, S*, H)
            attention_score = tf.reduce_sum(attention_score, axis=-1)                               # (B, S*)
            attention_score = tf.expand_dims(attention_score, axis=-1)                              # (B, S*, 1)

        elif self.score_function == 'concat':                                                       # Concat Score Function
            weighted_hidden_states = self.W_a(source_hidden_states)                                 # (B, S*, H)
            weighted_target_state = self.U_a(target_hidden_state)                                   # (B, 1, H)
            weighted_sum = weighted_hidden_states + weighted_target_state                           # (B, S*, H)
            weighted_sum = Activation('tanh')(weighted_sum)                                         # (B, S*, H)
            attention_score = self.v_a(weighted_sum)                                                # (B, S*, 1)

        attention_weights = Activation('softmax')(attention_score)                                  # (B, S*, 1)

        if self.alignment_type == 'local-p':                                                        # Gaussian Distribution
            gaussian_estimation = lambda s: tf.exp(-tf.square(s - aligned_position) /
                                                   (2 * tf.square(self.window_width / 2)))
            gaussian_factor = gaussian_estimation(0)
            for i in range(1, self.input_sequence_length):
                gaussian_factor = Concatenate(axis=1)([gaussian_factor, gaussian_estimation(i)])    # (B, S*, 1)
            attention_weights = attention_weights * gaussian_factor                                 # (B, S*, 1)

        context_vector = source_hidden_states * attention_weights                                   # (B, S*, H)

        if self.model_api == 'functional':
            return context_vector, attention_weights
        elif self.model_api == 'sequential':
            return context_vector


class SelfAttention(Layer):
    """
    Layer for implementing self-attention mechanism. Weight variables were preferred over Dense()
    layers in implementation because they allow easier identification of shapes. Softmax activation
    ensures that all weights sum up to 1.

    @param (int) size: a.k.a attention length, number of hidden units to decode the attention before
           the softmax activation and becoming annotation weights
    @param (int) num_hops: number of hops of attention, or number of distinct components to be
           extracted from each sentence.
    @param (bool) use_penalization: set True to use penalization, otherwise set False
    @param (int) penalty_coefficient: the weight of the extra loss
    @param (str) model_api: specify to use TF's Sequential OR Functional API, note that attention
           weights are not outputted with the former as it only accepts single-output layers
    """
    def __init__(self, size, num_hops=8, use_penalization=True,
                 penalty_coefficient=0.1, model_api='functional', **kwargs):
        if model_api not in ['sequential', 'functional']:
            raise ValueError("Argument for param @model_api is not recognized")
        self.size = size
        self.num_hops = num_hops
        self.use_penalization = use_penalization
        self.penalty_coefficient = penalty_coefficient
        self.model_api = model_api
        super(SelfAttention, self).__init__(**kwargs)

    def get_config(self):
        base_config = super(SelfAttention, self).get_config()
        base_config['size'] = self.size
        base_config['num_hops'] = self.num_hops
        base_config['use_penalization'] = self.use_penalization
        base_config['penalty_coefficient'] = self.penalty_coefficient
        base_config['model_api'] = self.model_api
        return base_config

    def build(self, input_shape):
        self.W1 = self.add_weight(name='W1',
                                  shape=(self.size, input_shape[2]),                                # (size, H)
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.W2 = self.add_weight(name='W2',
                                  shape=(self.num_hops, self.size),                                 # (num_hops, size)
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(SelfAttention, self).build(input_shape)

    def call(self, inputs): 
        W1, W2 = self.W1[None, :, :], self.W2[None, :, :]
        hidden_states_transposed = Permute(dims=(2, 1))(inputs)                                     # (B, H, S)
        attention_score = tf.matmul(W1, hidden_states_transposed)                                   # (B, size, S)
        attention_score = Activation('tanh')(attention_score)                                       # (B, size, S)
        attention_weights = tf.matmul(W2, attention_score)                                          # (B, num_hops, S)
        attention_weights = Activation('softmax')(attention_weights)                                # (B, num_hops, S)
        embedding_matrix = tf.matmul(attention_weights, inputs)                                     # (B, num_hops, H)
        embedding_matrix_flattened = Flatten()(embedding_matrix)                                    # (B, num_hops*H)

        if self.use_penalization:
            attention_weights_transposed = Permute(dims=(2, 1))(attention_weights)                  # (B, S, num_hops)
            product = tf.matmul(attention_weights, attention_weights_transposed)                    # (B, num_hops, num_hops)
            identity = tf.eye(self.num_hops, batch_shape=(inputs.shape[0],))                        # (B, num_hops, num_hops)
            frobenius_norm = tf.sqrt(tf.reduce_sum(tf.square(product - identity))) 
            self.add_loss(self.penalty_coefficient * frobenius_norm)  

        if self.model_api == 'functional':
            return embedding_matrix_flattened, attention_weights
        elif self.model_api == 'sequential':
            return embedding_matrix_flattened
