import tensorflow as tf


tf.set_random_seed(123456)


def show_params(name, params):
    print("Training Parameters for {}".format(name))
    for para in params:
        print(para.name + ': ' + str(para.get_shape()))


def embed_norm(x, axes):
    norm2 = tf.reduce_sum(tf.square(x), axes, keep_dims=True)
    norm = tf.sqrt(norm2) + 1e-4
    return x / norm


class UniMappingGAN:

    def __init__(self, s, seq_len, width, height, nlabels=10, channels=1,
                 batch_size=8, markov=1, g_lr=1e-4,
                 d_lr=1e-4, epsilon=1e-9):
        self.s = s
        self.batch_size = batch_size

        self.seq_len = seq_len
        self.width = width
        self.height = height
        self.nlabels = nlabels
        self.channels = channels

        self.beta1 = 0.5
        self.beta2 = .999
        self.d_lr = d_lr
        self.g_lr = g_lr
        self.lambda_ = 10.
        self.lambda_rec = 10.
        self.lambda_cycle = 10.
        self.n_train_critic = 10
        self.eps = epsilon
        self.n_experts = 5
        self.weight_decay = 0.01
        self.lambda_gp_g = 0.01
        self.n_concats = markov

        self.w_b = None
        self.w = None

        self.gp_b = None
        self.gp = None
        self.gs_op = None
        self.e_op = None

        self.g_loss = 0.
        self.g_b_loss = 0.
        self.d_loss = 0.
        
        self.d_op = None
        self.g_op = None
        self.vars = []
        self.merged = None
        self.writer = None
        self.saver = None
        self.use_embed = True
        self.pred, self.accuracy = None, None
        self.embed_dim = 128

        self.g_super_loss = 0.

        # placeholders
        self.image = \
            tf.placeholder(tf.float32,
                           [None, self.height, self.width,
                            self.channels], name='image')
        self.label = \
            tf.placeholder(tf.float32,
                           [None, self.nlabels], name='label')
        self.olabel = \
            tf.placeholder(tf.float32,
                           [None, self.nlabels], name='oracle-label')
        self.oimage = \
            tf.placeholder(tf.float32,
                           [None, self.height, self.width,
                            self.channels], name='oracle-image')
        self.lr_decay = tf.placeholder(tf.float32, shape=(),
                                       name='learning_rate-decay')
        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
        self.n_decay = tf.placeholder(tf.float32, shape=(), name='noise_decay')
        self.build_cyclegan()  # build CycleGAN

    def discriminator(self, x, reuse=None, name=""):
        with tf.variable_scope('discriminator-%s' % name, reuse=reuse):
            hidden = 512
            out = tf.layers.dense(x, hidden, activation=tf.nn.relu,
                                  name=name + '.1', reuse=reuse)
            out = tf.layers.dense(out, hidden, activation=tf.nn.relu,
                                  name=name + '.2', reuse=reuse)
            out = tf.layers.dense(out, hidden, activation=tf.nn.relu,
                                  name=name + '.3', reuse=reuse)
            out = tf.layers.dense(out, 1, activation=None,
                                  name=name + '.out', reuse=reuse)
            return out

    def discriminator_img(self, x, reuse=None, name=""):
        with tf.variable_scope('discriminator-image-%s' % name, reuse=reuse):
            is_training = self.is_training
            ndf = 32
            lc_x = tf.layers.conv2d(x, ndf, 5, strides=(2, 2),
                                    padding='same', use_bias=False)
            lc_x = tf.nn.relu(lc_x)
            lc_x = tf.layers.conv2d(lc_x, ndf * 2, 5, strides=(2, 2),
                                    padding='same', use_bias=False)
            lc_x = tf.nn.relu(lc_x)
            lc_x = tf.layers.conv2d(lc_x, ndf * 4, 5,
                                    use_bias=False)
            lc_x = tf.nn.relu(lc_x)

            lc_x = tf.reshape(lc_x, [-1, ndf * 4 * 3 * 3])
            critic = tf.layers.dense(lc_x, 1)
            return critic

    def gen_img2embed(self, x, reuse=None, name=""):
        if reuse is None:
            reuse = False
        with tf.variable_scope('generator-%s' % name, reuse=reuse):
            is_training = self.is_training
            ndf = 32
            lc_x = tf.layers.conv2d(x, ndf, 5, strides=(2, 2),
                                    padding='same', use_bias=False)
            lc_x = tf.nn.relu(lc_x)
            lc_x = tf.layers.conv2d(lc_x, ndf * 2, 5, strides=(2, 2),
                                    padding='same', use_bias=False)
            lc_x = tf.nn.relu(lc_x)
            lc_x = tf.layers.conv2d(lc_x, ndf * 4, 5,
                                    use_bias=False)
            lc_x = tf.nn.relu(lc_x)

            print(lc_x.get_shape())
            lc_x = tf.reshape(lc_x, [-1, ndf * 4 * 3 * 3])

            out = tf.layers.dense(lc_x, self.embed_dim, activation=tf.nn.tanh)
            return out

    def gen_embed2img(self, z, reuse=None, name=""):
        if reuse is None:
            reuse = False
        with tf.variable_scope('generator-%s' % name, reuse=reuse):
            ngf = 32

            in_z = tf.layers.dense(z, ngf * 4 * 1 * 1, use_bias=False)
            lx_z = tf.reshape(in_z, [-1, 1, 1, ngf * 4])
            lx_z = tf.layers.conv2d_transpose(lx_z, ngf * 4, 3, use_bias=False)
            lx_z = tf.nn.relu(lx_z)
            lx_z = tf.layers.conv2d_transpose(lx_z, ngf * 2, 5, use_bias=False)
            lx_z = tf.nn.relu(lx_z)
            lx_z = tf.layers.conv2d_transpose(lx_z, ngf, 5, strides=(2, 2),
                                              padding='same', use_bias=False)
            lx_z = tf.nn.relu(lx_z)
            lx_z = tf.layers.conv2d_transpose(
                lx_z, 1, 5, strides=(2, 2), padding='same', activation=tf.sigmoid)

            return lx_z

    def markov(self, x):
        return tf.reshape(x, [self.seq_len // self.n_concats, self.n_concats * self.embed_dim])
        #current = []
        #for i in range(self.n_concats):
        #    current.append(x[i: self.seq_len - self.n_concats + 1 + i, :])
        #return tf.concat(current, 1)

    def markov_interpolate(self, real, fake):
        #current = []
        real_sp = tf.reshape(real, [self.seq_len // self.n_concats, self.n_concats, self.embed_dim])
        fake_sp = tf.reshape(fake, [self.seq_len // self.n_concats, self.n_concats, self.embed_dim])
        alpha = tf.random_uniform([self.seq_len // self.n_concats, self.n_concats, 1], 0, 1)
        out = real_sp * alpha + (1 - alpha) * fake_sp
        print('M_I' + str(out.get_shape()))
        return self.markov(out)

    def markov_img(self, x):
        xr = tf.reshape(x, [self.seq_len // self.n_concats, self.n_concats,
                            self.height, self.width, self.channels])
        corp = []
        for i in range(self.n_concats):
            corp.append(xr[:, i, :, :, :])
        return tf.concat(corp, 3)

    def markov_interpolate_img(self, real, fake):
        real_sp = tf.reshape(real, [self.seq_len // self.n_concats, self.n_concats,
                             self.height, self.width, self.channels])
        fake_sp = tf.reshape(fake, [self.seq_len // self.n_concats, self.n_concats,
                             self.height, self.width, self.channels])
        current = []
        for i in range(self.n_concats):
            alpha = tf.random_uniform([self.seq_len // self.n_concats,
                                       1, 1, 1], 0.0, 1.0)
            real_x = real_sp[:, i, :, :, :]
            fake_x = fake_sp[:, i, :, :, :]
            current.append(real_x * alpha + (1 - alpha) * fake_x)
        return tf.concat(current, 3)

    def build_cyclegan(self):
        # Generator
        self.a = tf.identity(self.image)
        self.ora_a = tf.identity(self.oimage)

        with tf.variable_scope("label2embed"):
            self.embeddings = tf.get_variable('label-embed',
                                              [self.nlabels, self.embed_dim])
            self.embeddings = embed_norm(self.embeddings, 1)
            self.inner_product = tf.matmul(self.embeddings, self.embeddings,
                                           transpose_b=True)
            self.loss_inner_product = tf.reduce_sum(tf.square(self.inner_product))
            noise = tf.random_normal([self.seq_len, self.embed_dim], 0.0, 0.05 * self.n_decay)
            self.b = tf.matmul(self.label, self.embeddings) + noise
            self.ora_b = tf.matmul(self.olabel, self.embeddings) + noise

        with tf.variable_scope("generator-img2lb"):
            self.a2b = self.gen_img2embed(self.a, name='g_img2lb')
        print('img2lb shape = ' + str(self.a2b.get_shape()))
        with tf.variable_scope("generator-lb2img"):
            self.b2a = self.gen_embed2img(self.b, name='g_lb2img')
        print('lb2img shape = ' + str(self.b2a.get_shape()))

        with tf.variable_scope("generator-img2lb"):
            self.b2a2b = self.gen_img2embed(self.b2a, name='g_img2lb',
                                            reuse=True)
        with tf.variable_scope("generator-lb2img"):
            self.a2b2a = self.gen_embed2img(self.a2b, name='g_lb2img',
                                            reuse=True)

        cycle_loss_a = tf.reduce_mean(tf.abs(self.a - self.a2b2a))#* \
        #                              (1. - self.image))
        cycle_loss_b = tf.reduce_mean(tf.square(self.b - self.b2a2b))

        self.cycle_loss = cycle_loss_a + cycle_loss_b

        self.g_super_loss = tf.reduce_mean(tf.square(self.a2b - self.ora_b))
        self.g_ssuper_loss = tf.reduce_mean(tf.square(self.a2b - self.ora_b)) + \
            self.cycle_loss * 0.1

        logits = tf.matmul(self.a2b, self.embeddings,
                           transpose_b=True)
        self.pred = tf.one_hot(tf.argmax(logits, 1), self.nlabels)
        ncorrect = (tf.reduce_sum(self.pred * self.olabel) + 0.0)
        self.accuracy = ncorrect / self.seq_len

        # Classifier
        with tf.variable_scope("discriminator-b"):
            d_b = self.discriminator(self.markov(self.b))
            d_a2b = self.discriminator(self.markov(self.a2b),
                                       reuse=True)
            b_hat = self.markov_interpolate(self.b,
                                            self.a2b)
            d_b_hat = self.discriminator(b_hat, reuse=True)

        with tf.variable_scope("discriminator-a"):
            d_a = self.discriminator_img(self.markov_img(self.a))
            d_b2a = self.discriminator_img(self.markov_img(self.b2a),
                                       reuse=True)
            a_hat = self.markov_interpolate_img(self.a, self.b2a)
            d_a_hat = self.discriminator_img(a_hat, reuse=True)

        # Training Ops
        self.w_b = tf.reduce_mean(d_b) - tf.reduce_mean(d_a2b)
        self.w_a = tf.reduce_mean(d_a) - tf.reduce_mean(d_b2a)

        self.w = self.w_b + self.w_a

        self.gp_a = tf.reduce_mean(
            (tf.sqrt(tf.reduce_sum(tf.gradients(d_a_hat, a_hat)[0] ** 2,
                                   reduction_indices=[1, 2, 3])) - 1.) ** 2
        )
        self.gp_b = tf.reduce_mean(
            (tf.sqrt(tf.reduce_sum(tf.gradients(d_b_hat, b_hat)[0] ** 2,
                                   reduction_indices=[1])) - 1.) ** 2
        )
        self.gp = self.gp_a + self.gp_b

        self.d_loss = self.lambda_ * self.gp - self.w

        # using adv loss
        self.g_a_loss = -1. * tf.reduce_mean(d_b2a)
        self.g_b_loss = -1. * tf.reduce_mean(d_a2b)
        self.g_loss = self.g_b_loss + self.g_a_loss + \
            self.lambda_cycle * self.cycle_loss

        # Optimizer
        t_vars = tf.trainable_variables()
        show_params('Full', t_vars)

        d_params = [v for v in t_vars if v.name.startswith('discriminator')]
        g_params = [v for v in t_vars if v.name.startswith('generator')]
        e_params = [v for v in t_vars if v.name.startswith('label2embed')]

        show_params('Discriminator', d_params)
        show_params('Generator', g_params)
        show_params('Embedding', e_params)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.d_op = \
                tf.train.AdamOptimizer(learning_rate=self.d_lr * self.lr_decay,
                                       beta1=self.beta1, beta2=self.beta2).\
                minimize(self.d_loss, var_list=d_params)
            self.g_op = \
                tf.train.AdamOptimizer(learning_rate=self.g_lr * self.lr_decay,
                                       beta1=self.beta1, beta2=self.beta2).\
                minimize(self.g_loss, var_list=g_params)

            self.gs_op = \
                tf.train.AdamOptimizer(learning_rate=self.g_lr * self.lr_decay,
                                       beta1=self.beta1, beta2=self.beta2).\
                minimize(self.g_ssuper_loss, var_list=g_params)
        self.e_op = \
                tf.train.AdamOptimizer(learning_rate=self.g_lr * self.lr_decay,
                                       beta1=self.beta1, beta2=self.beta2).\
                minimize(self.loss_inner_product, var_list=e_params)

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)

