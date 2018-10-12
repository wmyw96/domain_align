from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import time
import mnist_model
from tsp import *
import itertools
from skimage import io, data
import argparse


try:
    import matplotlib.pyplot as plt
except:
    print('Cannot import matplotlib')


parser = argparse.ArgumentParser(description='Argument parser')

parser.add_argument('--seed', dest='seed', type=int, default=123456,
                    help='random seed')
parser.add_argument('--data', dest='data', type=str, default='random',
                    help='the transition matrix: zipcode/random')
parser.add_argument('--probscale', dest='probscale', type=float, default=1.0,
                    help='probability scale (>=1 prefer)')
parser.add_argument('--model', dest='model', type=str, default='full',
                    help='model configurations (full/nodop/nopd)')
parser.add_argument('--noise', dest='noise', type=float, default='noise parameter b',
                    help='noise parameter b')


def save_image_collections(seq_images, file_prefix):
    full_img = np.zeros((280, 280, 1))
    for cid in range(100):
        x = cid // 10
        y = cid % 10
        for i in range(28):
            for j in range(28):
                full_img[x * 28 + i, y * 28 + j, :] = seq_images[cid, i, j, :]
    io.imsave(file_prefix + 'c.png', np.squeeze(full_img))


args = parser.parse_args()
np.random.seed(args.seed)


mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_images = mnist.train.images # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
test_images = mnist.test.images # Returns np.array
test_labels = np.asarray(mnist.test.labels, dtype=np.int32)


nlabels = 10
prob_scale = args.probscale
bound = args.noise


def random_noise(pre_trans_mat, bound):
    ps = 1.0 + np.random.uniform(-bound, bound, (nlabels, nlabels))
    bias = np.random.uniform(-bound * 0.1, bound * 0.1, (nlabels, nlabels))
    trans_mat = ps * np.maximum(pre_trans_mat + bias, 1e-2)
    trans_mat = trans_mat / np.sum(trans_mat, 0, keepdims=True)
    return trans_mat


init_p = np.zeros((nlabels, ))

train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

image_corpus = []
for i in range(nlabels):
    image_corpus.append([])
for i in range(train_images.shape[0]):
    if train_labels[i] < nlabels:
        image_corpus[train_labels[i]].append(train_images[i, :, :, :])
        init_p[train_labels[i]] += 1.0
init_p /= np.sum(init_p)

print('Initial Distribution:')
print(init_p)

# t[i, j] the probability from i to j
trans_matrix = None
while True:
    trans_matrix = \
        np.array([[0.33217993,0.02076125,0.179930796,0.05190311,0.04152249,0.03460208,0.12110727,0.1107266436,0.107266436,0.00000000],
                  [0.08359133,0.09907121,0.049535604,0.22910217,0.07120743,0.04024768,0.05572755,0.2229102167,0.021671827,0.12693498],
                  [0.13060429,0.16569201,0.118908382,0.14814815,0.03898635,0.11500975,0.12670565,0.0370370370,0.118908382,0.00000000],
                  [0.09953704,0.04861111,0.171296296,0.05787037,0.12500000,0.12962963,0.15277778,0.1388888889,0.006944444,0.06944444],
                  [0.13875598,0.17464115,0.148325359,0.02392344,0.02631579,0.22248804,0.07894737,0.0287081340,0.023923445,0.13397129],
                  [0.09375000,0.09843750,0.084375000,0.12812500,0.12187500,0.08437500,0.14687500,0.0500000000,0.070312500,0.12187500],
                  [0.59558824,0.39705882,0.007352941,0.00000000,0.00000000,0.00000000,0.00000000,0.0000000000,0.000000000,0.00000000],
                  [0.00000000,0.00000000,0.000000000,0.00000000,0.00000000,0.00000000,1.00000000,0.0000000000,0.000000000,0.00000000],
                  [0.00000000,0.00000000,0.000000000,0.00000000,0.00000000,0.00000000,0.00000000,0.0000000000,0.000000000,1.00000000],
                  [0.10497639,0.11732655,0.186342172,0.15691972,0.15219760,0.23247367,0.04940065,0.0003632401,0.000000000,0.00000000]])
    trans_matrix = np.transpose(trans_matrix, [1, 0])

    if args.data == 'random':
        trans_matrix = np.random.normal(0.0, 1, [nlabels, nlabels])
        trans_matrix = np.exp(trans_matrix * prob_scale)
        trans_matrix = trans_matrix / np.sum(trans_matrix, 0, keepdims=True)

    trans_matrix2 = random_noise(trans_matrix, bound)

    t2 = np.dot(trans_matrix, trans_matrix)
    t4 = np.dot(t2, t2)
    t8 = np.dot(t4, t4)
    t16 = np.dot(t8, t8)
    t32 = np.dot(t16, t16)
    t64 = np.dot(t32, t32)
    t128 = np.dot(t64, t64)

    dif = np.dot(t128, np.reshape(init_p, (-1, 1))) - \
        np.reshape(init_p, (-1, 1))

    if np.sum(np.square(dif)) < 1e-3 or args.data == 'zipcode':
        print('OK')
        break
    else:
        print('WARNING: Zipcode\'s stationary distribution is not uniform')
        break

print('Transition Matrix')
print(trans_matrix)


def KM_match(transformed, ground_truth):
    from KM import min_KM
    weight = np.zeros((transformed.shape[0], transformed.shape[0]))
    for i in range(weight.shape[0]):
        for j in range(weight.shape[1]):
            weight[i, j] = np.mean(np.square(transformed[i, :] - ground_truth[j, :]))
    KMdist, cor = min_KM(weight)
    KMdist /= weight.shape[0]
    return KMdist, cor


train_step = {
    'epochs': 21,
    'logging_step': 100,
    'seq_len': 200,
    'dim': 23,
    'markov': 2,
    'embed': 128
}


# pair-wise data
def sample_image_seq(seq_len=10):
    choice = [i for i in range(nlabels)]
    horizon = seq_len
    image = np.zeros((horizon, 28, 28, 1))
    label = np.zeros((horizon, nlabels))
    g_size = train_step['markov']
    for i in range(seq_len // train_step['markov']):
        num = int(np.random.choice(choice, 1, p=init_p))
        for j in range(g_size):
            idx = np.random.randint(len(image_corpus[num]))
            image[i * g_size + j, :, :, :] = image_corpus[num][idx]
            label[i * g_size + j, num] = 1.0
            num = int(np.random.choice(choice, 1, p=trans_matrix[:, num]))
    return image, label


def sample_image_seq2(seq_len=10):
    choice = [i for i in range(nlabels)]
    horizon = seq_len
    image = np.zeros((horizon, 28, 28, 1))
    label = np.zeros((horizon, nlabels))
    g_size = train_step['markov']
    for i in range(seq_len // train_step['markov']):
        num = int(np.random.choice(choice, 1, p=init_p))
        for j in range(g_size):
            idx = np.random.randint(len(image_corpus[num]))
            image[i * g_size + j, :, :, :] = image_corpus[num][idx]
            label[i * g_size + j, num] = 1.0
            num = int(np.random.choice(choice, 1, p=trans_matrix2[:, num]))
    return image, label


def main():
    # GPU configure
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    path = 'mnist_%s_logs_%.2f/' % (args.model, bound)
    log_file_name = path + 'mnist-scale_' + str(prob_scale) + 'markov_' + \
                str(train_step['markov']) + 'd_op.log'
    np.save(log_file_name + '.data.trans', np.transpose(trans_matrix, (0, 1)))
    with open(log_file_name, 'w') as f:
        f.write(time.strftime('%Y.%m.%d',time.localtime(time.time())) + '\n')
        f.write('Initial distribution:\n')
        for digit in range(nlabels):
            if digit > 0:
                f.write(',')
            f.write('{:.8f}'.format(init_p[digit]))
        f.write('\n')
        f.write('Trans distribution:\n')
        for digit1 in range(nlabels):
            for digit2 in range(nlabels):
                if digit2 > 0:
                    f.write(',')
                f.write('{:.8f}'.format(trans_matrix[digit2, digit1]))
            f.write('\n')
        for digit1 in range(nlabels):
            for digit2 in range(nlabels):
                if digit2 > 0:
                    f.write(',')
                f.write('{:.8f}'.format(trans_matrix2[digit2, digit1]))
            f.write('\n')

    with tf.Session(config=config) as s:
        # CycleGAN Model

        n_concat = 2
        if args.model == 'nopd':
            n_concat = 1

        model = \
            mnist_model.UniMappingGAN(s,
                                      seq_len=train_step['seq_len'],
                                      width=28, height=28, nlabels=nlabels,
                                      markov=n_concat)

        # Initializing
        s.run(tf.global_variables_initializer())

        global_step = 0

        mses = []
        for my_iter in range(1000):
            _, pd, kk = s.run([model.e_op, model.loss_inner_product,
                              model.inner_product],
                              feed_dict={model.lr_decay: 1.0})
        for epoch in range(train_step['epochs']):
            # learning rate decay
            lr_decay = 1.
            if epoch >= 100 and epoch % 10 == 0:
                lr_decay = (train_step['epochs'] - epoch) / (
                    train_step['epochs'] / 2.)
            
            for i in range(1000):
                n_decay = 0.5 * max((20000 - global_step + 0.0) / 20000, 0) + 0.5
                if (global_step + 1) % train_step['logging_step'] == 0:
                    accs = []
                    bz = train_step['seq_len']
                    for ii in range(test_images.shape[0] // train_step['seq_len']):
                        image_t = test_images[ii * bz : (ii + 1) * bz, :, :, :]
                        label_t = np.zeros((bz, nlabels))
                        for j in range(bz):
                            label_t[j, test_labels[ii * bz + j]] = 1.0
                        acc = s.run(model.accuracy, feed_dict={model.is_training: False, model.image: image_t, model.olabel: label_t})
                        accs.append(acc)
                    print('[+] Test Accuracy = {}'.format(float(np.mean(accs))))

                    # write logs
                    mytrans = np.zeros((nlabels, nlabels))
                    mycnt = np.zeros((nlabels,))

                    cm = np.zeros((nlabels, nlabels))
                    for u in range(100):
                        u_gta, u_ = sample_image_seq(train_step['seq_len'])
                        u_gt = np.argmax(u_, 1)
                        u_pd = s.run(model.pred, feed_dict={model.image: u_gta,
                                                            model.is_training: False})
                        u_fs = np.argmax(u_pd, 1)
                        for k in range(train_step['seq_len'] - 1):
                            if (k + 1) % train_step['markov'] != 0:
                                mytrans[u_fs[k], u_fs[k + 1]] += 1.0
                                mycnt[u_fs[k]] += 1.0
                            cm[u_gt[k], u_fs[k]] += 1
                    for x in range(nlabels):
                        mytrans[x, :] /= max(1.0, mycnt[x])
                        cm[x, :] /= np.sum(cm[x, :])

                    # write logs
                    np.save(log_file_name + '.e{}'.format(global_step) + '.trans', mytrans)
                    np.save(log_file_name + '.e{}'.format(global_step) + '.pred', cm)
                    with open(log_file_name, 'a') as f:
                        f.write('Global Step {}\n'.format(global_step))
                        f.write('  Overall Test Accuracy: {}\n'.format(np.mean(accs)))
                        f.write('  Transition Matrix: ')
                        for digit1 in range(nlabels):
                            for digit2 in range(nlabels):
                                if digit2 > 0:
                                    f.write(',')
                                else:
                                    f.write('  ')
                                f.write('{:.8f}'.format(mytrans[digit1,
                                                                digit2]))
                            f.write('\n')
                        f.write('  Predictive Matrix: ')
                        for digit1 in range(nlabels):
                            for digit2 in range(nlabels):
                                if digit2 > 0:
                                    f.write(',')
                                else:
                                    f.write('  ')
                                f.write('{:.8f}'.format(cm[digit1, digit2]))
                            f.write('\n')

                    label_seq = np.zeros((200, nlabels))
                    for i in range(200):
                        label_seq[i, i % 10] = 1.0
                    imgs = s.run(model.b2a, feed_dict={model.label: label_seq,
                                                       model.n_decay: n_decay})
                    save_image_collections(imgs,
                                           path + 'img%d' % global_step)
                    
                    for k in range(10):
                        for digit in range(10):
                            label_seq = np.zeros((200, nlabels))
                            label_seq[:, digit] = 1.0
                            imgs = s.run(model.b2a, feed_dict={model.label: label_seq,
                                                               model.n_decay: n_decay})
                            save_image_collections(imgs, path + 'final_img_d%d_%d' % (digit, k))
                for _ in range(model.n_train_critic):
                    a_img, a_label = sample_image_seq(train_step['seq_len'])
                    b_img, b_label = sample_image_seq2(train_step['seq_len'])
                    _ = s.run([model.d_op],
                              feed_dict={model.image: a_img,
                                         model.label: b_label,
                                         model.is_training: True,
                                         model.lr_decay: lr_decay,
                                         model.n_decay: n_decay})

                a_img, a_label = sample_image_seq(train_step['seq_len'])
                b_img, b_label = sample_image_seq2(train_step['seq_len'])

                w, gp, g_loss, acc, _, mse = \
                    s.run([model.w, model.gp, model.g_loss, model.accuracy,
                           model.g_op, model.g_super_loss],
                          feed_dict={model.image: a_img, model.label: b_label,
                                     model.olabel: a_label,
                                     model.is_training: True,
                                     model.lr_decay: lr_decay,
                                     model.n_decay: n_decay})
                mses.append(mse)

                if global_step % train_step['logging_step'] == 0:
                    # Print loss
                    tr, gt, pred, bd = s.run([model.a2b, model.b, model.pred, model.embeddings],
                                   feed_dict={model.image: a_img,
                                              model.label: a_label,
                                              model.is_training: False, model.n_decay: n_decay})
                    pred = np.argmax(pred, 1)
                    kmdist, _ = KM_match(tr, gt)
                    print("[+] Global Step %08d =>" % global_step,
                          " G loss     : {:.8f}".format(g_loss),
                          " w          : {:.8f}".format(w),
                          " gp         : {:.8f}".format(gp),
                          " Accuracy   : {:.8f}".format(acc),
                          " L2 norm    : {:.8f}".format(mse),
                          " KM dist    : {:.8f}".format(kmdist))

                    # DEFIF DEBUG
                    cm = np.zeros((nlabels, nlabels))

                    gg = np.argmax(a_label, 1)
                    for i in range(pred.shape[0]):
                        cm[gg[i]][pred[i]] += 1.0

                    my_trans = np.zeros((nlabels, nlabels))
                    cnt = np.zeros((nlabels,))
                    for i in range(pred.shape[0] - 1):
                        my_trans[pred[i]][pred[i + 1]] += 1.0
                        cnt[pred[i]] += 1.0
                    for i in range(nlabels):
                        my_trans[i, :] /= max(cnt[i], 1.0)
                    # ENDIF

                    mses = []

                global_step += 1

            d_a, d_a2b = [], []

            if args.model == 'nodop':
                continue
            
            print('[+] Supervised Shuffling, A->B ...')

            horizon = train_step['seq_len']
            mytrans = np.zeros((nlabels, nlabels))
            mycnt = np.zeros((nlabels,))

            cm = np.zeros((nlabels, nlabels))
            for u in range(5000):
                u_gta, u_ = sample_image_seq(horizon)
                u_gt = np.argmax(u_, 1)
                u_pd = s.run(model.pred, feed_dict={model.image: u_gta,
                                                    model.is_training: False})
                u_fs = np.argmax(u_pd, 1)
                for k in range(horizon - 1):
                    if (k + 1) % train_step['markov'] != 0:
                        mytrans[u_fs[k], u_fs[k + 1]] += 1.0
                        mycnt[u_fs[k]] += 1.0
                    cm[u_gt[k], u_fs[k]] += 1
            for x in range(nlabels):
                mytrans[x, :] /= max(1.0, mycnt[x])

            myweight = np.zeros((nlabels, nlabels))
            for x in range(nlabels):
                for y in range(nlabels):
                    myweight[x, y] = trans_matrix2[y, x]

            best_se = 1e9
            reord_label = [0] * horizon
            solution = []
            for perm in itertools.permutations([k for k in range(nlabels)]):
                cur_se = 0.
                for x in range(nlabels):
                    for y in range(nlabels):
                        cur_se += np.abs(
                            mytrans[x, y] - myweight[perm[x], perm[y]])
                if cur_se < best_se:
                    best_se = cur_se
                    solution = [perm[k] for k in range(nlabels)]
            print('Discrete Optimization Solution')
            print(solution)

            for miter in range(100):
                gta, _ = sample_image_seq(horizon)
                tb, pd, bd = s.run(
                    [model.a2b, model.pred, model.embeddings],
                    feed_dict={model.image: gta,
                               model.is_training: False})
                
                pred_idx = np.argmax(pd, 1)
                gt_idx = np.argmax(_, 1)

                for k in range(horizon):
                    reord_label[k] = solution[pred_idx[k]]

                acc_o = sum([pred_idx[k] == gt_idx[k] for k in range(horizon)])
                acc_r = sum([reord_label[k] == gt_idx[k]
                             for k in range(horizon)])
                if miter == 0:
                    print('Origin Acc {}, Reordered Acc {}, '.
                          format(acc_o, acc_r))

                gtab = np.zeros((horizon, nlabels))
                for k in range(horizon):
                    gtab[k, reord_label[k]] = 1.0

                d_a.append(gta)
                d_a2b.append(gtab)

            decay = 1.0
            for k in range(2000):
                idx = np.random.randint(100)
                img, lb = sample_image_seq(train_step['seq_len'])
                n_decay = max((20000 - global_step + 0.0) / 20000, 0) * 0.5 + 0.5
                _, ls_ssp = \
                    s.run([model.gs_op, model.g_super_loss],
                          feed_dict={model.image: d_a[idx],
                                     model.label: lb,
                                     model.olabel: d_a2b[idx],
                                     model.lr_decay: decay,
                                     model.is_training: True,
                                     model.n_decay: n_decay})
                if k % 100 == 0:
                    decay *= 0.8
                if k % 100 == 0:
                    print('[+] Supervised Shuffling, MSE Loss = {}'.
                          format(ls_ssp))

    # Close tf.Session
    s.close()


if __name__ == '__main__':
    main()
