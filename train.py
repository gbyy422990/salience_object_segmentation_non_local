import cv2
import numpy as np
import model
import vgg16
import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_training_list():

    with open('pig1.txt') as f:
        lines = f.read().splitlines()

    files = []
    labels = []

    for line in lines:
        labels.append('./dataset/pig1/%s' % line)
        files.append('./dataset/piglabel1/%s' % line)

    print('num files',len(files))
    print('num label',len(labels))

    return files, labels


def load_train_val_list():

    files = []
    labels = []

    with open('pig1.txt') as f:

        lines = f.read().splitlines()

    for line in lines:
        labels.append('./dataset/piglabel1/%s' % line)
        files.append('./dataset/pig1/%s' % line)

    with open('pigtest1.txt') as f:
        lines = f.read().splitlines()

    for line in lines:
        labels.append('./dataset/pigtestlabel1/%s' % line)
        files.append('./dataset/pigtest1/%s' % line)

    print('num files', len(files))
    print('num label', len(labels))

    return files, labels


if __name__ == "__main__":

    model1 = model.Model()
    model1.build_model()
    pred = model1.Prob_C

    tf.add_to_collection('inputs', model1.input_holder)
    tf.add_to_collection('pred', model1.Prob)

    tf.summary.scalar('loss total',model1.Loss_Mean)
    tf.summary.scalar('loss iou',model1.C_IoU_LOSS)


    tf.summary.image('image',model1.input_holder)
    #tf.summary.image('label', tf.expand_dims(model1.label_holder[:,:,:,0],axis=-1))
    tf.summary.image('pred',tf.expand_dims(pred[:,:,:,0],axis=-1))

    summary_op = tf.summary.merge_all()

    sess = tf.Session()

    train_writer = tf.summary.FileWriter('./logs',sess.graph)

    max_grad_norm = 1
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(model1.Loss_Mean, tvars), max_grad_norm)
    opt = tf.train.AdamOptimizer(1e-6)
    train_op = opt.apply_gradients(zip(grads, tvars))

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    train_list, label_list = load_train_val_list()

    n_epochs = 5
    img_size = model.img_size
    label_size = model.label_size


    for i in xrange(n_epochs):
        whole_loss = 0.0
        whole_acc = 0.0
        count = 0
        for f_img, f_label in zip(train_list, label_list):

            img = cv2.imread(f_img).astype(np.float32)
            img_flip = cv2.flip(img, 1)

            label = cv2.imread(f_label)[:, :, 0].astype(np.float32)
            label_flip = cv2.flip(label, 1)

            img = cv2.resize(img, (img_size, img_size)) - vgg16.VGG_MEAN
            label = cv2.resize(label, (label_size, label_size))
            label = label.astype(np.float32) / 255.

            img = img.reshape((1, img_size, img_size, 3))
            #print(img.shape)
            label = np.stack((label, 1-label), axis=2)
            label = np.reshape(label, [-1, 2])
            #print(label.shape)

            _, loss, step_summary,acc = sess.run([train_op, model1.Loss_Mean, summary_op, model1.accuracy],
                                    feed_dict={model1.input_holder: img,
                                               model1.label_holder: label})

            whole_loss += loss
            whole_acc += acc
            count = count + 1

            # add horizon flip image for training
            img_flip = cv2.resize(img_flip, (img_size, img_size)) - vgg16.VGG_MEAN
            label_flip = cv2.resize(label_flip, (label_size, label_size))
            label_flip = label_flip.astype(np.float32) / 255.

            img_flip = img_flip.reshape((1, img_size, img_size, 3))
            label_flip = np.stack((label_flip, 1 - label_flip), axis=2)
            label_flip = np.reshape(label_flip, [-1, 2])

            _, loss, loss_iou, step_summary, acc = sess.run([train_op, model1.Loss_Mean, model1.C_IoU_LOSS, summary_op, model1.accuracy],
                                    feed_dict={model1.input_holder: img_flip,
                                               model1.label_holder: label_flip})



            whole_loss += loss
            whole_acc += acc
            count = count + 1

            train_writer.add_summary(step_summary, count)

            print('loss ce',loss)
            print('loss iou',loss_iou)
            if count % 2 == 0:
                print "Loss of %d images: %f, Accuracy: %f" % (count, (whole_loss/count), (whole_acc/count))

        print "Epoch %d: %f" % (i, (whole_loss/len(train_list)))
        try:

            os.mkdir('Model_352')
        except:
            print('model folder existed!')
        saver.save(sess, './Model_352/model.ckpt', global_step=n_epochs)
        print('saved model')
