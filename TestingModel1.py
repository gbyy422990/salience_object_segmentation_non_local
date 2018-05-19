import cv2
import numpy as np
import model
import os
import sys
import tensorflow as tf
import time
import vgg16

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

model_dir = './Model_352'

path = '/notebooks/huludao'

def load_model():
    file_meta = os.path.join(model_dir, 'model.ckpt-5.meta')
    file_ckpt = os.path.join(model_dir, 'model.ckpt-5')

    saver = tf.train.import_meta_graph(file_meta)
    # tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES

    sess = tf.InteractiveSession()
    saver.restore(sess, file_ckpt)
    # print(sess.run(tf.get_default_graph().get_tensor_by_name("v1:0")))
    return sess



def crop(pig_image):
    shape = pig_image.shape
    rate = float(shape[0]) / float(shape[1])
    if rate > 0.75:
        width = int(float(shape[1]) / 4 * 3)
        star = (shape[0] - width) / 2
        end = (shape[0] + width) / 2
        pig_image = pig_image[star:end, ...]
    elif rate < 0.75:
        height = int(float(shape[0]) / 3 * 4)
        star = (shape[1] - height) / 2
        end = (shape[1] + height) / 2
        pig_image = pig_image[:, star:end, :]

    return pig_image


if __name__ == "__main__":

    img_size = model.img_size
    label_size = model.label_size


    '''ckpt = tf.train.get_checkpoint_state('./model_bis/')
    saver = tf.train.Saver()
    saver.restore(sess, ckpt.model_checkpoint_path)'''

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./Model_352/model.ckpt-5.meta')
        saver.restore(sess,tf.train.latest_checkpoint('./Model_352'))
        X = tf.get_collection('inputs')[0]
        pred = tf.get_collection('pred')[0]
        #print(X)


        datasets = ['MSRA-B']


        img = os.listdir(path)
        print('img',img)
        for i in img:
            if i != '.DS_Store':
                #print('i',i)
                imgs = os.listdir(path + '/' + i)
                for f_img in imgs:
                    if f_img[-4:] == '.jpg' or f_img[-4:] == '.jpeg':
                        print('f_img',f_img)

                        img = cv2.imread(path + '/' + i + '/' + f_img)
                        print('before crop',img.shape)

                        img = crop(img)
                        print('after crop',img.shape)

                        img = cv2.resize(img, (img_size, img_size)) - vgg16.VGG_MEAN
                        img = img.reshape((1, img_size, img_size, 3))
                        print(img.shape)

                        start_time = time.time()
                        result = sess.run(pred,feed_dict={X:img})
                        print("--- %s seconds ---" % (time.time() - start_time))
                        print('shape',result.shape)

                        result = np.reshape(result, (label_size, label_size, 2))
                        result = result[:, :, 0]*255
                        #_, result = cv2.threshold(result, 50, 255, cv2.THRESH_BINARY)

                        result = cv2.resize(np.squeeze(result), (1600, 1200))

                        try:
                            os.mkdir('/notebooks/NLDF/huludao/' + i)
                        except Exception as e:
                            print(e)
                            #print('path existed')

                        cv2.imwrite('/notebooks/NLDF/huludao/' + i + '/' + f_img, result)

                        print('Ok')


