from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow as tf
from GAN import generator_g, generator_f, discriminator_x, discriminator_y
from tensorflow.keras.optimizers import Adam
import time
from utils import train_x,train_y,test_x,test_y, img_cols,img_rows,channels
import matplotlib.pyplot as plt
import os

epochs = 100
LAMBDA = 100

gen_g_optimizer = gen_f_optimizer = Adam(lr=0.002,beta_1=0.5)
dis_x_optimizer = dis_y_optimizer = Adam(lr=0.002,beta_1=0.5)
loss = BinaryCrossentropy(from_logits=True)

checkpoint_dirs = 'Sketch_2_Colour_training_checkpoints'
checkpoint_prefix_g = os.path.join(checkpoint_dirs,"ckpt_g")
checkpoint_prefix_f = os.path.join(checkpoint_dirs,"ckpt_f")
checkpoint_g = tf.train.Checkpoint(generator_optimizer = gen_g_optimizer, discriminator_optimizer = dis_x_optimizer, generator = generator_g, discriminator = discriminator_x)
checkpoint_f = tf.train.Checkpoint(generator_optimizer = gen_f_optimizer, discriminator_optimizer = dis_y_optimizer, generator = generator_f, discriminator = discriminator_y)

def generate_images():
    # Sample images
    x = next(iter(test_x.shuffle(1000))).numpy()
    y = next(iter(test_y.shuffle(1000))).numpy()

    # Get predictions for those images
    y_hat = generator_g.predict(x.reshape((1, img_rows, img_cols, channels)))
    x_hat = generator_f.predict(y.reshape((1, img_rows, img_cols, channels)))

    # Plot images
    plt.figure(figsize=(12, 12))

    images = [x[0], y_hat[0], y[0], x_hat[0]]

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def discriminator_loss(real,generated):

    return (loss(tf.ones_like(real),real) + loss(tf.zeros_like(generated),generated))*0.5

def gen_loss(validity):

    return loss(tf.ones_like(validity),validity)

def image_similarity(img1,img2):

    return tf.reduce_mean(tf.abs(img1 - img2))

#generator_g converts x to y
#generator_f converts y to x

@tf.function
def step(real_x,real_y):

    with tf.GradientTape(persistent=True) as tape:

        fake_y = generator_g(real_x,training = True)
        gen_g_validity = discriminator_y(fake_y,training=True)
        dis_y_loss = discriminator_loss(discriminator_y(real_y,training = True), gen_g_validity)

        with tape.stop_recording():

            discriminator_y_gradients = tape.gradient(dis_y_loss,discriminator_y.trainable_variables)
            dis_y_optimizer.apply_gradients(zip(discriminator_y_gradients,discriminator_y.trainable_variables))

        fake_x = generator_f(real_y,training = True)
        gen_f_validity = discriminator_x(fake_x,training=True)
        dis_x_loss = discriminator_loss(discriminator_x(real_x,training = True), gen_f_validity)

        with tape.stop_recording():

            discriminator_x_gradients = tape.gradient(dis_x_loss,discriminator_x.trainable_variables)
            dis_x_optimizer.apply_gradients(zip(discriminator_x_gradients,discriminator_x.trainable_variables))

        gen_g_adv_loss = gen_loss(gen_g_validity)
        gen_f_adv_loss = gen_loss(gen_f_validity)

        cyc_x = generator_f(fake_y,training = True)
        cyc_x_loss = image_similarity(cyc_x,real_x)

        cyc_y = generator_g(fake_x,training = True)
        cyc_y_loss = image_similarity(cyc_y,real_y)

        id_x = generator_f(real_x)
        id_x_loss = image_similarity(id_x,real_x)

        id_y = generator_g(real_y)
        id_y_loss = image_similarity(real_y,id_y)

        gen_g_loss = gen_g_adv_loss + (cyc_x_loss+cyc_y_loss)*LAMBDA + id_y_loss*LAMBDA*0.5
        gen_f_loss = gen_f_adv_loss + (cyc_x_loss+cyc_y_loss)*LAMBDA + id_x_loss*LAMBDA*0.5

        with tape.stop_recording():

            generator_g_gradients = tape.gradient(gen_g_loss,generator_g.trainable_variables)
            gen_g_optimizer.apply_gradients(zip(generator_g_gradients,generator_g.trainable_variables))

            generator_f_gradients = tape.gradient(gen_f_loss,generator_f.trainable_variables)
            gen_f_optimizer.apply_gradients(zip(generator_f_gradients, generator_f.trainable_variables))

checkpoint_g.restore(tf.train.latest_checkpoint(checkpoint_dirs))
checkpoint_f.restore(tf.train.latest_checkpoint(checkpoint_dirs))

for epoch in range(epochs):

    print('Epochs:{}'.format(epoch))
    start = time.time()

    #Each Batch
    for k, (real_x,real_y) in enumerate(tf.data.Dataset.zip((train_x, train_y))):
        if k%100 == 0:
            print(k)
        step(tf.reshape(real_x,(1,img_rows,img_cols,channels)),tf.reshape(real_y,(1,img_rows,img_cols,channels)))

    generate_images()
    print('Time Taken: {}'.format(time.time()-start))

    if epoch%5==0:
        checkpoint_g.save(file_prefix=checkpoint_prefix_g)
        checkpoint_f.save(file_prefix=checkpoint_prefix_f)