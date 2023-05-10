# GAN-Handwritten-digit-generation

## 总结
> 本项目基于GAN，实现了生成手写数字的图像，为GAN的学习实践提供了帮助。
> 在学习的过程中作者也体会到了GAN的强大之处，正如乔布斯说的站在人文学科与科学的交叉点上，GAN便是如此。
> 但是在学习GAN的同时也发现，GAN目前存在许多的不足，生成数据也存在普遍的质量不住，后续将会继续学习GAN的变种。

## 简介

生成式对抗神经网络GAN（**G**enerative **A**dversarial **N**etworks）是一种深度学习模型，由Ian Goodfellow等人在2014年提出。它是一种用于生成模拟数据的技术，如图像、音频和文本等。GAN是一种基于博弈论的模型，由两个神经网络组成：生成器（Generator）和判别器（Discriminator）。

生成器的作用是生成与真实数据相似的假数据，而判别器的作用是将真实数据和假数据进行区分。生成器和判别器之间通过博弈的方式进行学习和优化，最终生成器可以生成越来越接近真实数据的假数据。

GAN的工作原理类似于造假者和识别者之间的博弈。生成器试图以尽可能逼真的方式生成数据，并将其传递给判别器。判别器的任务是将生成器生成的数据与真实数据区分开来。生成器和判别器会反复进行这个过程，直到生成器能够生成与真实数据无法区分的数据。

GAN可以应用于许多领域，如图像生成、视频生成、语音合成、自然语言处理等。在图像生成方面，GAN已经可以生成逼真的人脸、风景和动物等图像。在自然语言处理方面，GAN已经可以用于生成逼真的文本和对话等。

GAN的优点是可以生成高质量的假数据，从而可以扩充数据集，提高模型的泛化能力。然而，GAN也存在一些缺点，如模型不稳定、生成的数据可能出现模糊和不真实等问题。此外，GAN的训练需要大量的计算资源和时间。

总的来说，GAN是一种强大的生成模型，可以应用于许多领域。随着深度学习技术的不断发展，GAN的应用前景也越来越广阔。



## 基本思想


GAN是一种生成式模型，其基本思想是通过两个神经网络模型的博弈来实现数据的生成。这两个模型分别被称为生成器（Generator）和判别器（Discriminator），它们通过不断地博弈来不断地提高自己的能力，最终实现数据的生成。

生成器的作用是生成与原始数据相似的虚假数据。它接受一个随机噪声作为输入，并生成一张图像、一段语音、一段文本等数据。生成器的输出会被送到判别器中进行判别，判别器会判断这个数据是否是真实的数据，如果是真实的数据，它会给出正面的评价，否则就给出负面的评价。

判别器的作用是判断输入的数据是否真实。它接受一个数据作为输入，并输出一个二元值（0或1），表示这个数据是否真实。如果输入的数据来自真实的数据集，判别器会给出正面的评价（1），否则就给出负面的评价（0）。

生成器和判别器通过对抗来提高自己的能力。生成器的目标是生成的数据越接近真实数据，判别器的目标是尽可能地区分真实数据和虚假数据。两个模型的博弈过程中，生成器会逐渐学习到真实数据的特征，从而生成越来越接近真实数据的虚假数据；判别器也会逐渐学习到真实数据和虚假数据的区别，从而判别越来越准确。

通过不断地迭代，生成器和判别器会逐渐趋于平衡，生成器生成的虚假数据越来越接近真实数据，判别器的准确率也越来越高。最终，GAN模型可以生成与原始数据相似的虚假数据，从而为数据增强、数据修复、数据融合等任务提供了有效的解决方案。



## 数学推导

GAN（Generative Adversarial Networks）的数学原理涉及到概率论、优化理论、信息论等多个数学领域。

GAN的基本思想是通过两个神经网络模型的博弈来实现数据的生成。生成器G和判别器D是两个对抗的模型，其中G生成虚假数据，D对这些虚假数据进行判别，并给出判别结果的概率。GAN的目标是通过不断的博弈过程，让生成器G生成的数据尽量接近真实数据，判别器D尽量正确地判别数据的真伪。具体来说，GAN的目标函数可以表示为：
$$
\min _{G} \max _{D} V(D, G):=\min _{G} \max _{D}\left(\mathbb{E}_{\mathbf{x} \sim \mu}[\log D(\mathbf{x})]+\mathbb{E}_{\mathbf{z} \sim \gamma}[\log (1-D(G(\mathbf{z})))]\right)\\
$$

其中，**X**代表真实数据，**P<sub>data</sub>(X)**表示真实数据的分布，**Z**代表生成器的输入噪声，P(Z)表示输入噪声的分布。$D(\boldsymbol{x})$表示判别器对真实数据的判别结果，**D(G(Z))**表示判别器对生成数据的判别结果。目标函数中的第一项表示真实数据被正确地判别的概率，第二项表示生成数据被错误地判别为真实数据的概率。目标函数的优化过程中，生成器G和判别器D会相互对抗，并不断地提高自己的能力，直到达到一个平衡点。

为了优化目标函数，通常采用交替优化的方法。具体来说，先固定生成器G，通过最大化目标函数来更新判别器D；再固定判别器D，通过最小化目标函数来更新生成器G。这个过程会不断地进行迭代，直到GAN模型收敛到一个稳定状态。

除了基本的GAN模型，还有许多变种，如DCGAN、WGAN、CGAN等，它们在数学原理上都有一定的区别和特点。例如，DCGAN中采用卷积神经网络来生成和判别图像数据，WGAN中采用Wasserstein距离作为损失函数来训练模型，CGAN则将生成器和判别器分别作用于条件数据和噪声输入。不同的变种模型采用不同的数学原理和算法，可以根据具体应用需求进行选择。


## 示例代码

### 项目地址

[GitHub地址-手写数字生成]([123spy/GAN-Handwritten-digit-generation: 基于GAN的手写数字生成 (github.com)](https://github.com/123spy/GAN-Handwritten-digit-generation))



### 部分代码

```python
# 定义生成模型
generator = tf.keras.Sequential([
    # 输入层
    layers.Dense(7*7*256, use_bias=False, input_shape=(latent_dim,)),
    # 标准化操作
    layers.BatchNormalization(),
    # leakRelu层
    layers.LeakyReLU(),
    # 重置大小
    layers.Reshape((7, 7, 256)),
    # 卷积层
    layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.LeakyReLU(),
    # 卷积层
    layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.LeakyReLU(),
    layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
])
```



```python
# 定义判别模型
discriminator = tf.keras.Sequential([
    layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
    layers.LeakyReLU(),
    layers.Dropout(0.3),
    layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
    layers.LeakyReLU(),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(1)
])
```

```python
# 定义生成损失函数
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

```

```python
# 定义判别损失函数
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

```

```python
# Define function to train the GAN
# 定义训练GAN的函数
def train(dataset, test_input, epochs):
    for epoch in range(epochs):
        for batch in dataset:
            gen_loss, disc_loss = train_step(batch)

        # Generate and save images every 10 epochs
        if (epoch + 1) % 10 == 0:
            generate_and_save_images(generator, epoch + 1, test_input)

        # Save model weights every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Epoch {}, Generator Loss: {:.4f}, Discriminator Loss: {:.4f}'.format(epoch+1, gen_loss, disc_loss))

```



## 参考视频

- GAN论文逐段精读【论文精读】- 李沫. [[Link\]](https://www.bilibili.com/video/BV1rb4y187vD/?spm_id_from=333.337.search-card.all.click&vd_source=f49218cb60f4f33426bf82f55cd9a775)
- 生成对抗网络GAN开山之作论文精读 - 同济子豪兄. [[Link\]](https://www.bilibili.com/video/BV1oi4y1m7np/?spm_id_from=333.337.search-card.all.click&vd_source=f49218cb60f4f33426bf82f55cd9a775)

## 参考文章

- Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. "Generative Adversarial Nets." In Advances in Neural Information Processing Systems, pp. 2672-2680, 2014. [[Link\]](https://papers.nips.cc/paper/5423-generative-adversarial-nets)
- Martin Arjovsky, Soumith Chintala, and Léon Bottou. "Wasserstein Generative Adversarial Networks." In International Conference on Machine Learning, pp. 214-223, 2017. [[Link\]](http://proceedings.mlr.press/v70/arjovsky17a.html)
- Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros. "Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks." In IEEE International Conference on Computer Vision, pp. 2223-2232, 2017. [[Link\]](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf)
- Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Andrew Tao, Jan Kautz, and Bryan Catanzaro. "High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs." In IEEE Conference on Computer Vision and Pattern Recognition, pp. 8798-8807, 2018. [[Link\]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_High-Resolution_Image_Synthesis_CVPR_2018_paper.pdf)
- Yujun Shen, Jinjin Gu, Xiaoou Tang, and Bolei Zhou. "Interpreting the Latent Space of GANs for Semantic Face Editing." In IEEE Conference on Computer Vision and Pattern Recognition, pp. 7266-7275, 2020. [[Link\]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Shen_Interpreting_the_Latent_Space_of_GANs_for_Semantic_Face_Editing_CVPR_2020_paper.pdf)
