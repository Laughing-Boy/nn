

![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)

## Term 2, Project 1 

### Abstract 

After sucessful completetion on term-1, we are moving to the practical part of RoboND. We are training and evaluating a neural network (google lenet in my case) to classify and detect objects in realtime. In this project, we are using Nvidia Digits 6.0 as our software and p2x.large GPU as our hardware. Digits 6.0 provides us, nice black box and interface on top of neural networks .The neural network framework used here is caffe.  We have trained neural network on two datasets (Bottles, Candybox, Nothing) and (Human, Cube, Nothing).  



### Background

Neural Networks were developed in early 1990's. But were not very useful until 2010 because they require huge data and computation. Number of architectures were proposed for classification and detection of Images. As years progressed, Network got deeper and deeper i.e., number of parameters increased. Out of 3 architectures given in Digits, Google lenet is most sophisticated network. It was the winner of 2014 ILSVRC. It has most number of parameters among the networks.



###![network diagram](https://joelouismarino.github.io/images/blog_images/blog_googlenet_keras/googlenet_diagram.png)

### Intro and Data Acquisition

My project is to make a network that classifies human, object and background. It has three classes. I have chosen `Rubix Cube` as my object because rubix cube as many orientations. If my network generalises on cube, then it can easily work on other objects. I took images on both mobile and laptop. 

After taking the images, I used following script to convert them into PNG and change their aspect ratio,

```python
from PIL import Image
import os, sys

path = "/home/ubuntu/data/"
dirs = os.listdir( path )

def resize():
    i = 1
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((500,500), Image.ANTIALIAS)
            imResize.save(f + '.png', 'PNG', quality=95)
            os.rename(os.path.join(path, item), os.path.join(path, 'human_'+str(i)+'.png'))
            i = i+1

resize()
```

For human images, I used images from Caltech 256 dataset. It has 430 images of more than 15 people.

I used three **different backgrounds** for robust training and testing.
![cube_87__fliph.png](https://udacity-reviews-uploads.s3.us-west-2.amazonaws.com/_attachments/128546/1512164004/cube_87__fliph.png) ![cube_71__fliph.png](https://udacity-reviews-uploads.s3.us-west-2.amazonaws.com/_attachments/128546/1512164046/cube_71__fliph.png)![cube_43__fliph.png](https://udacity-reviews-uploads.s3.us-west-2.amazonaws.com/_attachments/128546/1512164099/cube_43__fliph.png)

Finally I used data augumentation techniques to scaleup my dataset.
Here are some examples:
![Screenshot 2017-12-02 02.56.16.png](https://udacity-reviews-uploads.s3.us-west-2.amazonaws.com/_attachments/128546/1512163795/Screenshot_2017-12-02_02.56.16.png)
![Screenshot 2017-12-02 02.56.06.png](https://udacity-reviews-uploads.s3.us-west-2.amazonaws.com/_attachments/128546/1512163626/Screenshot_2017-12-02_02.56.06.png)

I finally split my dataset into 80% training and 20% testing.

### 

### Hyperparameters

```basic
- Epoch : 30
- Learning Rate : 0.001
- Optimiser : Adam

```

Initially I started with high learning of `0.1`, My accuracy did not increase with training. So, It steeped at 50%. So I reduced my learning rate to `0.001`. It gave me 98% percent accuracy after few epochs.

More the number of epochs better the accuracy. So I went by default value of `30`. It worked well.

Among optimisers, ADAM is first choice. But, I later changed the optimiser. There was no much difference in accuracy.

 

### Results

I used `evaluste` command to get the results. I got accuracy of 78% on **Bottles, Candybox, Nothing dataset**. Inference time was `4.76 ms`. 



![Screenshot 2017-12-01 04.27.09.png](https://udacity-reviews-uploads.s3.us-west-2.amazonaws.com/_attachments/128546/1512166625/Screenshot_2017-12-01_04.27.09.png)



![Screenshot 2017-12-01 04.02.39.png](https://udacity-reviews-uploads.s3.us-west-2.amazonaws.com/_attachments/128546/1512167101/Screenshot_2017-12-01_04.02.39.png)



On **Human, Cube, Nothing dataset**, My Inference time was `5.2ms`. The model generalised quite well on the data.



![alt-tag](https://raw.githubusercontent.com/jyoth1raditya/nn/master/Screenshot_2017-12-02_03.58.14%20(1).png)



![Screenshot 2017-12-02 01.18.14.png](https://udacity-reviews-uploads.s3.us-west-2.amazonaws.com/_attachments/128546/1512167428/Screenshot_2017-12-02_01.18.14.png)



### Inference time vs Accuracy

There is no straight answer for Inference time vs Accuracy. It mainly depends on system requirements.
As robotic tasks are intended to happen in real time, Inference time is very crucial. 
Ideal situation is small inference time with acceptable accuracy.

### Future Enhancements

- Due to limited time, There are only 3 categories. But more objects can be added to same model with new data.
- We can add more backgrounds for generalisation.
- More data, More accuracy. We can scrap internet for more images of these objects and feed them for training.

### Future of systems

- As it is my product is not commercially feasible. As Once you add more `objects and data`, it can be used on variety of data driven tasks. 
- Smart refrigerators are using these techniques to intelligently detect the objetcs inside it. So, that it can place orders online according to needs of customers.
- The bottleneck for these systems to become mainstream is accuracy. Neural networks are very high dimensional. They are many `false positives` in these systems.
- Currently, Digits only supports three NN architectures. But there are more advances in field like Resnet - 500. So, in a commercial product we would not use digits. Digits is more for `testing` the feasibility of product idea for non ML people.
- Other limitation of Digits is it only supports NVIDIA (CUDANN) hardware. AMD GPUs, Intel Nervana would not work with Digits, 









