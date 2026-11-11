# Angelus 2.0

Angelus 2.0 is software for **creating, training, and testing custom computer vision models**. It's made using **TensorFlow Object Detection**, and for the UI, I used **KivyMD**.

## Fun Fact

This project cost me an arm and a leg in terms of my hair. I was so nervous and stressed while doing this that I started balding.

Well, jokes aside, setting up TensorFlow was hard.

Angelus 2.0 is the project that I worked on the most, and it took me the most amount of time, but it was also the most fun to do.

## Setup

Since there are a lot of libraries that are just a pain in the ass to set up, here is a link to download the whole Python venv:

**[Download Python Virtual Environment](https://drive.google.com/drive/folders/1k0myqTlWjnB7F1-aAYvEowFn7UQnZbLK?usp=sharing)**

Models are from the **TensorFlow Model Zoo**, so if you want, you can change to a more advanced model from there. However, training and testing the model will be hard depending on which model you choose and what GPU you have.

### Running Angelus

You run Angelus by simply running `run.bat`.

I didn't make this an `.exe` because it's really slow, and running it from the console is much faster.

---

## Application Overview

Here is how Angelus looks when you first open it.

<img width="899" height="722" alt="Angelus 2.0 main interface" src="https://github.com/user-attachments/assets/8eaa8cc8-2fbf-45ee-be89-04caa0155994" />

There are some custom models that I or you have already trained, and you can choose them from the combo box on the left.

That big white space is going to be your camera. When you press **Start**, it will turn on, and your chosen model will be loaded.

## Creating Your Own Model

Now, on to the **Make Your Own Model** button.

When you press it, this window will open.

<img width="697" height="925" alt="Angelus 2.0 custom model creation" src="https://github.com/user-attachments/assets/9445072b-9cf0-4349-a137-c05291b7e604" />

Here, you can make your own model and see just how magical data collection is.

First, you input your model name. Then, when you press the **FOLDER** button, it will give you a folder where you can put images of your object(s).

Then you have to label them. That is done with **LabelImg**. There is an example there.

<img width="695" height="926" alt="Angelus 2.0 dataset preparation and image labeling" src="https://github.com/user-attachments/assets/5f5c8a93-f43e-41ae-acae-3409697b28d1" />

## Training Your Model

All that is left is to separate the images into training and test parts, pick the best ones for testing and all the others for training, and then press **Train**.

While Angelus is training, it will not be responding, and your GPU and CPU usage are going to spike to 100% (depending on what you have).

Personally, I used this to test how well I built my PC (I custom-built it). I used it to test the temperatures of my GPU and CPU while Angelus was training the models.

When Angelus starts responding again, you can just restart it, and you will have your custom model in the combo box.