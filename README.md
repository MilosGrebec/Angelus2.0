Angelus 2.0 is software for "creating", training, and testing custom computer vision models. It's made using TensorFlow object detection, and for UI I used KivyMD.

Fun fact: this project cost me an arm and a leg in terms of my hair. I was so nervus and stressed while doing this that I started balding. Well, jokes aside, it was hard setting up TensorFlow.

Since there are a lot of libraries that are just a pain in the ass to set up, here is a link to download the whole Python venv:

https://drive.google.com/drive/folders/1k0myqTlWjnB7F1-aAYvEowFn7UQnZbLK?usp=sharing

Angelus 2.0 is a project that I mostly worked on, and took me the most amount of time, but it's also the project that was most fun to do.

Models are from the TensorFlow models zoo, so if you want, you can change to a more advanced model from there, but training and testing the model will be hard depending on what model you choose and what GPU you have.

You ran Angelus by just running run.bat, I didn't do this as .exe because it's really slow, and just running it from the console is much faster.

Here is how Angelus looks when you first open it.

<img width="899" height="722" alt="image" src="https://github.com/user-attachments/assets/8eaa8cc8-2fbf-45ee-be89-04caa0155994" />

There are some custom models that I, or you, have already trained, and you can choose them from the combo box on the left.
That big white space is going to be your camera. When you press start, it will turn on, and your chosen model will be loaded.

Now on, make your own model button.
When you press it whis window will open.

<img width="697" height="925" alt="image" src="https://github.com/user-attachments/assets/9445072b-9cf0-4349-a137-c05291b7e604" />

Here you can make your own model and see just how magical data collection is. First you input your model name, then when you press the FOLDER button, it will give you a folder where to put your images of your object/s.
Then you have to label them. That is done with LabelImg. There is an example there.

<img width="695" height="926" alt="image" src="https://github.com/user-attachments/assets/5f5c8a93-f43e-41ae-acae-3409697b28d1" />

All that is left is to separate them into training and test parts, pick the best one for the test and all the others for training and, when you press train.

While Angelus is training, he will not be responding, and your GPU and CPU usage are going to spike to 100% (depending on what you have). Personally, I used this to test how well I built my PC (I custom-built it).
I used this to test the temperatures of my GPU and CPU while Angelus was training them.

When Angelus is responding again, you can just restart him, and you will have your custom model in combobox.


