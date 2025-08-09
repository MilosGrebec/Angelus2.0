import customtkinter as ck
import cv2
from PIL import Image, ImageTk
import tkinter as tk
import numpy as np
import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import subprocess

class HowApp(ck.CTk):
    def __init__(self):
        super().__init__()
        self.title("Make your own model")
        self.geometry("700x900")
        self.grid_columnconfigure(0,weight=1)
        self.rowconfigure(0,weight=1)

        self.FRAME=ck.CTkScrollableFrame(self,width=700, height=900,corner_radius=0, fg_color="transparent")
        self.FRAME.grid(row=0,column=0,sticky="nsew")
        self.FRAME.grid_columnconfigure(0,weight=1)
        self.label = ck.CTkLabel(self.FRAME, text="How to make and train your own model", font=("Roboto",20))
        self.label.grid(row=0,column=0,pady=20)

        self.label2 = ck.CTkLabel(self.FRAME, text="1.Data Collection", font=("Roboto", 15))
        self.label2.grid(row=1, column=0, pady=10)

        self.text = ck.CTkTextbox(self.FRAME, state="normal", width=600, height=80)
        self.text.grid(row=2,column=0,pady=20,padx=20)
        self.text.insert("0.0","The first step for making your own object detection model is to give it name, input your model name bel-ow, and how many object is it going to detect. For example if you want to make model of you and your  mom, name would be: 'mom,me' with number of objects detecting being 2 with first object name being 'mom' and second 'me'. ")
        self.text.configure(state="disabled")

        self.label2.grid(row=1, column=0, sticky="w", padx=(20, 0))
        self.text.grid(row=2, column=0, pady=20, padx=20)

        self.MFrame=ck.CTkFrame(self.FRAME, fg_color="transparent")
        self.MFrame.grid(row=3,column=0,pady=10)
        self.label3=ck.CTkLabel(self.MFrame,text="Input your model name: ")
        self.label3.grid(row=0,column=0,padx=(20,20))
        self.textbox2=ck.CTkTextbox(self.MFrame,width=100,height=20)
        self.textbox2.grid(row=0,column=1,padx=(0,20))

        self.text2=ck.CTkTextbox(self.FRAME,state="normal",width=600,height=80)
        self.text2.insert("0.0","Second step is to get the actual data and that is pictures, take as many pictures as you can and name    them after your object model name. For example in model: 'mom,me' pictures of your mom would be     mom, mom(1), mom(2) and so on, and me, me(1),me(2)... After you take those pictures with your phone or anything, press the button below and add those pictures to that folder.")
        self.text2.grid(row=4,column=0,pady=10)
        self.text2.configure(state="disabled")

        self.button=ck.CTkButton(self.FRAME,text="Folder", command=self.modelFolder)
        self.button.grid(row=5,column=0,pady=10)

        self.text3=ck.CTkTextbox(self.FRAME,state="normal",width=600,height=100)
        self.text3.insert("0.0","Now third step is to label those images with labelImg.py press button below to start it. When you open itpress Open Dir to open your folder with pictures and press W to draw anotation (name of the annotation should be same as object) on exact corner of object, see example below and do that for every object     you want to detect and for every picture. CAUTION: When you finish labeling one image press ctrl+s to  save it.")
        self.text3.grid(row=6,column=0,pady=10)
        self.text3.configure(state="disabled")

        self.button2=ck.CTkButton(self.FRAME,text="Label",command=self.LabelImg)
        self.button2.grid(row=7,column=0,pady=10)

        self.ImgFrame=ck.CTkFrame(self.FRAME,fg_color="transparent")
        self.ImgFrame.grid(row=8,column=0,pady=10)
        self.ImgFrame.grid_rowconfigure(0,weight=1)

        self.Cav1=ck.CTkCanvas(self.ImgFrame,width=200,height=100)
        self.Cav1.grid(row=0,column=0,padx=20)

        self.Cav2=ck.CTkCanvas(self.ImgFrame,width=200,height=100)
        self.Cav2.grid(row=0, column=1, padx=20)
        self.loadimg()

        self.label4= ck.CTkLabel(self.FRAME,text="2.Training", font=("Roboto",15))
        self.label4.grid(row=9,column=0,pady=10,sticky="w",padx=20)

        self.text4=ck.CTkTextbox(self.FRAME,state="normal",width=600,height=80)
        self.text4.grid(row=10,column=0,pady=10)
        self.text4.insert("0.0","Next thing to do is to separate your images to train and test partitions to do that press button below it    will open file explorer in which you will see 2 folders test and train now chose the best and move it to the test folder along with its xml pair, and all others to the train folder. Notice: if your model have 2 objects like 'mom,me' you will chose best looking one for both and move them to one test and train folder.")
        self.text4.configure(state="disabled")

        self.button3=ck.CTkButton(self.FRAME,text="Separate",command=self.traintest)
        self.button3.grid(row=11,column=0,pady=10)

        self.text5=ck.CTkTextbox(self.FRAME,state="normal",width=600,height=65)
        self.text5.insert("0.0","Final thing is just to press Train button and cmd will open and will start training and while it's doing that  your CPU usage will be near 100%, when it's done restart this program and you will have your model in    combo box.")
        self.text5.grid(row=12,column=0,pady=10)
        self.text5.configure(state="disabled")

        self.button4=ck.CTkButton(self.FRAME,text="Train",command=self.train)
        self.button4.grid(row=13,column=0,pady=10)
    def train(self):
        self.modelName = self.textbox2.get("0.0", "end")
        CUSTOM_MODEL_NAME = "model_"+self.modelName.strip()
        PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
        PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
        TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
        LABEL_MAP_NAME = 'label_map.pbtxt'
        paths = {
            'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
            'SCRIPTS_PATH': os.path.join('Tensorflow', 'scripts'),
            'APIMODEL_PATH': os.path.join('Tensorflow', 'models'),
            'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace', 'annotations'),
            'IMAGE_PATH': os.path.join('Tensorflow', 'workspace', 'images'),
            'MODEL_PATH': os.path.join('Tensorflow', 'workspace', 'models'),
            'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace', 'pre-trained-models'),
            'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME),
            'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'export'),
            'TFJS_PATH': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'tfjsexport'),
            'TFLITE_PATH': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'tfliteexport'),
            'PROTOC_PATH': os.path.join('Tensorflow', 'protoc')
        }
        files = {
            'PIPELINE_CONFIG': os.path.join('Tensorflow', 'workspace', 'models',CUSTOM_MODEL_NAME,'pipeline.config'),
            'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME),
            'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
        }
        for path in paths.values():
            if not os.path.exists(path):
                if os.name == 'nt':
                    proc = subprocess.run('mkdir '+path, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                    print(proc.stdout.decode())
        self.model=self.modelName.split(",")
        names_list = [name.strip() for name in self.modelName.split(',')]
        labels = [{'name': name, 'id': index + 1} for index, name in enumerate(names_list)]
        print("bitno jedan")
        with open(files['LABELMAP'], 'w') as f:
            for label in labels:
                print("bitno jako")
                f.write('item { \n')
                f.write('\tname:\'{}\'\n'.format(label['name']))
                f.write('\tid:{}\n'.format(label['id']))
                f.write('}\n')
        train_command = [
            os.path.join('zmaj', 'Scripts', 'python.exe'),
            files['TF_RECORD_SCRIPT'],
            '-x', os.path.join(paths['IMAGE_PATH'], 'train'),
            '-l', files['LABELMAP'],
            '-o', os.path.join(paths['ANNOTATION_PATH'], 'train.record')
        ]
        test_command = [
            os.path.join('zmaj', 'Scripts', 'python.exe'),
            files['TF_RECORD_SCRIPT'],
            '-x', os.path.join(paths['IMAGE_PATH'], 'test'),
            '-l', files['LABELMAP'],
            '-o', os.path.join(paths['ANNOTATION_PATH'], 'test.record')
        ]
        copy_command = [
            'cmd.exe',
            '/c',
            'copy',
            os.path.join(paths['PRETRAINED_MODEL_PATH'],PRETRAINED_MODEL_NAME,'pipeline.config'),
            os.path.join(paths['CHECKPOINT_PATH'].strip())
        ]
        if not os.path.exists(files['TF_RECORD_SCRIPT']):
            print("S")
            proc = subprocess.run('git clone https://github.com/nicknochnack/GenerateTFRecord' + paths['SCRIPTS_PATH'], shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            print(proc.stdout.decode())
        print("bitno")

        proc = subprocess.run(train_command, capture_output=True, text=True)
        if proc.returncode != 0:
            print("Error:", proc.stderr)
        else :
            print(proc.stdout)

        proc = subprocess.run(test_command, capture_output=True, text=True)
        if proc.returncode !=0:
            print("Error:", proc.stderr)
        else:
            print(proc.stdout)

        if os.name == 'nt':
            print(copy_command)
            proc = subprocess.run(copy_command, capture_output=True, text=True)
            if proc.returncode != 0:
                print("Error:", proc.stderr)
            else:
                print(proc.stdout)

        print(files['PIPELINE_CONFIG'])
        config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, pipeline_config)
        pipeline_config.model.ssd.num_classes = len(labels)
        pipeline_config.train_config.batch_size = 4
        pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'],
                                                                         PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
        pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
        pipeline_config.train_input_reader.label_map_path = files['LABELMAP']
        pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [
            os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
        pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
        pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [
            os.path.join(paths['ANNOTATION_PATH'], 'test.record')]
        config_text = text_format.MessageToString(pipeline_config)
        with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:
            f.write(config_text)
        TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')
        command2 = [
            'zmaj\\Scripts\\python.exe',
            'Tensorflow\\models\\research\\object_detection\\model_main_tf2.py',
            '--model_dir=Tensorflow\\workspace\\models\\'+CUSTOM_MODEL_NAME,
            '--pipeline_config_path=Tensorflow\\workspace\\models\\'+CUSTOM_MODEL_NAME+'\\pipeline.config',
            '--num_train_steps=4000'
        ]
        MakeFolderCommand=[
            'mkdir',
            os.path.join('zavrseni modeli', CUSTOM_MODEL_NAME)
        ]
        copyLabelMap = [
            'cmd.exe',
            '/c',
            'copy',
            os.path.join('Tensorflow', 'workspace', 'annotations', 'label_map.pbtxt'),
            os.path.join('zavrseni modeli',CUSTOM_MODEL_NAME)

        ]
        copyModel=[
            'cmd.exe',
            '/c',
            'copy',
            os.path.join('Tensorflow','workspace','models',CUSTOM_MODEL_NAME),
            os.path.join('zavrseni modeli', CUSTOM_MODEL_NAME)
        ]
        print(command2)
        proc = subprocess.run(command2, capture_output=True, text=True)
        if proc.returncode != 0:
            print("Error:", proc.stderr)
        else:
            print(proc.stdout)
            print("Kopiranje")
            proc = subprocess.run(MakeFolderCommand, shell=True,stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            print(proc.stdout.decode())
            labelmap = subprocess.run(copyLabelMap, capture_output=True, text=True)
            if labelmap.returncode != 0:
                print("Error:", labelmap.stderr)
            else:
                print(labelmap.stdout)
            model = subprocess.run(copyModel, capture_output=True, text=True)
            if model.returncode != 0:
                print("Error:", model.stderr)
            else:
                print(model.stdout)
    def traintest(self):
        path=os.path.join('Tensorflow','workspace','images')
        subprocess.run(['explorer', path])
    def loadimg(self):
        img = Image.open("image2.png")
        img = img.resize((200, 100), Image.ANTIALIAS)
        self.img_tk = ImageTk.PhotoImage(master=self.Cav1,image=img)
        self.Cav1.img = self.img_tk
        self.Cav1.create_image(0, 0, anchor=tk.NW, image=self.img_tk)

        img = Image.open("image1.png")
        img = img.resize((200, 100), Image.ANTIALIAS)
        self.img_tk = ImageTk.PhotoImage(master=self.Cav2,image=img)
        self.Cav2.img = self.img_tk
        self.Cav2.create_image(0, 0, anchor=tk.NW, image=self.img_tk)


    def LabelImg(self):
        self.modelName = self.textbox2.get("0.0", "end")
        self.labels=self.modelName.split(',')
        self.IMAGES_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'collectedimages')
        self.ZMAJ_PATH = os.path.join('Tensorflow', 'labelimg')
        proc = subprocess.run("cd "+self.ZMAJ_PATH+" && python labelImg.py", shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        print(proc.stdout.decode())

    def modelFolder(self):
        self.modelName = self.textbox2.get("0.0", "end")
        self.labels=self.modelName.split(',')
        self.IMAGES_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'collectedimages')
        print(self.labels)
        if not os.path.exists(self.IMAGES_PATH):
            if os.name == 'nt':
                proc = subprocess.Popen('dir D:\zmaj\TFODCourse\\', shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                out, err = proc.communicate()
                print(out.decode())
                proc=subprocess.run('mkdir nisam',shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                print(proc.stdout.decode())
        for label in self.labels:
            print(label)
            path = os.path.join(self.IMAGES_PATH, label)
            print(path)
            if not os.path.exists(path):
                print(path)
                proc=subprocess.run('mkdir '+path,shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                print(proc.stdout.decode())
                subprocess.run(['explorer', path])

class MainApp(ck.CTk):
    def  __init__(self):
        super().__init__()
        self.title("Angelus")
        self.geometry("900x700")
        self.c=0
        self.startB=False

        self.vid=cv2.VideoCapture(1)
        self.grid_columnconfigure(0,weight=1)

        self.canvas=ck.CTkCanvas(self,width=600,height=400)
        self.canvas.grid(row=0,column=0,padx=20,pady=20)

        self.modelFrame=ck.CTkFrame(self)
        self.modelFrame.grid(row=1,column=0)
        self.modelFrame.grid_rowconfigure(0,weight=1)
        self.modelFrame.configure(fg_color="transparent")

        self.combobox=ck.CTkComboBox(self.modelFrame,values=os.listdir('zavrseni modeli'),command=self.SetModel,state="readonly")
        self.combobox.grid(row=0,column=0,padx=(10,140))

        self.ModelText="model harrow"
        self.label=ck.CTkLabel(self.modelFrame,text="Selected: "+self.ModelText,font=("Roboto",20))
        self.label.grid(row=0,column=1,pady=20,padx=(140,10))

        self.button=ck.CTkButton(self,text="Start",command=self.start)
        self.button.grid(row=2,column=0,pady=20)

        self.buttonFrame=ck.CTkFrame(self)
        self.buttonFrame.grid(row=3,pady=50)
        self.buttonFrame.grid_rowconfigure(0,weight=1)
        self.buttonFrame.configure(fg_color="transparent")

        self.Tbutton=ck.CTkButton(self.buttonFrame,text="Make your own model",command=self.new)
        self.Tbutton.grid(row=0,column=0,pady=5,padx=(10,250))

        self.Sbutton=ck.CTkButton(self.buttonFrame,text="Stop",command=self.stop,state="disabled")
        self.Sbutton.grid(row=0,column=1,pady=5,padx=(250,10))
        self.updateFrame()
    def start(self):
        self.Sbutton.configure(state="normal")
        self.button.configure(state="disabled")
        self.c=1
        self.startB=True
        self.updateFrame()

    def stop(self):
        self.startB=False
        self.Sbutton.configure(state="disabled")
        self.button.configure(state="normal")

    def SetModel(self,choice):
        self.ModelText=choice
        self.label.configure(text="Selected: "+self.ModelText)

    @tf.function
    def detect_fn(self,image):
        print(image.shape)
        if len(image.shape) == 1:
            print("Received image with one dimension. Skipping detection.")
            return None
        image, shapes = self.detection_model.preprocess(image)
        prediction_dict = self.detection_model.predict(image, shapes)
        detections = self.detection_model.postprocess(prediction_dict, shapes)
        return detections
    def updateFrame(self):
        if(self.c==1):
            self.TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
            self.LABEL_MAP_NAME = 'label_map.pbtxt'

            self.paths = {
                'CHECKPOINT_PATH': os.path.join('zavrseni modeli', self.ModelText),
            }

            self.files = {
                'PIPELINE_CONFIG': os.path.join('zavrseni modeli', self.ModelText, 'pipeline.config'),
                'LABELMAP': os.path.join('zavrseni modeli', self.ModelText, 'label_map.pbtxt')
            }
            self.configs = config_util.get_configs_from_pipeline_file(self.files['PIPELINE_CONFIG'])
            self.detection_model = model_builder.build(model_config=self.configs['model'], is_training=False)

            self.ckpt = tf.compat.v2.train.Checkpoint(model=self.detection_model)
            self.ckpt.restore(os.path.join(self.paths['CHECKPOINT_PATH'], 'ckpt-5')).expect_partial()

            self.category_index = label_map_util.create_category_index_from_labelmap(self.files['LABELMAP'])

        if(self.startB==True):
            self.c=self.c+1
            ret, frame = self.vid.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.image_np = np.array(frame)
            if self.image_np is not None:

                self.input_tensor = tf.convert_to_tensor(np.expand_dims(self.image_np, 0), dtype=tf.float32)
                self.detections = self.detect_fn(self.input_tensor)
                if self.detections is not None:
                    self.num_detections = int(self.detections.pop('num_detections'))
                    self.detections = {key: value[0, :self.num_detections].numpy()
                                  for key, value in self.detections.items()}
                    self.detections['num_detections'] = self.num_detections

                    self.detections['detection_classes'] = self.detections['detection_classes'].astype(np.int64)

                    label_id_offset = 1
                    self.image_np_with_detectionsm = self.image_np.copy()

                    viz_utils.visualize_boxes_and_labels_on_image_array(
                        self.image_np_with_detectionsm,
                        self.detections['detection_boxes'],
                        self.detections['detection_classes'] + label_id_offset,
                        self.detections['detection_scores'],
                        self.category_index,
                        use_normalized_coordinates=True,
                        max_boxes_to_draw=1,
                        min_score_thresh=.6,
                        agnostic_mode=False)

                    img = Image.fromarray(self.image_np_with_detectionsm)
                    img_tk = ImageTk.PhotoImage(image=img)
                    self.canvas.img = img_tk
                    self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                    self.after(10, self.updateFrame)

    def new(self):
        ck.set_appearance_mode("dark")
        ck.set_default_color_theme("dark-blue")
        app2 = HowApp()
        app2.resizable(False, False)
        app2.mainloop()
ck.set_appearance_mode("dark")
ck.set_default_color_theme("dark-blue")
app=MainApp()
app.resizable(False,False)
app.mainloop()