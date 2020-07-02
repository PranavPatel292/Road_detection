# Road_detection
This project is done to find the road work currently going on the road and updte the google map with that information
### NOTE: - I am using the Google CO-LAB and my local machine to accomplish this project (road detection task), so I often switch back and forth between local machine and Google CO-LAB.
# Motivations
The sole motivation behind this project is to satisfy the inner passion for the Deep Learning.
# Dataset
There is no direct dataset available for this project, so we must create our own dataset by going out on the road and video record road work during Night and Day. After this the individual frames from the video must extract for the purpose of the image labelling. So, the steps of dataset collection are described as follow. You need to upload the provide dataset on the Google Drive (if you are going to use my Google CO-LAB notebook, which is something I highly recommended)

#### Note: - if you are unfamiler with Google CO-Lab please make your self confirtable

Step 1: - Find a road working side near you.

Step 2: - Video record the road work.

Step 3: - Extract the individual frames from the video.

Step 4: - Manually label all the images. (using [LabelMg](https://github.com/tzutalin/labelImg) or [LabelMe](http://labelme.csail.mit.edu/Release3.0/) [follow the link to get install in your local pc])

The only problem with this approach is that one must do the labelling of thousand and thousands of images. Most of the Smartphone nowadays record video using 30 FPS (frames per seconds) by default (you can change it if you want), so even with a 1 minute video you have as 30 * 60 = 900 images to label for one video so you can imagine if we have 15 videos each with 1 minute and 30 FPS than the images to label is round 45,000.  I have overcome this issue with one little change, the FPS of my video is changed from 30 to 240 FPS. So now I have a smaller number of images to label. The steps for these steps are describe as follow.

Step 1: - Find a road working side near you.

Step 2: - Video record the road work.

Step 3: - Configure the FPS. (I have used the Adobe Premier Pro for that, you can choose the video editor of your choice).

Step 4: - Import the video to Adobe Premier Pro (or the video editor of your choice)

Step 5: - Go to the Clip->Modify->Interpret Footage.

Step 6: - Select the option Assume this frame rate and set to 240 then click OK.

Step 7:  - Extract the individual frames from the video.

Step 8: - Manually label all the images. (using [LabelMg](https://github.com/tzutalin/labelImg) and [LabelMe](http://labelme.csail.mit.edu/Release3.0/))

# Model
I am using the Object Detection API with faster RCNN v2 version which can be download using this [link](https://github.com/tensorflow/models) (If you follow my [Google CO-LAB notebook](https://github.com/PranavPatel292/Road_detection/blob/master/RCNN_road_detection) then this will be done automatically for you no other step required). Alternatively, you can use the [GitHub repo by EdjeElectronic](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10) to install all the required library and TensorFlow GPU based version or just clone my repo  (before you get into the GitHub repo, please click [CUDA enable GPU cards](https://developer.nvidia.com/cuda-gpus) to check your GPU capability).

[Note: - I am not training my model on my local machine, I had used the Google – CO-LAB environment] 

If you are using my Google CO-LAB code, then after the training you will have an inference graph file which is the outcome of our entire training process. The inference graph file will be stored into your Google Drive, find it and download it to your local machine. Also, you need to download the label_map.pbtxt file (which is generated for you) by going into CO-LAB virtual environment temporary directory, followed by Annotation directory where you can search the labMap.pbtxt file (shown in the image GitHub image).
At this stage, you should have inference graph file and label map file download in your local machine.
Now, use my python file namely (already made available for you if you have clone my repo) Object_dectaction_video.py to see the result in your local machine. 

Path for the file:- path_of_where_my_repo_is_stored/models/research/object_detection/Object_detection_video.py. 

You need to change the file path for few variables in the Object_detection_video.py file such as pb_file_ssd to the inference_graph_file and label_map_ssd to the labelmap.pbtext file location where it is stored.

