







Convolutional Neural Networks for Segmentation of Brain Tumors



Convolutional Neural Networks for Segmentation of Brain Tumors











A project report submitted in partial
fulfillment of the requirements for the degree of
Master of Science







By









Michael Hoover
Purdue University, 2010
Bachelor of Arts in Psychology







May 2022
University of Colorado Denver
ABSTRACT

Segmentation of tumor regions in MRI volumes aids medical professionals in interpreting images. However, this is a difficult and computationally expensive task. Therefore, exploring architectures and frameworks for the efficient and accurate segmentation of medical images is a vital area of research. The project constructs several models that use a U-net architecture with different configurations and test them on MRI volumes from the MICCAI BraTS challenge. The project demonstrates the ability of the networks to learn how to segment brain tumors in medical images. 



This Project Report is approved for recommendation to the Graduate Committee.  



Project Advisor:



____________________________________ 
Ashis Biswas





MS Project Committee:




____________________________________  
Ellen Gethner


____________________________________  
Farnoush Banaei-Kashani 

TABLE OF CONTENTS

1.  Introduction    1
1.1  Problem     1
1.2  Project Report Statement    1
1.3  Approach    1
2.  Background    3
2.1  Key Concepts    3
2.1.1  U-net for Segmentation    3
2.1.2  Dilation for Convolution    4
2.1.3  Spatial Attention    5    
2.2  Related Work or Literature Review    6
3.  Model Design and Implementation    8
3.1  High Level Design    8
3.2  Implementation    8
   3.2.1 Model 1    8
   3.2.2 Model 2    9
   3.2.3 Model 3    9
   3.2.4 Model 4    10
4.  Methodology, Results and Analysis    11
4.1  Metrics    11
4.2  Methodology    13
4.3  Results    15
4.4  Analysis    21
5.  Conclusions    24
5.1  Summary    24
5.2  Potential Impact    24
References    25

LIST OF FIGURES

Figure 1:  U-net Architecture    4
Figure 2: Performance measures for Model 1 plotted over epochs.    15
Figure 3: Performance measures for Model 2 plotted over epochs.    16
Figure 4: Performance measures for Model 3 plotted over epochs.    16
Figure 5: Performance measures for Model 4 plotted over epochs.    17
Figure 6: Comparison of Model validation Dice score over epochs.    18
Figure 7: Performance Measures on Test Data.    19
Figure 8: Visualization of Segmentation.    21


    

1.  Introduction
1.1  Problem
Cancer is a category of diseases characterized by rampant, out of control, cell growth. It is a particularly dangerous group of diseases. It has been estimated that cancer is responsible for over 15% of deaths annually [1]. Brain tumors are abnormal growths in the brain and malignant brain tumors are one of the most deadly diseases of the central nervous system. Glioblastoma in particular is an aggressive type of cancer in the brain and patients who suffer from it have high mortality rates.  Detection of tumors is the first step in the treatment of any cancer. Obviously, the sooner a malignant growth is identified, the sooner health care professionals can begin treatment. 
The motivation for this project is to leverage recent advances in Machine Learning, with a particular focus on Convolution Neural Networks (CNNs), in building a model that can aid healthcare workers in the detection and identification of brain tumors. The goal is to develop techniques that will be able to detect tumors accurately and early.

1.2  Project Statement
The goal of this project is to utilize machine learning and neural networks to perform image segmentation of tumors in MRI brain scans. 
1.3  Approach
The project aims to create and test several types of 3D CNNs and test their performance on tumor segmentation in MRIs. This will first necessitate preprocessing the data, converting it into information that can be fed to a CNN and trimming the data to make it more manageable. 
The models will be developed using Tensorflow and Keras. We will begin with a basic implementation of a U-Net architecture as this is a common architecture that has shown success in image segmentation. From there we will proceed to add and test new elements in an effort to increase the accuracy of the model. 
The models will be evaluated using Dice coefficient, Intersection over Union (IoU), and Hausdorff distance. These are the metrics used to judge participants in the BraTS challenge. 

2.  Background
2.1  Key Concepts
Convolutional Neural Networks have demonstrated strong results in computer vision classification and now regularly perform as some of the top models for these types of tasks, beating out other advanced algorithms [2, 3]. However, in many instances it is desirable to go beyond mere classification of an entire image and label specific elements within an image. Segmentation divides the image up and seeks to label the sub-images, or segments. Often, this is done by treating each pixel as a segment. Being able to perform segmentation is useful in many areas of the medical field, for instance, the segmentation of blood vessels in the human eye [4], the segmentation of joints [5], and in the task taken on in this project, the labeling of tumors in MRI volumes.
2.1.1  U-net for Segmentation
One of the more successful network architectures in tasks of image segmentation is U-net [6]. Furthermore, many of the models that have succeeded in past iterations of the BraTs challenge, have used U-net architectures to great success [7, 8, 9].
U-net can be broken up into two separate processes. The first half of the architecture is an encoding path. This section uses convolution layers  to learn features of the input and then uses several pooling layers, interspersed between the convolutions, to reduce the feature space while at the same time expanding the number of filters in the kernel. The second half is a decoding path. This path is several convolution layers with upsampling stages that expand the feature space back to the original dimensions of the input image. The final convolution layer provides a label for each pixel. This architecture is able to consider the context of each pixel, i.e. its relationship to the other pixels, and provide a segmentation mask that corresponds to the spatial structure of the image. 


Figure 1: The above figure is taken from Ronneberger et al., the seminal paper that proposed the U-net architecture [6]. The feature maps are indicated by the blue rectangles and the arrows represent the transformations performed as the input moves through the network.
2.1.2  Dilation in Convolution
A standard convolution layer slides a kernel across the input in strides and multiplies the elements of the input by the weights in the kernel. In its typical implementation the kernel, or filter, operates on adjacent pixels or features of the input. One tradeoff that arises is that larger receptive kernels, i.e. filters with larger dimensions, have higher computational and memory costs. Dilation alters the kernel so that empty spaces are inserted between the weights and allows the window size of the kernel to be expanded without increasing the number of parameters, thereby avoiding any additional costs as far as computation and memory [10]. The use of dilation has been shown to aid models in the segmentation of medical images [11]. 
2.1.3  Spatial Attention
    Although there are complicated mechanisms for implementing attention in CNNs, such as Convolution block attention modules that combine channel-wise attention with spatial attention [12], this project makes use of a simple spatial attention layer that uses a 3D convolutional layer with a sigmoid activation function to produces an attention mask which is then used in element-wise multiplication with the input. This type of spatial attention mechanism has had success in the segmentation of brain tumors in medical images. [13]. If a(x) is a transformation that accepts a matrix x and outputs a x′, a matrix of the same dimensions as x, then the spatial attention function, s(x), used in this project can be described by the equation:

s(x) = a(x) ⊙ x                                                       (1)

where ⊙ is the element-wise multiplication operation. The trainable parameters of the attention layer, in essence, allow the network to learn which inputs to encourage and which inputs to suppress. 
2.2  Literature Review
Using CNNs for image segmentation is not a new task and thus there is an abundance of literature on the subject. An exhaustive summary of the work being done in the field is beyond the scope of this project. Instead this section will focus on a select number of research papers that provided instruction and guidance to the project. 
 The major dataset for this project comes from The Brain Tumor Segmentation (BraTS) challenge which is being held for the 10th year. A number of architectures have been used on this task in the past. 
Theophraste et al. used a 3-Dimensional U-Net architectures with good results on the BraTS dataset [14]. Once again this work shows the prominence of U-net architecture and its success for this application. L Jai et al. used a cascading CNN with an Attention Expectation-Maximization Algorithm for the task of segmenting tumors [15]. The team took home second place in last year’s BraTS challenge and is another instance of the high performance of model’s with attention mechanisms. This is further motivation testing attention in the project. 
The particular task of image segmentation has created a wide literature of techniques used to address the problem. Guosheng et al. demonstrated techniques to retain high-resolution features in deep neural networks with downsampling [16]. Kaiming et al. worked with spatial pyramid pooling to deal with differing image sizes and demonstrated that the technique improved the performance of a variety of CNNs in many classification and segmentation tasks [17]. Although these techniques were not implemented in the project, they pave the path for future exploration. 

3.  Model Design and Implementation
3.1  High Level Design
This section focuses on 4 different models that were trained and tested for the report. All models used a U-net architecture as described above. Because of the limitation of GPU and memory resources the models trained were not as robust as many of the neural networks reported in the Literature Review section, for instance the number of filters was greatly reduced in each convolution block. Although this was less than ideal, the results were still promising and if anything indicate that the models tested have fundamental structures and mechanisms that perform well in image segmentation and provide justification for further study. Dilation and spatial attention, as described in Background were implemented in some of the models. This allowed a comparison of the techniques. 
All the code for this project was written in Python. Tensorflow and Keras were used to build the neural networks. Nibabel, a library for use with medical images, was used to read the MRI volumes. Because the dataset did not fit into main memory, data generators were used to load the MRI volumes in batches and feed them to the models during training. The Matplotlib library was used to visualize slices of the volumes.
3.2  Implementation
3.2.1 Model 1
Model 1 uses a U-net architecture with 3 levels having 16, 32, and 64 filters respectively. A convolution block consists of two iterations of the following layers: 1) a convolution layer with a 3x3x3 kernel and a stride of 1 with no dilation; 2) a batch normalization layer; 3) an activation layer that uses the Rectified Linear Unit function (ReLU). Thus, each convolution block has a total of 6 layers. The encoding pathway of the model has max pooling layers, with the pool size having a 2x2x2 dimension, that follow each convolution block. The decoding layers, conversely, have upsampling layers that follow each convolution block. The final layer is a 1x1x1 convolution layer with a sigmoid activation that predicts between two labels: 1, a tumor, and 0, non-tumor. Model 1 was trained for 50 epochs.
3.2.2 Model 2
    Model 2 is identical to Model 1 except that it adds a dilation layer to the convolution block. Therefore, the input to a convolution block branches into two separate paths. The first path goes through the same layers as described in Model 1. The second path goes through a 3x3x3 convolution layer with a dilation rate of 2. The two branches are then concatenated together before exiting the convolution block. Model 2 was trained for 100 epochs. 
3.2.3 Model 3
    Model 3 adds spatial attention to the basic U-net described in Model 1. So, it has the convolution blocks identical to those in Model 1 but before each convolution block is an attention block. An attention block consists of 3 layers: 1) a 1x1x1 convolution layer; 2) a sigmoid activation layer; 3) a layer that performs element-wise multiplication between the output of the sigmoid activation and the input the attention block. The output of the attention block is then forwarded to the convolution block. The model was trained for 26 epochs and halted due to an early stopping callback that terminated training after the model failed to show improvements on the validation set for 10 consecutive epochs.
3.2.4 Model 4
    Model 4 adds both dilation and attention to the base U-net described in Model 1. Alternatively, it can be thought of as a hybrid of Model 2 and Model 3. The convolution blocks contain a dilation layer, identical to the layer described in Model 2. Each convolution block is preceded by an attention block, identical to the one described in Model 3. Model 4 was trained for 25 epochs and halted due to an early stopping callback that terminated training after the model failed to show improvements on the validation set for 10 consecutive epochs.
4.  Methodology, Results and Analysis
4.1 Metrics
The models were evaluated using three metrics: pixel-wise accuracy, Intersection over Union (IoU), and the Sørensen–Dice coefficient.
 Although pixel-wise accuracy was calculated, it is often a misleading measurement when judging the performance of models for medical segmentation. In this particular task, the pixels with a ground truth label of tumor make up a small portion of the total MRI volume. It is possible for a “dumb” model to label every pixel as a non-tumor and still record high pixel-wise accuracy. In the experiments all models ended up with similar high measures of accuracy even though they showed much more variation on other metrics. Accuracy, overall, does not appear to be a good metric to discriminate between models.
IoU and the Dice coefficient do a much better job of evaluating the model’s performance in regards to its ability to correctly label tumor pixels, i.e. models that incorrectly label the small portion of tumor subregions score significantly lower than these metrics than models that correctly label the tumor subregions. For this reason, most of the analysis was focused on the results of IoU and Dice coefficient. 
IoU is typically calculated by the following expression:

                     (2)

where Ypred is a matrix of the model’s predictions, Ytruth is the matrix of the ground truth segmentation. However, when measuring the performance of CNNs in binary segmentation, a few small changes are necessary in the calculation of IoU. The ground truth labels are either 0 or 1, i.e. elements of the set {0,1}. However, since the models use a sigmoid activation function in the final segmentation layer, the elements of the output will rarely, if ever, be precisely 0 or 1. This means it is likely that  Ypred  and Ytruth will not have any common elements and the intersection will be empty. Instead, we can use the element-wise multiplication between Ypred  and Ytruth to measure the similarity between tumor subregions and the model’s prediction of these subregions, and then take the sum of these elements to get a total. This total is divided by the union of the sets will be a measure of how closely the predictions matched the the ground truth segmentation. 
For this project the IoU was calculated as follows:

         (3)

where ⊙ is the operation for element-wise multiplication of matrices, and the sum function outputs the summation of all of a matrix’s elements.
    The elements of the output, Ypred , can also be rounded, i.e. values greater than or equal to 0.5 are predictions of tumor subregion and values less than 0.5 are predictions of non-tumor subregions. This is similar to how such a model would be used in practice and allows IoU to be calculated as in (2). When using IoU to create a loss function the expression in (3) should be used as it is continuous and therefore preferable for calculating a gradient.
The Dice coefficient was calculated as follows:

                (4)

Similarly, a rounded Dice coefficient can be calculated with output elements constrained to 0 or 1.
    Hausdorff distance was also used as a metric to measure the performance of the models. Hausdorff distance is the longest distance in a set to the nearest point in another set. For two sets A and B, it is calculated by the expression:

                                     (5)
     

where dist is a function for the Euclidean Distance between two points. Hausdorff Distance is a metric that provides a measure of similarity between the ground truth and model predictions.

4.2  Methodology
The models were trained and tested on data obtained from the 2021 RSNA-ASNR-MICCAI Brain Tumor Segmentation (BraTS) Challenge. The dataset consisted of 1,211 MRI volumes with corresponding segmentation ground truths. The dataset was split into training, validation, and test sets of 75%, 20%, and 5% of the total data, respectively. The features of the input files were scaled to floating point numbers between 0 and 1, and were set to have a mean of 0.5. For models that performed binary segmentation, the ground truth files were modified so that non-tumor regions were set to 0 and all tumor sub regions were set to 1. 
Models 1, 2, 3, and 4 were trained and tested on modified datasets that reduced the size of the MRI volumes. The original volumes measured 160 x 240 x 240 pixels and the modified volumes were resized to be 80 x 120 x 120 pixels using Tensorflow’s image resize method with bilinear interpolation. Given the limitations of the hardware used for the experiments, reducing the volume size helped speed up training time. 
Initially models were trained using the binary cross-entropy loss function provided by the Keras library. Although this did work to train models that scored high in pixel-wise accuracy, it produced poor results as far as IoU and Dice-coefficient. It was found that creating a custom loss function using IoU resulted in models with far superior performance in IoU and Dice coefficient, with little to no effect on pixel-wise accuracy. The loss function was calculated as follows:

lossIoU  =  1 - IoU                     (5)

This relatively simple loss function is bounded between 0 and 1 (because the range of IoU is 0 to 1) and has a low loss when the model scores high on IoU. This allows the training process to descend the gradient and creates models that score much better on IoU and Dice coefficients than models trained with binary-cross entropy and as a result are superior at identifying the tumor sub-regions.  
4.3  Results
The figures below show the IoU and Dice score during training of the 4 models plotted over epochs. As expected, the scores on the training data have smoother curves that show steady increase in performance, while the measures on validation data are more erratic, but still demonstrate an overall trend of increased performance in the beginning of training. For the validation set the models show a plateau at the end of the training session, while performance on training data still increases slightly. These are instances of typical overfitting. 

Figure 2: Performance measures for Model 1 plotted over epochs.

Figure 3: Performance measures for Model 2 plotted over epochs.

Figure 4: Performance measures for Model 3 plotted over epochs


Figure 5: Performance measures for Model 4 plotted over epochs.




Figure 6: Comparison of Model validation Dice score over epochs

The test dataset comprises 61 MRI volumes, about 5% of the total dataset, that were set aside and not used in training. In this way we can see how the models performed on samples they had not been exposed to until after the completion of training. The table in Figure # shows the performance metrics for each model on the test set. For the most part the models performed nearly as well on the test data as they did on the validation data.







IoU
Dice Coef.
IoU (rounded)


Dice Coef. (rounded)
Mean Hausdorff Distance
Model 1
0.858
0.923
0.929
0.962
9.78
Model 2
0.876
0.934
0.939
0.967
9.13
Model 3
0.821
0.902
0.910
0.950
11.08
Model 4
0.732 
0.845
0.864
0.921
11.61


Figure 7: Performance Measures on Test Data

    
    When comparing the performance of these models to other state-of-the-art models, there is a difficulty in that this project simplified the output to a binary segmentation task, whereas other projects participating in the BraTS challenge aimed to complete the full segmentation task, which has 3 distinct tumor sub-regions: Enhancing tumor, invaded tissue, and necrotic tumor core. So although the models in this project were able, in some instances, to achieve higher Dice Coefficients and Mean Hausdorff Distance, it should be kept in mind that this is not a fair comparison, as other models are attempting a more challenging and complex task. However, a somewhat more fair comparison can be had by considering the union of the 3 labels. Many papers already calculate the scores for the whole tumor. Still, it should be noted that the models in this project were trained on volumes that were reduced in size. Therefore, the resolution of the ground truth and the predictions does not match that of other models that made predictions on the full volumes. 
    Jun et al. succeeded in training a U-net model for segmentation and reported a mean Dice score of 0.861 and a mean Hausdorff distance of 7.45 on the whole tumor  for a test set of volumes [18]. Similarly, Henry et al. created a model that when measured on a test set for whole tumor segmentation reported a Dice score of 0.886 and a Hausdorff distance of 6.67 [19]. Also, the U-net developed and trained by Sundaresan et al. reported a Dice score of 0.89 and a Hausdorff distance of 6.3 for whote tumor segmentation [20]. The three previous models were all contestants in the BraTS 2020 challenge and therefore represent some of the latest models in tumor segmentation.
    It is interesting that the models trained in this project were able to achieve Dice scores better than some of the best models in the 2020 BraTS challenge and yet all models in this project scored decidedly worse when it came to Hausdorff distance. The superior performance when it comes to Dice score is encouraging, even though the models in this project were trained and tested on a dataset that has been modified to make the task easier. On the other hand, the inferior performance when it comes to Hausdorff distance hints that there are still areas for improvement, especially since the models in this project were trained on a dataset of lower resolution volumes which might lead one to think it should score higher in all metrics. At the very least, this suggests similar techniques would perform on par with state of the art networks on the full segmentation task.  
Figure 8 visualizes the segmentation on slices taken from three distinct samples in the test dataset. The MRI scan of the brain is denoted by the green and blue. In the Ground Truth column, the areas marked in red are the tumor subregions that have been labeled manually. These are the pixels we want the model to identify. In the other columns the red areas mark the model’s prediction as to the tumor subregions. This allows us to visualize each model’s performance and to see how such models would be used in practice with medical professionals. We can see that the models are to varying degrees successful at approximating the tumor regions. 


Figure 8: Visualization of Segmentation

4.4  Analysis
    The results of the project confirm the viability of U-net architecture in segmentation tasks. In particular it shows that this type of model works well with medical images and offers a robust area of research. 
The experiments also exhibit how dilation is a useful mechanism in improving the performance of the segmentation models. Dilation allows the filter to expand the width from which the layer is gathering information, but does not increase the number of computations necessary to execute the layer. In this way it is a useful technique to help the network contextualize pixels without an increase in processing cost. 
Adding spatial attention did not result in higher performance in either model 3 or 4 in comparison to simpler implementations. It’s possible that the mechanism does not function well with a low number of filters, a choice which was necessary due to limited main memory. Models with a higher number of filters exhausted main memory and crashed the training process. We can see that when the model’s were evaluated on test data, 61 MRI volumes that were set aside and not used in training, the performance gap between models with spatial attention and those without it becomes clear. In particular, the mean Hausdorff Distance in Model 3 and Model 4 increases significantly. In the literature, spatial attention is demonstrated to improve the performance of models on this type of task, so the findings here disagree with conclusions of other papers, which is the main reason why we suspect that with better hardware implementations of spatial attention may provide increases to performance that this project was not able to demonstrate. 
Model 2 scored the highest on all performance measures on the test dataset. Furthermore, Model 2 is interesting in that it trained for 100 epochs and showed a steady increase in performance measures on the validation dataset for the entirety of training, that is it did not halt due to early stopping, which suggests it may be possible the model will continue to improve with additional training. With more training the incremental improvements may be within reach. Given the serious nature of the task even small improvements to a model used in practice could be critical.  
 
5.  Conclusions
5.1  Summary
The project tested 4 distinct models on segmentation of brain tumors in MRI volumes. All models used a U-net architecture. They made use of different combinations of layers, most notably dilation layers and spatial attention layers. The models were measured using the metrics of Intersection over Union, Dice Coefficient, and Mean Hausdorff Distance on a test set of 5% of the MRI volumes. Overall, model 2, which used dilation layers but not spatial attention, exhibited the highest performance across all measures. 
5.2  Potential Impact
The project underscores the power of U-net architecture and dilation layers in convolutional neural networks. It shows that the choice of loss function is a crucial decision when training neural networks. In preliminary experiments, loss functions such as binary cross-entropy resulted in high pixel-wise accuracy but failed to produce models that scored well on other metrics and ultimately did not produce desirable results for the task. The IoU loss function was much better suited for training and shows that the choice of loss function is often task specific. Overall the project demonstrates that convolutional neural networks in medical image segmentation is a fertile area of research.


References
 [1]     GBD 2015 Mortality and Causes of Death, Collaborators. "Global, regional, and national life expectancy, all-cause mortality, and cause-specific mortality for 249 causes of death, 1980–2015: a systematic analysis for the Global Burden of Disease Study 2015" Lancet. 388 (10053): 1459–1544.

[2]     LeCun, Y., Boser, B., Denker, J.S., Henderson, D., Howard, R.E., Hubbard, W.,
Jackel, L.D.: Backpropagation applied to handwritten zip code recognition. Neural
Computation 1(4), 541–551 (1989)

[3]     Krizhevsky, A., Sutskever, I., Hinton, G.E.: Imagenet classification with deep con-
volutional neural networks. In: NIPS. pp. 1106–1114 (2012)

[4]    Zhang, Z., Wu, C., Coleman, S., Kerr, D. “DENSE-INception U-net for medical image segmentation” In: Computer Methods and Programs in Biomedicine, vol. 192, 2020.

[5]     Trullo, R., Petitjean, C., Nie, D., Shen, D., Ruan, S. “ Joint Segmentation of Multiple Thoracic Organs in CT Images with Two Collaborative Deep Architectures.” In: Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support . DLMIA ML-CDS 2017 2017. Lecture Notes in Computer Science(), vol 10553. Springer, Cham. 2017.

[6]    O. Ronneberger, P. Fischer, T. Brox, U-net: Convolutional networks for biomedical image segmentation, in: International Conference on Medical image computing and computer-assisted intervention, Springer, 2015, pp. 234–241.
[7]    Kermi, A., Mahmoudi, I., Khadir, M. T. “Brain Tumor Segmentation in Multimodal 3D-MRI of BraTS’2018 Datasets using Deep Convolutional Neural Networks” In: Pre-Conference Proceedings of the 7th MICCAI BraTS Challenge (2018), pp. 251-263. 2018
[8]    Lachinov, D, Vasiliev, E., Turlapov, V. “Glioma Segmentation with Cascaded Unet: Preliminary results” In: Pre-Conference Proceedings of the 7th MICCAI BraTS Challenge (2018), pp. 272-279. 2018

[9]    Kotowski, K., Nalepa, J.,  Dudzik, D. “Detection and Segmentation of Brain
Tumors from MRI Using U-Nets” In: Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries. 5th International Workshop, BrainLes 2019, Revised Selected Papers, vol. 2, pp. 179-190. 2019.
 
[10]    F. Yu and V. Koltun, "Multi-Scale Context Aggregation by Dilated Convolutions," 2015. arXiv:1511.07122

[11] Y. Guo, J. Bernal and B. J. Matuszewski, "Polyp Segmentation with Fully Convolutional Deep Neural Networks—Extended Evaluation Study," Journal of Imaging, vol. 6, (7), pp. 69, 2020.

[12]    Y. Xiao, H. Yin, S. Wang, Y. Zhang, "TReC: Transferred ResNet and CBAM for Detecting Brain Diseases," Frontiers in Neuroinformatics, vol. 15, pp. 781551-781551, 2021.

[13]    C. Liu, W. Ding, L. Li, Z. Zhang, C. Pei, L. Huang, and X. Zhuang, “Brain Tumor Segmentation Network Using Attention-Based Fusion and Spatial Relationship Constraint” In: Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries. 5th International Workshop, BrainLes 2020, Revised Selected Papers, vol. 1, pp. 219-229, 2020.

[14]    Henry, Théophraste & Carre, Alexandre & Lerousseau, Marvin & Estienne, Théo & Robert, Charlotte & Paragios, Nikos & Deutsch, Eric. (2020). Top 10 BraTS 2020 challenge solution: Brain tumor segmentation with self-ensembled, deeply-supervised 3D-Unet like neural networks.

[15]    Haozhe Jia, Weidong Cai, Heng Huang, and Yong Xia “H2NF-Net for Brain Tumor Segmentation using Multimodal MR Imaging: 2nd Place Solution to BraTS Challenge 2020 Segmentation Task” arXiv:2012.15318.

[16]    Guosheng Lin, Anton Milan, Chunhua Shen and Ian Reid, "Refinenet: Multi-path refinement networks for high-resolution semantic segmentation", CVPR, pp. 1925-1934, 2017.

[17]    Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun “Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition”, IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 37, no. 9, pp. 1904-1916, 2015.

[18]    W. Jun, X. Haoxiang, and Z. Wang, “Brain Tumor Segmentation Using Dual-Path Attention U-Net in 3D MRI Images” In: Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries. 5th International Workshop, BrainLes 2020, Revised Selected Papers, vol. 1, pp 183-193, 2020.

[19]    T. Henry, A. Carrè, M. Lerousseau, T. Estienne, C. Robert, N. Paragios, and E. Deutsch, “Brain Tumor Segmentation with Self-ensembled, Deeply-Supervised 3D U-Net Neural Networks: A BraTS 2020 Challenge Solution” In: Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries. 5th International Workshop, BrainLes 2020, Revised Selected Papers, vol. 1, pp 327-339, 2020.

[20]    V. Sundaresan, L. Griffanti , and M. Jenkinson,  “Brain Tumour Segmentation Using a Triplanar Ensemble of U-Nets on MR Images”  In: Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries. 5th International Workshop, BrainLes 2020, Revised Selected Papers, vol. 1, pp 340-353, 2020.



This is the final page of a Project Report and should be a blank page
