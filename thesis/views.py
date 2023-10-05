from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.core.files.storage import FileSystemStorage
from keras.models import load_model
from keras.preprocessing import image
import json
from tensorflow import Graph
import tensorflow as tf
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")  # Use the "Agg" backend for non-interactive use
import matplotlib.pyplot as plt
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from .models import CoconutDamages
from .forms import CoconutDamagesForm
from keras.models import Sequential
import cv2
from keras.layers import Flatten, Dense  # Import Flatten class
import mpld3
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# model2 = load_model('./models/ResNet50_Architecture_AnomalyDetection_ai_training.h5')
# multiclass_model2 = load_model('./models/MobileNetV2_Categorical_Architecture_coconutDamages_ai_training-test1.h5')

model = load_model('./models/ResNet50_Architecture_AnomalyDetection_ai_training-testFinal.h5')
multiclass_model = load_model('./models/ResNet50_Categorical_Architecture_coconutDamages_ai_training-testFinal.h5')


from keras.applications import ResNet50
# The line of code initializes a ResNet50 convolutional base model with pre-trained weights from ImageNet, excluding the top (classification) layer, and setting the input shape to (150, 150, 3).  
conv_base = ResNet50(
                include_top=False,
                input_shape=(150, 150, 3)
                )

multiclass_conv_base = ResNet50(
                include_top=False,
                input_shape=(150, 150, 3),
                pooling='avg',
                classes=4,
                weights='imagenet'
                )

# Build the rest of the Sequential model
# model = Sequential()
# model.add(conv_base)
# model.add(Flatten())  # Add a Flatten layer to convert the output to a flattened vector
# model.add(Dense(256, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))  # Assuming you have 3 classes

# multiclass_model = Sequential()
# multiclass_model.add(multiclass_conv_base)
# multiclass_model.add(Flatten())  # Add a Flatten layer to convert the output to a flattened vector
# multiclass_model.add(Dense(256, activation='relu'))
# multiclass_model.add(Dense(4, activation='softmax'))  # Assuming you have 3 classes


@login_required(login_url='admin-login')
def index(request):
    context={'a': 1}
    print("Hello World")
    return render(request, 'index.html', context)

# Create your views here.

@login_required(login_url='user-login')
def predictImage(request):

    
    print("Hello Jasper")
    print("This is the request:", request)
    print("This is the WSGI:", request.POST.dict())
    print("This is the fileName:", request.FILES['filePath'])

    fileObj = request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName = fs.save(fileObj.name, fileObj) 
    filePathName= fs.url(filePathName)
    print("Saved successfully!")

    test_img = '.'+filePathName
    # img = cv2.imread(test_img)
    # img_resized = cv2.resize(img, (150, 150))
    # img_normalized = img_resized / 255.0 
    # features_batch = conv_base.predict(np.array([img_normalized]))
    # features_flattened = features_batch.flatten()
    # predi = model.predict([features_flattened])

    # img = image.load_img(test_img, target_size=(150, 150))
    # test_image_array = image.img_to_array(img)
    # test_image_array = np.expand_dims(test_image_array, axis=0)
    # test_image_array=test_imct(test_image_array)
    # features_flattened = features_batch.flatten()  # Keep it flattened

    # predi = model.predict(np.array([features_flattened]))
    # print("This is the predi: ", predi)age_array/225.0 # Normalize the pixel values (optional)
    
    # features_batch = conv_base.predi

    #-----------------

    # test_image = image.load_img(test_img, target_size=(150, 150))

    #------------------

#*---------------------

    img = image.load_img(test_img, target_size=(150, 150))
    test_image_array = image.img_to_array(img)
    test_image_array = np.expand_dims(test_image_array, axis=0)
    test_image_array = test_image_array / 255.0  # Normalize the pixel values (optional)


    features_batch = conv_base.predict(test_image_array)
    test_image_array = np.reshape(features_batch, (1, 5 * 5 * 2048))
    print("This is the img shape: ", test_image_array)

    pred = model.predict(test_image_array)
    print("This is the prediction: ", pred)

    score = float(pred[0])
    noSignsDamage = round(100 * score)
    damaged = round(100 * (1 - score))

    class_names_labels = ['no_signs_of_damage']
    output_class=class_names_labels[np.argmax(pred)]

    class_names_labels = ['no signs of damage']
    output_class=class_names_labels[np.argmax(pred)]

    percentage = pred[0][np.argmax(pred)]*100

    if (percentage < 60):
        print("The predicted class is Damaged")
        result = 100 - percentage
        print(f"Damage Percentage = {result}%")
        
        test_image_array = image.img_to_array(img)
        test_image_array = np.expand_dims(test_image_array, axis=0)
        print(test_image_array.shape)
        test_image_array = test_image_array / 255.0  # Normalize the pixel values (optional)


        features_batch = multiclass_conv_base.predict(test_image_array)
        test_image_array = np.reshape(features_batch, (1, 2048))
        print("This is the image shape: ", test_image_array)

        pred = multiclass_model.predict(test_image_array)
        print("This is the prediction: ", pred)
        
        class_names_labels = ['asiatic_palm_weevil_damage', 'leaf_beetle_damage','rhinoceros_beetle_damage',  'scale_insect_damage']
        output_class=class_names_labels[np.argmax(pred)]
        print("The predicted class is", output_class)
        
        prediction_percentages = {}
        
        for z in zip(['asiatic_palm_weevil_damage', 'leaf_beetle_damage','rhinoceros_beetle_damage',  'scale_insect_damage'], pred[0]):
            # result = '%s %% = %.2f%%' % (z[0], z[1] * 100)
            # print(result)
            
            pest_class = z[0]
            percentage = z[1] * 100
            prediction_percentages[pest_class] = percentage
            print(f"{pest_class} % = {percentage: .2f}%")
        
            output_class=class_names_labels[np.argmax(pred)]
            print("The predicted class is", output_class)

            if output_class == 'asiatic_palm_weevil_damage':
                identification = "Diagnosis: Asiatic Palm Weevil Damage"
                solution1 = "Management Strategies" 
                solution2 = "Cultural Method:Regulated prunning of infested fronds" 
                solution3= "Chemical Method: Chemical control using systemic insecticides, FPA prescribed for coconut pests by trunk injection(for emergency only)."
                solution4 = "Topical application of vegetable oil in young palms can be done during cooler months." 
                solution5 = "Biological Method: Sustained release of biological control agents like Chilochorus sp., Telsimia sp., Pseudoscymnus anomalus, Cybocephalus sp., Bathracedra sp., Comperiella calauanica."
                solution6= "Regulatory Control: Implementation of quarantine regulations and establishment of checkpoints (PCA Davao Research Center, 2023)"
            elif output_class == 'leaf_beetle_damage':
                identification = "Diagnosis: Coconut Leaf Beetle Damage"
                solution1 = "Management Strategies"
                solution2 = "Biological Control: Establishment of coconut log traps inoculated with Green Muscardine Fungus (GMF) granules. Concentrated Oryctes nudivurus is dropped to the mouth of beetles. Infection spreads through mating and visit in breeding sites"
                solution3 = "Cultural Control: Collection and utilization of coconut debris or farm waste t avoid piling. PRactice farm sanitation. Regular inspection of all possible breeding sites and collection of all stages of the beetle. Plant covercrops if intercropping is not practiced" 
                solution4 = "Chemical Control: Installation of rhinoceros beetle pheromone in traps, enhanced with food bait (PCA Davao Research Center, 2023)"
                solution5= ""
                solution6= ""
            elif output_class == 'rhinoceros_beetle_damage':
                identification = "Diagnosis: Coconut Rhinoceros Beetle Damage"
                solution1 = "Management Strategies" 
                solution2 = "Cultural Method:Regulated prunning of infested fronds" 
                solution3= "Chemical Method: Chemical control using systemic insecticides, FPA prescribed for coconut pests by trunk injection(for emergency only)."
                solution4 = "Topical application of vegetable oil in young palms can be done during cooler months." 
                solution5 = "Biological Method: Sustained release of biological control agents like Chilochorus sp., Telsimia sp., Pseudoscymnus anomalus, Cybocephalus sp., Bathracedra sp., Comperiella calauanica."
                solution6= "Regulatory Control: Implementation of quarantine regulations and establishment of checkpoints (PCA Davao Research Center, 2023)"
            elif output_class == 'scale_insect_damage':
                identification = "Diagnosis: Coconut Scale Insect Damage"
                solution1 = "Management Strategies"
                solution2 = "Biological Control: Establishment of coconut log traps inoculated with Green Muscardine Fungus (GMF) granules. Concentrated Oryctes nudivurus is dropped to the mouth of beetles. Infection spreads through mating and visit in breeding sites"
                solution3 = "Cultural Control: Collection and utilization of coconut debris or farm waste t avoid piling. PRactice farm sanitation. Regular inspection of all possible breeding sites and collection of all stages of the beetle. Plant covercrops if intercropping is not practiced" 
                solution4 = "Chemical Control: Installation of rhinoceros beetle pheromone in traps, enhanced with food bait (PCA Davao Research Center, 2023)"
                solution5= ""
                solution6= ""
            else :
                identification = "Diagnosis: Record of this data does not exist!"
                solution1 = "Not applicable"
                solution1 = "Not applicable" 
                solution2 = ""
                solution3 = "" 
                solution4 = ""
                solution5 = ""
                solution6 = ""

        Coconut_Rhinoceros_Beetle_Damage = prediction_percentages['rhinoceros_beetle_damage']   
        Coconut_Scale_Insect_Damage = prediction_percentages['scale_insect_damage']
        Coconut_Leaf_Beetle_Damage = prediction_percentages['leaf_beetle_damage']
        Asiatic_Palm_Weevil_Damage = prediction_percentages['asiatic_palm_weevil_damage']
            
        pests = ['Asiatic Palm Weevil Damage', 'Coconut Leaf Beetle Damage', 'Coconut Rhinoceros Beetle Damage','Coconut Scale Insect Damage' ]
        pestScores = [Asiatic_Palm_Weevil_Damage,Coconut_Leaf_Beetle_Damage, Coconut_Rhinoceros_Beetle_Damage, Coconut_Scale_Insect_Damage ]

        bar_colors = ['#3B7A57', '#29AB87', '#23AB87', '#27AB87']
        fig = make_subplots(rows=1, cols=1)
        bar_trace = go.Bar(y=pests, x=pestScores, orientation='h', text=[f'{score: .2f}%' for score in pestScores],  marker_color=bar_colors, width=0.5)
        fig.add_trace(bar_trace)
        fig.update_layout(
            title="Predicted Regions",
            xaxis_title="Probability", 
            yaxis_title="Pest Damage",
            width = 500,
            height = 300,
            margin=dict(l=20, r=0, t=40, b=20)
            )
        interactive_plot = fig.to_html(full_html=False, config={'displayModeBar': False})
        contexttwo={'filePathName': filePathName, 'result': result, 'identification': identification, "scaleInsectDamageScore": Coconut_Scale_Insect_Damage, "rhinocerosBeetleDamageScore": Coconut_Rhinoceros_Beetle_Damage, "g": interactive_plot, "solution1": solution1, "solution2":solution2, "solution3":solution3, "solution4" : solution4, "solution5":solution5, "solution6" : solution6  }
       
    else :

        print("The predicted image has", output_class)
        identification = "Diagnosis: No signs of damage"
        result = 100 - percentage
        print(f"Damage Percentage = {result}%")

        pests = ['No Signs of Damage', 'damaged' ]
        pestScores = [noSignsDamage, damaged ]

        bar_colors = ['#3B7A57', '#29AB87']
        fig = make_subplots(rows=1, cols=1)
        bar_trace = go.Bar(y=pests, x=pestScores, orientation='h', text=[f'{score: .2f}%' for score in pestScores],  marker_color=bar_colors, width=0.5)
        fig.add_trace(bar_trace)
        fig.update_layout(
            title="Predicted Regions",
            xaxis_title="Probability", 
            yaxis_title="Pest Damage",
            width = 500,
            height = 300,
            margin=dict(l=20, r=0, t=40, b=20)
            )
        interactive_plot = fig.to_html(full_html=False, config={'displayModeBar': False})

        contexttwo={'filePathName': filePathName, 'identification': identification, "g": interactive_plot}
    

        
    # print(f"{output_class} % = {percentage:2f}%")
#*------------------

#*****----------------

    # class_names = ['Asiatic_Palm_Weevil_Damage', 'Coconut_Leaf_Beetle_Damage', 'Coconut_Rhinoceros_Beetle_Damage', 'Coconut_Scale_Insect_Damage']
    # img = image.load_img(test_img, target_size=(150, 150))
    # test_image_array = image.img_to_array(img)
    # test_img_array = np.expand_dims(test_image_array, axis=0)
    # print("This is the img shape: ", test_img_array)

    # pred = model.predict(test_img_array)
    # print("This is the prediction: ", pred)

#*****------------------
    # for z in zip(['Asiatic_Palm_Weevil_Damage', 'Coconut_Leaf_Beetle_Damage', 'Coconut_Rhinoceros_Beetle_Damage', 'Coconut_Scale_Insect_Damage', 'Coconut_Undiseased'], pred[0]):
    #     result = '%s %% = %.2f%%' % (z[0], z[1] * 100)
    #     print(result)

    #     pest_class = z[0]
    #     percentage = z[1] * 100
    #     prediction_percentages[pest_class] = percentage
    #     print(f"{pest_class} % = {percentage: .2f}%")


#***--------------------------------
    # prediction_percentages = {}
    # for z in zip(['Asiatic_Palm_Weevil_Damage', 'Coconut_Leaf_Beetle_Damage', 'Coconut_Rhinoceros_Beetle_Damage', 'Coconut_Scale_Insect_Damage'], pred[0]):
    #     result = '%s %% = %.2f%%' % (z[0], z[1] * 100)
    #     print(result)
        
    #     pest_class = z[0]
    #     percentage = z[1] * 100
    #     prediction_percentages[pest_class] = percentage
    #     print(f"{pest_class} % = {percentage: .2f}%")
    
    #     output_class=class_names[np.argmax(pred)]
    #     print("The predicted class is", output_class)

    #     if output_class == 'Asiatic_Palm_Weevil_Damage':
    #         identification = "Diagnosis: Asiatic Palm Weevil Damage"
    #         solution1 = "Management Strategies" 
    #         solution2 = "Cultural Method:Regulated prunning of infested fronds" 
    #         solution3= "Chemical Method: Chemical control using systemic insecticides, FPA prescribed for coconut pests by trunk injection(for emergency only)."
    #         solution4 = "Topical application of vegetable oil in young palms can be done during cooler months." 
    #         solution5 = "Biological Method: Sustained release of biological control agents like Chilochorus sp., Telsimia sp., Pseudoscymnus anomalus, Cybocephalus sp., Bathracedra sp., Comperiella calauanica."
    #         solution6= "Regulatory Control: Implementation of quarantine regulations and establishment of checkpoints (PCA Davao Research Center, 2023)"
    #     elif output_class == 'Coconut_Leaf_Beetle_Damage':
    #         identification = "Diagnosis: Coconut Leaf Beetle Damage"
    #         solution1 = "Management Strategies"
    #         solution2 = "Biological Control: Establishment of coconut log traps inoculated with Green Muscardine Fungus (GMF) granules. Concentrated Oryctes nudivurus is dropped to the mouth of beetles. Infection spreads through mating and visit in breeding sites"
    #         solution3 = "Cultural Control: Collection and utilization of coconut debris or farm waste t avoid piling. PRactice farm sanitation. Regular inspection of all possible breeding sites and collection of all stages of the beetle. Plant covercrops if intercropping is not practiced" 
    #         solution4 = "Chemical Control: Installation of rhinoceros beetle pheromone in traps, enhanced with food bait (PCA Davao Research Center, 2023)"
    #         solution5= ""
    #         solution6= ""
    #     elif output_class == 'Coconut_Rhinoceros_Beetle_Damage':
    #         identification = "Diagnosis: Coconut Rhinoceros Beetle Damage"
    #         solution1 = "Management Strategies" 
    #         solution2 = "Cultural Method:Regulated prunning of infested fronds" 
    #         solution3= "Chemical Method: Chemical control using systemic insecticides, FPA prescribed for coconut pests by trunk injection(for emergency only)."
    #         solution4 = "Topical application of vegetable oil in young palms can be done during cooler months." 
    #         solution5 = "Biological Method: Sustained release of biological control agents like Chilochorus sp., Telsimia sp., Pseudoscymnus anomalus, Cybocephalus sp., Bathracedra sp., Comperiella calauanica."
    #         solution6= "Regulatory Control: Implementation of quarantine regulations and establishment of checkpoints (PCA Davao Research Center, 2023)"
    #     elif output_class == 'Coconut_Scale_Insect_Damage':
    #         identification = "Diagnosis: Coconut Scale Insect Damage"
    #         solution1 = "Management Strategies"
    #         solution2 = "Biological Control: Establishment of coconut log traps inoculated with Green Muscardine Fungus (GMF) granules. Concentrated Oryctes nudivurus is dropped to the mouth of beetles. Infection spreads through mating and visit in breeding sites"
    #         solution3 = "Cultural Control: Collection and utilization of coconut debris or farm waste t avoid piling. PRactice farm sanitation. Regular inspection of all possible breeding sites and collection of all stages of the beetle. Plant covercrops if intercropping is not practiced" 
    #         solution4 = "Chemical Control: Installation of rhinoceros beetle pheromone in traps, enhanced with food bait (PCA Davao Research Center, 2023)"
    #         solution5= ""
    #         solution6= ""
    #     else :
    #         identification = "Diagnosis: Record of this data does not exist!"
    #         solution1 = "Not applicable"
    #         solution1 = "Not applicable" 
    #         solution2 = ""
    #         solution3 = "" 
    #         solution4 = ""
    #         solution5 = ""
    #         solution6 = ""

    # Coconut_Rhinoceros_Beetle_Damage = prediction_percentages['Coconut_Rhinoceros_Beetle_Damage']   
    # Coconut_Scale_Insect_Damage = prediction_percentages['Coconut_Scale_Insect_Damage']
    # Coconut_Leaf_Beetle_Damage = prediction_percentages['Coconut_Leaf_Beetle_Damage']
    # Asiatic_Palm_Weevil_Damage = prediction_percentages['Asiatic_Palm_Weevil_Damage']
        
    # pests = ['Asiatic Palm Weevil Damage', 'Coconut Leaf Beetle Damage', 'Coconut Rhinoceros Beetle Damage', 'Coconut Scale Insect Damage']
    # pestScores = [Asiatic_Palm_Weevil_Damage,  Coconut_Leaf_Beetle_Damage, Coconut_Rhinoceros_Beetle_Damage, Coconut_Scale_Insect_Damage]

    # bar_colors = ['#3B7A57', '#29AB87', '#23AB87', '#27AB87']
    # fig = make_subplots(rows=1, cols=1)
    # bar_trace = go.Bar(y=pests, x=pestScores, orientation='h', text=[f'{score: .2f}%' for score in pestScores],  marker_color=bar_colors, width=0.5)
    # fig.add_trace(bar_trace)
    # fig.update_layout(
    #     title="Predicted Regions",
    #     xaxis_title="Probability", 
    #     yaxis_title="Pest Damage",
    #     width = 500,
    #     height = 300,
    #     margin=dict(l=20, r=0, t=40, b=20)
    #     )
    # interactive_plot = fig.to_html(full_html=False, config={'displayModeBar': False})

#****--------------------------------   

    # Coconut_Leaf_Beetle_Damage = prediction_percentages['Coconut_Leaf_Beetle_Damage']
    # Asiatic_Palm_Weevil_Damage = prediction_percentages['Asiatic_Palm_Weevil_Damage']
    # Coconut_Rhinoceros_Beetle_Damage = prediction_percentages['Coconut_Rhinoceros_Beetle_Damage']   
    # Coconut_Scale_Insect_Damage = prediction_percentages['Coconut_Scale_Insect_Damage']
    # Coconut_Undiseased = prediction_percentages['Coconut_Undiseased']
    
    # pests = ['Asiatic Palm Weevil Damage', 'Coconut Leaf Beetle Damage', 'Coconut Rhinoceros Beetle Damage', 'Coconut Scale Insect Damage', 'Coconut Undiseased']
    # pestScores = [Asiatic_Palm_Weevil_Damage,  Coconut_Leaf_Beetle_Damage, Coconut_Rhinoceros_Beetle_Damage, Coconut_Scale_Insect_Damage, Coconut_Undiseased]

    # fig, ax = plt.subplots(figsize=(4, 4))
    # ax.clear()
    # width = 0.5
    # bar_labels = ['Scale Insect Damage', 'Rhinoceros Beetle Damage', 'Undiseased Coconut']
    # bar_colors = ['#3B7A57', '#29AB87', '#23AB87']

    # bars = ax.barh(range(len(pests)), pestScores, width, label=bar_labels, color=bar_colors)
    # ypos = np.arange(len(pests))
        
    # ax.set_yticks(ypos, pests)
    # for bar, score in zip(bars, pestScores):
    #     ax.text(score + 1, bar.get_y() + bar.get_height() / 2, f'{score:.2f}%', va='center')
    # ax.set_xlabel('probability')
    # ax.set_title('Predicted regions')
    # ax.set_xlim(0, 105)
    # ax.legend(title='Pest Damage', loc='lower right')

    # interactive_plot = mpld3.fig_to_html(plt.gcf())

    # for z in zip(['Coconut_Rhinoceros_Beetle_Damage', 'Coconut_Scale_Insect_Damage', 'Coconut_Undiseased'], predi[0]):
    #     percentage = '%s %% = %.2f%%' % (z[0], z[1] * 100)
    #     print(percentage)

    # identification = class_names[np.argmax(predi)]
    # print("The predicted class is", identification)
    
    
    # score = float(predi[0])
    # scaleInsectDamageScore = round(100 * score)
    # rhinocerosBeetleDamageScore = round(100 * (1 - score))

    # result = f"This image is {percentage}"

    # if scaleInsectDamageScore > rhinocerosBeetleDamageScore:
    #     identification = "Diagnosis: Coconut Scale Insect Damage"
    #     solution1 = "Management Strategies" 
    #     solution2 = "Cultural Method:Regulated prunning of infested fronds" 
    #     solution3= "Chemical Method: Chemical control using systemic insecticides, FPA prescribed for coconut pests by trunk injection(for emergency only)."
    #     solution4 = "Topical application of vegetable oil in young palms can be done during cooler months." 
    #     solution5 = "Biological Method: Sustained release of biological control agents like Chilochorus sp., Telsimia sp., Pseudoscymnus anomalus, Cybocephalus sp., Bathracedra sp., Comperiella calauanica."
    #     solution6= "Regulatory Control: Implementation of quarantine regulations and establishment of checkpoints (PCA Davao Research Center, 2023)"
    # elif scaleInsectDamageScore < rhinocerosBeetleDamageScore:
    #     identification = "Diagnosis: Coconut Rhinoceros Beetle Damage"
    #     solution1 = "Management Strategies"
    #     solution2 = "Biological Control: Establishment of coconut log traps inoculated with Green Muscardine Fungus (GMF) granules. Concentrated Oryctes nudivurus is dropped to the mouth of beetles. Infection spreads through mating and visit in breeding sites"
    #     solution3 = "Cultural Control: Collection and utilization of coconut debris or farm waste t avoid piling. PRactice farm sanitation. Regular inspection of all possible breeding sites and collection of all stages of the beetle. Plant covercrops if intercropping is not practiced" 
    #     solution4 = "Chemical Control: Installation of rhinoceros beetle pheromone in traps, enhanced with food bait (PCA Davao Research Center, 2023)"
    #     solution5= ""
    #     solution6= ""
    # else:
    #     identification = "Diagnosis: Record of this data does not exist!"
    #     solution1 = "Not applicable"

    # fig, ax = plt.subplots(figsize=(4, 4))
    # ax.clear()
    # width = 0.5
    # pests = ['Coconut Scale Insect Damage', 'Rhinoceros Beetle Damage', 'Undiseased Coconut']
    # pestScores = [scaleInsectDamageScore, rhinocerosBeetleDamageScore]
    # bar_labels = ['Scale Insect Damage', 'Rhinoceros Beetle Damage', 'Undiseased Coconut']
    # bar_colors = ['#3B7A57', '#29AB87', '#23AB87']

    # ax.barh(pests, pestScores, width, label=bar_labels, color=bar_colors)
    # ypos = np.arange(len(pests))
    
    # ax.set_yticks(ypos, pests)
    # ax.set_xlabel('probability')
    # ax.set_title('Predicted regions')
    # ax.set_xlim(0, 105)
    # ax.legend(title='Pest Damage', loc='lower right')

   
    # interactive_plot = mpld3.fig_to_html(plt.gcf())
    
 
    # contexttwo={'filePathName': filePathName, 'result': result, 'identification': identification, "scaleInsectDamageScore": scaleInsectDamageScore, "rhinocerosBeetleDamageScore": rhinocerosBeetleDamageScore, "g": interactive_plot, "solution1": solution1, "solution2":solution2, "solution3":solution3, "solution4" : solution4, "solution5":solution5, "solution6" : solution6  }

    # contexttwo={'filePathName': filePathName  }
    return render(request, 'result.html', contexttwo)

@login_required(login_url='user-login')
def viewDatabase(request):
    listofImages = os.listdir('./media/')
    listofImagePath = ['./media/'+i for i in listofImages]
    fs=FileSystemStorage()
    context = {'listofImagePath': listofImagePath}
    return render(request, 'viewDB.html', context)

@login_required(login_url='user-login')
def coconutDamages(request):
    # items = CoconutDamages.objects.all()
    items = CoconutDamages.objects.raw('SELECT * FROM thesis_CoconutDamages')
    context = {
        'items': items,
    }
    # context = {}
    return render(request, 'coconutdamages.html', context)

def add_coconutDamages(request):
    if request.method == 'POST':
        form = CoconutDamagesForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('coconutDamages')
    else:
        form = CoconutDamagesForm()
    context = {
            'form': form,
    }
    return render(request, 'add_coconutdamages.html', context)

@login_required(login_url='user-login')
def captureImage(request):
    context = {}
    return render(request, 'captureMedia.html', context)

def delete_coconutDamages(request, pk):

    item = CoconutDamages.objects.get(id=pk)
    if request.method == 'POST':
        item.delete()
        return redirect('coconutDamages')
    return render(request, 'delete_coconutdamages.html')

def update_coconutDamages(request, pk):
    item = CoconutDamages.objects.get(id=pk)
    if request.method == 'POST':
        form = CoconutDamagesForm(request.POST, instance=item)
        if form.is_valid():
            form.save()
            return redirect('coconutDamages')
    else:
        form = CoconutDamagesForm(instance=item)
    context ={
        'form': form
    }
    return render(request, 'update_coconutdamages.html', context)
# def login(request):
#     context = {}
#     return render(request, 'sample.html', context)