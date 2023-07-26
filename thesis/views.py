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
import mpld3
import matplotlib
matplotlib.use("Agg")  # Use the "Agg" backend for non-interactive use
import matplotlib.pyplot as plt
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from .models import CoconutDamages
from .forms import CoconutDamagesForm

model = load_model('./models/ResNet50_Architecture_coconutDamages_ai_training.h5')


from keras.applications import ResNet50
# The line of code initializes a ResNet50 convolutional base model with pre-trained weights from ImageNet, excluding the top (classification) layer, and setting the input shape to (150, 150, 3).  
conv_base = ResNet50(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

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

    testimage = '.'+filePathName
    img = image.load_img(testimage, target_size=(150, 150))
    test_image_array = image.img_to_array(img)
    test_image_array = np.expand_dims(test_image_array, axis=0)
    test_image_array=test_image_array/225 # Normalize the pixel values (optional)
    features_batch = conv_base.predict(test_image_array)
    test_image_array = np.reshape(features_batch, (1, 5 * 5 * 2048))
    predi = model.predict(test_image_array)
    
    
    score = float(predi[0])
    scaleInsectDamageScore = round(100 * score)
    rhinocerosBeetleDamageScore = round(100 * (1 - score))

    result = f"This image is {rhinocerosBeetleDamageScore}% Coconut Rhinoceros Beetle Damage and {scaleInsectDamageScore}% Coconut Scale Insect Damage."

    if scaleInsectDamageScore > rhinocerosBeetleDamageScore:
        identification = "Diagnosis: Coconut Scale Insect Damage"
        solution1 = "Management Strategies" 
        solution2 = "Cultural Method:Regulated prunning of infested fronds" 
        solution3= "Chemical Method: Chemical control using systemic insecticides, FPA prescribed for coconut pests by trunk injection(for emergency only)."
        solution4 = "Topical application of vegetable oil in young palms can be done during cooler months." 
        solution5 = "Biological Method: Sustained release of biological control agents like Chilochorus sp., Telsimia sp., Pseudoscymnus anomalus, Cybocephalus sp., Bathracedra sp., Comperiella calauanica."
        solution6= "Regulatory Control: Implementation of quarantine regulations and establishment of checkpoints (PCA Davao Research Center, 2023)"
    elif scaleInsectDamageScore < rhinocerosBeetleDamageScore:
        identification = "Diagnosis: Coconut Rhinoceros Beetle Damage"
        solution1 = "Management Strategies"
        solution2 = "Biological Control: Establishment of coconut log traps inoculated with Green Muscardine Fungus (GMF) granules. Concentrated Oryctes nudivurus is dropped to the mouth of beetles. Infection spreads through mating and visit in breeding sites"
        solution3 = "Cultural Control: Collection and utilization of coconut debris or farm waste t avoid piling. PRactice farm sanitation. Regular inspection of all possible breeding sites and collection of all stages of the beetle. Plant covercrops if intercropping is not practiced" 
        solution4 = "Chemical Control: Installation of rhinoceros beetle pheromone in traps, enhanced with food bait (PCA Davao Research Center, 2023)"
        solution5= ""
        solution6= ""
    else:
        identification = "Diagnosis: Record of this data does not exist!"
        solution1 = "Not applicable"

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.clear()
    width = 0.5
    pests = ['Coconut Scale Insect Damage', 'Rhinoceros Beetle Damage']
    pestScores = [scaleInsectDamageScore, rhinocerosBeetleDamageScore]
    bar_labels = ['Scale Insect Damage', 'Rhinoceros Beetle Damage']
    bar_colors = ['#3B7A57', '#29AB87']

    ax.barh(pests, pestScores, width, label=bar_labels, color=bar_colors)
    ypos = np.arange(len(pests))
    
    ax.set_yticks(ypos, pests)
    ax.set_xlabel('probability')
    ax.set_title('Predicted regions')
    ax.set_xlim(0, 105)
    ax.legend(title='Pest Damage', loc='lower right')
    # interactive_plot = plt.show()

   
    interactive_plot = mpld3.fig_to_html(plt.gcf())
    
    # Convert the plot to an interactive HTML representation using mpld3

    # _, train_acc = model.evaluate(train_features, train_labels, verbose=2)

    # _, validation_acc = model.evaluate(validation_features, validation_labels)
 
    contexttwo={'filePathName': filePathName, 'result': result, 'identification': identification, "scaleInsectDamageScore": scaleInsectDamageScore, "rhinocerosBeetleDamageScore": rhinocerosBeetleDamageScore, "g": interactive_plot, "solution1": solution1, "solution2":solution2, "solution3":solution3, "solution4" : solution4, "solution5":solution5, "solution6" : solution6  }
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