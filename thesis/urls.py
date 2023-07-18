from django.urls import path


from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("predictImage/",views.predictImage, name="predictImage"),
    path("viewDatabase", views.viewDatabase, name="viewDatabase"),
    path("captureImage", views.captureImage, name='captureImage')
]
