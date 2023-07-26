from django.urls import path
from user import views as user_view

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("predictimage/",views.predictImage, name="predictImage"),
    path("viewdatabase", views.viewDatabase, name="viewDatabase"),
    path("captureimage", views.captureImage, name='captureImage'),
    path('pcadrcregister', user_view.register, name='user-register'),
    path('coconutdamages', views.coconutDamages, name="coconutDamages"),
    path('coconutdamages/delete/<int:pk>', views.delete_coconutDamages, name='delete_coconutDamages' ),
    path('coconutdamages/update/<int:pk>', views.update_coconutDamages, name='update_coconutDamages' ),
    path('coconutdamages/add/', views.add_coconutDamages, name='add_coconutDamages' ),
    # path("login", views.login, name='login'),

]
