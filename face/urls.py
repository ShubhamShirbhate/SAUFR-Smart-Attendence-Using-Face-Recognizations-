from django.urls import path,include
from . import views
#from .project import views

urlpatterns = [
    path('', views.first_page, name='first_page'),
    
    path('create_dataset/',views.create_dataset, name='create_dataset'),
    path('trainer/', views.trainer, name='trainer'),
    path('detect/', views.detect, name='detect'),
    
]
