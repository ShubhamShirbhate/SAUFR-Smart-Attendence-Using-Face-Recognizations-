
from django.contrib import admin
from django.urls import path, include
#from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',include('face.urls')),
    path('faculty',include('faculty.urls')),
  #  path('detect/', views.detect, name='detect')
]
