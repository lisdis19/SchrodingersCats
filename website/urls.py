from django.urls import path
from . import views

# root , filename.function ,  name the variable

urlpatterns = [
    path("", views.home, name="home")
]