from django.urls import path
from . import views
#urls is where you link everything together
urlpatterns = [
    path("", views.home, name="home")
]