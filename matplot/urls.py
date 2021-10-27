from django.urls import path, include
from . import views

urlpatterns = [
    #path("", views.home, name="home"),
    path("", views.matplot, name="home.html"),
    #path("matplot", views.matplot, name="home.html"),
    #path("add", views.add, name="add")
]