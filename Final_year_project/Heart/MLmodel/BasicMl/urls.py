from django.urls import path
from . import views

urlpatterns = [
    path('', views.welcome),
    path('predict/', views.predict),
    path('predict/result/', views.result),
    path('result/', views.result, name='result'),
    path('visualization/', views.visualization, name='visualization'),
]