from django.urls import path

from detections import views

app_name = 'detections'

urlpatterns = [
    path('', views.DetectionsView.as_view(), name='lista_detectii'),
    path('adaugare/', views.CreateDetectionsView.as_view(), name='adauga'),
    path('<int:pk>/update/', views.UpdateDetectionsView.as_view(), name='modifica'),
    path('<int:pk>/stergere/', views.delete_detections, name='sterge'),
    path('<int:pk>/activeaza/', views.activate_detections, name='activeaza'),
]
