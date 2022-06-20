from django.urls import path

from cars import views

app_name = 'cars'

urlpatterns = [
    path('', views.CarsView.as_view(), name='lista_masini'),
    path('adaugare/', views.CreateCarsView.as_view(), name='adauga'),
    path('<int:pk>/update/', views.UpdateCarsView.as_view(), name='modifica'),
    path('<int:pk>/stergere/', views.delete_cars, name='sterge'),
    path('<int:pk>/activeaza/', views.activate_cars, name='activeaza'),
]
