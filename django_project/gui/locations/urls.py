from django.urls import path

from locations import views

app_name = 'locations'

urlpatterns = [
    path('', views.LocationsView.as_view(), name='lista_locatii'),
    path('adaugare/', views.CreateLocationsView.as_view(), name='adauga'),
    path('<int:pk>/update/', views.UpdateLocationsView.as_view(), name='modifica'),
    path('<int:pk>/stergere/', views.delete_locations, name='sterge'),
    path('<int:pk>/activeaza/', views.activate_locations, name='activeaza'),
]
