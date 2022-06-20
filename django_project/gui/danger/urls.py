from django.urls import path

from danger import views

app_name = 'danger'

urlpatterns = [
    path('', views.DangerView.as_view(), name='lista_pericol'),
    path('adaugare/', views.CreateDangerView.as_view(), name='adauga'),
    path('<int:pk>/update/', views.UpdateDangerView.as_view(), name='modifica'),
    path('<int:pk>/stergere/', views.delete_danger, name='sterge'),
    path('<int:pk>/activeaza/', views.activate_danger, name='activeaza'),
]
