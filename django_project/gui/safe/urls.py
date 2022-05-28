from django.urls import path

from safe import views

app_name = 'safe'

urlpatterns = [
    path('', views.SafeView.as_view(), name='lista_sigur'),
    path('adaugare/', views.CreateSafeView.as_view(), name='adauga'),
    path('<int:pk>/update/', views.UpdateSafeView.as_view(), name='modifica'),
    path('<int:pk>/stergere/', views.delete_safe, name='sterge'),
    path('<int:pk>/activeaza/', views.activate_safe, name='activeaza'),
]
