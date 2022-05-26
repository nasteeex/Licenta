from django.urls import path

from aplicatie2 import views

app_name = 'companies'

urlpatterns = [
    path('', views.CompaniesView.as_view(), name='lista_companii'),
    path('adaugare/', views.CreateCompaniesView.as_view(), name='adauga'),
    path('<int:pk>/update/', views.UpdateCompaniesView.as_view(), name='modifica'),
    path('<int:pk>/stergere/', views.delete_company, name='sterge'),
    path('<int:pk>/activeaza/', views.activate_company, name='activeaza'),
]
