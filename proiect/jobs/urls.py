from django.urls import path

from jobs import views

app_name = 'jobs'

urlpatterns = [
    path('', views.JobsView.as_view(), name='lista_joburi'),
    path('adaugare/', views.CreateJobView.as_view(), name='adauga'),
    path('<int:pk>/update/', views.UpdateJobView.as_view(), name='modifica'),
    path('<int:pk>/stergere/', views.delete_job, name='sterge'),
    path('<int:pk>/activeaza/', views.activate_job, name='activeaza'),
]
