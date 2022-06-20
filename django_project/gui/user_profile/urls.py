from django.urls import path
from user_profile import views

app_name = 'userprofile'

urlpatterns = [
    path('start_tracking/', views.start_track, name='start_tracker'),
    path('stop_tracking/', views.stop_track, name='stop_tracker'),
]
