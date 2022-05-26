from django.urls import path, include
from rest_framework import routers

from myapi import views

routerLocation = routers.DefaultRouter()
routerLocation.register('location', views.LocationViewSet)
routerCompany = routers.DefaultRouter()
routerCompany.register('company', views.CompanyViewSet)

urlpatterns = [
    path('', include(routerLocation.urls)),
    path('', include(routerCompany.urls)),
]
