from django.shortcuts import render
from rest_framework import viewsets
from aplicatie1.models import Locations
from aplicatie2.models import Companies
from myapi.serializers import LocationSerializers, CompanySerializers


class LocationViewSet(viewsets.ModelViewSet):
    queryset = Locations.objects.all().order_by('city')
    serializer_class = LocationSerializers


class CompanyViewSet(viewsets.ModelViewSet):
    queryset = Companies.objects.all().order_by('nume')
    serializer_class = CompanySerializers
