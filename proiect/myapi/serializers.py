from rest_framework import serializers

from aplicatie1.models import Locations
from aplicatie2.models import Companies


class LocationSerializers(serializers.HyperlinkedModelSerializer):

    class Meta:
        model = Locations
        fields = ['id', 'city', 'country']


class CompanySerializers(serializers.HyperlinkedModelSerializer):

    class Meta:
        model = Companies
        fields = ['id', 'nume', 'website']
