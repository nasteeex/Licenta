from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import redirect
from django.urls import reverse
from django.views.generic import ListView, CreateView, UpdateView

from locations.models import Locations


class LocationsView(LoginRequiredMixin, ListView):
    model = Locations
    template_name = 'locations/locations_index.html'
    paginate_by = 2


class CreateLocationsView(LoginRequiredMixin, CreateView):
    model = Locations
    fields = ['name']
    template_name = 'locations/locations_form.html'

    def get_success_url(self):
        return reverse('locations:lista_locatii')


class UpdateLocationsView(LoginRequiredMixin, UpdateView):
    model = Locations
    fields = ['name', 'coordinates', 'last_accessed', 'mark']
    template_name = 'locations/locations_form.html'

    def get_success_url(self):
        return reverse('locations:lista_locatii')


@login_required
def delete_locations(request, pk):
    Locations.objects.filter(id=pk).update(active=0)
    return redirect('locations:lista_locatii')


@login_required
def activate_locations(request, pk):
    Locations.objects.filter(id=pk).update(active=1)
    return redirect('locations:lista_locatii')
