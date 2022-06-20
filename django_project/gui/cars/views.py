from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import redirect
from django.urls import reverse
from django.views.generic import ListView, CreateView, UpdateView

from cars.models import Cars


class CarsView(LoginRequiredMixin, ListView):
    model = Cars
    template_name = 'cars/cars_index.html'
    paginate_by = 2


class CreateCarsView(LoginRequiredMixin, CreateView):
    model = Cars
    fields = ['name', 'plate']
    template_name = 'cars/cars_form.html'

    def get_success_url(self):
        return reverse('cars:lista_masini')


class UpdateCarsView(LoginRequiredMixin, UpdateView):
    model = Cars
    fields = ['name', 'brand', 'model', 'year', 'plate']
    template_name = 'cars/cars_form.html'

    def get_success_url(self):
        return reverse('cars:lista_masini')


@login_required
def delete_cars(request, pk):
    Cars.objects.filter(id=pk).update(active=0)
    return redirect('cars:lista_masini')


@login_required
def activate_cars(request, pk):
    Cars.objects.filter(id=pk).update(active=1)
    return redirect('cars:lista_masini')
