from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import redirect
from django.urls import reverse
from django.views.generic import ListView, CreateView, UpdateView

from danger.models import Danger


class DangerView(LoginRequiredMixin, ListView):
    model = Danger
    template_name = 'danger/danger_index.html'
    paginate_by = 2


class CreateDangerView(LoginRequiredMixin, CreateView):
    model = Danger
    fields = ['plate']
    template_name = 'danger/danger_form.html'

    def get_success_url(self):
        return reverse('danger:lista_pericol')


class UpdateDangerView(LoginRequiredMixin, UpdateView):
    model = Danger
    fields = ['name', 'brand', 'model', 'year', 'plate']
    template_name = 'danger/danger_form.html'

    def get_success_url(self):
        return reverse('danger:lista_pericol')


@login_required
def delete_danger(request, pk):
    Danger.objects.filter(id=pk).update(active=0)
    return redirect('danger:lista_pericol')


@login_required
def activate_danger(request, pk):
    Danger.objects.filter(id=pk).update(active=1)
    return redirect('danger:lista_pericol')
