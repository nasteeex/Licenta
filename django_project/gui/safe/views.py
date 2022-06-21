from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import redirect
from django.urls import reverse
from django.views.generic import ListView, CreateView, UpdateView

from safe.models import Safe


class SafeView(LoginRequiredMixin, ListView):
    model = Safe
    template_name = 'safe/safe_index.html'
    paginate_by = 5


class CreateSafeView(LoginRequiredMixin, CreateView):
    model = Safe
    fields = ['plate']
    template_name = 'safe/safe_form.html'

    def get_success_url(self):
        return reverse('safe:lista_sigur')


class UpdateSafeView(LoginRequiredMixin, UpdateView):
    model = Safe
    fields = ['name', 'brand', 'model', 'year', 'plate']
    template_name = 'safe/safe_form.html'

    def get_success_url(self):
        return reverse('safe:lista_sigur')


@login_required
def delete_safe(request, pk):
    Safe.objects.filter(id=pk).update(active=0)
    return redirect('safe:lista_sigur')


@login_required
def activate_safe(request, pk):
    Safe.objects.filter(id=pk).update(active=1)
    return redirect('safe:lista_sigur')
