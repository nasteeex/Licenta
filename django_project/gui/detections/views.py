from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import redirect
from django.urls import reverse
from django.views.generic import ListView, CreateView, UpdateView

from detections.models import Detections


class DetectionsView(LoginRequiredMixin, ListView):
    model = Detections
    template_name = 'detections/detections_index.html'
    paginate_by = 2


class CreateDetectionsView(LoginRequiredMixin, CreateView):
    model = Detections
    fields = ['plate', 'date', 'time', 'file']
    template_name = 'detections/detections_form.html'

    def get_success_url(self):
        return reverse('detections:lista_detectii')


class UpdateDetectionsView(LoginRequiredMixin, UpdateView):
    model = Detections
    fields = ['plate', 'date', 'time', 'file', 'mark']
    template_name = 'detections/detections_form.html'

    def get_success_url(self):
        return reverse('detections:lista_detectii')


@login_required
def delete_detections(request, pk):
    Detections.objects.filter(id=pk).update(active=0)
    return redirect('detections:lista_detectii')


@login_required
def activate_detections(request, pk):
    Detections.objects.filter(id=pk).update(active=1)
    return redirect('detections:lista_detectii')
