from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import redirect
from django.urls import reverse
from django.views.generic import ListView, CreateView, UpdateView

from jobs.models import Jobs


class JobsView(LoginRequiredMixin, ListView):
    model = Jobs
    template_name = 'jobs/jobs_index.html'
    paginate_by = 7


class CreateJobView(LoginRequiredMixin, CreateView):
    model = Jobs
    fields = ['title', 'description', 'type', 'url', 'how_to_apply']
    template_name = 'jobs/jobs_form.html'

    def get_success_url(self):
        return reverse('jobs:lista_joburi')


class UpdateJobView(LoginRequiredMixin, UpdateView):
    model = Jobs
    fields = ['title', 'description', 'type', 'url', 'how_to_apply']
    template_name = 'jobs/jobs_form.html'

    def get_success_url(self):
        return reverse('jobs:lista_joburi')


@login_required
def delete_job(request, pk):
    Jobs.objects.filter(id=pk).update(active=0)
    return redirect('jobs:lista_joburi')


@login_required
def activate_job(request, pk):
    Jobs.objects.filter(id=pk).update(active=1)
    return redirect('jobs:lista_joburi')