import datetime

from django.contrib.auth.decorators import login_required
from django.http import HttpResponseRedirect
from django.shortcuts import render

from user_profile.models import Tracking


@login_required
def start_track(request):
    Tracking.objects.create(user_id=request.user.id, start_date=datetime.datetime.now())
    return HttpResponseRedirect(request.META.get('HTTP_REFERER'))


@login_required
def stop_track(request):
    Tracking.objects.filter(user_id=request.user.id, end_date=None).update(end_date=datetime.datetime.now())
    return HttpResponseRedirect(request.META.get('HTTP_REFERER'))
