from user_profile.models import Tracking


def is_ready_to_track(request):
    if request.user.is_authenticated:
        if Tracking.objects.filter(user_id=request.user.id, end_date=None).exists():
            return {'ready_to_track': False}
        return {'ready_to_track': True}
    return {}
