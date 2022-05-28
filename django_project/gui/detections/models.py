from django.db import models

mark_options = (('Not marked', 'Not Marked'), ('Danger', 'Danger'), ('Safe', 'Safe'))


class Detections(models.Model):

    plate = models.CharField(max_length=20)
    date = models.CharField(max_length=20)
    time = models.CharField(max_length=10)
    file = models.CharField(max_length=120)
    mark = models.CharField(max_length=20, default="Not marked", choices=mark_options)
    active = models.BooleanField(default=1)

    def __str__(self):
        return f"{self.plate} - {self.mark}"
