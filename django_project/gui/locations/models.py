from django.db import models


class Locations(models.Model):

    name = models.CharField(max_length=20)
    coordinates = models.CharField(max_length=100)
    last_accessed = models.CharField(max_length=20)
    mark = models.CharField(max_length=20, default="Not marked")
    active = models.BooleanField(default=1)

    def __str__(self):
        return f"{self.name} - {self.mark}"
