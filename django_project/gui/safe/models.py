from django.db import models


class Safe(models.Model):

    name = models.CharField(max_length=20)
    brand = models.CharField(max_length=20)
    model = models.CharField(max_length=50)
    year = models.CharField(max_length=4)
    plate = models.CharField(max_length=20)
    active = models.BooleanField(default=1)

    def __str__(self):
        return f"{self.name} - {self.plate}"
