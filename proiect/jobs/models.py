from django.db import models

tipuri_joburi = (("Part-Time", "Part-Time"), ("Full-Time", "Full-Time"))


class Jobs(models.Model):

    type = models.CharField(max_length=11, choices=tipuri_joburi)
    url = models.CharField(max_length=100)
    title = models.CharField(max_length=100)
    description = models.TextField()
    how_to_apply = models.TextField()
    active = models.BooleanField(default=1)

    def __str__(self):
        return f"{self.title} - {self.description}"
