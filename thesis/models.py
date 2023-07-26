from django.db import models

# Create your models here.
class CoconutDamages(models.Model):
    name = models.CharField(max_length=100, null=True)
    biologicalcontrol = models.CharField(max_length=500, null=True)
    culturalcontrol = models.CharField(max_length=500, null=True)
    chemicalcontrol = models.CharField(max_length=500, null=True)

    def __str__(self):
        return f'{self.name}-{self.biologicalcontrol}'
    