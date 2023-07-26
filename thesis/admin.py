from django.contrib import admin
from .models import CoconutDamages

admin.site.site_header = 'PCADRC Dashboard'
# Register your models here.

class CoconutDamagesAdmin(admin.ModelAdmin):
    list_display = ('name', 'biologicalcontrol', 'culturalcontrol', 'chemicalcontrol')

admin.site.register(CoconutDamages, CoconutDamagesAdmin)
