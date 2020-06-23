from django.db import models

class Student(models.Model):
    name = models.CharField(max_length=255)
    Skey = models.CharField(max_length=255, null=True)
   
    def __str__(self):
        return self.name



class Attendence(models.Model):
    Skey = models.CharField(max_length=255)
    #name = models.CharField(max_length=255)
    date=  models.CharField(max_length=100)
    time = models.CharField(max_length=100)

    def __str__(self):
        return self.date