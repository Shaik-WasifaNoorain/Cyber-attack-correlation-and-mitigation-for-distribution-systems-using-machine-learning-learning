from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)


class cyber_attack_detection(models.Model):

    Fid= models.CharField(max_length=3000)
    pid= models.CharField(max_length=3000)
    ptime= models.CharField(max_length=3000)
    date_time= models.CharField(max_length=3000)
    src_ip_address= models.CharField(max_length=3000)
    dst_ip_address= models.CharField(max_length=3000)
    frame_protos= models.CharField(max_length=3000)
    src_port= models.CharField(max_length=3000)
    dst_port= models.CharField(max_length=3000)
    sbytes= models.CharField(max_length=3000)
    dbytes= models.CharField(max_length=3000)
    uid= models.CharField(max_length=3000)
    Prediction= models.CharField(max_length=30000)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



