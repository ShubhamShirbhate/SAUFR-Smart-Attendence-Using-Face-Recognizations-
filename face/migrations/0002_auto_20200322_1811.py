# Generated by Django 3.0.4 on 2020-03-22 12:41

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('face', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='attendence',
            old_name='Sname',
            new_name='Skey',
        ),
        migrations.RemoveField(
            model_name='attendence',
            name='name',
        ),
        migrations.RemoveField(
            model_name='student',
            name='Sname',
        ),
        migrations.AddField(
            model_name='student',
            name='Skey',
            field=models.CharField(max_length=255, null=True),
        ),
    ]