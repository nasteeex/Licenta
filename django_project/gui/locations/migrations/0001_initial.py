# Generated by Django 4.0.4 on 2022-05-28 10:19

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Locations',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=20)),
                ('coordinates', models.CharField(max_length=100)),
                ('last_accessed', models.CharField(max_length=20)),
                ('mark', models.CharField(default='Not marked', max_length=20)),
                ('active', models.BooleanField(default=1)),
            ],
        ),
    ]