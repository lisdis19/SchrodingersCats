# Hacking the Gender Stack 2024

This template can be used as a starting point for developing your own application during the hackathon. The template is a [Django](https://www.djangoproject.com/start/) project - you can find a quick tutorial about Django and its best practices [here](https://docs.djangoproject.com/en/5.0/intro/tutorial01/).

## Common commands
```sh
# To start the server
$> python manage.py runserver

# To add a new application to the project
$> python manage.py startapp my_app

# To create migrations for your models
$> python manage.py makemigrations my_app

# To apply migrations to your database
$> python manage.py migrate
```

## The `main` module

The `main` module is the entrypoint of your Django project. The `main` module includes the `settings.py` file where you will find settings related to your Django project, notably the list of applications included in your project. You can read more about configuring applications in your Django project [here](https://docs.djangoproject.com/en/5.0/ref/applications/)

## The `shared` module

The `shared` module is a Django app containing boilerplate code that should enable your team to quickly prototype your application. The shared app includes a base template that we recommend extending in any of your other apps (learn more about extending templates in Django [here](https://docs.djangoproject.com/en/5.0/ref/templates/language/#template-inheritance)):
```html
<!-- my_app/templates/my_app/my-template.html -->

{% extends 'shared/base.html' %}

{% block content %}
<p>
  The HTML markup for your page should be included in the 'content' block
</p>
{% endblock %}

{% block styles %}
<style>
  // Any custom CSS you may need to write should be included in a style tag within the 'styles' block
</style>
{% endblock %}

{% block scripts %}
<script>
  // Any JavaScript you write should be included in the 'scripts' block
</script>
{% endblock %}
```
The base template includes [Bootstrap](https://getbootstrap.com/), a frontend toolkit, to allow you to quickly create polished UI elements. You can learn more about Bootstrap [here](https://getbootstrap.com/docs/5.3/getting-started/introduction/). (NOTE: The base template includes a maximal bundle of Bootstrap utilities that includes all necessary modules for supporting advanced features like tooltips and [Bootstrap icons](https://icons.getbootstrap.com/))
