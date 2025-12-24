Project Documentation
=====================

This document describes the full setup, development workflow, automation,
authentication, and data model used in the Django CRM system.

-------------------------------------
Environment & Dependency Management
-------------------------------------

This project uses **UV** as the Python package manager.

Install UV:

.. code-block:: bash

   brew install uv

Add dependencies:

.. code-block:: bash

   uv add django
   uv add pip --dev

Create the Django project:

.. code-block:: bash

   mkdir src
   uv run django-admin startproject core src

-------------------------------------
Development Workflow
-------------------------------------

Makefile Commands:

.. code-block:: makefile

   .PHONY: run-server
   run-server:
       uv run python src/manage.py runserver 0.0.0.0:8000

   .PHONY: install
   install:
       uv install --no-root

   .PHONY: migrate
   migrate:
       uv run python src/manage.py migrate

   .PHONY: migrations
   migrations:
       uv run python src/manage.py makemigrations

   .PHONY: superuser
   superuser:
       uv run python src/manage.py createsuperuser

   .PHONY: install-pre-commit
   install-pre-commit:
       uv run pre-commit install

   .PHONY: lint
   lint:
       uv run pre-commit run --all-files

   .PHONY: update
   update: install install-pre-commit migrate

-------------------------------------
Pre-Commit Automation
-------------------------------------

This project uses pre-commit to enforce formatting, linting, and auto-export of
production dependencies.

Auto-export hook:

.. code-block:: yaml

   - repo: local
     hooks:
       - id: export-requirements
         name: Export prod requirements.txt
         entry: bash -c 'uv export --no-dev --no-hashes -o requirements.prod.txt && git add requirements.prod.txt'
         language: system
         pass_filenames: false
         files: ^(pyproject\.toml|uv\.lock)$
         always_run: false

-------------------------------------
Secrets & Environment Variables
-------------------------------------

This project uses **python-dotenv** to manage secrets.

Install:

.. code-block:: bash

   uv add python-dotenv

Load `.env` in `manage.py`:

.. code-block:: python

   from dotenv import load_dotenv
   load_dotenv()

Example `.env`:

.. code-block:: bash

   DJANGO_SECRET_KEY=your-secret-key
   GOOGLE_OAUTH_CLIENT_ID=xxxx
   GOOGLE_OAUTH_CLIENT_SECRET=xxxx

-------------------------------------
Authentication & Social Login
-------------------------------------

Google OAuth:

.. code-block:: bash

   uv add django-googler

Callback URL:

::

   http://127.0.0.1:8000/auth/google/callback/

LinkedIn OAuth:

.. code-block:: bash

   uv add django-allauth
   uv add "django-allauth[socialaccount]"

Follow setup:

- https://docs.allauth.org/en/latest/installation/quickstart.html
- https://docs.allauth.org/en/latest/socialaccount/providers/linkedin.html

-------------------------------------
CRM Data Model
-------------------------------------

Django uses the default user model:

.. code-block:: python

   AUTH_USER_MODEL = "auth.User"

The CRM defines a ``contacts`` model with ForeignKey relationships to users.

-------------------------------------
Result
-------------------------------------

This documentation describes:

• UV based dependency management
• Automated linting & formatting
• Automatic production requirements export
• Secure secret handling
• Google & LinkedIn OAuth
• CRM data modeling principles

This forms a solid production-ready Django CRM foundation.
