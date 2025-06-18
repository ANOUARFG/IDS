# Utilise l'image officielle Airflow avec la version souhaitée
FROM apache/airflow:2.9.0-python3.11

# Passe en root pour installer des paquets système si besoin
USER root

RUN apt-get update && apt-get install -y build-essential libpq-dev

# Retour à l'utilisateur airflow
USER airflow

# Installe directement les dépendances Python via pip
RUN pip install --no-cache-dir \
    scikit-learn \
    matplotlib \
    pandas \
    numpy \
    seaborn \
    requests \
    h5py \
    apache-airflow-providers-postgres \
    apache-airflow-providers-redis \
    apache-airflow-providers-http \
    apache-airflow-providers-docker

# (Optionnel) Copier des plugins custom si nécessaire
# COPY plugins /opt/airflow/plugins

# Déclare le dossier des logs
VOLUME ["/opt/airflow/logs"]

# Point d'entrée laissé par défaut à celui de l'image Airflow
