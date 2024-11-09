from setuptools import setup, find_packages

setup(
    name='pagri_data_tools',  # Имя вашего пакета
    version='0.1',  # Версия вашего пакета
    packages=find_packages(),  # Найти все пакеты
    install_requires=[
        'dash==2.18.2'
        'ipython==8.27.0'
        'ipywidgets==8.1.5'
        'nbformat==5.10.4'
        'numpy==2.1.3'
        'pandas==2.2.3'
        'pingouin==0.5.5'
        'plotly==5.24.1'
        'pyaspeller==2.0.0'
        'pymystem3==0.2.0'
        'scikit_learn==1.5.2'
        'scipy==1.14.1'
        'statsmodels==0.14.4'
        'termcolor==2.5.0'
        'tqdm==4.67.0'
        ],
    description='This repository contains data analysis modules, including tools for data preprocessing, visualization, statistical analysis.',  # Описание
    long_description=open('README.md').read(),  # Долгое описание из README
    long_description_content_type='text/markdown',  # Тип контента
    author='Pavel Grigoryev',  # Ваше имя
    author_email='pagri.analytics@gmail.com',  # Ваш email
    url='https://github.com/PAGriAnalytics/pagri_analytics_modules',  # URL репозитория
)
